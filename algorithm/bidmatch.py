# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
import argparse
from algorithm.core.algorithmbase import AlgorithmBase
from algorithm.core.utils import ALGORITHMS
from algorithm.core.hooks import PseudoLabelingHook, DistAlignEMAHook


@ALGORITHMS.register('bidmatch')
class BidMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p)
    
    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        super().set_hooks()    

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label)


            self.class_entropy_t += self.cal_entropy(probs_x_ulb_w, probs_x_ulb_w)
            self.update_class()
            mask = self.bid_weight(probs_x_ulb_w)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            
            total_loss = sup_loss +  1.0 * unsup_loss
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         # semantic_loss = semantic_loss.item(),
                                         # class_loss = class_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint
    
    
    @torch.no_grad()
    def class_weight(self, probs_x_ulb_w):
        _, max_indices_w = torch.max(probs_x_ulb_w, dim=1)
        class_weight = self.class_entropy[max_indices_w]
        return class_weight
    
    
    # Compute the entropy for each class at every iteration
    @torch.no_grad()
    def cal_entropy(self, probs_x_ulb_w, probs_y_lb):
        entropy = -torch.sum(probs_y_lb * torch.log(torch.clamp(probs_x_ulb_w, 1e-9, 1)), dim=1) 
        max_indices = torch.argmax(probs_y_lb, dim=1)
        sum_entropy = torch.zeros(probs_x_ulb_w.size(1), device=probs_x_ulb_w.device) 
        count = torch.zeros(probs_x_ulb_w.size(1), device=probs_x_ulb_w.device) 
        sum_entropy.index_add_(0, max_indices, entropy)  
        count.index_add_(0, max_indices, torch.ones_like(entropy))  
        self.num_samples += count
        return sum_entropy

    @torch.no_grad()
    def update_sample(self, probs_x_ulb_w):
        if self.distributed and self.world_size > 1:
            probs_x_ulb_w = self.concat_all_gather(probs_x_ulb_w)
        max_probs, max_idx = probs_x_ulb_w.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu = self.m * self.prob_max_mu + (1 - self.m) * prob_max_mu_t.item()
        self.prob_max_var = self.m * self.prob_max_var + (1 - self.m) * prob_max_var_t.item()
        return 
    
    @torch.no_grad()
    def masking(self, probs_x_ulb_w):
        self.update_sample(probs_x_ulb_w)
        max_probs, max_idx = probs_x_ulb_w.max(dim=-1)
        # compute weight
        probs = (max_probs - self.prob_max_mu) / math.sqrt(self.prob_max_var)
        condition = probs < 0
        probs[condition] = probs[condition] * math.sqrt(2)
        probs[~condition] = probs[~condition] / math.sqrt(2)
        return probs
    
    @torch.no_grad()
    def bid_weight(self, probs_x_ulb_w):
        mask = self.masking(probs_x_ulb_w)
        condition = mask < 0
        weight = self.class_weight(probs_x_ulb_w)
        mask[condition] = torch.exp(mask[condition]) / torch.exp(weight[condition])
        mask[~condition] = torch.exp(mask[~condition]) * torch.exp(weight[~condition])        
        return mask
        
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, math.sqrt(2)),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
        ]


class SSL_Argument(object):
    """
    Algorithm specific argument
    """
    def __init__(self, name, type, default, help=''):
        """
        Model specific arguments should be added via this class.
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

name2alg = ALGORITHMS
def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')