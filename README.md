# BidMatch

The official repository for **BidMatch: Boosting Semi-Supervised Learning by Bi-Dimensional Sample Weight Guidance**. This repository provides the implementation of the BidMatch algorithm, which is built upon the USB architecture ([USB](https://github.com/microsoft/Semi-supervised-learning)).

------

## Installation

### 1. Create a Virtual Environment

Use `conda` to set up the virtual environment named `bidmatch` for running the project:

```bash
conda create --name bidmatch python=3.8
```

### 2. Activate the Environment

Activate the virtual environment before proceeding:

```bash
conda activate bidmatch
```

### 3. Install Dependencies

Once the environment is active, install the required dependencies:

```bash
pip install -r requirements.txt
```

------

## Running the Code

### 1. Configure Parameters

Hyperparameters for training are pre-defined in the configuration files located in the `config` directory.

### 2. Train the Model

To train the model, activate the environment in the `bidmatch` directory:

```bash
conda activate bidmatch
```

Then run the training script with the desired configuration file:

```bash
python train.py --c config/bidmatch_cifar10_10.yaml
```

------

## Data

### Small Datasets

This repository supports five datasets for experiments: **CIFAR-10**, **CIFAR-100**, **STL-10**, **SVHN**, and **ImageNet-1K**.
 For the smaller datasets (**CIFAR-10**, **CIFAR-100**, **STL-10**, **SVHN**), the code will automatically download the required data during runtime. Ensure you have an active internet connection.

### Large Dataset

For the **ImageNet-1K** dataset, download it manually from [Hugging Face](https://huggingface.co/datasets/imagenet-1k) and place the data in the `data` directory of the repository before running the training script.

------

