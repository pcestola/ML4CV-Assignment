# Deep Learning Project

## Project Overview
This project implements a deep learning model for semantic segmentation of unexpected objects on roads.
The main components of the project include a Jupyter Notebook for execution and visualization, Python scripts for modularity, and pre-trained model weights for inference.

## Repository Structure
```
.
├── main.ipynb          # Main Jupyter Notebook for execution
├── src/                # Directory containing Python scripts
│   ├── data.py         # Dataset handling utility functions
│   ├── loss.py         # Loss functions
│   ├── test.py         # Test functions
│   ├── train.py        # Train functions
│   ├── utils.py        # Generic utility functions
├── network/            # Directory containing Python scripts
├── data/               # Folder for dataset
├── ckpts/              # Folder containing trained model weights (.pt files)
├── README.md           # This document
```
The `main.ipynb` file is designed to summarize the entire workflow, including training, testing, and the ablation study.

All custom functions and classes are contained within the `src` library. However, for clarity and ease of access, I have copied the most relevant functions directly into `main.ipynb`, ensuring that key parts of the implementation can be easily referenced without navigating through the library files. The remaining functions are imported from `src`.

For completeness, I have also included the `train.py` and `test.py` files, which were used to perform the training and testing processes. Their results are documented within `main.ipynb`.

Additionally, the results of the hyperparameter tuning and ablation study are stored in `tuning.csv` and `ablation.csv`, respectively.

## Setup
This project is designed to run on Google Colab.
1. Open `main.ipynb` in Colab and follow the execution cells.
2. The training code is disabled using `%%script echo skipping`. This ensures that training scripts are automatically skipped during execution, allowing the notebook to run smoothly.

## Model Weights
Pretrained model weights are stored in the weights/ directory as .pt files. Ensure the notebook loads the correct path to these files when running inference.
