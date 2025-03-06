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

## Setup
This project is designed to run on Google Colab.
1. Open `main.ipynb` in Colab and follow the execution cells.
2. The training code is disabled using `%%script echo skipping`. This ensures that training scripts are automatically skipped during execution, allowing the notebook to run smoothly.

## Model Weights
Pretrained model weights are stored in the weights/ directory as .pt files. Ensure the notebook loads the correct path to these files when running inference.
