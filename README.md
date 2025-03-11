# Semantic Segmentation with Anomaly Detection on StreetHazards
This project focuses on semantic segmentation with anomaly detection in autonomous driving scenes. It implements a DeepLabv3+ based model to segment known classes (like roads, cars, pedestrians, etc.) while detecting unknown objects or anomalies in the environment. The approach is evaluated on the StreetHazards dataset – a synthetic benchmark for anomaly segmentation – to identify unexpected objects in driving scenes. The goal is to achieve high segmentation accuracy on familiar classes and reliably flag unfamiliar objects by measuring the model’s prediction uncertainty.

## Installation and Requirements
To run this project, you will need the following environment and dependencies:
- Python 3.x (tested on Python 3.8+).
- PyTorch (Deep learning framework for the model. Version 1.8.0+ recommended for compatibility).
- Torchmetrics (for evaluating segmentation metrics like mIoU).
- Other Python packages: numpy, opencv-python (for image processing), matplotlib (for plotting results), and gdown or wget (for downloading data and weights).
- GPU: A CUDA-compatible GPU is highly recommended for training, due to the computational demands of the model.

## Project Structure
```
.
├── main.ipynb          # Main Jupyter Notebook for execution
├── lib/                # Directory containing custom functions and classes
│   ├── data.py         # Dataset handling utility functions
│   ├── train.py        # Train functions
│   ├── test.py         # Test functions
│   ├── utils.py        # Generic utility functions
├── network/            # Directory containing Python scripts from external repository
├── data/               # Folder for dataset
├── ckpts/              # Folder containing model weights and test results (.pt and .csv files)
│   ├── tuning.csv      # Hyperparameter tuning results
│   ├── ablation.csv    # Ablation study results
├── README.md           # This document
```
The `main.ipynb` file is designed to summarize the entire workflow, including training, testing, and the ablation study. All custom functions and classes used in tha main file are contained within the `lib` directory. The only external code used in this project is located in the network folder, sourced from this [repository](https://git01lab.cs.univie.ac.at/est-gan/deeplabv3plus-pytorch).

The results of the hyperparameter tuning and ablation study are stored respectively in `tuning.csv` and `ablation.csv` inside the `ckpts` directory.

## Setup
This project is designed to run on Google Colab.
1. Upload the entire project folder to your Google Drive.
2. Open the `main.ipynb` file with Google Colab.
3. The first cell in the notebook will prompt you to mount your Google Drive:
    
    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    ```

    Run this cell and follow the authentication steps.
4. Update the path to your project folder in the second cell by modifying the following line:  
    ```bash
    %cd /content/drive/MyDrive/YOUR_FOLDER_PATH
    ```
5. Execute the remaining cells as instructed in the notebook. All necessary steps are documented and implemented there, including:  
   - Installing dependencies (torchmetrics)
   - Downloading the dataset (11GB)
   - Downloading model weights (674MB)

The training code and all non-essential cells are disabled using `%%script echo skipping`. This prevents them from running automatically, ensuring the notebook executes smoothly. To enable these cells, simply remove the first line.

The notebook is structured into sections:
- **Setup**: mounting drive, setting paths, importing libraries, download weights.
- **Dataset**: downloading data.
- **Model**: building the model.
- **Training**: training loop for the model.
- **Test**: computing mIoU and AUPR, and comparing results.
- **Ablation Study**: ...

- A cell to download and extract the dataset using wget (this may take some time,
  and requires about 6 GB of space).
- A cell to download the pre-trained Cityscapes weights using gdown. These cells are
  included under the "Download model weights" and "Download" sections of the
  notebook. If you already manually put the files in the correct locations (as per
  the Installation steps), you can skip these or keep them commented out.

## Custom library
- `lib.train`: Contains training routines, model architectures, loss functions and
  utilities for logging and checkpointing during model training.

- `lib.test`: Provides testing utilities and evaluation metrics. This includes
  scoring functions (for AUPR) and functions to compute AUPR and mIoU.

- `lib.data`: Handles data loading and preprocessing for segmentation tasks. It
  includes a custom dataset class, synchronized data augmentation and transformations
  to prepare images/masks pairs ensuring consistency between the transformations
  applied.

- `lib.utils`: Offers helper functions for visualization and plotting, such as displaying segmentation results, plotting performance metrics and generating comparison bar plots.

## Expected Results
After successfully running the training and evaluation, you can expect the following performance from the model on the StreetHazards test set:
- **Segmentation Performance**: The model achieves around 65% mIoU on the known
  classes of the StreetHazards test set. This indicates that the segmentation quality
  for standard objects is high and on par with the performance of state-of-the-art
  models on this dataset. In practical terms, the model accurately segments most
  visible road scene objects (roads, vehicles, etc.) with good overlap against the
  ground truth.
- **Anomaly Detection Performance**: Using the entropy-based uncertainty measure,
  the model achieves an AUPR (Average Precision) of approximately 17.3% for anomaly
  detection.

## Extra
For completeness, I have included the original train.py and test.py files, which were used for the training and testing processes. However, these files are not strictly necessary, as everything required is already provided in the main file. To start training or testing, run the following commands:

Train: `python train.py --backbone resnet101 --head distance --dataset cityscapes --loss fl+h --gamma 0.1 --gamma_focal 2.0`

Test: `python test.py --file <train_file_name>.py --backbone resnet101 --head distance --dataset cityscapes`

The training script will generate the `/results` folder if it does not already exist. Inside this folder, it will create a `train_n.py` file, where n is a strictly increasing identifier (starting from 0) that distinguishes different runs.


https://chatgpt.com/c/67d09a12-1b70-800e-9179-0bc7fbd42fc1
