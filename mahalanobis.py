import re
import os
import gc
import sys
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import deeplab.network as network
import torchvision.transforms.functional as FUNC

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex, AveragePrecision

# https://github.com/CVLAB-Unibo/ml4cv-assignment/tree/master


# ====== UTILS ======
def fix_random(seed: int) -> None:
    """Fix all possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ====== Dataset e Dataloader ======
class ToTensorSync:
    def __call__(self, img, mask):
        # Converti l'immagine in un tensore normalizzato
        img_tensor = FUNC.to_tensor(img)
        
        # Converti la maschera in un tensore grezzo senza normalizzazione
        mask_tensor = FUNC.pil_to_tensor(mask)
        
        return img_tensor, mask_tensor
    
class NormalizeSync:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = FUNC.normalize(img, mean=self.mean, std=self.std)
        return img, mask
        
class ComposeSync:
    """
    Composizione di trasformazioni che accettano immagine e maschera.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask
    
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = mask.squeeze(0)
        mask = mask.type(torch.int64)-1

        return image, mask


# ====== TEST ======
def evaluate_means(model: nn.Module, dataloader: DataLoader, num_classes: int):

    device = next(model.parameters()).device

    # TODO: hard coded, il 256 va calcolato
    means = [torch.zeros(256,device='cpu') for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]

    model.eval()

    with torch.no_grad():
        L = len(dataloader)
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            features = model(images).permute((1, 0, 2, 3))  #(c,b,h,w)
            
            del images

            for c in range(num_classes):
                mask_indices = (masks == c)
                
                if mask_indices.any():
                    means[c] += features[:, mask_indices].sum(dim=1).cpu()
                    count[c] += mask_indices.sum().item()

    # Calcola la media finale
    final_means = [
        mean / c if c > 0 else torch.zeros_like(mean)
        for c, mean in zip(count, means)
    ]

    return final_means

def evaluate_covariance(model: nn.Module, dataloader: DataLoader, means:list, num_classes: int):

    device = next(model.parameters()).device

    # TODO: hard coded, il 256 va calcolato
    for i in range(num_classes):
        means[i] = means[i].unsqueeze(dim=1).to(device)
    variance = torch.zeros((256,256),device='cpu')
    count = 0

    model.eval()

    with torch.no_grad():
        L = len(dataloader)
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            features = model(images).permute((1, 0, 2, 3))  # (c, b, h, w)

            del images

            for c in range(num_classes):
                mask_indices = (masks == c)

                if mask_indices.any():
                    centered_features = features[:, mask_indices] - means[c]
                    
                    del mask_indices
                    
                    variance += (centered_features @ centered_features.T).cpu()

                    # Aggiorna il conteggio
                    count += centered_features.shape[1]

            #print(f'{(i + 1) * 100 / L:.2f}%', end='\r')

    # Normalizza la covarianza finale
    variance /= count if count > 0 else 1

    for i in range(num_classes):
        means[i] = means[i].squeeze().to('cpu')

    return variance

def mahalanobis_score(features: torch.Tensor, block_size: int = 1000):
    global mean, cov
    _, c, _, _ = features.shape
    
    # Flatten features to (b * h * w, c)
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
    
    # Result container
    num_features = features_flat.shape[0]
    scores = torch.empty(num_features, device=features.device)
    
    # Process in blocks
    for start in range(0, num_features, block_size):
        end = min(start + block_size, num_features)
        
        # Slice the current block
        block = features_flat[start:end]  # (block_size, c)
        
        # Compute the centered features (block_size, n, c)
        block_centered = block.unsqueeze(1) - mean.unsqueeze(0)  # Broadcasting
        
        # Compute cov multiplication (block_size, n, c)
        block_cov = torch.einsum('ij,bnc->bni', cov, block_centered)
        
        # Compute Mahalanobis score (block_size, n)
        block_result = torch.einsum('bnc,bnc->bn', block_centered, block_cov)
        
        # Take the maximum score for this block
        scores[start:end] = torch.max(-block_result, dim=1)[0]

    return scores


def calculate_aupr(dataloader, model, score_function, device):
    # Sposta il modello su eval mode
    model.eval()

    # Inizializza la metrica
    eval_AUPR = AveragePrecision(task='binary')

    with torch.no_grad():
        for image, mask in dataloader:

            # Prepara immagine
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Calcola le feature utilizzando il modello
            logits = model(image)

            # Applica la funzione di score
            score = score_function(logits).view(-1).cpu()

            # Prepara la maschera come etichetta binaria
            label = (mask == 13).to(torch.int32).view(-1).cpu()

            # Aggiorna la metrica AUPR
            eval_AUPR.update(score, label)

    # Calcola il valore finale di AUPR
    return eval_AUPR.compute()

# ====== Utils ======
def find_next_folder_name(parent_directory, prefix="folder_c_"):
    # Pattern per estrarre il numero dai nomi delle cartelle
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    
    max_ci = -1  # Inizializza il massimo a -1 (nel caso non ci siano cartelle)
    
    # Scansiona le sottocartelle nella directory
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        if os.path.isdir(item_path):  # Controlla che sia una cartella
            match = pattern.match(item)
            if match:
                ci = int(match.group(1))  # Estrai il numero c_i
                max_ci = max(max_ci, ci)
    
    # Genera il nuovo nome della cartella e creala
    next_ci = max_ci + 1
    new_folder_name = f"{prefix}{next_ci}"
    return os.path.join(parent_directory, new_folder_name)

def find_highest_file(parent_directory, prefix="file_c_", next=False):
    # Pattern per estrarre il numero dai nomi dei file
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    
    max_ci = -1  # Inizializza il massimo a -1 (nel caso non ci siano file)
    highest_file = None  # Per salvare il file con il numero più alto
    
    # Scansiona i file nella directory
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        if os.path.isfile(item_path):  # Controlla che sia un file
            match = pattern.match(item)
            if match:
                ci = int(match.group(1))  # Estrai il numero c_i
                if ci > max_ci:
                    max_ci = ci
                    highest_file = item_path  # Aggiorna il file con il numero più alto
    
    if next:
        # Calcola il nome del file con c+1
        next_file_name = f"{prefix}{max_ci + 1}"
        next_file_path = os.path.join(parent_directory, next_file_name)
        return highest_file, next_file_path

    return highest_file

def find_best_device() -> str:
    if torch.cuda.is_available():   
        # Esegui il comando nvidia-smi per ottenere le informazioni sulla memoria delle GPU
        result = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader').read()
        # Analizza l'output per ottenere la memoria libera per ogni GPU
        memory_free = [int(x) for x in result.strip().split('\n')]
        # Pedicini - evito la gpu 2
        memory_free[2] = -1
        # Trova l'indice della GPU con più memoria libera
        gpu = memory_free.index(max(memory_free))
        gpu = f'cuda:{gpu}'
    else:
        gpu = 'cpu'
    return gpu

# Funzione per configurare il file di log
def setup_logger(log_dir:str, log_file:str) -> logging.Logger:
    
    # Percorso completo del file di log
    log_path = os.path.join(log_dir, log_file)

    # Configura il logger
    logging.basicConfig(
        level=logging.INFO,
        #format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("TrainingLogger")

def setup_train_folder(dir:str, resume:int = None) -> Tuple[str,str,logging.Logger]:
    
    # Creo o recupero la cartella
    if resume:
        path = os.path.join(dir,f'train_{resume}')
    else:
        path = find_next_folder_name(dir, 'train_')
        os.makedirs(path, exist_ok=True)
    
    # Creo il logger
    logger = setup_logger(path,'train.log')

    # Creo la ckpts folder
    ckpts_dir = os.path.join(path,'ckpts')
    os.makedirs(ckpts_dir, exist_ok=True)

    return path, ckpts_dir, logger
    

if __name__ == '__main__':
    
    fix_random(0)

    '''
        PREPARAZIONE
    '''
    parser = argparse.ArgumentParser(description="Accetta valori dalla linea di comando")
    
    # Aggiungi argomenti
    parser.add_argument('--device', type=str, default = find_best_device())
    parser.add_argument('--folder', type=int)
    parser.add_argument('--file', type=str)
    parser.add_argument('--test', action='store_true')

    # Parsing degli argomenti
    args = parser.parse_args()

    # Creo il logger
    results_dir = f'/home/piecestola/space/ML4CV/results_{args.folder}'
    test_dir = os.path.join(results_dir, args.file)
    ckpt_dir = os.path.join(test_dir, 'ckpts')
    logger = setup_logger(test_dir, 'mahalanobis.log')

    if not args.test:
        
        for name, value in args.__dict__.items():
            logger.info(f"{name:<{15}} : {value}")
        logger.info("end\n")

        '''
            DATASET
        '''
        path_train = './data/train'
        path_train_images =  path_train + '/images/training/t1-3'
        path_valid_images =  path_train + '/images/validation/t4'
        path_train_masks  =  path_train + '/annotations/training/t1-3'
        path_valid_masks  =  path_train + '/annotations/validation/t4'

        train_transforms = ComposeSync([
                ToTensorSync(),
                NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Training dataset
        dataset_train = SegmentationDataset(
            images_dir = path_train_images,
            masks_dir = path_train_masks,
            transforms = train_transforms
        )

        logger.info("Dataset Creato")
        logger.info(f"Training images: {len(dataset_train)}")

        # Dataloader
        num_workers = 2
        train_batch_size = 4

        dataloader_train = DataLoader(dataset_train,
                                    batch_size=train_batch_size,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    drop_last=True)

        logger.info("Dataloader Creato")
        logger.info(f"Training batches: {len(dataloader_train)}")

        '''
            MODELLO
        '''

        # Carico il modello
        weight_path = find_highest_file(ckpt_dir,'weights_mIoU_')
        #model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
        model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)

        model.classifier.classifier[3] = nn.Identity()
        
        state_dict = torch.load(weight_path,map_location=args.device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'],strict=False)
        else:
            model.load_state_dict(state_dict,strict=False)

        model.eval()
        model.to(args.device)
        logger.info(f'Caricati i pesi: {weight_path}')
        logger.info(f"Modello caricato correttamente su {args.device}\n")
        
        '''
            CALCOLO DELLA DISTANZA
        '''
        means = evaluate_means(model, dataloader_train, 13)
        covariance = evaluate_covariance(model, dataloader_train, means, 13)
        
        torch.save({'means': means, 'covariance': covariance}, os.path.join(test_dir,'mahalanobis.pth'))
    else:
        path_test = './data/test'
        path_test_images            =  path_test + '/images/test/t5'
        path_test_masks             =  path_test + '/annotations/test/t5'
        path_test_anomaly_images    =  path_test + '/images/test/t6'
        path_test_anomaly_masks     =  path_test + '/annotations/test/t6'

        # Trasformazioni
        transforms = ComposeSync([
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Training dataset
        dataset_test = SegmentationDataset(
            images_dir = path_test_images,
            masks_dir = path_test_masks,
            transforms = transforms
        )

        # Validation dataset
        dataset_test_anomaly = SegmentationDataset(
            images_dir = path_test_anomaly_images,
            masks_dir = path_test_anomaly_masks,
            transforms = transforms
        )

        logger.info("Dataset Creati")
        logger.info(f"Test images: {len(dataset_test)}")
        logger.info(f"Anomalies images: {len(dataset_test_anomaly)}\n")

        # Dataloader
        num_workers = 2
        batch_size = 4

        dataloader_test = DataLoader(dataset_test,
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    drop_last=True)

        dataloader_test_anomaly = DataLoader(dataset_test_anomaly,
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    drop_last=True)

        logger.info("Dataloader Creati")
        logger.info(f"Test batches: {len(dataloader_test)}")
        logger.info(f"Anomalies batches: {len(dataloader_test_anomaly)}\n")
        
        # Carico il modello
        weight_path = find_highest_file(ckpt_dir,'weights_mIoU_')
        #model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
        model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)

        model.classifier.classifier[3] = nn.Identity()
        
        state_dict = torch.load(weight_path,map_location=args.device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'],strict=False)
        else:
            model.load_state_dict(state_dict,strict=False)

        model.eval()
        model.to(args.device)
        logger.info(f'Caricati i pesi: {weight_path}')
        logger.info(f"Modello caricato correttamente su {args.device}\n")
        
        '''
            CALCOLO DELLA DISTANZA
        '''
        data = torch.load(os.path.join(test_dir,'mahalanobis.pth'),map_location=args.device)
        mean = torch.stack(data['means'])
        cov = torch.linalg.inv(data['covariance'])
        
        aupr = calculate_aupr(dataloader_test,model,mahalanobis_score,args.device)
        logger.info(f'AUPR t5: {aupr}')

        aupr = calculate_aupr(dataloader_test_anomaly,model,mahalanobis_score,args.device)
        logger.info(f'AUPR t6: {aupr}')

    '''
        FINAL
    '''
    del model
    gc.collect()

''' TODO
- trova il modo di assicurarti che la distanza di mahalanobis sia calcolate bene
- aggiungi il mahalanobis score a test.py
- aggiorna il file excel con le metriche
- inizia a scrivere qualcosa da presentare
- inventa qualcosa
'''