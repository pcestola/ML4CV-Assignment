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


# region definizioni
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

class NormedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, scale=1.0, margin=0.0, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None):
        super(NormedConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=device, dtype=dtype)
        self.scale = scale
        self.margin = margin

    def forward(self, x):
        # Normalizza i pesi lungo la dimensione dei filtri (output channels)
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        # Normalizza l'input lungo la dimensione dei canali (dim=1)
        norm_x = F.normalize(x, p=2, dim=1)
        # Chiama il forward con input e pesi normalizzati
        return self.scale * F.conv2d(norm_x, norm_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PrototypeHead(nn.Module):
    def __init__(self, embed_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        nn.init.xavier_normal_(self.prototypes)
    
    def forward(self, x):
        B, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, D)

        dists = torch.cdist(x_flat, self.prototypes, p=2).pow(2)
        dists = dists.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return -dists

class PrototypeHeadMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: list, num_prototypes: int):
        super().__init__()
        
        # 1) Prototipi nello spazio originale (es. 256)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        
        # 2) MLP come serie di conv1x1 + ReLU
        #    (equivale a una MLP su ogni pixel in parallelo)
        layers = []
        in_channels = embed_dim
        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor shape (B, embed_dim, H, W)
        Ritorna: Tensor shape (B, num_prototypes, H, W) con i logits = -distanza
        """
        # 1) Trasforma le feature del backbone in uno spazio nascosto
        x = self.mlp(x)
        
        # 2) Trasforma i prototipi con la stessa MLP (f(w_c))
        proto = self.prototypes.unsqueeze(-1).unsqueeze(-1)
        proto = self.mlp(proto)
        proto = proto.view(proto.size(0), -1)
        
        # 3) Distanza
        B, D2, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, D2)
        
        dists = torch.cdist(x_flat, proto, p=2).pow(2)
        dists = dists.view(B, H, W, -1).permute(0, 3, 1, 2)

        return -dists
    
class PrototypeHeadMLP2(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: list, num_prototypes: int):
        super().__init__()
        
        # 1) Prototipi nello spazio originale (es. 256)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dims[-1]))
        
        # 2) MLP come serie di conv1x1 + ReLU
        #    (equivale a una MLP su ogni pixel in parallelo)
        layers = []
        in_channels = embed_dim
        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor shape (B, embed_dim, H, W)
        Ritorna: Tensor shape (B, num_prototypes, H, W) con i logits = -distanza
        """
        # 1) Trasforma le feature del backbone in uno spazio nascosto
        x = self.mlp(x)
        
        # 3) Distanza
        B, D2, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, D2)
        
        dists = torch.cdist(x_flat, self.prototypes, p=2).pow(2)
        dists = dists.view(B, H, W, -1).permute(0, 3, 1, 2)

        return -dists
    
class PrototypeHeadArcFace(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: list, num_prototypes: int):
        super().__init__()
        
        # 1) Prototipi nello spazio finale
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        
        # 2) MLP per trasformare le feature (equivalente a una MLP per pixel)
        layers = []
        in_channels = embed_dim
        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        
        # 1) Trasforma le feature con la MLP
        x = self.mlp(x)
        
        # 2) Normalizziamo le feature pixel-wise
        B, D2, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, D2)
        x_flat = F.normalize(x_flat, p=2, dim=1)

        # 3) Trasforma i prototipi con la stessa MLP (f(w_c))
        proto = self.prototypes.unsqueeze(-1).unsqueeze(-1)
        proto = self.mlp(proto)
        proto = proto.view(proto.size(0), -1)

        # 4) Normalizziamo i prototipi
        proto = F.normalize(proto, p=2, dim=1)
        
        # 5) Prodotto scalare normalizzato tra feature e prototipi
        logits = torch.mm(x_flat, proto.T)

        # 6) Reshape a (B, num_prototypes, H, W)
        logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  
        
        return logits

class PrototypeHeadArcFace2(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: list, num_prototypes: int):
        super().__init__()
        
        # 1) Prototipi nello spazio finale
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dims[-1]))
        
        # 2) MLP per trasformare le feature (equivalente a una MLP per pixel)
        layers = []
        in_channels = embed_dim
        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        
        # 1) Trasforma le feature con la MLP
        x = self.mlp(x)
        
        # 2) Normalizziamo le feature pixel-wise
        B, D2, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, D2)
        x_flat = F.normalize(x_flat, p=2, dim=1)

        # 3) Normalizziamo i prototipi
        proto = F.normalize(self.prototypes, p=2, dim=1)
        
        # 5) Prodotto scalare normalizzato tra feature e prototipi
        logits = torch.mm(x_flat, proto.T)

        # 6) Reshape a (B, num_prototypes, H, W)
        logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  
        
        return logits


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
def msp_score(logits:torch.Tensor):
    score, _ = torch.max(torch.softmax(logits,dim=1),dim=1)
    return -score

def max_logit_score(logits:torch.Tensor):
    score, _ = torch.max(logits,dim=1)
    return -score

def entropy_score(logits:torch.Tensor):
    probs = torch.softmax(logits,dim=1).clamp(min=1e-12)
    return - torch.sum(probs * torch.log(probs), dim=1)

def energy_score(logits:torch.Tensor):
    return -torch.logsumexp(logits, dim=1)


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

def calculate_miou(dataloader, model, device):
    # Sposta il modello su eval mode
    model.eval()

    # Inizializza la metrica
    eval_IoU = JaccardIndex(
        task='multiclass',
        num_classes=14,
        ignore_index=13,
        average="none",
        zero_division=0
    ).to(device)

    with torch.no_grad():
        for image, mask in dataloader:
            
            # Prepara immagine
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Calcola le predizioni del modello
            logits = model(image)
            predicted_classes = torch.argmax(logits, dim=1)

            # Aggiorna la metrica IoU
            eval_IoU.update(predicted_classes, mask)

    # Calcola il valore finale di mIoU
    return eval_IoU.compute()[:-1]

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

# Funzione per scrivere aggiornamenti nel log
def log_training_progress(logger:logging.Logger, epoch:int, loss:float, val_loss:float) -> None:
    lr_list = [group['lr'] for group in logger.optimizer.param_groups]
    lr_list = f"LR: {', '.join(f'{lr:.6f}' for lr in lr_list)}"
    logger.info(f"Epoch: {epoch:3d} - Loss: {loss:10.6f} - Val Loss: {val_loss:10.6f} - " + lr_list)

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
# endregion

if __name__ == '__main__':
    
    fix_random(0)
    
    # region preparazione
    parser = argparse.ArgumentParser(description="Accetta valori dalla linea di comando")
    
    # Aggiungi argomenti
    parser.add_argument('--device', type=str, default = find_best_device())
    parser.add_argument('--folder', type=int)
    parser.add_argument('--file', type=str)
    parser.add_argument('--head', type=int)

    # Parsing degli argomenti
    args = parser.parse_args()

    # Creo il logger
    results_dir = f'/raid/homespace/piecestola/space/ML4CV/results' #_{args.folder}'
    test_dir = os.path.join(results_dir, args.file)
    ckpt_dir = os.path.join(test_dir, 'ckpts')
    logger = setup_logger(test_dir, 'test.log')

    # Scrivo i parametri
    for name, value in args.__dict__.items():
        logger.info(f"{name:<{15}} : {value}")
    logger.info("end\n")
    # endregion
    

    # region dataset
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
    batch_size = 8

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
    # endregion


    # region model
    # Carico il modello
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)
    if args.head == 0:
        head = PrototypeHead(embed_dim=256, num_prototypes=13)
    elif args.head == 1:
        head = PrototypeHeadMLP(embed_dim=256, hidden_dims=[128], num_prototypes=13)
    elif args.head == 2:
        head = PrototypeHeadMLP2(embed_dim=256, hidden_dims=[128], num_prototypes=13)
    elif args.head == 3:
        head = PrototypeHeadArcFace(embed_dim=256, hidden_dims=[128], num_prototypes=13)
    elif args.head == 4:
        head = PrototypeHeadArcFace2(embed_dim=256, hidden_dims=[128], num_prototypes=13)
    model.classifier.classifier[3] = head

    # Carico i pesi
    weight_path = find_highest_file(ckpt_dir,'weights_mIoU_')
    model.load_state_dict(torch.load(weight_path, map_location=args.device)['model_state_dict'])

    # Impostazioni del modello
    model.eval()
    model.to(args.device)
    logger.info(f'Caricati i pesi: {weight_path}')
    logger.info(f"Modello caricato correttamente su {args.device}\n")
    # endregion

    
    # region test-t5
    logger.info("Testing t5 dataset")
    msp_aupr = calculate_aupr(dataloader_test, model, msp_score, args.device)
    logger.info(f'MSP AUPR:\t{msp_aupr}')

    max_logit_aupr = calculate_aupr(dataloader_test, model, max_logit_score, args.device)
    logger.info(f'MAXLOG AUPR:\t{max_logit_aupr}')

    entropy_aupr = calculate_aupr(dataloader_test, model, entropy_score, args.device)
    logger.info(f'ENTROPY AUPR:\t{entropy_aupr}')

    energy_aupr = calculate_aupr(dataloader_test, model, energy_score, args.device)
    logger.info(f'ENERGY AUPR:\t{energy_aupr}\n')
    
    miou = calculate_miou(dataloader_test, model, args.device)
    stringa = 'IoU per class\n'
    for i, x in enumerate(miou):    stringa += f'classe {i}: {x}\n'
    logger.info(stringa)
    logger.info(f'mIoU: {miou.mean()}')
    logger.info(f'mIoU_no_zero: {miou[miou!=0].mean()}')
    logger.info(f'minIoU: {miou.min()}')
    logger.info(f'maxIoU: {miou.max()}\n\n')
    # endregion
    
    
    # region test-t6
    logger.info("Testing t6 dataset")
    msp_aupr = calculate_aupr(dataloader_test_anomaly, model, msp_score, args.device)
    logger.info(f'MSP AUPR:\t{msp_aupr}')

    max_logit_aupr = calculate_aupr(dataloader_test_anomaly, model, max_logit_score, args.device)
    logger.info(f'MAXLOG AUPR:\t{max_logit_aupr}')

    entropy_aupr = calculate_aupr(dataloader_test_anomaly, model, entropy_score, args.device)
    logger.info(f'ENTROPY AUPR:\t{entropy_aupr}')
    
    energy_aupr = calculate_aupr(dataloader_test_anomaly, model, energy_score, args.device)
    logger.info(f'ENERGY AUPR:\t{energy_aupr}\n')
    
    miou = calculate_miou(dataloader_test_anomaly, model, args.device)
    stringa = 'IoU per class\n'
    for i, x in enumerate(miou):    stringa += f'classe {i}: {x}\n'
    logger.info(stringa)
    logger.info(f'mIoU: {miou.mean()}')
    logger.info(f'mIoU_no_zero: {miou[miou!=0].mean()}')
    logger.info(f'minIoU: {miou.min()}')
    logger.info(f'maxIoU: {miou.max()}\n\n')
    # endregion

    # region final
    del model
    gc.collect()
    # endregion