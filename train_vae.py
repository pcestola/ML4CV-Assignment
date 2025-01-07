import re
import os
import gc
import sys
import pickle
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import deeplab.network as network
import torchvision.transforms.functional as FUNC

from PIL import Image
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop
from typing import Tuple

# https://github.com/CVLAB-Unibo/ml4cv-assignment/tree/master
# https://github.com/cc-ai/Deeplabv3?tab=readme-ov-file
# https://github.com/kerrgarr/SemanticSegmentationCityscapes?tab=readme-ov-file

'''
    HO MODIFICATO
    /raid/homespace/piecestola/ML4CV/deeplab/network/utils.py

    RIGA 13
'''

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

def get_allocated_memory(device):
    # Memoria attualmente utilizzata
    used_memory = torch.cuda.memory_allocated(device)

    # Memoria totale disponibile
    total_memory = torch.cuda.get_device_properties(device).total_memory

    # Percentuale di memoria utilizzata
    return (used_memory / total_memory) * 100

# region DATASET
class RandomCropSync:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        # Ottieni i parametri per il ritaglio casuale
        i, j, h, w = RandomCrop.get_params(img, output_size=self.size)
        # Applica il ritaglio sia all'immagine che alla maschera
        img = FUNC.crop(img, i, j, h, w)
        mask = FUNC.crop(mask, i, j, h, w)
        return img, mask

class RandomCropResizeSync:
    def __init__(self, size, min_crop_size=(128, 128), aspect_ratio_range=(0.5, 2.0)):
        """
        Inizializza il Random Crop con controlli sulle dimensioni minime e il range di aspect ratio,
        seguito da un Resize fisso.
        :param size: Tuple con la dimensione finale del resize (width, height).
        :param min_crop_size: Tuple con le dimensioni minime del crop (min_width, min_height).
        :param aspect_ratio_range: Tuple con il range valido di rapporto (width / height).
        """
        self.size = size
        self.min_crop_size = min_crop_size
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img, mask):
        # Ottieni dimensioni originali dell'immagine
        img_width, img_height = img.size
        min_width, min_height = self.min_crop_size
        min_aspect, max_aspect = self.aspect_ratio_range

        # TODO: sarebbe meglio togliere il for
        # Genera dimensioni valide per il crop
        for _ in range(10):  # Tentativi per trovare un crop valido
            crop_width = random.randint(min_width, img_width)
            crop_height = random.randint(min_height, img_height)

            # Calcola il rapporto tra larghezza e altezza
            aspect_ratio_min = min(crop_width / crop_height, crop_height / crop_width)
            aspect_ratio_max = 1 / aspect_ratio_min

            # Verifica che il rapporto sia nel range specificato
            if min_aspect <= aspect_ratio_min and aspect_ratio_max <= max_aspect:
                break
        else:
            # Se non trova un crop valido, usa l'intera immagine
            crop_width, crop_height = img_width, img_height

        # Genera coordinate valide per il crop
        i = random.randint(0, img_height - crop_height)
        j = random.randint(0, img_width - crop_width)

        # Applica il crop casuale all'immagine e alla maschera
        img = FUNC.crop(img, i, j, crop_height, crop_width)
        mask = FUNC.crop(mask, i, j, crop_height, crop_width)

        # Applica il resize fisso all'immagine e alla maschera
        img = FUNC.resize(img, self.size)
        mask = FUNC.resize(mask, self.size, interpolation=FUNC.InterpolationMode.NEAREST)

        return img, mask

class RandomHorizontalFlipSync:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = FUNC.hflip(img)
            mask = FUNC.hflip(mask)
        return img, mask
    
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

class RandomResizeSync:
    def __init__(self, scale_range=(0.5, 1.5)):
        """
        Inizializza il resize dinamico sincronizzato.
        :param scale_range: Tuple contenente il range di scaling (min_scale, max_scale).
        """
        self.scale_range = scale_range

    def __call__(self, img, mask):
        # Scegli un fattore di scala casuale nell'intervallo specificato
        scale = random.uniform(*self.scale_range)
        
        # Calcola la nuova dimensione dell'immagine e della maschera
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Applica il resize sia all'immagine che alla maschera
        img = FUNC.resize(img, (new_height, new_width))
        mask = FUNC.resize(mask, (new_height, new_width), interpolation=FUNC.InterpolationMode.NEAREST)
        
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

        mask = mask.squeeze()
        mask = mask.type(torch.int64)-1

        return image, mask
# endregion

# region TRAINING
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        input_dim: Numero di canali in input (C).
        latent_dim: Dimensione dello spazio latente per pixel.
        """
        super(VAE, self).__init__()

        # Encoder: Riduce progressivamente i canali tramite convoluzioni 1x1
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Latent Space: Calcola \mu e \log\sigma^2
        self.fc_mu = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)  # Media
        self.fc_logvar = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)  # Log-varianza

        # Decoder: Ricostruisce progressivamente i canali tramite convoluzioni 1x1
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_dim, kernel_size=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# TODO: controllare se è corretto
# TODO: modificare il codice di test per aggiungere la VAE
class VAELoss(nn.modules.loss._Loss):
    def __init__(self, beta=1.0, reduction='mean'):
        """
        Classe per la perdita della VAE con reshaping dei tensori per trattare (B, C, H, W)
        come (C, BHW).

        Args:
            beta: Peso del termine KL divergence.
            reduction: Tipo di riduzione ('mean', 'sum', 'none').
        """
        super(VAELoss, self).__init__(reduction=reduction)
        self.beta = beta

    def forward(self, reconstructed, original, mu, logvar):
        """
        Calcola la perdita totale.

        Args:
            reconstructed: Feature ricostruite (output del decoder), shape (B, C, H, W).
            original: Feature originali (input del VAE), shape (B, C, H, W).
            mu: Media (\(\mu\)) della distribuzione latente, shape (B, Z, H, W).
            logvar: Logaritmo della varianza (\(\log\sigma^2\)) della distribuzione latente, shape (B, Z, H, W).

        Returns:
            Perdita totale (\(\mathcal{L}_{reconstruction} + \beta \cdot \mathcal{L}_{KL}\)).
        """
        # Ridimensiona (B, C, H, W) a (C, BHW)
        B, C, _, _ = original.shape
        original_reshaped = original.view(B, C, -1)  # (B, C, H * W)
        reconstructed_reshaped = reconstructed.view(B, C, -1)  # (B, C, H * W)

        # Reconstruction Loss (MSE su (C, BHW))
        reconstruction_loss = F.mse_loss(reconstructed_reshaped, original_reshaped, reduction='none')  # (B, C, H * W)
        reconstruction_loss = reconstruction_loss.mean(dim=[1, 2])  # Media su (C, H * W)

        if self.reduction == 'mean':
            reconstruction_loss = reconstruction_loss.mean()  # Media su tutti i batch
        elif self.reduction == 'sum':
            reconstruction_loss = reconstruction_loss.sum()  # Somma su tutti i batch

        # KL Divergence Loss (su ogni pixel vettore)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # Somma lungo Z (feature latenti)
        kl_loss = kl_loss.view(B, -1).mean(dim=1)  # Media su H * W

        if self.reduction == 'mean':
            kl_loss = kl_loss.mean()  # Media su tutti i batch
        elif self.reduction == 'sum':
            kl_loss = kl_loss.sum()  # Somma su tutti i batch

        # Loss Totale
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss

def train_epoch(model: nn.Module,
                vae:nn.Module,
                loader_train: utils.data.DataLoader,
                device: torch.device,
                optimizer: torch.optim,
                criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    
    loss_train = 0

    vae.train()
    for images, _ in loader_train:

        # Trasferisci tensori al device
        images = images.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # Calcolo della forward pass
        features = model(images)
        del images
        
        reconstructed, mu, logvar = vae(features)
        #torch.Size([8, 256, 512, 512])
        #torch.Size([8, 256, 512, 512])
        #torch.Size([8, 64, 512, 512])
        #torch.Size([8, 64, 512, 512])
        
        loss = criterion(reconstructed, features, mu, logvar)
        del reconstructed, features, mu, logvar
        loss.backward()

        optimizer.step()

        loss_train += loss.item()
        del loss

        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

    loss_train /= len(dataloader_train)
    return loss_train

def validate(model: nn.Module,
             vae:nn.Module,
             loader_val: utils.data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    
    loss_val = 0

    vae.eval()
    with torch.no_grad():
        for images, _ in loader_val:
            images = images.to(device)
            
            features = model(images)
            del images

            reconstructed, mu, logvar = vae(features)

            loss = criterion(reconstructed, features, mu, logvar)
            del reconstructed, features, mu, logvar

            loss_val += loss.item()

    loss_val /= len(loader_val)
    return loss_val

def training_loop(num_epochs: int,
                  optimizer: torch.optim,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  model: nn.Module,
                  vae:nn.Module,
                  criterion: nn.Module,
                  loader_train: utils.data.DataLoader,
                  loader_val: utils.data.DataLoader,
                  device: torch.device,
                  run_name: str,
                  path_ckpts:str,
                  logger) -> None:
    
    path_ckpts = Path(path_ckpts)
    path_ckpts.mkdir(parents=True, exist_ok=True)

    losses_train = list()
    losses_valid = list()

    for epoch in range(num_epochs):
        loss_train = train_epoch(model,
                                 vae,
                                 loader_train,
                                 device,
                                 optimizer,
                                 criterion)

        loss_val = validate(model,
                            vae,
                            loader_val,
                            device,
                            criterion)

        if epoch % 15 == 0:
            logger.info(f"{'Epoch':8} {'Loss':10} {'Val Loss':10} {'LR':8}")
        
        log_training_progress(optimizer, logger, epoch, loss_train, loss_val)
        losses_train.append(loss_train)
        losses_valid.append(loss_val)

        # Update learning rate
        scheduler.step(loss_val)

        # Save model with best validation loss
        flag = False
        if epoch == 0:
            flag = True
            loss_val_best = loss_val
        elif loss_val < loss_val_best:
            flag = True
            loss_val_best = loss_val
        
        if flag:
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss_val_best': loss_val_best,
            }, path_ckpts / f"{run_name}.pt")
    
    return losses_train, losses_valid

def train(model: nn.Module,
          vae:nn.Module,
          criterion: nn.Module,
          optimizer,
          epochs: int,
          data_loader_train: torch.utils.data.DataLoader,
          data_loader_val: torch.utils.data.DataLoader,
          device: torch.device,
          run_name: str,
          path_ckpts:str,
          logger:logging.Logger) -> None:

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    logger.info("Inizio del training")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"loss:      {criterion}")
    logger.info(f"epochs:    {epochs}")
    logger.info(f"name:      {run_name}\n")

    a, b = training_loop(epochs,
                        optimizer,
                        scheduler,
                        model,
                        vae,
                        criterion,
                        data_loader_train,
                        data_loader_val,
                        device,
                        run_name,
                        path_ckpts,
                        logger)
    
    logger.info("Fine del training\n")

    return a, b
# endregion

# region MODELS
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
# endregion

# region UTILS
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

def log_training_progress(optimizer:torch.optim.Optimizer, logger:logging.Logger, epoch:int, loss:float, val_loss:float) -> None:
    lr_list = [group['lr'] for group in optimizer.param_groups]
    lr_list = f"{', '.join(f'{lr:<8.1e}' for lr in lr_list)}"
    logger.info(f"{epoch:<8d} {loss:<10.6f} {val_loss:<10.6f}" + lr_list)

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

    '''
        PREPARAZIONE
    '''
    parser = argparse.ArgumentParser(description="Accetta valori dalla linea di comando")
    
    # Aggiungi argomenti
    parser.add_argument('--device', type=str, default = find_best_device())
    parser.add_argument('--model', type=str, default = 'mobilenet')
    parser.add_argument('--folder', type=int)
    parser.add_argument('--train', type=int)
    
    # Parametri addestramento
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam')

    # Parsing degli argomenti
    args = parser.parse_args()

    # Creo la directory ed il logger
    results_dir = f'/home/piecestola/space/ML4CV/results_{args.folder}'
    train_dir = results_dir + f'/train_{args.train}'
    ckpts_dir = train_dir + '/ckpts'
    logger = setup_logger(train_dir, 'vae.log')

    # Scrivo i parametri
    for name, value in args.__dict__.items():
        logger.info(f"{name:<{15}} : {value}")
    
    '''
        DATASET
    '''
    # region DATASET
    path_train = './data/train'
    path_train_images =  path_train + '/images/training/t1-3'
    path_valid_images =  path_train + '/images/validation/t4'
    path_train_masks  =  path_train + '/annotations/training/t1-3'
    path_valid_masks  =  path_train + '/annotations/validation/t4'

    # Trasformazioni
    
    '''
    RandomCropResizeSync(
        size=(args.crop, args.crop),      # Dimensione fissa dopo il resize
        min_crop_size=(128, 128),         # Dimensioni minime del crop
        aspect_ratio_range=(0.75, 1.3)    # Range del rapporto altezza/larghezza
    )
    '''

    if args.crop:
        train_transforms = ComposeSync([
            RandomCropSync((args.crop,args.crop)),
            RandomHorizontalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = ComposeSync([
            RandomHorizontalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    valid_transforms = ComposeSync([
        RandomCropSync((args.crop,args.crop)),
        ToTensorSync(),
        NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Training dataset
    dataset_train = SegmentationDataset(
        images_dir = path_train_images,
        masks_dir = path_train_masks,
        transforms = train_transforms
    )

    # Validation dataset
    dataset_valid = SegmentationDataset(
        images_dir = path_valid_images,
        masks_dir = path_valid_masks,
        transforms = valid_transforms
    )

    logger.info("Dataset Creati")
    logger.info(f"Training images: {len(dataset_train)}")
    logger.info(f"Validation images: {len(dataset_valid)}\n")

    # Dataloader
    num_workers = 2
    train_batch_size = args.batch
    valid_batch_size = 8

    dataloader_train = DataLoader(dataset_train,
                                batch_size=train_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True)

    dataloader_valid = DataLoader(dataset_valid,
                                batch_size=valid_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True)

    logger.info("Dataloader Creati")
    logger.info(f"Training batches: {len(dataloader_train)}")
    logger.info(f"Validation batches: {len(dataloader_valid)}\n")
    # endregion

    '''
        MODELLO
    '''
    # region MODEL

    # Carico il modello
    if args.model == 'mobilenet':
        model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
        logger.info('Model: deeplabv3plus_mobilenet\n')
    elif args.model == 'resnet101':
        model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)
        logger.info('Model: deeplabv3plus_resnet101\n')
    
    model.classifier.classifier[3] = nn.Identity()

    path = find_highest_file(ckpts_dir, 'weights_mIoU_')
    state_dict = torch.load(path, map_location=args.device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'],strict=False)
    else:
        model.load_state_dict(state_dict,strict=False)
    logger.info(f'Weights: {path}')

    vae = VAE(input_dim=256, latent_dim=64)

    model.eval()
    model.to(args.device)
    vae.to(args.device)
    
    logger.info(f"Modello caricato correttamente su {args.device}\n")

    '''
        TRAINING
    '''
    # region TRAINING1

    # Freeze backbone (TODO: ha senso sbloccare solo classifier.classifier invece di classifier? (chatgpt dice si))
    for param in model.parameters():
        param.requires_grad = False

    for param in vae.parameters():
        param.requires_grad = True

    # Ottimizzatore
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(vae.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = VAELoss()
    criterion = criterion.to(args.device)

    # Avvio del training
    loss_train, loss_valid = train(
        model,
        vae,
        criterion,
        optimizer,
        args.epochs,
        dataloader_train,
        dataloader_valid,
        args.device,
        'weights_vae',
        ckpts_dir,
        logger)
    # endregion

    # Salvo i parametri
    all_parameters = dict()
    for name, value in args.__dict__.items():
        if not name in ['device','lr','epochs']:
            all_parameters[name] = value
    torch.save(all_parameters, train_dir+'/vae_params.pt')

    # Salvo le loss
    with open(os.path.join(train_dir,'vae_losses.pkl'), "wb") as f:
        pickle.dump((loss_train, loss_valid), f)

    # Rilascia il modello e i tensori
    del model
    gc.collect()
    
    # endregion
