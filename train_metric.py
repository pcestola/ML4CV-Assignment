import re
import os
import gc
import sys
import copy
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
from torchmetrics import JaccardIndex
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

class TopScoringCrop:
    def __init__(self, size, freqs):
        self.size = size
        self.freqs = torch.tensor(freqs)

    def calculate_iou(self, box1, box2):
        side = self.size
        x1_min, y1_min = box1
        x1_max, y1_max = box1[0] + side, box1[1] + side

        x2_min, y2_min = box2
        x2_max, y2_max = box2[0] + side, box2[1] + side

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box_area = side * side
        union_area = 2 * box_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def filter_squares(self, vertices, threshold=0.5):
        filtered = [vertices[0]]
        for idx in range(1, len(vertices)):
            iou = max(self.calculate_iou(x, vertices[idx]) for x in filtered)
            if iou <= threshold:
                filtered.append(vertices[idx])
        return filtered

    def __call__(self, img, mask):
        scores = self.freqs[mask.type(torch.int64)-1].squeeze()

        values = []
        indices = []
        for x in range(0, mask.shape[1] - self.size, 40):
            for y in range(0, mask.shape[2] - self.size, 40):
                value = scores[x:x + self.size, y:y + self.size].sum().item()
                values.append(value)
                indices.append((x, y))

        sorted_indices = [obj for _, obj in sorted(zip(values, indices), reverse=True)]
        sorted_indices = self.filter_squares(sorted_indices, threshold=0.3)

        # Select the top scoring crop
        top_index = sorted_indices[random.randint(0,min(3,len(sorted_indices)))]
        img_crop = FUNC.crop(img, top_index[0], top_index[1], self.size, self.size)
        mask_crop = FUNC.crop(mask, top_index[0], top_index[1], self.size, self.size)

        return img_crop, mask_crop

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
    
class RandomVerticalFlipSync:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = FUNC.vflip(img)
            mask = FUNC.vflip(mask)
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

# region TEST
# TODO: controlla che faccia esattamente quello che credi
# TODO: ious torna un solo numero!!!!
def test_model(model: nn.Module, test_loader: DataLoader, num_classes: int):
    device = next(model.parameters()).device
    model.eval()

    # Metriche sulla CPU
    eval_IoU = JaccardIndex(
        task='multiclass',
        num_classes=num_classes,    #13
        average="none",
        zero_division=0
    ).to(device)

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            # Sposta input sulla GPU
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            max_class = torch.argmax(logits, dim=1)

            # Aggiorna le metriche sulla CPU
            eval_IoU.update(max_class, masks)

    # Compute metriche finali
    ious = eval_IoU.compute().cpu()

    return ious.mean().item(), ious.min().item(), ious.max().item()
# endregion

# region TRAINING
class ParallelBlock(nn.Module):
    def __init__(self, *modules):
        super(ParallelBlock, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        outputs = [module(x) for module in self.modules_list]
        return torch.concat(outputs, dim=1)

def old_train_epoch(model: nn.Module,
                loader_train: utils.data.DataLoader,
                device: torch.device,
                optimizer: torch.optim,
                criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    samples_train = 0
    loss_train = 0

    model.train()
    for images, masks in loader_train:
        images = images.to(device)
        masks = masks.to(device)

        out = model(images)

        loss = criterion(out, masks)
        loss_train += loss.item()
        samples_train += images.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= samples_train
    return loss_train

# MEMO: hai modificato l'ordine dei del per far funzionare
# i norm_weights con ArcFace Loss
def train_epoch(model: nn.Module,
                loader_train: utils.data.DataLoader,
                device: torch.device,
                optimizer: torch.optim,
                criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    
    loss_train = 0

    model.train()
    for images, masks in loader_train:

        # Trasferisci tensori al device
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # Calcolo della forward pass
        out = model(images)
        del images
        
        # Calcolo della loss
        loss = criterion(out, masks)
        del masks, out
        
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        del loss
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            #gc.collect()
            #torch.cuda.synchronize()

    loss_train /= len(dataloader_train)
    return loss_train

def validate(model: nn.Module,
             loader_val: utils.data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    
    loss_val = 0

    model = model.eval()
    with torch.no_grad():
        for images, masks in loader_val:
            images = images.to(device)
            masks = masks.to(device)

            out = model(images)

            loss = criterion(out, masks)
            loss_val += loss.item()

    loss_val /= len(loader_val)
    return loss_val

def training_loop(num_epochs: int,
                  optimizer: torch.optim,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  model: nn.Module,
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
                                loader_train,
                                device,
                                optimizer,
                                criterion)

        loss_val = validate(model,
                            loader_val,
                            device,
                            criterion)
        
        mIoU, minIoU, maxIoU = test_model(model, loader_val, 13)

        if epoch % 15 == 0:
            logger.info(f"{'Epoch':8} {'Loss':10} {'Val Loss':10} {'mIoU':10} {'minIoU':10} {'maxIoU':10} {'LR':8}")
        
        log_training_progress(optimizer, logger, epoch, loss_train, loss_val, mIoU, minIoU, maxIoU)
        losses_train.append(loss_train)
        losses_valid.append(loss_val)

        # Update learning rate
        scheduler.step(loss_val)

        # Save model with best validation loss
        flag = False
        if epoch == 0:
            flag = True
            name = run_name.replace('_','_mIoU_')
            mIoU_best = mIoU
            minIoU_best = minIoU
            loss_val_best = loss_val
        else:
            #if loss_val < loss_val_best:
                #flag = True
                #name = run_name
                #loss_val_best = loss_val
            if mIoU > mIoU_best:
                flag = True
                name = run_name.replace('_','_mIoU_')
                mIoU_best = mIoU
                logger.info("> saved mIoU")
            elif minIoU_best > minIoU:
                flag = True
                name = run_name.replace('_','_minIoU_')
                minIoU_best = minIoU
                logger.info("> saved minIoU")
        
        if flag:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'mIoU_best': mIoU_best,
                'loss_val_best': loss_val_best,
            }, path_ckpts / f"{name}.pt")
    
    return losses_train, losses_valid

def train(model: nn.Module,
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

# region LOSSES
class BCELossModified(nn.BCEWithLogitsLoss):
    def __init__(self, weight = None, size_average=None, reduce=None, reduction = 'mean', pos_weight = None):
        if pos_weight != None:
            for _ in range(3-pos_weight.dim()):
                pos_weight = pos_weight.unsqueeze(dim=-1)
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
    
    def forward(self, input, target):
        target = torch.nn.functional.one_hot(target, num_classes=input.shape[1]).permute((0,3,1,2)).float()
        return super().forward(input, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma:int=0, alpha=None, size_average:bool=True, activation:str='sigmoid'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-7
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        if activation=='sigmoid':
            self.activation = lambda x: torch.sigmoid(x)
        else:
            self.activation = lambda x: torch.softmax(x,dim=1)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        pt = self.activation(input)
        pt = pt.gather(1,target).view(-1)
        logpt = torch.log(pt + self.eps)

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

class ArcFaceLoss(nn.Module):
    def __init__(self, scale=30.0, margin=0.5, weights=None, loss=None):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.loss = nn.CrossEntropyLoss(weight=weights) if loss is None else loss

    def forward(self, logits, labels):
        # Calcola l'angolo theta aggiungendo il margine
        target_logits = torch.cos(torch.arccos(torch.clamp(logits, -1.0+1e-7, 1.0-1e-7)) + self.margin)

        # Crea un one-hot encoding dei label
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)

        # Applica il margine solo alle classi target
        target_logits = (one_hot * target_logits) + ((1.0 - one_hot) * logits)
        target_logits *= self.scale

        # Calcola e restituisce la perdita
        loss = self.loss(target_logits, labels)
        return loss

class CE_EntropyMinimization(nn.Module):
    def __init__(self, entropy_weight=0.1):
        super(CE_EntropyMinimization, self).__init__()
        self.entropy_weight = entropy_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Calcolo della Cross Entropy
        ce_loss = self.ce_loss(logits, targets)

        # Calcolo dell'entropia del softmax
        softmax_probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1).mean()

        # Loss combinata: CE + penalizzazione dell'entropia
        total_loss = ce_loss + self.entropy_weight * entropy

        return total_loss
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

def log_training_progress(optimizer:torch.optim.Optimizer, logger:logging.Logger, epoch:int, loss:float, val_loss:float, mIoU:float, minIoU:float, maxIoU:float) -> None:
    lr_list = [group['lr'] for group in optimizer.param_groups]
    lr_list = f"{', '.join(f'{lr:<8.1e}' for lr in lr_list)}"
    logger.info(f"{epoch:<8d} {loss:<10.6f} {val_loss:<10.6f} {mIoU:<10.6f} {minIoU:<10.6f} {maxIoU:<10.6f} " + lr_list)

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
    parser.add_argument('--model', type=str, default = 'resnet101')
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--p', type=float, default=0)
    # Parametri addestramento
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam')
    # Parametri rete
    parser.add_argument('--focal_loss', type=int, default=0)
    parser.add_argument('--activation', type=str, default='softmax')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--arc_face', action='store_true')
    parser.add_argument('--entropy', type=float, default=0)
    # Caricamento

    # Parsing degli argomenti
    args = parser.parse_args()

    # Creo la directory ed il logger
    results_dir = '/raid/homespace/piecestola/space/ML4CV/results'
    train_dir, ckpts_dir, logger = setup_train_folder(results_dir, False)

    # Scrivo i parametri
    for name, value in args.__dict__.items():
        logger.info(f"{name:<{15}} : {value}")
    
    '''
        DATASET
    '''
    # region DATASET
    path_train = '/raid/homespace/piecestola/space/ML4CV/data/train'
    path_train_images =  path_train + '/images/training/t1-3'
    path_valid_images =  path_train + '/images/validation/t4'
    path_train_masks  =  path_train + '/annotations/training/t1-3'
    path_valid_masks  =  path_train + '/annotations/validation/t4'

    # Trasformazioni
    scores_ = [3.223254919052124, 8.324562072753906, 1000.07819366455078, 214.23089599609375, 500441.0986328125, 80.23406219482422, 74.35269927978516, 3.00116229057312, 14.900472640991211, 11.333002090454102, 404.04559326171875, 29.11901092529297, 995.0892333984375]

    if args.crop:
        train_transforms = ComposeSync([
            RandomCropSync((args.crop,args.crop)),
            RandomHorizontalFlipSync(p=0.5),
            #RandomVerticalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #TopScoringCrop(100,scores_)
        ])
    else:
        train_transforms = ComposeSync([
            RandomHorizontalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    valid_transforms = ComposeSync([
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

    def seed_worker(worker_id):
        worker_seed = 0
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g_train = torch.Generator()
    g_train.manual_seed(0)

    g_valid = torch.Generator()
    g_valid.manual_seed(0) 

    dataloader_train = DataLoader(dataset_train,
                                batch_size=train_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=g_train)

    dataloader_valid = DataLoader(dataset_valid,
                                batch_size=valid_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=g_valid)

    logger.info("Dataloader Creati")
    logger.info(f"Training batches: {len(dataloader_train)}")
    logger.info(f"Validation batches: {len(dataloader_valid)}\n")
    # endregion

    '''
        MODELLO
    '''
    # region MODEL

    # Carico il modello
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)
    path = '/raid/homespace/piecestola/space/ML4CV/weights/best_deeplabv3plus_resnet101_voc_os16.pth'
    model.load_state_dict(torch.load(path, map_location=args.device)['model_state'])
    logger.info('Model: deeplabv3plus_resnet101\n')
    
    # Parametri
    inv_freq = [3.223254919052124, 8.324562072753906, 80.07819366455078, 214.23089599609375, 5441.0986328125, 80.23406219482422, 74.35269927978516, 3.00116229057312, 14.900472640991211, 11.333002090454102, 404.04559326171875, 29.11901092529297, 995.0892333984375]

    # Modifico la testa di classificazione
    #model.classifier.classifier[3] = PrototypeHeadMLP(embed_dim=256, hidden_dims=[128], num_prototypes=13)
    model.classifier.classifier[3] = PrototypeHeadArcFace2(embed_dim=256, hidden_dims=[128], num_prototypes=13)

    # Carico il modello sul device
    model.to(args.device)

    logger.info(f"Modello caricato correttamente su {args.device}\n")
    # endregion



    # region Training
    
    # Ottimizzatore
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.classifier.parameters(),'lr': args.lr},
            {'params': model.backbone.parameters(),'lr': args.lr * 0.1}
        ], weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.classifier.parameters(),'lr': args.lr},
            {'params': model.backbone.parameters(),'lr': args.lr * 0.1}
        ], weight_decay=1e-4)
    
    # Loss
    if args.class_weights:
        weights = torch.tensor(inv_freq, device=args.device, requires_grad=False)
    else:
        weights = None

    if args.arc_face:
        if args.activation == 'softmax':
            criterion = ArcFaceLoss(weights=weights)
        elif args.activation == 'sigmoid':
            criterion = ArcFaceLoss(loss=BCELossModified(pos_weight=weights))
    elif args.entropy != 0:
        criterion = CE_EntropyMinimization(entropy_weight=args.entropy)
    else:
        if args.activation == 'softmax':
            criterion = nn.CrossEntropyLoss(weight=weights)
        elif args.activation == 'sigmoid':
            criterion = BCELossModified(pos_weight=weights)
    
    criterion = criterion.to(args.device)
    print('=== CRITERIO ===>',criterion)
    
    # Avvio del training
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.classifier.parameters():
        param.requires_grad = True

    loss_train, loss_valid = train(
        model,
        criterion,
        optimizer,
        15,
        dataloader_train,
        dataloader_valid,
        args.device,
        'weights_0',
        ckpts_dir,
        logger)
    
    # Carico i pesi migliori del passo precedente
    directory = os.path.join(ckpts_dir,'weights_mIoU_0.pt')
    model.load_state_dict(torch.load(directory, map_location=args.device)['model_state_dict'])
    logger.info(f"\nCaricati i pesi in: {directory}\n")

    # Addestro tutti i parametri
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer.param_groups[1]['lr'] = 0.1 * optimizer.param_groups[0]['lr']

    loss_train, loss_valid = train(
        model,
        criterion,
        optimizer,
        args.epochs,
        dataloader_train,
        dataloader_valid,
        args.device,
        'weights_1',
        ckpts_dir,
        logger)

    # Salvo le loss
    with open(os.path.join(train_dir,'losses.pkl'), "wb") as f:
        pickle.dump((loss_train, loss_valid), f)
    
    # endregion

    torch.save(model.prototypes, os.path.join(ckpts_dir, "prototypes.pth"))

    # Rilascia il modello e i tensori
    del model
    gc.collect()