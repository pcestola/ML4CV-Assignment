import re
import os
import gc
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import deeplab.network as network

from torch.utils.data import DataLoader

from lib.train import train
from lib.test import test_model
from lib.losses import BCELossModified, FocalLoss, ArcFaceLoss
from lib.data import RandomCropSync, RandomHorizontalFlipSync, NormalizeSync, ToTensorSync, ComposeSync, SegmentationDataset



# https://github.com/CVLAB-Unibo/ml4cv-assignment/tree/master
# https://github.com/cc-ai/Deeplabv3?tab=readme-ov-file

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
        return highest_file, next_file_name

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

# endregion

if __name__ == '__main__':
    
    fix_random(0)

    '''
        PREPARAZIONE
    '''
    # region PREPARAZIONE
    parser = argparse.ArgumentParser(description="Accetta valori dalla linea di comando")
    
    # Aggiungi argomenti
    # TODO: alcuni di questi parametri andrebbero letti dal file .pt
    # ti conviene rifare tutti i vecchi addestramenti
    # dopo esserti assicurato che salvino tutto il necessario per continuare
    parser.add_argument('--device', type=str, default = find_best_device())
    parser.add_argument('--folder', type=int)
    # Parametri addestramento
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default='adam')
    # Parametri rete
    parser.add_argument('--biases', action='store_true')
    parser.add_argument('--focal_loss', type=int, default=0)
    parser.add_argument('--activation', type=str, default='softmax')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--norm_weights', action='store_true')

    # Parsing degli argomenti
    args = parser.parse_args()

    # Directories + Logger
    train_dir = '/raid/homespace/piecestola/space/ML4CV/results' + f'/train_{args.folder}'
    ckpts_dir = train_dir + '/ckpts'
    logger = setup_logger(train_dir,'train.log')
    # endregion

    ''' DATASET '''
    # region DATASET
    path_train = './data/train'
    path_train_images =  path_train + '/images/training/t1-3'
    path_valid_images =  path_train + '/images/validation/t4'
    path_train_masks  =  path_train + '/annotations/training/t1-3'
    path_valid_masks  =  path_train + '/annotations/validation/t4'

    # Trasformazioni
    if args.crop:
        train_transforms = ComposeSync([
            RandomCropSync(size=(args.crop, args.crop)),
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

    # endregion

    logger.info(f"\n ========= INIZIO NUOVO ADDESTRAMENTO =========\n")

    '''
        MODELLO
    '''
    # region MODEL

    # Estraggo i dati del modello precedente
    directory, name = find_highest_file(ckpts_dir,'weights_mIoU_',next=True)
    checkpoint = torch.load(directory, map_location=args.device)

    normalized_head = 'classifier.classifier.3.bias' in checkpoint['model_state_dict']

    # Carico il modello
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)

    # Modifico la testa di classificazione
    if normalized_head:
        model.classifier.classifier[3] = nn.Conv2d(256, 13, kernel_size=(1, 1), stride=(1, 1))
    else:
        model.classifier.classifier[3] = NormedConv(256,13,(1,1),(1,1))

    # Carico il modello sul device
    model.to(args.device)

    # endregion

    '''
        TRAINING
    '''
    # region TRAINING

    # Parametri
    inv_freq = [3.2232590372125256, 8.324592511196576, 80.0768737988469, 214.22450728363324, 5434.782608695652, 80.23106546854943, 74.3549706297866, 3.001164451807301, 14.900464894504708, 11.332985788435822, 404.0404040404041, 29.118863199580684, 995.0248756218905]    

    # Loss
    if args.class_weights:
        weights = torch.tensor(inv_freq, device=args.device, requires_grad=False)
    else:
        weights = None

    if args.norm_weights:
        criterion = ArcFaceLoss(weights=weights)
    elif args.activation == 'softmax':
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.activation == 'sigmoid':
        if not args.focal_loss:
            criterion = BCELossModified(pos_weight=weights)
        else:
            criterion = FocalLoss(gamma=args.focal_loss, alpha=weights)

    # Carico i pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"\nCaricati i pesi di: {directory}\n")

    for param in model.parameters():
        param.requires_grad = True

    # Ottimizzatore
    # TODO: DA SISTEMARE
    args.lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr*0.1}
        ], weight_decay=1e-4)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr*0.1}
        ], weight_decay=1e-4)

    # sovrapponeva pesi
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Avvio del training
    loss_train, loss_valid = train(
        model,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        dataloader_train,
        dataloader_valid,
        args.device,
        name,
        ckpts_dir,
        logger,
        checkpoint['mIoU_best'],
        checkpoint['loss_val_best']
    )

    # Rilascia il modello e i tensori
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # endregion
