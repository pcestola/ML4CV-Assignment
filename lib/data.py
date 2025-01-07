import os
import torch
import random
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

class RandomCropSync:
    def __init__(self, size):
        """
        Inizializza l'operazione di ritaglio casuale.
        :param size: Tuple con la dimensione finale del ritaglio (altezza, larghezza).
        """
        self.size = size

    def __call__(self, img, mask):
        """
        Applica un ritaglio casuale sincronizzato tra immagine e maschera.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera ritagliate.
        """
        # Ottieni i parametri per il ritaglio casuale
        i, j, h, w = RandomCrop.get_params(img, output_size=self.size)
        
        # Applica il ritaglio
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask

class RandomCropResizeSync:
    def __init__(self, size, min_crop_size=(128, 128), aspect_ratio_range=(0.5, 2.0)):
        """
        Inizializza un ritaglio casuale con controlli sulle dimensioni e il range di aspect ratio,
        seguito da un ridimensionamento fisso.
        :param size: Tuple con la dimensione finale del resize (larghezza, altezza).
        :param min_crop_size: Dimensioni minime del crop (min_larghezza, min_altezza).
        :param aspect_ratio_range: Range valido di rapporto (larghezza / altezza).
        """
        self.size = size
        self.min_crop_size = min_crop_size
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img, mask):
        """
        Applica un ritaglio casuale seguito da un ridimensionamento fisso.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera processate.
        """
        img_width, img_height = img.size
        min_width, min_height = self.min_crop_size
        min_aspect, max_aspect = self.aspect_ratio_range

        for _ in range(10):  # Tentativi per trovare un crop valido
            crop_width = random.randint(min_width, img_width)
            crop_height = random.randint(min_height, img_height)
            
            aspect_ratio = crop_width / crop_height
            if min_aspect <= aspect_ratio <= max_aspect:
                break
        else:
            crop_width, crop_height = img_width, img_height

        i = random.randint(0, img_height - crop_height)
        j = random.randint(0, img_width - crop_width)

        # Applica il ritaglio
        img = F.crop(img, i, j, crop_height, crop_width)
        mask = F.crop(mask, i, j, crop_height, crop_width)

        # Ridimensiona immagine e maschera
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return img, mask

class RandomHorizontalFlipSync:
    def __init__(self, p=0.5):
        """
        Inizializza l'operazione di flip orizzontale casuale.
        :param p: Probabilità di applicare il flip.
        """
        self.p = p

    def __call__(self, img, mask):
        """
        Applica un flip orizzontale casuale.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera processate.
        """
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class ToTensorSync:
    def __call__(self, img, mask):
        """
        Converte immagine e maschera in tensori.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera come tensori.
        """
        img_tensor = F.to_tensor(img)
        mask_tensor = F.pil_to_tensor(mask)
        return img_tensor, mask_tensor

class NormalizeSync:
    def __init__(self, mean, std):
        """
        Inizializza la normalizzazione per le immagini.
        :param mean: Media per la normalizzazione.
        :param std: Deviazione standard per la normalizzazione.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        """
        Applica la normalizzazione all'immagine.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine normalizzata e maschera invariata.
        """
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask

class ComposeSync:
    def __init__(self, transforms):
        """
        Composizione di trasformazioni sincronizzate per immagine e maschera.
        :param transforms: Lista di trasformazioni da applicare.
        """
        self.transforms = transforms

    def __call__(self, img, mask):
        """
        Applica in sequenza le trasformazioni.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera processate.
        """
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask

class RandomResizeSync:
    def __init__(self, scale_range=(0.5, 1.5)):
        """
        Inizializza il ridimensionamento dinamico sincronizzato.
        :param scale_range: Range di scaling (min_scale, max_scale).
        """
        self.scale_range = scale_range

    def __call__(self, img, mask):
        """
        Applica un ridimensionamento casuale.
        :param img: Immagine da processare.
        :param mask: Maschera corrispondente.
        :return: Immagine e maschera ridimensionate.
        """
        scale = random.uniform(*self.scale_range)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        img = F.resize(img, (new_height, new_width))
        mask = F.resize(mask, (new_height, new_width), interpolation=F.InterpolationMode.NEAREST)
        return img, mask

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        """
        Dataset per la segmentazione semantica.
        :param images_dir: Directory contenente le immagini.
        :param masks_dir: Directory contenente le maschere.
        :param transforms: Trasformazioni da applicare a immagini e maschere.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Recupera un'immagine e la relativa maschera.
        :param idx: Indice dell'immagine.
        :return: Coppia (immagine, maschera) processata.
        """
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = mask.squeeze()
        mask = mask.type(torch.int64) - 1
        return image, mask