import os
import torch
import random
import torchvision.transforms.functional as F

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

class RandomCropSync:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        i, j, h, w = RandomCrop.get_params(img, output_size=self.size)

        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        
        return img, mask


class RandomCropResizeSync:
    def __init__(self, size, min_crop_size=(128, 128), aspect_ratio_range=(0.5, 2.0)):
        self.size = size
        self.min_crop_size = min_crop_size
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img, mask):
        img_width, img_height = img.size
        min_width, min_height = self.min_crop_size
        min_aspect, max_aspect = self.aspect_ratio_range

        for _ in range(10):
            crop_width = random.randint(min_width, img_width)
            crop_height = random.randint(min_height, img_height)
            aspect_ratio_min = min(crop_width / crop_height, crop_height / crop_width)
            aspect_ratio_max = 1 / aspect_ratio_min
            if min_aspect <= aspect_ratio_min and aspect_ratio_max <= max_aspect:
                break
        else:
            crop_width, crop_height = img_width, img_height

        i = random.randint(0, img_height - crop_height)
        j = random.randint(0, img_width - crop_width)

        img = F.crop(img, i, j, crop_height, crop_width)
        mask = F.crop(mask, i, j, crop_height, crop_width)

        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)

        return img, mask


class RandomHorizontalFlipSync:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask
    

class RandomVerticalFlipSync:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask
    

class ToTensorSync:
    def __call__(self, img, mask):
        img_tensor = F.to_tensor(img)
        mask_tensor = F.pil_to_tensor(mask)
        return img_tensor, mask_tensor
    

class NormalizeSync:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask
        

class ComposeSync:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class RandomResizeSync:
    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, img, mask):
        scale = random.uniform(*self.scale_range)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        img = F.resize(img, (new_height, new_width))
        mask = F.resize(mask, (new_height, new_width), interpolation=F.InterpolationMode.NEAREST)
        
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

        top_index = sorted_indices[random.randint(0,min(3,len(sorted_indices)))]
        img_crop = F.crop(img, top_index[0], top_index[1], self.size, self.size)
        mask_crop = F.crop(mask, top_index[0], top_index[1], self.size, self.size)

        return img_crop, mask_crop


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
        mask = mask.type(torch.long)-1

        return image, mask