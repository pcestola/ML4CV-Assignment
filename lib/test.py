import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

def calculate_intersection_union(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """
    Calcola l'intersezione e l'unione per ogni classe.
    :param pred: Array delle predizioni.
    :param target: Array dei target.
    :param num_classes: Numero totale di classi.
    :return: Array di intersezioni e unioni per classe.
    """
    intersection = []
    union = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection.append((pred_cls & target_cls).sum())
        union.append((pred_cls | target_cls).sum())
    return np.array(intersection), np.array(union)

def test_model(model: nn.Module,
               test_loader: DataLoader,
               num_classes: int):
    """
    Testa il modello calcolando la mIoU.
    :param model: Modello PyTorch da testare.
    :param test_loader: DataLoader con i dati di test.
    :param num_classes: Numero di classi nel dataset.
    :return: mIoU medio, minimo e massimo.
    """
    device = next(model.parameters()).device

    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Previsioni del modello
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            # Calcola mIoU per la segmentazione semantica
            for pred, mask in zip(predictions, masks):
                intersection, union = calculate_intersection_union(pred.cpu().numpy(), mask.cpu().numpy(), num_classes)
                total_intersection += intersection
                total_union += union

    # Calcola mIoU medio
    if (total_union == 0).all():
        return 0, 0, 0
    else:
        iou = total_intersection / total_union

    return iou.mean(), iou.min(), iou.max()
