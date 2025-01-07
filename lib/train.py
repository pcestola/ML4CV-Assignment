import torch
import logging
import torch.nn as nn
import torch.utils as utils

from pathlib import Path
from typing import Callable

from lib.test import test_model

def log_training_progress(optimizer: torch.optim.Optimizer, logger: logging.Logger, epoch: int, loss: float, val_loss: float, mIoU: float, minIoU: float, maxIoU: float) -> None:
    """
    Logga il progresso del training, includendo informazioni sull'epoch, le perdite, le metriche mIoU e il learning rate.

    :param optimizer: Ottimizzatore del modello.
    :param logger: Logger per registrare i dettagli del progresso.
    :param epoch: Numero dell'epoch corrente.
    :param loss: Perdita media del training per l'epoch.
    :param val_loss: Perdita media di validazione per l'epoch.
    :param mIoU: Mean Intersection over Union (mIoU).
    :param minIoU: Valore minimo di IoU.
    :param maxIoU: Valore massimo di IoU.
    """
    # Ottieni la lista dei learning rate correnti
    lr_list = [group['lr'] for group in optimizer.param_groups]
    formatted_lrs = ', '.join(f'{lr:<8.1e}' for lr in lr_list)

    # Logga i dettagli formattati
    logger.info(f"{epoch:<8d} {loss:<10.6f} {val_loss:<10.6f} {mIoU:<10.6f} {minIoU:<10.6f} {maxIoU:<10.6f} " + formatted_lrs)

def train_epoch(model: nn.Module,
                loader_train: utils.data.DataLoader,
                device: torch.device,
                optimizer: torch.optim,
                criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    """
    Esegue un'epoca di training per il modello dato.
    :param model: Modello da addestrare.
    :param loader_train: DataLoader con i dati di training.
    :param device: Dispositivo su cui eseguire il training.
    :param optimizer: Ottimizzatore.
    :param criterion: Funzione di perdita.
    :return: Perdita media sull'epoca.
    """
    samples_train = 0
    loss_train = 0

    model.train()
    for images, masks in loader_train:
        images = images.to(device)
        masks = masks.to(device)

        # Predizione del modello
        out = model(images)

        # Calcolo della perdita
        loss = criterion(out, masks)
        loss_train += loss.item()
        samples_train += images.shape[0]

        # Aggiornamento dei pesi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= samples_train
    return loss_train

def validate(model: nn.Module,
             loader_val: utils.data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> float:
    """
    Valida il modello sui dati di validazione.
    :param model: Modello da validare.
    :param loader_val: DataLoader con i dati di validazione.
    :param device: Dispositivo su cui eseguire la validazione.
    :param criterion: Funzione di perdita.
    :return: Perdita media sui dati di validazione.
    """
    samples_val = 0
    loss_val = 0

    model.eval()
    with torch.no_grad():
        for images, masks in loader_val:
            images = images.to(device)
            masks = masks.to(device)

            # Predizione del modello
            out = model(images)

            # Calcolo della perdita
            loss = criterion(out, masks)
            loss_val += loss.item()
            samples_val += images.shape[0]

    loss_val /= samples_val
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
                  path_ckpts: str,
                  logger,
                  mIoU_best:float,
                  loss_val_best:float) -> None:
    """
    Esegue il ciclo di training e validazione per un numero specificato di epoche.
    :param num_epochs: Numero di epoche.
    :param optimizer: Ottimizzatore.
    :param scheduler: Scheduler del learning rate.
    :param model: Modello da addestrare.
    :param criterion: Funzione di perdita.
    :param loader_train: DataLoader con i dati di training.
    :param loader_val: DataLoader con i dati di validazione.
    :param device: Dispositivo su cui eseguire il training.
    :param run_name: Nome dell'esperimento.
    :param path_ckpts: Percorso per salvare i checkpoint.
    :param logger: Logger per registrare il progresso.
    :return: Liste delle perdite di training e validazione.
    """
    path_ckpts = Path(path_ckpts)
    path_ckpts.mkdir(parents=True, exist_ok=True)

    losses_train = []
    losses_valid = []

    for epoch in range(num_epochs):
        loss_train = train_epoch(model, loader_train, device, optimizer, criterion)
        loss_val = validate(model, loader_val, device, criterion)
        mIoU, minIoU, maxIoU = test_model(model, loader_val, 13)

        if epoch % 15 == 0:
            logger.info(f"{'Epoch':8} {'Loss':10} {'Val Loss':10} {'mIoU':10} {'minIoU':10} {'maxIoU':10} {'LR':8}")

        log_training_progress(optimizer, logger, epoch, loss_train, loss_val, mIoU, minIoU, maxIoU)
        losses_train.append(loss_train)
        losses_valid.append(loss_val)

        # Aggiorna il learning rate
        scheduler.step(loss_val)

        # Salva il modello con la miglior perdita di validazione
        if loss_val < loss_val_best:
            loss_val_best = loss_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'mIoU_best': mIoU_best,
                'loss_val_best': loss_val_best,
            }, path_ckpts / f"{run_name}.pt")
        if mIoU > mIoU_best:
            mIoU_best = mIoU
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'mIoU_best': mIoU_best,
                'loss_val_best': loss_val_best,
            }, path_ckpts / f"{run_name}.pt")

    return losses_train, losses_valid

def train(model: nn.Module,
          criterion: nn.Module,
          optimizer,
          scheduler,
          epochs: int,
          data_loader_train: torch.utils.data.DataLoader,
          data_loader_val: torch.utils.data.DataLoader,
          device: torch.device,
          run_name: str,
          path_ckpts: str,
          logger: logging.Logger,
          mIoU_best:float,
          loss_val_best:float) -> None:
    """
    Funzione principale per addestrare il modello.
    :param model: Modello da addestrare.
    :param criterion: Funzione di perdita.
    :param optimizer: Ottimizzatore.
    :param epochs: Numero di epoche di addestramento.
    :param data_loader_train: DataLoader per il training.
    :param data_loader_val: DataLoader per la validazione.
    :param device: Dispositivo su cui eseguire il training.
    :param run_name: Nome dell'esperimento.
    :param path_ckpts: Percorso per i checkpoint.
    :param logger: Logger per registrare l'output.
    :return: Liste delle perdite di training e validazione.
    """

    logger.info("Inizio del training")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"loss:      {criterion}")
    logger.info(f"epochs:    {epochs}")
    logger.info(f"name:      {run_name}\n")

    losses_train, losses_valid = training_loop(epochs, optimizer, scheduler, model, criterion,
                                               data_loader_train, data_loader_val, device,
                                               run_name, path_ckpts, logger, mIoU_best, loss_val_best)

    logger.info("Fine del training\n")

    return losses_train, losses_valid
