import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELossModified(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        """
        Modifica della BCEWithLogitsLoss per supportare pos_weight di dimensioni arbitrarie.

        :param weight: Peso per ogni classe.
        :param size_average: Non utilizzato, per compatibilità.
        :param reduce: Non utilizzato, per compatibilità.
        :param reduction: Modalità di riduzione ('mean', 'sum', ecc.).
        :param pos_weight: Peso positivo per le classi sbilanciate.
        """
        if pos_weight is not None:
            for _ in range(3 - pos_weight.dim()):
                pos_weight = pos_weight.unsqueeze(dim=-1)
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, input, target):
        """
        Calcola la perdita BCE con logit e one-hot encoding per il target.

        :param input: Predizioni del modello.
        :param target: Target ground-truth.
        :return: Perdita calcolata.
        """
        target = torch.nn.functional.one_hot(target, num_classes=input.shape[1]).permute((0, 3, 1, 2)).float()
        return super().forward(input, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Inizializza la Focal Loss per il bilanciamento delle classi sbilanciate.

        :param gamma: Fattore di focalizzazione.
        :param alpha: Peso opzionale per le classi (float, lista o tensor).
        :param size_average: Se True, ritorna la perdita media.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Calcola la Focal Loss.

        :param input: Predizioni del modello (logit).
        :param target: Target ground-truth.
        :return: Perdita calcolata.
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pt = torch.sigmoid(input)
        pt = pt.gather(1, target)
        pt = pt.view(-1)
        logpt = torch.log(pt)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

class ArcFaceLoss(nn.Module):
    def __init__(self, scale=30.0, margin=0.5, weights=None, loss=None):
        """
        Inizializza la ArcFace Loss per il riconoscimento facciale.

        :param scale: Fattore di scaling.
        :param margin: Margine angolare aggiuntivo.
        :param weights: Pesi opzionali per le classi.
        :param loss: Funzione di perdita personalizzata (predefinita: CrossEntropyLoss).
        """
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.loss = nn.CrossEntropyLoss(weight=weights) if loss is None else loss(weight=weights)

    def forward(self, logits, labels):
        """
        Calcola la ArcFace Loss.

        :param logits: Logit del modello (angoli coseni).
        :param labels: Target ground-truth.
        :return: Perdita calcolata.
        """
        # Calcola l'angolo con arccoseno
        theta = torch.acos(torch.clamp(logits, -1.0, 1.0))
        # Aggiungi il margine alla classe corretta
        target_logits = torch.cos(theta + self.margin)
        # Crea un one-hot encoding per il target
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype).permute((0, 3, 1, 2))
        # Aggiorna i logit
        logits_with_margin = logits * (1 - one_hot) + target_logits * one_hot
        # Scala i logit
        logits_with_margin *= self.scale
        # Calcola la perdita
        return self.loss(logits_with_margin, labels)
