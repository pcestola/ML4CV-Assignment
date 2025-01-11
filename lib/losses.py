import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLossModified(nn.BCEWithLogitsLoss):
    def __init__(self, weight = None, size_average=None, reduce=None, reduction = 'mean', pos_weight = None):
        if pos_weight != None:
            for _ in range(3-pos_weight.dim()):
                pos_weight = pos_weight.unsqueeze(dim=-1)
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
    
    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=input.shape[1]).permute((0,3,1,2)).float()
        return super().forward(input, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma:int=0, alpha=None, size_average:bool=True, activation:str='sigmoid'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-7
        self.size_average = size_average

        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)

        if activation=='sigmoid':
            self.activation = lambda x: torch.sigmoid(x)
        else:
            self.activation = lambda x: torch.softmax(x,dim=1)

    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1, 2).contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        # Calculate probability and log-probability
        pt = self.activation(input).gather(1, target).view(-1)
        logpt = torch.log(pt + self.eps)

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        # Compute Focal Loss
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


class ArcFaceLoss(nn.Module):
    def __init__(self, scale:float=30.0, margin:float=0.5, weights=None, loss=None):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.loss = nn.CrossEntropyLoss(weight=weights) if loss is None else loss

    def forward(self, input:torch.Tensor, labels:torch.Tensor):
        # Apply margin
        target_logits = torch.cos(torch.acos(torch.clamp(input, -1.0, 1.0)) + self.margin)

        # Create one-hot encoded labels
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)

        # Apply scale and margin adjustments
        target_logits = (one_hot * target_logits) + ((1.0 - one_hot) * input)
        target_logits *= self.scale

        # Compute loss
        return self.loss(target_logits, labels)


class CE_EntropyMinimization(nn.Module):
    def __init__(self, entropy_weight=0.1):
        super(CE_EntropyMinimization, self).__init__()
        self.entropy_weight = entropy_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input:torch.Tensor, targets:torch.Tensor):
        # Compute standard cross-entropy loss
        ce_loss = self.ce_loss(input, targets)

        # Compute entropy regularization term
        softmax_probs = F.softmax(input, dim=1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1).mean()

        # Combine losses
        total_loss = ce_loss + self.entropy_weight * entropy

        return total_loss
