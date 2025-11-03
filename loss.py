import torch
from torch import nn
import torch.nn.functional as F


def compute_class_weights(labels, num_classes):
    """
    Compute class weights based on sample class labels.

    Args:
        labels: 1D Tensor containing labels for all samples (e.g., [0, 1, 1, 2]).
        num_classes: Integer representing the total number of classes.

    Returns:
        Tensor: Class weights with shape [num_classes].
    """
    class_counts = torch.bincount(labels, minlength=num_classes)  # Count samples for each class
    total_samples = len(labels)  # Total number of samples
    class_weights = total_samples / (num_classes * class_counts.float())  # Weight per class: total samples / num classes / samples in class
    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.
        Args:
            gamma: Focusing parameter. Default is 2.0.
            alpha: Class weight balance. Default is None.
            reduction: Specify the reduction to apply to the output.
                       'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=None if self.alpha is None else self.alpha.to(inputs.device))    #None if self.alpha is None else self.alpha.to(inputs.device)
        pt = torch.exp(-ce_loss)  # Compute prediction probability for each sample
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss