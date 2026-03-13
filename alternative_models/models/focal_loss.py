#!/usr/bin/env python3
"""
Focal Loss for handling hard examples.

Focal Loss = -alpha * (1 - p)^gamma * log(p)

Focuses training on hard-to-classify examples by down-weighting easy examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance classes
               For balanced data (1:1), use alpha=0.5
        gamma: Focusing parameter >= 0. Higher = more focus on hard examples
               gamma=0 is equivalent to CrossEntropyLoss
               gamma=2 is standard for Focal Loss
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        
        Returns:
            focal_loss: scalar or [B] depending on reduction
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)  # Probability of correct class
        
        # Focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test Focal Loss
    focal = FocalLoss(alpha=0.5, gamma=2.0)
    ce = nn.CrossEntropyLoss()
    
    # Easy example (high confidence, correct)
    logits_easy = torch.tensor([[5.0, -5.0]])  # Very confident real
    target = torch.tensor([0])  # Real
    
    # Hard example (low confidence, correct)
    logits_hard = torch.tensor([[0.5, 0.4]])  # Barely correct
    
    print("Easy example (confident & correct):")
    print(f"  CE Loss:    {ce(logits_easy, target).item():.4f}")
    print(f"  Focal Loss: {focal(logits_easy, target).item():.4f}")
    print(f"  Focal/CE:   {focal(logits_easy, target).item() / ce(logits_easy, target).item():.4f}x")
    
    print("\nHard example (uncertain & correct):")
    print(f"  CE Loss:    {ce(logits_hard, target).item():.4f}")
    print(f"  Focal Loss: {focal(logits_hard, target).item():.4f}")
    print(f"  Focal/CE:   {focal(logits_hard, target).item() / ce(logits_hard, target).item():.4f}x")
    
    print("\nFocal Loss focuses more on hard examples (higher relative loss)")
