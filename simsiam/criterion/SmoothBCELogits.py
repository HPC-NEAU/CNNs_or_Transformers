import torch
from torch import binary_cross_entropy_with_logits
from torch.nn.modules.loss import _WeightedLoss
import config_moco



class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, label):
        # inputs = inputs["logits"]
        if inputs.dim() != label.dim():
            label = torch.unsqueeze(label, dim=1).long()
            targets = torch.zeros(config_moco.batch_size, config_moco.Data.NUM_CLASSES).to("cuda")
            targets = targets.scatter(1, label, 1)
        else:
            targets = label
        targets = SmoothBCEwLogits._smooth(targets, self.smoothing)
        loss = binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss