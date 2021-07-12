import torchsort
from torch import nn
from torch.nn import functional as F

__all__ = ['TopKClassificationLoss']


class TopKClassificationLoss(nn.Module):
    def __init__(self, nclasses, k=1, regularization='l2', regularization_strength=1.0):
        super().__init__()
        self.nclasses = nclasses
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        self.k = k

    def forward(self, output, target):
        ranks = torchsort.soft_rank(
            output, self.regularization, self.regularization_strength
        )
        ranks_label = ranks.gather(-1, target.view(-1, 1))
        loss = F.relu(self.nclasses - ranks_label - self.k + 1).mean()
        return loss
