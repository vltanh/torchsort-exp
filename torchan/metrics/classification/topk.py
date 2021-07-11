import torch
from sklearn.metrics import top_k_accuracy_score

__all__ = ['TopK']


class TopK():
    def __init__(self, k, nclasses):
        self.k = k
        self.labels = list(range(nclasses))
        self.reset()

    def update(self, output, target):
        self.pred += output.cpu().tolist()
        self.target += target.cpu().tolist()

    def reset(self):
        self.pred = []
        self.target = []

    def value(self):
        return top_k_accuracy_score(self.target, self.pred,
                                    k=self.k, labels=self.labels)

    def summary(self):
        topk = top_k_accuracy_score(self.target, self.pred,
                                    k=self.k, labels=self.labels)
        print(f'+ Top-{self.k}: {topk}')


if __name__ == '__main__':
    auc = TopK(1, 4)
    auc.update(
        torch.tensor([[.1, .2, .8, .3],
                      [.1, .1, .8, .0],
                      [.1, .5, .2, .2]]),
        torch.tensor([2, 1, 3])
    )
    auc.summary()
    auc.update(
        torch.tensor([[.9, .1, .0, .0],
                      [.6, .2, .1, .1],
                      [.7, .0, .3, .0]]),
        torch.tensor([1, 1, 2])
    )
    auc.summary()
