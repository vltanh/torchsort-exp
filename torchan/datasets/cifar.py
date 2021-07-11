from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as tvtf

__all__ = ['CIFARDataset']


def CIFAR(version):
    assert version in ['cifar10', 'cifar100'], f'Invalid version [{version}]!'
    if version == 'cifar10':
        return CIFAR10
    elif version == 'cifar100':
        return CIFAR100


class CIFARDataset:
    def __init__(self, version='cifar10', data_dir='data', is_train=True):
        self.data = CIFAR(version)(data_dir, train=is_train, download=True)
        if is_train:
            self.transform = tvtf.Compose([
                tvtf.ToTensor(),
                tvtf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            self.transform = tvtf.Compose([
                tvtf.ToTensor(),
                tvtf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

    def __getitem__(self, i):
        img, lbl = self.data[i]
        img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.data)
