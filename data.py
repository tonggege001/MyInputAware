import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np




def get_data(batch_size):
    train_dataset = torchvision.datasets.CIFAR10("data/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10("data/", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader









if __name__ == "__main__":
    get_data(100)

