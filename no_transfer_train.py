# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/11
Time: 14:58

"""
import torch
import torch.nn as nn
import numpy as np
import random
import torchvision
from torchvision import datasets, transforms
from data.datasets.triplet_mnist import TripletMNIST
from torch.utils.data import DataLoader, TensorDataset
from models.losses import TripletLoss
from models.network import TripletNet, EmbeddingNet, MLP_Embedding, Generator, Discriminator, MetaLearner
import torch.optim as optim
from matplotlib import pyplot as plt
from utils import util
import json
import time

torch.manual_seed(21)  # cpu
torch.cuda.manual_seed(21)  # gpu
np.random.seed(21)  # numpy
random.seed(21)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    for idx, (data, _) in enumerate(data_loader):
        data = tuple(d.to(device) for d in data)
        output = model(*data)
        loss = criterion(*output)
        total_loss += loss.item()
    return total_loss / len(data_loader)


def train(model, train_loader, optimizer, criterion, epochs, test_loader=None):
    plt_train_loss = []
    plt_val_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for idx, (data, _) in enumerate(train_loader):
            data = tuple(d.to(device) for d in data)
            optimizer.zero_grad()
            output = model(*data)
            loss = criterion(*output)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        plt_train_loss.append(train_loss)
        val_loss = None
        if test_loader is not None:
            val_loss = val(model, test_loader, criterion)
            plt_val_loss.append(val_loss)
        print('[epochs: {:3}/{:3}] Train loss: {:.4f} Val loss: {:.4f}'.format(epoch + 1, epochs, train_loss, val_loss))

    plt.plot(plt_train_loss, c='b')
    plt.plot(plt_val_loss, c='r')
    plt.show()

    loss_dict = {"train": plt_train_loss, "val": plt_val_loss}
    save_time = time.strftime("%Y-%m-%d--", time.localtime())
    util.save_loss("./results/" + save_time + 'no_meta_transfer_loss.json', loss_dict)


def main():
    samples_per_class = 600
    invert_rate = 0.1
    train_epochs = 200
    learning_rate = 1e-3
    batch_size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    triplet_train_dataset = TripletMNIST(train_dataset, samples_per_class, invert_rate)
    triplet_test_dataset = TripletMNIST(test_dataset, samples_per_class, invert_rate)
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = TripletNet(MLP_Embedding()).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss()

    train(model, train_loader, optimizer, criterion, epochs=train_epochs, test_loader=test_loader)


if __name__ == "__main__":
    main()
