# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/9
Time: 15:13

"""
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing

    sample_per_label: number of samples per class
    pro: probability of inverted samples
    """

    def __init__(self, mnist_dataset, samples_per_class=None, prob=None):
        np.random.seed(21)
        torch.manual_seed(21)
        torch.cuda.manual_seed_all(21)
        random_state = np.random.RandomState(21)

        self.rs = random_state
        self.prob = prob

        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        self.targets = self.mnist_dataset.targets
        self.data = self.mnist_dataset.data

        self.targets_set = set(self.targets.numpy())
        self.targets_to_indices = {target: np.where(self.targets.numpy() == target)[0]
                                   for target in self.targets_set}

        # 每一张测试数据都给了一个triplet. e.g.【2,3211,3869】
        self.triplets = []
        for i in range(len(self.data)):
            positive_index = random_state.choice(self.targets_to_indices[self.targets[i].item()])
            negative_target = np.random.choice(list(self.targets_set - set([self.targets[i].item()])))
            negative_index = random_state.choice(self.targets_to_indices[negative_target])
            self.triplets.append([i, positive_index, negative_index])
        # self.test_triplets = triplets

        # for small data: targets_to_indices changed to small data indices.
        if samples_per_class is not None:
            self.targets_to_indices = {target: random_state.choice(self.targets_to_indices[target],
                                                                   samples_per_class)
                                       for target in self.targets_set}

        self.data_indices = []
        for target in self.targets_set:
            self.data_indices += list(self.targets_to_indices[target])

    def __getitem__(self, index):
        if self.train:
            index = self.data_indices[index]
            img, target = self.data[index], self.targets[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.targets_to_indices[target])
            negative_target = np.random.choice(list(self.targets_set - set([target])))
            negative_index = np.random.choice(self.targets_to_indices[negative_target])
            positive = self.data[positive_index]
            negative = self.data[negative_index]
        else:
            img, positive, negative = self.data[self.triplets[index]]

        img = Image.fromarray(img.numpy(), mode='L')
        positive = Image.fromarray(positive.numpy(), mode='L')
        negative = Image.fromarray(negative.numpy(), mode='L')

        if self.prob is not None:
            r = self.rs.rand(3)
            if r[0] <= self.prob:
                img = ImageOps.invert(img)
            if r[1] <= self.prob:
                positive = ImageOps.invert(positive)
            if r[2] <= self.prob:
                negative = ImageOps.invert(negative)

        if self.transform is not None:
            img = self.transform(img)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return (img, positive, negative), []

    def __len__(self):
        return len(self.data_indices)
