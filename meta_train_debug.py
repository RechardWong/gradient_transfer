# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/9
Time: 16:16

"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import random
from data.datasets.triplet_mnist import TripletMNIST
from torch.utils.data import DataLoader, TensorDataset
from models.losses import TripletLoss
from models.network import TripletNet, EmbeddingNet, MLP_Embedding, Generator, Discriminator, MetaLearner
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import util

torch.manual_seed(21)  # cpu
torch.cuda.manual_seed(21)  # gpu
np.random.seed(21)  # numpy
random.seed(21)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

batch_size = 128
epochs_B = 20
epochs_S = 10 * epochs_B
mid_epoch = 8  # 8
epochs_GAN = 200  # 200
GAN_batch_size = 4
invert_rate = 0.1
samples_per_class = 600
learning_rate = 1e-3
meta_alpha = 0.1
meta_beta = 5e-3

top_layer_name = "fc2"
layer_size = (256, 16)
meta_train = True
is_train_GAN = False
is_debug = False

if is_debug:
    epochs_B = 2
    epochs_S = 10 * epochs_B
    mid_epoch = 1  # 8

plt_train_loss = []
plt_val_loss = []

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

cuda = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mini_triplet_train_dataset = TripletMNIST(train_dataset, samples_per_class, invert_rate)
mini_triplet_test_dataset = TripletMNIST(test_dataset, samples_per_class, invert_rate)
triplet_train_dataset = TripletMNIST(train_dataset)
triplet_test_dataset = TripletMNIST(test_dataset)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader_B = DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader_B = DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_loader_S = DataLoader(mini_triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader_S = DataLoader(mini_triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_loader_S_query = DataLoader(mini_triplet_train_dataset, batch_size=batch_size, shuffle=False, **kwargs)

triplet_loss = nn.TripletMarginLoss()

triplet_net_B = TripletNet(MLP_Embedding()).to(device)
triplet_net_S = TripletNet(MLP_Embedding()).to(device)
triplet_net_Assist = TripletNet(MLP_Embedding()).to(device)
netG = Generator(layer_size).to(device)
netD = Discriminator(layer_size).to(device)

meta = MetaLearner(triplet_net_S, triplet_net_Assist, triplet_loss, device, meta_alpha=meta_alpha, meta_beta=meta_beta)

optim_B = optim.SGD(triplet_net_B.parameters(), lr=learning_rate)
optim_S = optim.SGD(triplet_net_S.parameters(), lr=learning_rate)
optim_G = optim.SGD(netG.parameters(), lr=learning_rate)
optim_D = optim.SGD(netD.parameters(), lr=learning_rate)


def train_GAN(netG, netD, data_loader, optimizerG, optimizerD, criterion, num_epochs):
    netD.train()
    netG.train()
    print("---------strat train GAN-----------------")
    print("len(GAN_data_loader):{}".format(len(data_loader)))
    for epoch in range(num_epochs):
        for i, (fake_data, real_data) in enumerate(data_loader):
            fake_label = torch.zeros(fake_data.size(0), 1).to(device)
            real_label = torch.ones(real_data.size(0), 1).to(device)
            fake_data.to(device)
            real_data.to(device)

            ############################
            # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizerD.zero_grad()
            # 使用fake data 训练
            gen_grads = netG(fake_data)
            fake_out = netD(gen_grads.detach())
            fake_loss = criterion(fake_out, fake_label)
            fake_loss.backward()
            D_fake1 = fake_out.mean().item()
            # 使用real data 训练
            real_out = netD(real_data)
            real_loss = criterion(real_out, real_label)
            real_loss.backward()
            D_real = real_out.mean().item()
            loss_D = real_loss + fake_loss
            optimizerD.step()

            ############################
            # (2) 更新 G 网络: 最大化 log(D(G(z)))
            ###########################
            optimizerG.zero_grad()
            output = netD(gen_grads)
            loss_G = criterion(output, real_label)
            loss_G.backward()
            D_fake2 = output.mean().item()
            optimizerG.step()

            if (epoch + 1) % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(data_loader), loss_D.item(), loss_G.item(), D_real, D_fake1,
                         D_fake2))


def val(model, data_loader, criterion):
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = tuple(d.to(device) for d in data)
            outputs = model(*data)
            loss = criterion(*outputs)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def train_one_batch(model, data, optimizer, criterion):
    model.train()
    data = tuple(d.to(device) for d in data)
    optimizer.zero_grad()
    outputs = model(*data)
    loss = criterion(*outputs)
    loss_value = loss.item()
    loss.backward()
    optimizer.step()

    return loss_value


def train_one_epoch(model, train_loader, optimizer, criterion):
    total_loss = 0.0
    model.train()
    for i, (data, _) in enumerate(train_loader):
        data = tuple(d.to(device) for d in data)
        optimizer.zero_grad()
        outputs = model(*data)
        loss = criterion(*outputs)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss_value

    return total_loss / len(train_loader)


def get_params(model, lay_name='fc2'):
    params = None
    for name, param in model.named_parameters():
        if lay_name in name and "weight" in name:
            params = param
    return params


def plot_save_loss(compare_file, save_name):
    no_meta_train_loss, no_meta_val_loss = util.load_loss(compare_file)
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(plt_train_loss, c="r")
    ax2.plot(plt_val_loss, c="r")

    ax1.plot(no_meta_train_loss, c="b")
    ax2.plot(no_meta_val_loss, c="b")
    ax1.set_title('Train loss')
    ax2.set_title('Val  loss')
    plt.show()

    save_loss_dict = {'train': plt_train_loss, 'val': plt_val_loss}
    util.save_loss(save_name, save_loss_dict)


def trainer(triplet_net_S, triplet_net_B, mid_epoch, meta_train=True):
    print("start training...")
    print("len(train_loader_B)={}, len(train_loader_S)={}.".format(len(train_loader_B), len(train_loader_S)))

    epoch_S = 0
    dif_params_B = []
    dif_params_S = []

    params_B1 = get_params(triplet_net_B).clone()
    params_S1 = get_params(triplet_net_S).clone()

    if not meta_train:
        mid_epoch = epochs_B

    batch_num_S = len(train_loader_S)
    for epoch_B in range(mid_epoch):
        total_loss_B = 0.0

        for idx_B, (data_B, _) in enumerate(train_loader_B):
            if meta_train:
                total_loss_B += train_one_batch(triplet_net_B, data_B, optim_B, triplet_loss)

            # train small data model one epoch
            if idx_B % batch_num_S == 0:
                train_loss_S = train_one_epoch(triplet_net_S, train_loader_S, optim_S, triplet_loss)
                val_loss_S = val(triplet_net_S, test_loader_S, triplet_loss)
                plt_train_loss.append(train_loss_S)
                plt_val_loss.append(val_loss_S)
                template = '[Small Dataset {:3}/{:3}] Train loss: {:.4f} Val loss: {:.4f}'
                print(template.format(epoch_S + 1, epochs_S, train_loss_S, val_loss_S))
                epoch_S += 1

                params_B2 = get_params(triplet_net_B).clone()
                params_S2 = get_params(triplet_net_S).clone()
                dif_params_B.append(params_B2 - params_B1)
                dif_params_S.append(params_S2 - params_S1)
                params_B1 = params_B2
                params_S1 = params_S2

        train_loss_B = total_loss_B / len(train_loader_B)
        val_loss_B = val(triplet_net_B, test_loader_B, triplet_loss)
        template = '[Big   Dataset {:3}/{:3}] Train loss: {:.4f} Val loss: {:.4f}'
        print(template.format(epoch_B + 1, epochs_B, train_loss_B, val_loss_B))

    if not meta_train:
        return None

    if is_train_GAN:
        print("start train  GAN...")
        dif_params_B, dif_params_S = torch.stack(dif_params_B), torch.stack(dif_params_S)
        print("GAN train size ", dif_params_B.size())

        GAN_dataset = TensorDataset(dif_params_B, dif_params_S)
        GAN_loader = DataLoader(GAN_dataset, batch_size=GAN_batch_size)
        GAN_criterion = torch.nn.BCELoss()
        train_GAN(netG, netD, GAN_loader, optim_G, optim_D, GAN_criterion, epochs_GAN)

    for epoch_B in range(mid_epoch, epochs_B):
        total_loss_B = 0.0
        total_loss_S = 0.0
        for idx_B, (data_B, _) in enumerate(train_loader_B):
            total_loss_B += train_one_batch(triplet_net_B, data_B, optim_B, triplet_loss)
            if idx_B % batch_num_S == 0:
                support_data_list = list(enumerate(train_loader_S))
                query_data_list = list(enumerate(train_loader_S_query))

            _, (support_data, _) = support_data_list[idx_B % batch_num_S]
            _, (query_data, _) = query_data_list[idx_B % batch_num_S]

            # 原来是所有 'fc' 层？
            params_B = get_params(triplet_net_B)  # params_B.size(): torch.Size([16, 256])

            grad_B2S = grad_B = params_B.grad.clone()
            if is_train_GAN:
                netG.eval()
                x, y = grad_B.size()
                grad_B2S = netG(grad_B.unsqueeze(0))
                grad_B2S = grad_B2S.data.view(x, y)
            para_map = dict()
            para_map[top_layer_name] = grad_B2S
            optim_S.zero_grad()
            loss = meta(para_map, support_data, query_data)
            total_loss_S += loss

            # small data finished one epoch
            if idx_B % batch_num_S == batch_num_S - 1 or idx_B == len(train_loader_B) - 1:
                batch_loss_num = (idx_B % batch_num_S) + 1
                meta_loss_S = total_loss_S / batch_loss_num
                total_loss_S = 0.0
                val_loss_S = val(triplet_net_S, test_loader_S, triplet_loss)
                train_loss_S = val(triplet_net_S, train_loader_S, triplet_loss)
                plt_train_loss.append(train_loss_S)
                plt_val_loss.append(val_loss_S)

                template = "[Meta][Small Dataset {:3}/{:3}] Meta loss: {:.4f} Val loss: {:.4f}"
                print(template.format(epoch_S + 1, epochs_S, meta_loss_S, val_loss_S))

                # template = "[Meta][Small Dataset {:3}/{:3}] Meta loss: {:.4f} Train loss: {:.4f} Val loss: {:.4f}"
                # print(template.format(epoch_S + 1, epochs_S, meta_loss_S, train_loss_S, val_loss_S))

                epoch_S += 1

        # big data finished one epoch
        train_loss_B = total_loss_B / len(train_loader_B)
        val_loss_B = val(triplet_net_B, test_loader_B, triplet_loss)
        template = "[Big   Dataset {:3}/{:3}] Train loss: {:.4f} Val loss: {:.4f}"
        print(template.format(epoch_B + 1, epochs_B, train_loss_B, val_loss_B))


if __name__ == "__main__":
    trainer(triplet_net_S, triplet_net_B, mid_epoch, meta_train)
    compare_file = './results/2019-03-14-11-20--no_meta_benchmark.json'
    plot_save_loss(compare_file, save_name='only_small_grad.json')
