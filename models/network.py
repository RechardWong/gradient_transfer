# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/9
Time: 16:11

"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from torch.optim import lr_scheduler
import torch.optim as optim

torch.manual_seed(21)
torch.cuda.manual_seed_all(21)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MLP_Embedding(nn.Module):
    def __init__(self):
        super(MLP_Embedding, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28 * 28, 256),
                                 # nn.PReLU(),
                                 nn.Linear(256, 256),
                                 # nn.PReLU(),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(256, 16, bias=False))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        output = self.fc2(output)
        return output


class Embed_1_layer(nn.Module):
    def __init__(self):
        super(Embed_1_layer, self).__init__()
        self.fc = nn.Linear(28 * 28, 16)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        output = self.fc(x)
        return output


class Embed_2_layers(nn.Module):
    def __init__(self):
        super(Embed_2_layers, self).__init__()
        self.embed = nn.Linear(28 * 28, 256)
        self.fc = nn.Linear(256, 16)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.embed(x)
        output = self.fc(x)
        return output


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.dim = dim
        self.size = self.dim[0] * self.dim[1]
        self.fc = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Linear(256, self.size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), self.dim[0], self.dim[1])


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.size = self.dim[0] * self.dim[1]
        self.fc = nn.Sequential(
            nn.Linear(self.size, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LinearRegression(nn.Module):
    def __init__(self, dim):
        super(LinearRegression, self).__init__()
        self.dim = dim
        self.size = self.dim[0] * self.dim[1]
        self.fc = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Linear(256, self.size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), self.dim[0], self.dim[1])


class MetaLearner(nn.Module):
    def __init__(self, learner, assistLearner, criterion, device, meta_alpha=1e-4, meta_beta=5e-3):
        super(MetaLearner, self).__init__()
        self.learner = learner
        self.assistLearner = assistLearner
        self.criterion = criterion
        self.device = device
        self.meta_alpha = meta_alpha
        self.meta_beta = meta_beta
        self.optim_a = optim.Adam(self.assistLearner.parameters(), lr=1e-3)

    def forward(self, para_map, support_set, query_set):
        self.assistLearner.train()
        self.assistLearner.load_state_dict(copy.deepcopy(self.learner.state_dict()))

        data = tuple(d.to(self.device) for d in support_set)
        query = tuple(d.to(self.device) for d in query_set)

        top_layer_name = list(para_map.keys())[0]
        grad_B2S = list(para_map.values())[0]
        # print("g_B2S",g_B2S)

        self.optim_a.zero_grad()
        assist_out = self.assistLearner(*data)
        loss_outputs = self.criterion(*assist_out)
        loss_0 = loss_outputs.item()
        loss_outputs.backward()

        s_grads = [(name, param.grad.clone()) for name, param in self.assistLearner.named_parameters()]
        b_grads = [(name, param.grad.clone()) for name, param in self.assistLearner.named_parameters()]
        # print("sss", s_grads)
        for i, (name, grad) in enumerate(b_grads):
            if top_layer_name in name and "weight" in name:
                # print(b_grads[i][1].size(),g_B2S.size())
                b_grads[i] = (name, grad_B2S.clone())
        # print("bbb", b_grads)

        # theta1:(theta - delta theta) 2:(theta - delta theta)
        theta_small_state_dict = copy.deepcopy(self.learner.state_dict())
        theta_big_state_dict = copy.deepcopy(self.learner.state_dict())

        for (name, grad) in s_grads:
            theta_small_state_dict[name] -= self.meta_alpha * grad
        for (name, grad) in b_grads:
            theta_big_state_dict[name] -= self.meta_alpha * grad

        self.assistLearner.train()
        self.assistLearner.load_state_dict(theta_small_state_dict)
        assist_out = self.assistLearner(*query)
        loss_outputs = self.criterion(*assist_out)
        loss_1 = loss_outputs.item()
        loss_outputs.backward()

        # get grad
        grad_1 = [(name, param.grad.clone()) for name, param in self.assistLearner.named_parameters()]

        self.assistLearner.train()
        self.assistLearner.load_state_dict(theta_big_state_dict)
        assist_out = self.assistLearner(*query)
        loss_outputs = self.criterion(*assist_out)
        loss_2 = loss_outputs.item()
        loss_outputs.backward()
        # get grad
        grad_2 = [(name, param.grad.clone()) for name, param in self.assistLearner.named_parameters()]
        # print("grad_1", grad_1[0][1].size())
        # print("grad_2", grad_2[0][1].size())
        # print("before", self.learner.state_dict())
        for i, (name, grad) in enumerate(grad_2):
            self.learner.state_dict()[name] -= (self.meta_beta * (grad + grad_1[i][1]) / 2)
        # print("after",self.learner.state_dict())
        # query_loss = (loss_2+loss_1)/2
        query_loss = loss_0
        return query_loss