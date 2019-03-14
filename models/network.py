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
    def __init__(self, learner, assist_learner, criterion, device, meta_alpha=1e-4, meta_beta=5e-3):
        super(MetaLearner, self).__init__()
        self.learner = learner
        self.assist_learner = assist_learner
        self.criterion = criterion
        self.device = device
        self.meta_alpha = meta_alpha
        self.meta_beta = meta_beta
        self.optim_a = optim.Adam(self.assist_learner.parameters(), lr=1e-3)

    def forward(self, para_map, support_set, query_set):
        self.assist_learner.train()
        self.assist_learner.load_state_dict(copy.deepcopy(self.learner.state_dict()))

        data = tuple(d.to(self.device) for d in support_set)
        query = tuple(d.to(self.device) for d in query_set)

        top_layer_name = list(para_map.keys())[0]
        grad_B2S = list(para_map.values())[0]
        # print("g_B2S",g_B2S)

        self.optim_a.zero_grad()
        assist_out = self.assist_learner(*data)
        loss_outputs = self.criterion(*assist_out)
        loss_0 = loss_outputs.item()
        loss_outputs.backward()

        s_grads = [(name, param.grad.clone()) for name, param in self.assist_learner.named_parameters()]
        b_grads = [(name, param.grad.clone()) for name, param in self.assist_learner.named_parameters()]
        # print("sss", s_grads)
        for i, (name, grad) in enumerate(b_grads):
            if top_layer_name in name and "weight" in name:
                b_grads[i] = (name, grad_B2S.clone())

        # theta1:(theta - delta theta) 2:(theta - delta theta)
        theta_small_state_dict = copy.deepcopy(self.learner.state_dict())
        theta_big_state_dict = copy.deepcopy(self.learner.state_dict())

        for (name, grad) in s_grads:
            theta_small_state_dict[name] -= self.meta_alpha * grad
        for (name, grad) in b_grads:
            theta_big_state_dict[name] -= self.meta_alpha * grad

        self.assist_learner.train()
        self.assist_learner.load_state_dict(theta_small_state_dict)

        self.optim_a.zero_grad()
        assist_out = self.assist_learner(*query)
        loss_outputs = self.criterion(*assist_out)
        loss_s = loss_outputs.item()
        loss_outputs.backward()
        # get grad
        grad_s = [(name, param.grad.clone()) for name, param in self.assist_learner.named_parameters()]

        self.assist_learner.train()
        self.assist_learner.load_state_dict(theta_big_state_dict)

        self.optim_a.zero_grad()
        assist_out = self.assist_learner(*query)
        loss_outputs = self.criterion(*assist_out)
        loss_b = loss_outputs.item()
        loss_outputs.backward()
        # get grad
        grad_b = [(name, param.grad.clone()) for name, param in self.assist_learner.named_parameters()]

        # diff_grad_sum = sum(torch.sum(torch.abs(grad - grad_s[i][1])) for i, (name, grad) in enumerate(grad_b))
        # print("diff between grad_s and grad_b: %4.f" % diff_grad_sum)

        # print("before", self.learner.state_dict())
        for i, (name, grad) in enumerate(grad_b):
            # meta learn
            self.learner.state_dict()[name] -= (self.meta_beta * (grad + grad_s[i][1]) / 2)

            # only big grad
            # self.learner.state_dict()[name] -= (self.meta_beta * grad)
            
            # only small grad
            # self.learner.state_dict()[name] -= (self.meta_beta * grad_s[i][1])

        # only small first order grad
        # self.learner.load_state_dict(theta_small_state_dict)

        # print("after",self.learner.state_dict())
        # query_loss = (loss_2+loss_1)/2
        query_loss = loss_0
        return query_loss
