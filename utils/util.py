# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/11
Time: 20:23

"""
import json
import time
from datetime import datetime
from matplotlib import pyplot as plt


def save_loss(file_name, loss_dict):
    save_path = './results/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '--'
    with open(save_path + file_name, 'w') as f:
        json.dump(loss_dict, f)


def load_loss(file_name):
    with open(file_name, 'r') as f:
        loss = json.load(f)
        return loss["train"], loss["val"]


def plot_loss(loss_files):
    colors = ['r', 'b', 'g', 'y', 'k', 'm', 'w', 'c']
    f, (ax1, ax2) = plt.subplots(1, 2)
    for idx, loss_file in enumerate(loss_files):
        train_loss, val_loss = load_loss(loss_file)
        ax1.plot(train_loss, c=colors[idx])
        ax2.plot(val_loss, c=colors[idx])
    ax1.set_title('Train loss')
    ax2.set_title('Val  loss')
    plt.show()


if __name__ == "__main__":
    meta_loss_file = "../results/2019-03-12-18-06--meta_loss.json"
    no_meta_loss_file = "../results/2019-03-12-08-48--no_meta_loss.json"
    no_GAN_meta_loss_file = '../results/2019-03-13-18-20--meta_loss_no_GAN.json'
    # first_order_small_loss = '../results/2019-03-13-17-13--only_small_first_order_grad_meta_loss.json'
    only_small_grad_meta1 = '../results/2019-03-13-21-34--with_clear_grad_no_GAN.json'
    only_small_grad_meta2 = '../results/2019-03-14-01-18--with_clear_grad_no_GAN.json'
    plot_loss((only_small_grad_meta1, only_small_grad_meta2))
