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
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    f, (ax1, ax2) = plt.subplots(1, 2)
    for idx, loss_file in enumerate(loss_files):
        train_loss, val_loss = load_loss(loss_file)
        ax1.plot(train_loss, c=colors[idx])
        ax2.plot(val_loss, c=colors[idx])
    ax1.set_title('Train loss')
    ax2.set_title('Val  loss')
    plt.show()


if __name__ == "__main__":
    no_meta_benchmark = "../results/2019-03-14-11-20--no_meta_benchmark.json"
    meta_learn = "../results/2019-03-14-13-10--meta_learn.json"
    meta_no_GAN = '../results/2019-03-14-12-15--meta_no_GAN.json'
    only_small_grad = '../results/2019-03-14-12-12--only_small_grad.json'
    only_big_grad = '../results/2019-03-14-13-08--only_big_grad.json'
    # first_order_small_loss = '../results/'
    # plot_loss((no_meta_benchmark, meta_learn, meta_no_GAN, only_small_grad, only_big_grad))
    plot_loss((meta_no_GAN, only_small_grad, only_big_grad))
