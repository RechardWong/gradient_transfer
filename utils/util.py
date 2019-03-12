# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/11
Time: 20:23

"""
import json
import time
from matplotlib import pyplot as plt

save_path = './results/' + time.strftime("%Y-%m-%d-%H-%M", time.localtime()) + '--'


def save_loss(file_name, loss_dict):
    with open(save_path + file_name, 'w') as f:
        json.dump(loss_dict, f)


def load_loss(file_name):
    with open(file_name, 'r') as f:
        loss = json.load(f)
        return loss["train"], loss["val"]


def plot_loss(meta_loss_file, no_meta_loss_file):
    meta_train_loss, meta_val_loss = load_loss(meta_loss_file)
    no_meta_train_loss, no_meta_val_loss = load_loss(no_meta_loss_file)
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(meta_train_loss, c="r")
    ax2.plot(meta_val_loss, c="r")

    ax1.plot(no_meta_train_loss, c="b")
    ax2.plot(no_meta_val_loss, c="b")
    ax1.set_title('Train loss')
    ax2.set_title('Val  loss')
    plt.show()


if __name__ == "__main__":
    meta_loss_file = "../results/2019-03-12-08-49--meta_loss.json"
    no_meta_loss_file = "../results/2019-03-12-08-48--no_meta_loss.json"
    plot_loss(meta_loss_file, no_meta_loss_file)
