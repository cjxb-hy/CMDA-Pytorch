import os
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from myloss import MyLoss
from dataset import Data_Loader
from lib import _read_lists, _label_decomp
import source_segmenter as drn


def train():

    train_fid = "./lists/mr_train_list"
    val_fid = "./lists/mr_val_list"
    output_path = "./tmp_exps/mr_baseline"

    restore = True  # set True if resume training from stored model
    restored_path = output_path
    lr_update_flag = False  # Set True if want to use a new learning rate for fine-tuning

    num_cls = 2
    batch_size = 2
    epochs = 5
    optimizer = 'adam'

    cost_kwargs = {
        "cross_flag": True,  # use cross entropy loss
        "miu_cross": 1.0,
        "dice_flag": True,  # use dice loss
        "miu_dice": 1.0,
        "regularizer": 1e-4
    }

    # try:
    #     os.makedirs(output_path)
    # except:
    #     print("folder exist!")

    # train_list = _read_lists(train_fid)
    # val_list = _read_lists(val_fid)

    train_set = Data_Loader('data/train')
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)

    net = drn.Full_DRN(channels=3, n_class=num_cls, batch_size=batch_size)
    criterion = MyLoss(net, num_cls, cost_kwargs)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        net.train()
        loss = 0
        for image, label in train_loader:
            #image, label = image.to(), label.to()
            label = _label_decomp(num_cls, label)
            output = net(image)
            print('one batch finish,start to get loss!')
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Loss/Train: {}'.format(epoch, loss.item()))


if __name__ == "__main__":
    train()
