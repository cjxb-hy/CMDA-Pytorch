import os
import numpy as np

import torch
from torch import embedding, optim
from torch.utils.data import DataLoader

from myloss import MyLoss
from datasetnpy import Data_Loader
from lib import _label_decomp
import source_segmenter as drn


def train():

    output_path = "./models/mr_baseline/"

    restore = True  # set True if resume training from stored model
    restored_path = output_path
    lr_update_flag = False  # Set True if want to use a new learning rate for fine-tuning

    num_cls = 5
    batch_size = 2
    epochs = 5
    optimizer = 'adam'
    device = 'cpu'

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

    train_set = Data_Loader('./data/mr_train')
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)

    device = torch.device(device)
    net = drn.Full_DRN(channels=3, n_class=num_cls, batch_size=batch_size)
    net = net.to(device)
    criterion = MyLoss(net, num_cls, cost_kwargs).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    best_loss = 1000
    for epoch in range(epochs):
        net.train()
        cost = 0
        for batch, [image, label] in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            label = _label_decomp(num_cls, label)
            output = net(image)

            cost, reg = criterion(output, label)
            loss = torch.add(cost, reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:{}, Batch:{} finish!'.format(epoch, batch))

        if epoch % 500 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.9

        print('Training at Epoch: {}, Dice Loss is: {}'.format(epoch, cost.item()))

        if cost < best_loss:
            best_loss = cost
            path = os.path.join(output_path, 'model{}.pth'.format(epoch))
            torch.save(net.state_dict(), path)
            print('Epoch: {}, save model:{}'.format(epoch, path))


if __name__ == "__main__":
    train()
