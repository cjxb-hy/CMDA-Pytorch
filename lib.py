import os
import numpy as np

import torch


def _read_lists(fid):
    if not os.path.isfile(fid):
        return None

    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 3:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])

    return my_list


def _label_decomp(num_cls, label_vol):

    label_vol = label_vol.numpy()
    _batch_shape = list(label_vol.shape)  # 2,1,512,512
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    #_vol = _vol[..., np.newaxis]
    for i in range(num_cls):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i] = 1
        _vol = np.concatenate((_vol, _n_slice), axis=1)
    return torch.from_numpy(_vol)


if __name__ == "__main__":
    # fid = "./lists/mr_train_list"
    # _list = _read_lists(fid)
    # print(_list)
    x = torch.randn([2, 1, 2, 2])
    # print(x.shape)
    # y = torch.nn.functional.softmax(x, dim=1)
    # print(y.shape)
    #x = np.zeros([2, 2, 2, 1])
    print(x.shape)
    y = _label_decomp(5, x)
    print(y.shape)
