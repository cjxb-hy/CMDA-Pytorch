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


def pixel_wise_softmax_2(output_map):
    b, c, h, w = output_map.shape
    exponential_map = torch.exp(output_map)
    sum_exp = torch.sum(exponential_map, dim=1, keepdim=True)
    tensor_sum_exp = sum_exp.expand(b, c, h, w)
    return torch.clamp(torch.div(exponential_map, tensor_sum_exp), -1.0*1e15, 1.0*1e15)


def _label_decomp(num_cls, label_vol):

    label_vol = label_vol.numpy()
    _batch_shape = list(label_vol.shape)
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


def _dice_eval(logits, label, n_class):

    dice_arr = []
    dice = 0
    eps = 1e-7

    label = _label_decomp(num_cls=n_class, label_vol=label)  # b,5,h,w

    predicter = pixel_wise_softmax_2(logits)  # b,5,h,w
    predicter = predicter.permute(0, 2, 3, 1)
    compact_pred = torch.argmax(predicter, dim=3)  # b,h,w
    pred = torch.nn.functional.one_hot(compact_pred, n_class)  # b,h,w,5
    pred = pred.permute(0, 3, 1, 2)
    for i in range(n_class):
        inse = torch.sum(pred[:, i, :, :] * label[:, i, :, :])
        union = torch.sum(pred[:, i, :, :]) + torch.sum(label[:, i, :, :])
        dice = dice + 2.0 * inse / (union + eps)
        dice_arr.append(2.0 * inse / (union + eps))

    return 1.0 * dice / n_class, dice_arr


def load_model(net, model_path):

    save_model = torch.load(model_path)
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items()
                  if k in model_dict.keys()}
    model_dict.update(state_dict)
    return model_dict


def ct_copy_model(net, model_path):

    save_model = torch.load(model_path)
    model_dict = net.state_dict()
    state_dict = {k1: v2 for (k1, v1), (k2, v2) in zip(
        model_dict.items(), save_model.items())}
    model_dict.update(state_dict)
    return model_dict


if __name__ == "__main__":
    # fid = "./lists/mr_train_list"
    # _list = _read_lists(fid)
    # print(_list)
    # x = torch.randn([2, 1, 2, 2])
    # print(x.shape)
    # y = torch.nn.functional.softmax(x, dim=1)
    # print(y.shape)
    #x = np.zeros([2, 2, 2, 1])
    # print(x.shape)
    # y = _label_decomp(5, x)
    # print(y.shape)
    # predicter = torch.randn([2, 5, 256, 256])
    # predicter = pixel_wise_softmax_2(predicter)
    # predicter = predicter.permute(0, 2, 3, 1)
    # compact_pred = torch.argmax(predicter, dim=3)  # b,h,w
    # print(compact_pred.shape)
    # pred = torch.nn.functional.one_hot(compact_pred, 5)  # b,h,w,5
    # pred = pred.permute(0, 3, 1, 2)
    # print(pred.shape)
    pass
