import torch
from torch import nn


class MyLoss(nn.Module):
    def __init__(self, net, n_class, cost_kwargs):
        super(MyLoss, self).__init__()
        self.net = net
        self.n_class = n_class
        self.cost_kwargs = cost_kwargs

    def _softmax_weighted_loss(self, logits, label):

        softmaxpred = torch.nn.functional.softmax(logits, dim=1)
        raw_loss = 0
        for i in range(self.n_class):
            gti = label[:, i, :, :]
            predi = softmaxpred[:, i, :, :]
            weighted = 1 - (torch.sum(gti) / torch.sum(label))
            if i == 0:
                raw_loss = -1.0 * weighted * gti * \
                    torch.log(torch.clamp(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * weighted * gti * \
                    torch.log(torch.clamp(predi, 0.005, 1))

        loss = torch.mean(raw_loss)
        return loss

    def _dice_loss_fun(self, logits, label):

        dice = 0
        eps = 1e-7
        softmaxpred = torch.nn.functional.softmax(logits, dim=1)

        for i in range(self.n_class):
            inse = torch.sum(softmaxpred[:, i, :, :] * label[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
            r = torch.sum(label[:, i, :, :] * label[:, i, :, :])
            dice += 2.0 * inse / (l + r + eps)

        return 1.0 - dice / self.n_class

    def _get_cost(self, logits, label):

        loss = 0
        # dice_flag = self.cost_kwargs.pop("dice_flag", True)
        # cross_flag = self.cost_kwargs.pop("cross_flag", False)
        # miu_dice = self.cost_kwargs.pop("miu_dice", 1.0)
        # miu_cross = self.cost_kwargs.pop("miu_cross", 1.0)
        # reg_coeff = self.cost_kwargs.pop("regularizer", 1e-4)
        dice_flag = True
        cross_flag = False
        miu_dice = 1.0
        miu_cross = 1.0
        reg_coeff = 1e-4

        if cross_flag is True:
            weighted_loss = self._softmax_weighted_loss(logits, label)
            loss += miu_cross * weighted_loss

        if dice_flag is True:
            dice_loss = self._dice_loss_fun(logits, label)
            loss += miu_dice * dice_loss

        regularizers = sum([torch.norm(parameters) for name,
                            parameters in self.net.state_dict().items() if 'weight' in name and '.2.' not in name and 'extra.1' not in name])

        return loss, reg_coeff * regularizers

    def forward(self, logits, label):
        return self._get_cost(logits, label)


# # class My_loss(nn.Module):
# #     def __init__(self):
# #         super(My_loss, self).__init__()

# #     def forward(self, x, y):  # 定义前向的函数运算
# #         return torch.mean(torch.pow((x - y), 2))

# # 在使用这个损失函数的时候只需要如下即可：
# # criterion = My_loss()
# # loss = criterion(outputs, targets)
# def pixel_wise_softmax_2(output_map):
#     b, c, h, w = output_map.shape
#     exponential_map = torch.exp(output_map)
#     sum_exp = torch.sum(exponential_map, dim=1, keepdim=True)
#     tensor_sum_exp = sum_exp.expand(b, c, h, w)
#     return torch.clamp(torch.div(exponential_map, tensor_sum_exp), -1.0*1e15, 1.0*1e15)


# def _get_cost(net, logits, label, cost_kwargs):

#     loss = 0
#     dice_flag = cost_kwargs.pop("dice_flag", True)
#     cross_flag = cost_kwargs.pop("cross_flag", False)
#     miu_dice = cost_kwargs.pop("miu_dice", None)
#     miu_cross = cost_kwargs.pop("miu_cross", None)
#     reg_coeff = cost_kwargs.pop("regularizer", 1e-4)

#     if cross_flag is True:
#         weighted_loss = _softmax_weighted_loss(logits, label)
#         loss += miu_cross * weighted_loss

#     if dice_flag is True:
#         dice_loss = _dice_loss_fun(logits, label)
#         loss += miu_dice * dice_loss

#     dice_eval, dice_eval_arr = _dice_eval(logits, label)
#     dice_eval_c1 = dice_eval_arr[1]
#     dice_eval_c2 = dice_eval_arr[2]
#     dice_eval_c3 = dice_eval_arr[3]
#     dice_eval_c4 = dice_eval_arr[4]

#     regularizers = sum([torch.norm(parameters) for name,
#                         parameters in net.state_dict().items() if 'weight' in name])

#     return loss + reg_coeff * regularizers


# def _softmax_weighted_loss(logits, label):

#     softmaxpred = torch.nn.Softmax(logits)
#     raw_loss = 0
#     for i in range(n_class):
#         gti = label[:, i, :, :]
#         predi = softmaxpred[:, i, :, :]
#         weighted = 1 - (torch.sum(gti) / torch.sum(label))
#         if i == 0:
#             raw_loss = -1.0 * weighted * gti * \
#                 torch.log(torch.clamp(predi, 0.005, 1))
#         else:
#             raw_loss += -1.0 * weighted * gti * \
#                 torch.log(torch.clamp(predi, 0.005, 1))

#     loss = torch.mean(raw_loss)
#     return loss


# def _dice_loss_fun(logits, label):

#     dice = 0
#     eps = 1e-7
#     softmaxpred = torch.nn.Softmax(logits)

#     for i in range(n_class):
#         inse = torch.sum(softmaxpred[:, i, :, :] * label[:, i, :, :])
#         l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
#         r = torch.sum(label[:, i, :, :])
#         dice += 2.0 * inse / (l + r + eps)

#     return -1.0 * dice / n_class


# def _dice_eval(logits, label):

#     dice_arr = []
#     dice = 0
#     eps = 1e-7

#     predicter = pixel_wise_softmax_2(logits)
#     compact_pred = torch.argmax(predicter, 1)
#     pred = torch.nn.functional.one_hot(compact_pred, n_class)
#     pred = pred.permute(0, 3, 1, 2)
#     for i in range(n_class):
#         inse = torch.sum(pred[:, i, :, :] * label[:, i, :, :])
#         union = torch.sum(pred[:, i, :, :]) + torch.sum(label[:, i, :, :])
#         dice = dice + 2.0 * inse / (union + eps)
#         dice_arr.append(2.0 * inse / (union + eps))

#     return 1.0 * dice / n_class, dice_arr
