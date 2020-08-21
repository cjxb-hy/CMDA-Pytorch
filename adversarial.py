import __future__
import numpy as np

import torch
from torch import logical_and, nn
from torch.nn import functional as F

keep_prob = 0.75


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, padding=1, dilation=1):
        super(ResBlk, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
        )

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1,
                          stride=1, dilation=dilation),
                nn.BatchNorm2d(ch_out)
            )

        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        out = self.extra(x) + out
        out = self.lrelu(out)
        return out


class Conv2d_Sym(nn.Module):
    def __init__(self, ch_in, ch_out,  offset, kernel_size=3, stride=1, keep_prob=0.75):
        super(Conv2d_Sym, self).__init__()

        self.offset = [offset, offset, offset, offset]
        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
                      stride=stride, padding=0),
            nn.Dropout(keep_prob)
        )

    def forward(self, x):

        x = F.pad(x, self.offset, mode='replicate')
        x = self.model(x)

        return x


class PS(nn.Module):
    def __init__(self, r, n_channel, batch_size):
        super(PS, self).__init__()
        self.r = r
        self.n_channel = n_channel
        self.batch_size = batch_size

    def _phase_shift(self, image, r, batch_size):
        _, c, h, w = image.shape  # b,c,h,w
        X = image.permute(0, 2, 3, 1)  # b,h,w,c
        X = image.view(batch_size, h, w, r, r)  # b,h,w,r,r
        X = X.permute(0, 1, 2, 4, 3)  # b,h,w,r,r
        X = torch.chunk(X, h, 1)  # b,w,r,r *h
        X = torch.cat([x.squeeze() for x in X], 2)  # b,w,h*r,r
        X = torch.chunk(X, w, 1)  # b,h*r,r *w
        X = torch.cat([x.squeeze() for x in X], 2)  # b,h*r,w*r
        X = X.unsqueeze(3)  # b,h*r,w*r,1
        out = X.permute(0, 3, 1, 2)  # b,1,h*r,w*r
        return out

    def forward(self, x):
        X = torch.chunk(x, self.n_channel, 1)
        X = torch.cat([self._phase_shift(x, self.r, self.batch_size)
                       for x in X], 1)
        return X


class Group_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Group_1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(keep_prob),
        )
        self.blk = ResBlk(ch_out, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv(x)
        x = self.blk(x)
        x = self.pool(x)
        #print('Group1', x.shape)
        return x


class Group_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Group_2, self).__init__()

        self.blk = ResBlk(ch_in, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.blk(x)
        x = self.pool(x)
        #print('Group2', x.shape)

        return x


class Group_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Group_3, self).__init__()

        self.blk1 = ResBlk(ch_in, ch_out)
        self.blk2 = ResBlk(ch_out, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.pool(x)
        #print('Group3', x.shape)
        return x


class Group_4_5_6_7_8(nn.Module):
    def __init__(self, ch_in, ch_out, padding=1, dilation=1):
        super(Group_4_5_6_7_8, self).__init__()

        self.blk1 = ResBlk(ch_in, ch_out, padding=padding, dilation=dilation)
        self.blk2 = ResBlk(ch_out, ch_out, padding=padding, dilation=dilation)

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        #print('Group4', x.shape)

        return x


class Group_9(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Group_9, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.model1(x)
        x = self.model2(x)
        #print('Group9', x.shape)

        return x


class Group_10(nn.Module):
    def __init__(self, ch_in, ch_out, n_class, batch_size):
        super(Group_10, self).__init__()

        self.model = Conv2d_Sym(ch_in, ch_out, offset=1, keep_prob=1)
        self.ps = PS(r=8, n_channel=n_class*8, batch_size=batch_size)

    def forward(self, x):

        x = self.model(x)
        x = self.ps(x)
        #print('Group10', x.shape)

        return x


class Output(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Output, self).__init__()

        self.model = Conv2d_Sym(ch_in, ch_out, kernel_size=5, offset=2)

    def forward(self, x):

        x = self.model(x)
        #print('output', x.shape)
        return x


class Adapt_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Adapt_1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(keep_prob),
        )
        self.blk = ResBlk(ch_out, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv(x)
        x = self.blk(x)
        x = self.pool(x)

        return x


class Adapt_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Adapt_2, self).__init__()

        self.blk = ResBlk(ch_in, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.blk(x)
        x = self.pool(x)

        return x


class Adapt_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Adapt_3, self).__init__()

        self.blk1 = ResBlk(ch_in, ch_out)
        self.blk2 = ResBlk(ch_out, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.pool(x)

        return x


class Adapt_4_5_6(nn.Module):
    def __init__(self, ch_in, ch_out, padding=1, dilation=1):
        super(Adapt_4_5_6, self).__init__()

        self.blk1 = ResBlk(ch_in, ch_out, padding=padding, dilation=dilation)
        self.blk2 = ResBlk(ch_out, ch_out, padding=padding, dilation=dilation)

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)

        return x


class Cls_1_2_3_4_5(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Cls_1_2_3_4_5, self).__init__()

        self.blk = ResBlk(ch_in, ch_out)
        self.model = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.blk(x)
        x = self.model(x)

        return x


class Cls_6(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Cls_6, self).__init__()

        self.model1 = Conv2d_Sym(ch_in, ch_out, offset=1,
                                 kernel_size=3, stride=2)
        self.model2 = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.model1(x)
        x = self.model2(x)

        return x


class Cls_out(nn.Module):
    def __init__(self):
        super(Cls_out, self).__init__()

        self.linear = nn.Linear(512*2*2, 1)

    def forward(self, x):

        x = x.view(-1, 512*2*2)
        x = self.linear(x)

        return x


class Mask_Cls_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Mask_Cls_1, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=2, padding=1),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.model(x)

        return x


class Mask_Cls_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Mask_Cls_2, self).__init__()

        self.blk = ResBlk(ch_in, ch_in)
        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5,
                      stride=4, padding=2),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.blk(x)
        x = self.model(x)

        return x


class Mask_Cls_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Mask_Cls_3, self).__init__()

        self.blk = ResBlk(ch_in, ch_in*2)
        self.model = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_out, kernel_size=5,
                      stride=4, padding=2),
            nn.Dropout(keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.blk(x)
        x = self.model(x)

        return x


class Mask_Cls_4(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Mask_Cls_4, self).__init__()

        self.model = Conv2d_Sym(ch_in, ch_out, offset=2,
                                kernel_size=5, stride=4)

    def forward(self, x):

        x = self.model(x)

        return x


class Mask_Cls_out(nn.Module):
    def __init__(self):
        super(Mask_Cls_out, self).__init__()

        self.linear = nn.Linear(256*2*2, 1)

    def forward(self, x):

        x = x.view(-1, 256*2*2)
        x = self.linear(x)

        return x


class Create_Mr_Network(nn.Module):
    def __init__(self, channels):
        super(Create_Mr_Network, self).__init__()

        self.channels = channels

        self.g_1 = Group_1(ch_in=self.channels, ch_out=16)
        self.g_2 = Group_2(ch_in=16, ch_out=32)
        self.g_3 = Group_3(ch_in=32, ch_out=64)
        self.g_4 = Group_4_5_6_7_8(ch_in=64, ch_out=128)
        self.g_5 = Group_4_5_6_7_8(ch_in=128, ch_out=256)
        self.g_6 = Group_4_5_6_7_8(ch_in=256, ch_out=256)

    def forward(self, mr):

        gx_1 = self.g_1(mr)
        gx_2 = self.g_2(gx_1)
        gx_3 = self.g_3(gx_2)
        gx_4 = self.g_4(gx_3)
        gx_5 = self.g_5(gx_4)
        gx_6 = self.g_6(gx_5)

        return gx_4, gx_6


class Create_Ct_Network(nn.Module):
    def __init__(self, channels):
        super(Create_Ct_Network, self).__init__()

        self.channels = channels

        self.a_1 = Adapt_1(ch_in=self.channels, ch_out=16)
        self.a_2 = Adapt_2(ch_in=16, ch_out=32)
        self.a_3 = Adapt_3(ch_in=32, ch_out=64)
        self.a_4 = Adapt_4_5_6(ch_in=64, ch_out=128)
        self.a_5 = Adapt_4_5_6(ch_in=128, ch_out=256)
        self.a_6 = Adapt_4_5_6(ch_in=256, ch_out=256)

    def forward(self, ct):

        ax_1 = self.a_1(ct)
        ax_2 = self.a_2(ax_1)
        ax_3 = self.a_3(ax_2)
        ax_4 = self.a_4(ax_3)
        ax_5 = self.a_5(ax_4)
        ax_6 = self.a_6(ax_5)

        return ax_4, ax_6


class Create_Second_Half(nn.Module):
    def __init__(self, n_class, batch_size):
        super(Create_Second_Half, self).__init__()

        self.n_class = n_class
        self.batch_size = batch_size

        self.g_7 = Group_4_5_6_7_8(ch_in=256, ch_out=512)
        self.g_8 = Group_4_5_6_7_8(
            ch_in=512, ch_out=512, padding=2, dilation=2)
        self.g_9 = Group_9(ch_in=512, ch_out=512)
        self.g_10 = Group_10(ch_in=512, ch_out=2560, n_class=self.n_class,
                             batch_size=self.batch_size)
        self.g_out = Output(ch_in=40, ch_out=self.n_class)

    def forward(self, x):

        gx_7 = self.g_7(x)
        gx_8 = self.g_8(gx_7)
        gx_9 = self.g_9(gx_8)
        gx_10 = self.g_10(gx_9)
        logit = self.g_out(gx_10)

        return gx_9, gx_8, gx_7, logit


class Create_Classifier(nn.Module):
    def __init__(self, batch_size):
        super(Create_Classifier, self).__init__()

        self.batch_size = batch_size

        self.ps_4 = PS(r=8, n_channel=2, batch_size=self.batch_size)
        self.ps_6 = PS(r=8, n_channel=4, batch_size=self.batch_size)
        self.ps_7 = PS(r=8, n_channel=8, batch_size=self.batch_size)
        self.ps_9 = PS(r=8, n_channel=8, batch_size=self.batch_size)

        self.c_1 = Cls_1_2_3_4_5(ch_in=32, ch_out=64,
                                 kernel_size=3, stride=2, padding=1)
        self.c_2 = Cls_1_2_3_4_5(ch_in=64, ch_out=128,
                                 kernel_size=5, stride=2, padding=2)
        self.c_3 = Cls_1_2_3_4_5(ch_in=128, ch_out=256,
                                 kernel_size=3, stride=2, padding=1)
        self.c_4 = Cls_1_2_3_4_5(ch_in=256, ch_out=512,
                                 kernel_size=3, stride=2, padding=1)
        self.c_5 = Cls_1_2_3_4_5(ch_in=512, ch_out=512,
                                 kernel_size=5, stride=4, padding=2)
        self.c_6 = Cls_6(ch_in=512, ch_out=512)
        self.c_out = Cls_out()

    def forward(self, ax_4, ax_6, gx_7, gx_9, seg_logit):

        a4 = self.ps_4(ax_4)    # b,2,512,512
        a4 = a4.repeat(1, 3, 1, 1)  # b,6,512,512
        a6 = self.ps_6(ax_6)    # b,4,512,512
        g7 = self.ps_7(gx_7)    # b,8,512,512
        g9 = self.ps_9(gx_9)    # b,8,512,512
        input_comp = torch.cat((a4, a6), dim=1)  # 10
        input_comp = torch.cat((input_comp, g7), dim=1)  # 18
        input_comp = torch.cat((input_comp, g9), dim=1)  # 26
        input_comp = torch.cat((input_comp, seg_logit), dim=1)  # 31
        input_comp = torch.cat((input_comp, seg_logit.argmax(
            dim=1, keepdim=True).type(torch.FloatTensor)), dim=1)  # 32

        cls_logits = self.c_1(input_comp)
        cls_logits = self.c_2(cls_logits)
        cls_logits = self.c_3(cls_logits)
        cls_logits = self.c_4(cls_logits)
        cls_logits = self.c_5(cls_logits)
        cls_logits = self.c_6(cls_logits)
        cls_logits = self.c_out(cls_logits)

        return cls_logits


class Create_Mask_Critic(nn.Module):
    def __init__(self, n_class):
        super(Create_Mask_Critic, self).__init__()

        self.n_class = n_class
        self.m_1 = Mask_Cls_1(ch_in=self.n_class, ch_out=16)
        self.m_2 = Mask_Cls_2(ch_in=16, ch_out=32)
        self.m_3 = Mask_Cls_3(ch_in=32, ch_out=128)
        self.m_4 = Mask_Cls_4(ch_in=128, ch_out=256)
        self.m_o = Mask_Cls_out()

    def forward(self, x):

        x = self.m_1(x)
        x = self.m_2(x)
        x = self.m_3(x)
        x = self.m_4(x)
        x = self.m_o(x)

        return x


def main():
    pass
    # x = torch.randn([10, 3, 256, 256])
    # y = torch.randn([10, 3, 256, 256])
    # net1 = Create_Zip_Network(channels=3, n_class=5, batch_size=10)
    # o1, o2, o3, o4 = net1(x, y)
    # print(o1.shape)
    # print(o2.shape)
    # print(o3.shape)

    # x1 = torch.randn([10, 256, 32, 32])
    # net2 = Create_Second_Half(channels=3, n_class=5, batch_size=10)
    # _1, _2, _3, output = net2(x1)
    # print(_1.shape)
    # print(_2.shape)
    # print(_3.shape)
    # print(output.shape)

    # x2 = torch.randn([10, 5, 256, 256])
    # ax_4 = torch.randn([10, 128, 32, 32])
    # ax_6 = torch.randn([10, 256, 32, 32])
    # gx_7 = torch.randn([10, 512, 32, 32])
    # gx_9 = torch.randn([10, 512, 32, 32])
    # net = Create_Classifier(batch_size=10)
    # output = net(ax_4, ax_6, gx_7, gx_9, x2)
    # print(output.shape)

    # x1 = torch.randn([10, 5, 256, 256])
    # net = Create_Mask_Critic(n_class=5)
    # output = net(x1)
    # print(output.shape)

    # print(net.state_dict().keys())
    # and 'a_' in name and '.2.' not in name and 'extra.1' not in name
    # count = 0
    # for name, parameters in net.state_dict().items():
    #     if 'weight' in name and '.2.' not in name and 'extra.1' not in name:
    #         print(name, ':', type(parameters), ':', parameters.shape, '\n')
    #         count += 1
    # print(count)

    # total_weights = {**dict(net1.state_dict().items()),
    #                  **dict(net2.state_dict().items())}
    # count = 0
    # for name, parameters in total_weights.items():
    #     if 'weight' in name and 'a_' not in name and '.2.' not in name and 'extra.1' not in name and 'g_out' not in name:
    #         print(name, ':', type(parameters), ':', parameters.shape, '\n')
    #         count += 1
    # print(count)


if __name__ == "__main__":
    main()
