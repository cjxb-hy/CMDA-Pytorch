import __future__

import torch
from torch import nn
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
    def __init__(self, ch_in, ch_out,  offset, kernel_size=3):
        super(Conv2d_Sym, self).__init__()

        self.offset = [offset, offset, offset, offset]
        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
                      stride=1, padding=0),
            nn.Dropout(keep_prob)
        )

    def forward(self, x):

        x = F.pad(x, self.offset, mode='replicate')
        x = self.model(x)

        return x


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

        self.model = Conv2d_Sym(ch_in, ch_out, offset=1)
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


class Full_DRN(nn.Module):
    def __init__(self, channels, n_class, batch_size):
        super(Full_DRN, self).__init__()

        self.channels = channels
        self.n_class = n_class
        self.batch_size = batch_size

        self.g_1 = Group_1(ch_in=self.channels, ch_out=16)
        self.g_2 = Group_2(ch_in=16, ch_out=32)
        self.g_3 = Group_3(ch_in=32, ch_out=64)
        self.g_4 = Group_4_5_6_7_8(ch_in=64, ch_out=128)
        self.g_5 = Group_4_5_6_7_8(ch_in=128, ch_out=256)
        self.g_6 = Group_4_5_6_7_8(ch_in=256, ch_out=256)
        self.g_7 = Group_4_5_6_7_8(ch_in=256, ch_out=512)
        self.g_8 = Group_4_5_6_7_8(
            ch_in=512, ch_out=512, padding=2, dilation=2)
        self.g_9 = Group_9(ch_in=512, ch_out=512)
        self.g_10 = Group_10(ch_in=512, ch_out=2560,
                             n_class=self.n_class, batch_size=self.batch_size)
        self.g_out = Output(ch_in=40, ch_out=self.n_class)

        # self.model = nn.Sequential(
        #     Group_1(ch_in=self.channels, ch_out=16),
        #     Group_2(ch_in=16, ch_out=32),
        #     Group_3(ch_in=32, ch_out=64),
        #     Group_4_5_6_7_8(ch_in=64, ch_out=128),
        #     Group_4_5_6_7_8(ch_in=128, ch_out=256),
        #     Group_4_5_6_7_8(ch_in=256, ch_out=256),
        #     Group_4_5_6_7_8(ch_in=256, ch_out=512),
        #     Group_4_5_6_7_8(ch_in=512, ch_out=512, padding=2, dilation=2),
        #     Group_9(ch_in=512, ch_out=512),
        #     Group_10(ch_in=512, ch_out=2560, n_class=self.n_class,
        #              batch_size=self.batch_size),
        #     Output(ch_in=40, ch_out=self.n_class)
        # )

    def forward(self, x):

        x = self.g_1(x)
        x = self.g_2(x)
        x = self.g_3(x)
        x = self.g_4(x)
        x = self.g_5(x)
        x = self.g_6(x)
        x = self.g_7(x)
        x = self.g_8(x)
        x = self.g_9(x)
        x = self.g_10(x)
        logits = self.g_out(x)
        # logits = self.model(x)

        return logits


def main():
    x = torch.randn([2, 3, 256, 256])
    net = Full_DRN(channels=3, n_class=5, batch_size=2)
    out = net(x)
    # print(out.shape)
    # print(net.state_dict().keys())
    # count = 0
    for name, parameters in net.state_dict().items():
        # if 'weight' in name and '.2.' not in name and 'extra.1' not in name:
        print(name, ':', type(parameters), ':', parameters.shape, '\n')
    #         count += 1
    # print(count)


if __name__ == "__main__":
    main()
