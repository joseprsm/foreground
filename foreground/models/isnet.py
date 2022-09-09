import torch
import torch.nn as nn
from torch.nn.functional import interpolate, log_softmax, softmax

from foreground.models.rebnconv import BaseREBNCONV
from foreground.models.rsu import RSU4, RSU4F, RSU5, RSU6, RSU7

bce_loss = nn.BCELoss()


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


fea_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()
l1_loss = nn.L1Loss()
smooth_l1_loss = nn.SmoothL1Loss()


def muti_loss_fusion_kl(preds, target, dfs, fs, mode="MSE"):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss

    for i in range(0, len(dfs)):
        if mode == "MSE":
            loss = loss + fea_loss(
                dfs[i], fs[i]
            )  # add the mse loss of features as additional constraints
        elif mode == "KL":
            loss = loss + kl_loss(log_softmax(dfs[i], dim=1), softmax(fs[i], dim=1))
        elif mode == "MAE":
            loss = loss + l1_loss(dfs[i], fs[i])
        elif mode == "SmoothL1":
            loss = loss + smooth_l1_loss(dfs[i], fs[i])

    return loss0, loss


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):

    src = interpolate(src, size=tar.shape[2:], mode="bilinear")

    return src


class ISNetGTEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(ISNetGTEncoder, self).__init__()

        self.conv_in = BaseREBNCONV(in_ch, 16, stride=2)

        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 64, 512)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    @staticmethod
    def compute_loss(preds, targets):
        return muti_loss_fusion(preds, targets)

    def forward(self, x):

        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        ], [hx1, hx2, hx3, hx4, hx5, hx6]


class ISNetDIS(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS, self).__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    @staticmethod
    def compute_loss_kl(preds, targets, dfs, fs, mode="MSE"):
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    @staticmethod
    def compute_loss(preds, targets):
        return muti_loss_fusion(preds, targets)

    def forward(self, x):

        hx = x

        hxin = self.conv_in(hx)
        _ = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        ], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
