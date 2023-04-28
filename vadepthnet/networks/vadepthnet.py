import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .loss import VarLoss, SILogLoss
########################################################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=4),
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            #ModulatedDeformConvPack(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )

        self.bt = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        skip = self.bt(x)

        x = self.channel_shuffle(x, 4)

        x = self.conv1(x)

        x = self.conv2(x)

        return x + skip

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape

        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(
            in_channels, out_channels, in_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean = 1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)


class VarLayer(nn.Module):
    def __init__(self, in_channels, h, w):
        super(VarLayer, self).__init__()

        self.gr = 16

        self.grad = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, 4*self.gr, kernel_size=3, padding=1))

        self.att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, 4*self.gr, kernel_size=3, padding=1),
                nn.Sigmoid())


        num = h * w

        a = torch.zeros(num, 4, num, dtype=torch.float16)

        for i in range(num):

            #a[i, 0, i] = 1.0
            #if i + 1 < num:
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i+1] = -1.0

            #a[i, 1, i] = 1.0
            if i + w < num:
                a[i, 1, i] = 1.0
                a[i, 1, i+w] = -1.0

            if (i+2) % w != 0 and (i+2) < num:
                a[i, 2, i] = 1.0
                a[i, 2, i+2] = -1.0

            if i + w + w < num:
                a[i, 3, i] = 1.0
                a[i, 3, i+w+w] = -1.0

        a[-1, 0, -1] = 1.0
        a[-1, 1, -1] = 1.0

        a[-1, 2, -1] = 1.0
        a[-1, 3, -1] = 1.0

        self.register_buffer('a', a.unsqueeze(0))

        self.ins = nn.GroupNorm(1, self.gr)

        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels//2, self.gr, kernel_size=1, padding=0),
                nn.Sigmoid())

        self.post = nn.Sequential(
                nn.Conv2d(self.gr, 8*self.gr, kernel_size=3, padding=1))

    def forward(self, x):
        skip = x.clone()
        att = self.att(x)
        grad = self.grad(x)


        se = self.se(x)

        n, c, h, w = x.shape

        att = att.reshape(n*self.gr, 4, h*w, 1).permute(0, 2, 1, 3)
        grad = grad.reshape(n*self.gr, 4, h*w, 1).permute(0, 2, 1, 3)

        A = self.a * att
        B = grad * att

        A = A.reshape(n*self.gr, h*w*4, h*w)
        B = B.reshape(n*self.gr, h*w*4, 1)

        AT = A.permute(0, 2, 1)

        ATA = torch.bmm(AT, A)
        ATB = torch.bmm(AT, B)

        jitter = torch.eye(n=h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        x, _ = torch.solve(ATB, ATA+jitter)

        x = x.reshape(n, self.gr, h, w)

        x = self.ins(x)

        x = se * x

        x = self.post(x)

        return x



class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        s = c1 + c2
        self.fw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c1, kernel_size=3, padding=1))

        self.dw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c2, kernel_size=3, padding=1))

    def forward(self, feat, depth):
        cc = torch.cat([feat, depth], 1)
        feat_new = self.fw(cc)
        depth_new = self.dw(cc)
        return feat_new, depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
                nn.Linear(c, c//4),
                nn.LeakyReLU(),
                nn.Linear(c//4, 2))

    def forward(self, x):

        x = x.squeeze(-1).squeeze(-1)
        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x


class VADepthNet(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640)):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
        window_size = 12

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )


        self.backbone = SwinTransformer(**backbone_cfg)
        
        self.backbone.init_weights(pretrained=pretrained)

        self.up_4 = Up(1536 + 768, 512)
        self.up_3 = Up(512 + 384, 256)
        self.up_2 = Up(256 + 192, 64)

        self.outc = OutConv(128, 1, self.prior_mean)

        self.vlayer = VarLayer(512, img_size[0]//16, img_size[1]//16)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        self.var_loss = VarLoss(128, 512)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        self.mlayer = nn.Sequential(
                nn.AdaptiveMaxPool2d((1,1)),
                MetricLayer(1536))

    def forward(self, x, gts=None):

        x2, x3, x4, x5 = self.backbone(x)

        outs = {}

        metric = self.mlayer(x5)

        x = self.up_4(x5, x4)
        
        d = self.vlayer(x)

        if self.training:
            var_loss = self.var_loss(x, d, gts)


        x, d  = self.ref_4(x, d)

        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up_3(x, x3)

        x, d = self.ref_3(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u3 = F.interpolate(d, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up_2(x, x2)

        x, d = self.ref_2(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u2 = F.interpolate(d, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4

        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))

        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            return outs['scale_1'], var_loss + si_loss
        else:
            return outs['scale_1']



