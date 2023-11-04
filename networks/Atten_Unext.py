import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .utils.Attention_blocks import Attention_block, conv_block


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class AttenUNext(nn.Module):
    """
        U-Next Implementation

        Courtesy of: https://github.com/jeya-maria-jose/UNeXt-pytorch
        Paper: https://arxiv.org/abs/2203.04967
    """
    def __init__(self, num_classes=2, img_size=1024, embed_dims=[128, 160, 256], mlp_ratio=1, drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(160)

        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.dbn2 = nn.BatchNorm2d(128)
        self.att2 = Attention_block(F_g=128, F_l=128, F_int=64) # F_l = x, F_g = g
        self.Up_conv2 = conv_block(ch_in=256, ch_out=128)

        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.dbn3 = nn.BatchNorm2d(32)
        self.att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(16)
        self.att4 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_conv4 = conv_block(ch_in=32, ch_out=16)

        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        # region Encoder
        # region Convolutional Stage

        ### Stage 1 (1, 16, 512, 512)
        x1 = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))

        ### Stage 2 (1, 32, 256, 256)
        x2 = F.relu(F.max_pool2d(self.ebn2(self.encoder2(x1)), 2, 2))

        ### Stage 3 (1, 128, 128, 128)
        x3 = F.relu(F.max_pool2d(self.ebn3(self.encoder3(x2)), 2, 2))

        # endregion

        # region MLP stage
        ### Stage 4 (1, 160, 64, 64)

        x4, H, W = self.patch_embed3(x3)
        for blk in self.block1:
            x4 = blk(x4, H, W)
        x4 = self.norm3(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # endregion

        # endregion

        # region Bottleneck stage
        ### Bottleneck (1, 256, 32, 32)

        x5, H, W = self.patch_embed4(x4)
        for blk in self.block2:
            x5 = blk(x5, H, W)
        x5 = self.norm4(x5)
        x5 = x5.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4 (1, 4096, 160)

        d5 = F.relu(F.interpolate(self.dbn1(self.decoder1(x5)), scale_factor=(2, 2), mode='bilinear'))

        d5 = torch.add(d5, x4)
        _, _, H, W = d5.shape
        d5 = d5.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            d5 = blk(d5, H, W)
        # endregion

        # region Decoder stage
        # region MLP stage

        ### Stage 3 (1, 128, 128, 128)

        d4 = self.dnorm3(d5)
        d4 = d4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d4 = F.relu(F.interpolate(self.dbn2(self.decoder2(d4)), scale_factor=(2, 2), mode='bilinear'))
        x3 = self.att2(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv2(d4)
        #d4 = torch.add(d4, x3)
        _, _, H, W = d4.shape
        d4 = d4.flatten(2).transpose(1, 2)

        for blk in self.dblock2:
            d4 = blk(d4, H, W)

        d4 = self.dnorm4(d4)
        d4 = d4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # endregion

        # region Convolution stage

        d3 = F.relu(F.interpolate(self.dbn3(self.decoder3(d4)), scale_factor=(2, 2), mode='bilinear'))
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # d3 = torch.add(d3, x2)

        d2 = F.relu(F.interpolate(self.dbn4(self.decoder4(d3)), scale_factor=(2, 2), mode='bilinear'))
        x1 = self.att4(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv4(d2)
        # d2 = torch.add(d2, x1)

        d1 = F.relu(F.interpolate(self.decoder5(d2), scale_factor=(2, 2), mode='bilinear'))

        # endregion
        # endregion

        return self.final(d1)
