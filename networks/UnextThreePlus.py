import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
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


class UNext(nn.Module):
    """
        U-Next Implementation

        Courtesy of: https://github.com/jeya-maria-jose/UNeXt-pytorch
        Paper: https://arxiv.org/abs/2203.04967
    """

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False, img_size=1024, patch_size=512,
                 in_chans=3, embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[1, 1, 1, 1], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
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
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

        # conn_1 160 in
        # 16 -> 160
        self.encoder1_2 = nn.Sequential(
            nn.Conv2d(16, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
        )
        # 32 -> 160
        self.encoder2_2 = nn.Sequential(
            nn.Conv2d(32, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
        )
        # 128 -> 160
        self.encoder3_2 = nn.Sequential(
            nn.Conv2d(128, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
        )

        self.Up1 = nn.Sequential(
            nn.Conv2d(800, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )

        # conn_2 128 in
        # 16 -> 128
        self.encoder1_3 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )

        # 32 -> 128
        self.encoder2_3 = nn.Sequential(
            nn.Conv2d(32, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )

        # 256 -> 128
        self.decoder5_3 = nn.Sequential(
            nn.Conv2d(160, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        self.Up2 = nn.Sequential(
            nn.Conv2d(640, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # conn_3 32 in
        # 16 -> 32
        self.encoder1_4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )

        # 256 -> 32
        self.decoder5_4 = nn.Sequential(
            nn.Conv2d(160, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        # 160 -> 32
        self.decoder4_4 = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        self.Up3 = nn.Sequential(
            nn.Conv2d(160, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # conn_4 16 in
        # 256 -> 16
        self.decoder5_5 = nn.Sequential(
            nn.Conv2d(160, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        # 160 -> 16
        self.decoder4_5 = nn.Sequential(
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        # 128 -> 16
        self.decoder3_5 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        self.Up4 = nn.Sequential(
            nn.Conv2d(80, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        x1 = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))

        ### Stage 2
        x2 = F.relu(F.max_pool2d(self.ebn2(self.encoder2(x1)), 2, 2))

        ### Stage 3
        x3 = F.relu(F.max_pool2d(self.ebn3(self.encoder3(x2)), 2, 2))

        ### Tokenized MLP Stage
        ### Stage 4

        x4, H, W = self.patch_embed3(x3)
        for blk in self.block1:
            x4 = blk(x4, H, W)
        x4 = self.norm3(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Bottleneck

        x5, H, W = self.patch_embed4(x4)
        for blk in self.block2:
            x5 = blk(x5, H, W)
        x5 = self.norm4(x5)
        x5 = x5.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        d5 = F.relu(F.interpolate(self.dbn1(self.decoder1(x5)), scale_factor=(2, 2), mode='bilinear'))

        _, _, H, W = d5.shape
        d5 = d5.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            d5 = blk(d5, H, W)

        ### Stage 3

        d5 = self.dnorm3(d5)
        d5 = d5.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        d4_1 = F.relu(F.max_pool2d(self.encoder1_2(x1), 8, 8))
        d4_2 = F.relu(F.max_pool2d(self.encoder2_2(x2), 4, 4))
        d4_3 = F.relu(F.max_pool2d(self.encoder3_2(x3), 2, 2))

        d4 = torch.cat((d4_1, d4_2, d4_3, x4, d5), dim=1)
        d4 = self.Up1(d4)

        d4 = F.relu(F.interpolate(self.dbn2(self.decoder2(d4)), scale_factor=(2, 2), mode='bilinear'))
        _, _, H, W = d4.shape
        d4 = d4.flatten(2).transpose(1, 2)

        for blk in self.dblock2:
            d4 = blk(d4, H, W)

        d4 = self.dnorm4(d4)
        d4 = d4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        d3_1 = F.relu(F.max_pool2d(self.encoder1_3(x1), 4, 4))
        d3_2 = F.relu(F.max_pool2d(self.encoder2_3(x2), 2, 2))
        upscale_5 = F.relu(F.interpolate(self.decoder5_3(d5), scale_factor=(2, 2), mode='bilinear'))
        d3 = torch.cat((d3_1, d3_2, x3, d4, upscale_5), dim=1)
        d3 = self.Up2(d3)

        d3 = F.relu(F.interpolate(self.dbn3(self.decoder3(d3)), scale_factor=(2, 2), mode='bilinear'))

        d2_1 = F.relu(F.max_pool2d(self.encoder1_4(x1), 2, 2))
        upscale_5 = F.relu(F.interpolate(self.decoder5_4(d5), scale_factor=(4, 4), mode='bilinear'))
        upscale_4 = F.relu(F.interpolate(self.decoder4_4(d4), scale_factor=(2, 2), mode='bilinear'))
        d2 = torch.cat((d2_1, x2, d3, upscale_4, upscale_5), dim=1)
        d2 = self.Up3(d2)

        d2 = F.relu(F.interpolate(self.dbn4(self.decoder4(d2)), scale_factor=(2, 2), mode='bilinear'))
        upscale_5 = F.relu(F.interpolate(self.decoder5_5(d5), scale_factor=(8, 8), mode='bilinear'))
        upscale_4 = F.relu(F.interpolate(self.decoder4_5(d4), scale_factor=(4, 4), mode='bilinear'))
        upscale_3 = F.relu(F.interpolate(self.decoder3_5(d3), scale_factor=(2, 2), mode='bilinear'))
        d1 = torch.cat((x1, d2, upscale_3, upscale_4, upscale_5), dim=1)
        d1 = self.Up4(d1)

        d1 = F.relu(F.interpolate(self.decoder5(d1), scale_factor=(2, 2), mode='bilinear'))

        return self.final(d1)
