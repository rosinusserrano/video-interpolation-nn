"Model file"
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class Config:
    patch_width: int = 32
    patch_height: int = 32
    num_channels: list[int] = field(
        default_factory=lambda: [3, 4, 8, 16, 32])
    skip_connections: list[bool] = field(
        default_factory=lambda: [True, True, True, True, True])
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    num_heads: int = 8
    max_width_patches: int = 8  # 256px
    max_height_patches: int = 8  # 256px
    model_name: str = "PatchAttentionUNETNextFrame"


class PatchAttentionUNETNextFrame(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.num_channels[-1] * 4, num_heads=config.num_heads)

        self.patch_pos_embed = nn.Embedding(
            config.max_width_patches * config.max_height_patches,
            config.num_channels[-1] * 4)

    def forward(self, inputs):
        x, z, p = inputs
        bs1, c1, h, w = x.shape

        # SPLIT PATCHES

        x = torch.stack(torch.split(x, self.config.patch_width, dim=3), dim=1)
        x = torch.stack(torch.split(x, self.config.patch_height, dim=3), dim=1)

        z = torch.stack(torch.split(z, self.config.patch_width, dim=3), dim=1)
        z = torch.stack(torch.split(z, self.config.patch_height, dim=3), dim=1)

        bs2, num_height_patches, num_width_patches, c2, h_patch, w_patch = x.shape

        assert (bs1 == bs2 and c1 == c2 and h == h_patch * num_height_patches
                and w == w_patch * num_width_patches
                ), "Something went wrong when splitting the image into patches"

        x = x.reshape((-1, c2, h_patch, w_patch))
        z = z.reshape((-1, c2, h_patch, w_patch))

        # ENCODER

        outs_x = self.encoder(x)
        outs_z = self.encoder(z)

        outs_x = [
            torch.reshape(out, (bs2, num_height_patches * num_width_patches, c,
                                int(h_patch * 0.5**i), int(w_patch * 0.5**i)))
            for out, c, i in zip(outs_x, self.config.num_channels,
                                 range(len(outs_x)))
        ]
        outs_z = [
            torch.reshape(out, (bs2, num_height_patches * num_width_patches, c,
                                int(h_patch * 0.5**i), int(w_patch * 0.5**i)))
            for out, c, i in zip(outs_z, self.config.num_channels,
                                 range(len(outs_z)))
        ]

        # ATTENTION

        for outs in [outs_x, outs_z]:
            patch_embeds = torch.reshape(
                outs[-1], (bs2, num_height_patches * num_width_patches, -1))
            position_ids = torch.arange(num_height_patches *
                                        num_width_patches)[None, :]
            position_embeds = self.patch_pos_embed(position_ids)
            
            patch_embeds = patch_embeds + position_embeds

            if self.config.skip_connections[-1]:
                attn_out = self.attention(patch_embeds, patch_embeds,
                                          patch_embeds)[0]
                outs[-1] = attn_out + patch_embeds
            else:
                outs[-1] = self.attention(patch_embeds, patch_embeds,
                                          patch_embeds)[0]

            outs[-1] = torch.reshape(
                outs[-1], (bs2, num_height_patches * num_width_patches,
                           self.config.num_channels[-1], 2, 2))

        outs_x = [
            torch.reshape(out,
                          (bs2 * num_height_patches * num_width_patches, c,
                           int(h_patch * 0.5**i), int(w_patch * 0.5**i)))
            for out, c, i in zip(outs_x, self.config.num_channels,
                                 range(len(outs_x)))
        ]

        outs_z = [
            torch.reshape(out,
                          (bs2 * num_height_patches * num_width_patches, c,
                           int(h_patch * 0.5**i), int(w_patch * 0.5**i)))
            for out, c, i in zip(outs_z, self.config.num_channels,
                                 range(len(outs_z)))
        ]

        # DECODER

        out = self.decoder((outs_x, outs_z), p)
        out = torch.reshape(
            out,
            (bs2, num_height_patches, num_width_patches, c2, h_patch, w_patch))
        out = torch.cat(
            [out[:, i, :, :, :, :] for i in range(num_height_patches)], dim=-2)
        out = torch.cat([out[:, i, :, :, :] for i in range(num_width_patches)],
                        dim=-1)

        return out


class Encoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.convblocks = nn.ModuleList([
            ConvBlock(in_channels, out_channels, config.kernel_size,
                      config.stride, config.padding)
            for in_channels, out_channels in zip(config.num_channels[:-1],
                                                 config.num_channels[1:])
        ])

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        outs = [x]
        for block in self.convblocks:
            out = block(outs[-1])
            out = self.maxpool(out)
            outs.append(out)
        return outs


class Decoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.upconvlayers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=2,
                               stride=2) for in_channels, out_channels in
            zip(config.num_channels[:0:-1], config.num_channels[-2::-1])
        ])

        self.convblocks = nn.ModuleList([
            ConvBlock(channels * (3 if skip_conn else 1), channels,
                      config.kernel_size, config.stride, config.padding)
            for channels, skip_conn in zip(config.num_channels[-2::-1],
                                           config.skip_connections[-2::-1])
        ])

    def forward(self, outs, p):
        outs_x, outs_z = outs
        bs = p.shape[0]
        p = p[:, None, None, None, None]

        out = (1 - p) * outs_x[-1].reshape(
            (bs, -1, self.config.num_channels[-1], 2,
             2)) + p * outs_z[-1].reshape(
                 (bs, -1, self.config.num_channels[-1], 2, 2))

        out = torch.reshape(out, (-1, self.config.num_channels[-1], 2, 2))

        for upconv, conv, skip_x, skip_z, skip_conn in zip(
                self.upconvlayers, self.convblocks, outs_x[-2::-1],
                outs_z[-2::-1], self.config.skip_connections[-2::-1]):
            out = upconv(out)
            if skip_conn:
                out = torch.cat([out, skip_x, skip_z], dim=1)
            out = conv(out)
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride,
                               padding)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride,
                               padding)
        self.conv3 = ConvLayer(out_channels, out_channels, kernel_size, stride,
                               padding)
        self.conv_skip = ConvLayer(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.conv_skip(x)
        return out


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
