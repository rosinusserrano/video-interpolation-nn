"Model file"
from dataclasses import dataclass

import torch
from torch import nn

# TODO:
# Encoding the first and last patch embeddings separately may not be the best idea.
# Instead concatenate x and z and then just do a single forward pass through the encoder.
# Simply define more models and let them compete with ech other


@dataclass
class Config:
    patch_width = 32
    patch_height = 32
    num_channels = [3, 64, 128, 256, 512]
    kernel_size = 3
    stride = 1
    padding = 1
    num_heads = 8
    max_width_patches = 8  # 256px
    max_height_patches = 8  # 256px
    model_name = "PatchAttentionUNET"


class PatchAttentionUNET(nn.Module):

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
        x, z = inputs
        bs1, c1, h, w = x.shape

        # SPLIT PATCHES

        x = torch.stack(torch.split(x, self.config.patch_width, dim=3), dim=1)
        x = torch.stack(torch.split(x, self.config.patch_height, dim=3), dim=1)

        z = torch.stack(torch.split(x, self.config.patch_width, dim=3), dim=1)
        z = torch.stack(torch.split(x, self.config.patch_height, dim=3), dim=1)

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

        out = self.decoder((outs_x, outs_z))
        out = torch.reshape(
            out,
            (bs2, num_height_patches, num_width_patches, c2, h_patch, w_patch))
        out = torch.cat([out[:, i, :, :, :, :] for i in range(num_height_patches)],
                        dim=-2)
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
            ConvBlock(config.num_channels[-1] * 2, config.num_channels[-1],
                      config.kernel_size, config.stride, config.padding)
        ] + [
            ConvBlock(channels * 3, channels, config.kernel_size,
                      config.stride, config.padding)
            for channels in config.num_channels[-2::-1]
        ])

    def forward(self, outs):
        outs_x, outs_z = outs

        out = torch.cat([outs_x[-1], outs_z[-1]], dim=1)
        out = self.convblocks[0](out)

        for upconv, conv, skip_x, skip_z in zip(self.upconvlayers,
                                                self.convblocks[1:],
                                                outs_x[-2::-1],
                                                outs_z[-2::-1]):
            out = upconv(out)
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
        self.conv_skip = ConvLayer(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.conv_skip(x)
        return out


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))
