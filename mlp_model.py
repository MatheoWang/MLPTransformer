import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.nn as nn
import layers
from modelio import LoadableModel, store_config_args
import copy
import torch
from swin_trans_utils import *

#========================================================================================================
# Useful functions
#========================================================================================================
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding2D(nn.Module):
    """
    :param d_model: dimension of the model
    :param max_len: number of the position along one dimension
    adapted from # https://github.com/wzlxjtu/PositionalEncoding2D
    """
    def __init__(self, d_model, dropout=0., max_len=100):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(max_len, max_len, d_model)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., max_len).unsqueeze(1)
        pos_h = torch.arange(0., max_len).unsqueeze(1)
        pe[:, :, 0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, max_len, 1)
        pe[:, :, 1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, max_len, 1)
        pe[:, :, d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)
        pe[:, :, d_model + 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, dim_len):
        batch = x.size(0)
        pos = self.pe[:dim_len, :dim_len].reshape(1, -1, self.d_model).repeat(batch, 1, 1)
        x = x + pos
        return self.dropout(x)


#========================================================================================================
# MLP-related modules
#========================================================================================================
class MLPblock(nn.Module):
    def __init__(self, dim_in, dim_forward):
        super(MLPblock, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_forward)
        self.linear2 = nn.Linear(dim_forward, dim_in)
        self.activation = nn.GELU()

    def forward(self, inp):
        out = self.linear2(self.activation(self.linear1(inp)))
        return out


class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_patch, dim_forward):
        super(MLPMixerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-05)
        self.mlp1 = MLPblock(d_patch, dim_forward)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-05)
        self.mlp2 = MLPblock(d_model, dim_forward)

    def forward(self, inp, src_mask = None):
        out1 = self.norm1(inp)
        out2 = self.mlp1(out1.transpose(1,2))
        out2 = out2.transpose(1,2) + inp
        out3 = self.norm2(out2)
        out4 = self.mlp2(out3)
        out4 = out4 + out2
        return out4


class MLPEncoderLayer(nn.Module):
    def __init__(self, d_model, d_patch, dim_forward):
        super(MLPEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-05)
        self.mlp1 = MLPblock(d_model, dim_forward)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-05)
        self.mlp2 = MLPblock(d_model, dim_forward)

    def forward(self, inp, src_mask = None):
        out1 = self.norm1(inp)
        out2 = self.mlp1(out1)
        out2 = out2 + inp
        out3 = self.norm2(out2)
        out4 = self.mlp2(out3)
        out4 = out4 + out2
        return out4


class MLP_encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm_layer = None):
        super(MLP_encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self, src, mask = None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask = mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class MLP_decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm_layer = None):
        super(MLP_decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask = tgt_mask, memory_mask = memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


#========================================================================================================
# MLP net
# No mixer layer
# MLP layer for cross feature
#========================================================================================================
class PureMLP_subnet(LoadableModel):
    @store_config_args
    def __init__(self, image_size, patch_size, out_dim, d_model, depth, dim_forward, num_heads = 8, window_size=8,
                 pool='cls', channels=1,
                bidir=True, int_steps = 7, int_downsize=1, learn_pos=True, out_dis = True, int_first=True):
        super(PureMLP_subnet, self).__init__()
        self.learn_pos = learn_pos
        self.int_first = int_first
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        input_resolution = [image_size // patch_size, image_size // patch_size]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_along_dim = (image_size // patch_size)
        self.patch_sz = patch_size
        self.img_sz = image_size
        self.out_dim = out_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model),
        )
        self.bidir = bidir
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.displacement_flag = out_dis

        encoder_layer1 = MLPEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder1 = MLP_encoder(encoder_layer1, num_layers=depth)
        encoder_layer2 = MLPEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder2 = MLP_encoder(encoder_layer2, num_layers=depth)

        self.mixer = MLPEncoderLayer(d_model=d_model, d_patch=num_patches, dim_forward=dim_forward)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=out_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )
        self.upsample = nn.Upsample(scale_factor=int(patch_size/int_downsize), mode='bilinear')
        self.sin_embedding = PositionalEncoding2D(d_model=d_model)

        if self.displacement_flag:
            if self.int_first:
                vecshape = [int(image_size / patch_size), int(image_size / patch_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = int_downsize
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)
            else:
                vecshape = [int(image_size), int(image_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = patch_size
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, img_mov, img_fix):

        x_fix = self.to_patch_embedding(img_fix)
        b, n, pp = x_fix.shape
        if self.learn_pos:
            x_fix += self.pos_embedding[:, :(n + 1)]
        else:
            x_fix = self.sin_embedding(x_fix, self.patch_along_dim)

        x_mov = self.to_patch_embedding(img_mov)
        b, n, pp = x_mov.shape  # Size batch, size channels, patch_size^2
        if self.learn_pos:
            x_mov += self.pos_embedding[:, :(n + 1)]
        else:
            x_mov = self.sin_embedding(x_mov, self.patch_along_dim)

        memory1 = self.transformer_encoder1(x_fix)
        memory2 = self.transformer_encoder2(x_mov)
        vf = self.mixer(memory1 + memory2)

        vf = self.to_latent(vf)
        vf = self.mlp_head(vf)
        vf = torch.reshape(torch.transpose(vf, -2, -1), (b, self.out_dim, np.int(self.patch_along_dim), np.int(
            self.patch_along_dim)))
        if self.displacement_flag:
            if self.bidir:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    neg_flow = self.integrate(-vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                        neg_flow = self.fullsize(neg_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                        neg_flow = self.fullsize(-vf)
                    pos_flow = self.integrate(pos_flow)
                    neg_flow = self.integrate(neg_flow)

                trans_mov = self.transformer(img_mov, pos_flow)
                trans_fix = self.transformer(img_fix, neg_flow)
                return trans_mov, trans_fix, pos_flow, neg_flow
            else:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                    pos_flow = self.integrate(pos_flow)
                trans_mov = self.transformer(img_mov, pos_flow)
                return trans_mov, pos_flow
        else:
            vf = self.upsample(vf)
            return vf


#=======================================================================================================================
# MLP-mixer net
# with mixer layer for cross feature
#=======================================================================================================================

class MLPMixer_subnet(LoadableModel):
    @store_config_args
    def __init__(self, image_size, patch_size, out_dim, d_model, depth, dim_forward, num_heads = 8, window_size=8, pool='cls', channels=1,
                bidir=True, int_steps = 7, int_downsize=1, learn_pos=True, out_dis = True, int_first=True):
        super(MLPMixer_subnet, self).__init__()
        self.learn_pos = learn_pos
        self.int_first = int_first
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        input_resolution = [image_size // patch_size, image_size // patch_size]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_along_dim = (image_size // patch_size)
        self.patch_sz = patch_size
        self.img_sz = image_size
        self.out_dim = out_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model),
        )
        self.bidir = bidir
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.displacement_flag = out_dis

        encoder_layer1 = MLPMixerEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder1 = MLP_encoder(encoder_layer1, num_layers=depth)
        encoder_layer2 = MLPMixerEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder2 = MLP_encoder(encoder_layer2, num_layers=depth)

        self.mixer = MLPMixerEncoderLayer(d_model=d_model, d_patch=num_patches, dim_forward=dim_forward)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=out_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )
        self.upsample = nn.Upsample(scale_factor=int(patch_size / int_downsize), mode='bilinear')
        self.sin_embedding = PositionalEncoding2D(d_model=d_model)

        if self.displacement_flag:
            if self.int_first:
                vecshape = [int(image_size / patch_size), int(image_size / patch_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = int_downsize
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)
            else:
                vecshape = [int(image_size), int(image_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = patch_size
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, img_mov, img_fix):

        x_fix = self.to_patch_embedding(img_fix)
        b, n, pp = x_fix.shape
        if self.learn_pos:
            x_fix += self.pos_embedding[:, :(n + 1)]
        else:
            x_fix = self.sin_embedding(x_fix, self.patch_along_dim)

        x_mov = self.to_patch_embedding(img_mov)
        b, n, pp = x_mov.shape  # Size batch, size channels, patch_size^2
        if self.learn_pos:
            x_mov += self.pos_embedding[:, :(n + 1)]
        else:
            x_mov = self.sin_embedding(x_mov, self.patch_along_dim)

        memory1 = self.transformer_encoder1(x_fix)
        memory2 = self.transformer_encoder2(x_mov)
        vf = self.mixer(memory1 + memory2)

        vf = self.to_latent(vf)
        vf = self.mlp_head(vf)
        vf = torch.reshape(torch.transpose(vf, -2, -1), (b, self.out_dim, np.int(self.patch_along_dim), np.int(
            self.patch_along_dim)))
        if self.displacement_flag:
            if self.bidir:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    neg_flow = self.integrate(-vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                        neg_flow = self.fullsize(neg_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                        neg_flow = self.fullsize(-vf)
                    pos_flow = self.integrate(pos_flow)
                    neg_flow = self.integrate(neg_flow)

                trans_mov = self.transformer(img_mov, pos_flow)
                trans_fix = self.transformer(img_fix, neg_flow)
                return trans_mov, trans_fix, pos_flow, neg_flow
            else:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                    pos_flow = self.integrate(pos_flow)
                trans_mov = self.transformer(img_mov, pos_flow)
                return trans_mov, pos_flow
        else:
            vf = self.upsample(vf)
            return vf

#========================================================================================================
# MLP net
# No mixer layer
# swin transformer for feature cross attention
#========================================================================================================

class SwinTrans_subnet(LoadableModel):
    @store_config_args
    def __init__(self, image_size, patch_size, out_dim, d_model, depth, dim_forward, num_heads = 8, window_size=8, pool='cls', channels=1,
                bidir=True, int_steps = 7, int_downsize=1, learn_pos=True, out_dis = True,int_first=True):
        super(SwinTrans_subnet, self).__init__()
        self.learn_pos = learn_pos
        self.int_first = int_first
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        input_resolution = [image_size // patch_size, image_size // patch_size]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_along_dim = (image_size // patch_size)
        self.patch_sz = patch_size
        self.img_sz = image_size
        self.out_dim = out_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model),
        )
        self.bidir = bidir
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.displacement_flag = out_dis

        encoder_layer1 = MLPEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder1 = MLP_encoder(encoder_layer1, num_layers=depth)
        encoder_layer2 = MLPEncoderLayer(d_model=d_model, d_patch= num_patches, dim_forward=dim_forward)
        self.transformer_encoder2 = MLP_encoder(encoder_layer2, num_layers=depth)

        self.mixer1 = SwinTransformerBlock(
            d_model, input_resolution, num_heads, window_size=window_size, shift_size=0,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm
        )
        self.mixer2 = SwinTransformerBlock(
            d_model, input_resolution, num_heads, window_size=window_size, shift_size=window_size//2,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm
        )

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=out_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )
        self.upsample = nn.Upsample(scale_factor=int(patch_size / int_downsize), mode='bilinear')
        self.sin_embedding = PositionalEncoding2D(d_model=d_model)

        if self.displacement_flag:
            if self.int_first:
                vecshape = [int(image_size / patch_size), int(image_size / patch_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = int_downsize
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)
            else:
                vecshape = [int(image_size), int(image_size)]
                inshape = [image_size, image_size]
                self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
                ndims = len(inshape)
                self.int_downsize = patch_size
                self.fullsize = nn.Upsample(scale_factor=self.int_downsize, mode='bilinear')
                self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, img_mov, img_fix):

        x_fix = self.to_patch_embedding(img_fix)
        b, n, pp = x_fix.shape
        if self.learn_pos:
            x_fix += self.pos_embedding[:, :(n + 1)]
        else:
            x_fix = self.sin_embedding(x_fix, self.patch_along_dim)

        x_mov = self.to_patch_embedding(img_mov)
        b, n, pp = x_mov.shape  # Size batch, size channels, patch_size^2
        if self.learn_pos:
            x_mov += self.pos_embedding[:, :(n + 1)]
        else:
            x_mov = self.sin_embedding(x_mov, self.patch_along_dim)

        memory1 = self.transformer_encoder1(x_fix)
        memory2 = self.transformer_encoder2(x_mov)
        vf1 = self.mixer1(memory1, memory2)
        vf2 = self.mixer2(memory1, memory2)

        vf = self.to_latent(vf1 + vf2)
        vf = self.mlp_head(vf)
        vf = torch.reshape(torch.transpose(vf, -2, -1), (b, self.out_dim, np.int(self.patch_along_dim), np.int(
            self.patch_along_dim)))
        if self.displacement_flag:
            if self.bidir:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    neg_flow = self.integrate(-vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                        neg_flow = self.fullsize(neg_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                        neg_flow = self.fullsize(-vf)
                    pos_flow = self.integrate(pos_flow)
                    neg_flow = self.integrate(neg_flow)

                trans_mov = self.transformer(img_mov, pos_flow)
                trans_fix = self.transformer(img_fix, neg_flow)
                return trans_mov, trans_fix, pos_flow, neg_flow
            else:
                if self.int_first:
                    pos_flow = self.integrate(vf)
                    if self.fullsize:
                        pos_flow = self.fullsize(pos_flow)
                else:
                    if self.fullsize:
                        pos_flow = self.fullsize(vf)
                    pos_flow = self.integrate(pos_flow)
                trans_mov = self.transformer(img_mov, pos_flow)
                return trans_mov, pos_flow
        else:
            vf = self.upsample(vf)
            return vf


#=======================================================================================================================
# Multi-scale model
#=======================================================================================================================

class PackNet(LoadableModel):
    @store_config_args
    def __init__(self, inshape, int_steps=7, patch_size=[4, 8, 16], out_dim=[6, 6, 6], dim=[128, 128, 128],
                 depth=[1, 1, 1], forward_list=[512, 512, 512], num_heads=[32, 32, 32], window_size=[8, 4, 2],
                int_downsize=1, bidir=True, learn_pos=True, int_first=True,
                 subnet_type='PureMLP', num_scale=3):
        super(PackNet, self).__init__()
        self.bidir = bidir
        self.scale_layers = nn.ModuleList()
        self.num_scale = num_scale
        for i in range(num_scale):
            if subnet_type=='PureMLP':
                self.scale_layers.append(PureMLP_subnet(image_size=inshape[0], patch_size=patch_size[i], out_dim=out_dim[i],
                                    d_model=dim[i], depth=depth[i], dim_forward=forward_list[i],
                                    num_heads=num_heads[i], int_downsize=int_downsize, int_steps=int_steps,
                                    window_size=window_size[i], bidir=bidir, learn_pos=learn_pos, out_dis=False,
                                                        int_first=int_first))
            if subnet_type=='MLPMixer':
                self.scale_layers.append(MLPMixer_subnet(image_size=inshape[0], patch_size=patch_size[i], out_dim=out_dim[i],
                                    d_model=dim[i], depth=depth[i], dim_forward=forward_list[i],
                                    num_heads=num_heads[i],int_downsize=int_downsize, int_steps=int_steps,
                                    window_size=window_size[i], bidir=bidir, learn_pos=learn_pos, out_dis=False,
                                                         int_first=int_first))
            else:
                self.scale_layers.append(
                    SwinTrans_subnet(image_size=inshape[0], patch_size=patch_size[i], out_dim=out_dim[i],
                                   d_model=dim[i], depth=depth[i], dim_forward=forward_list[i],
                                   num_heads=num_heads[i],int_downsize=int_downsize, int_steps=int_steps,
                                   window_size=window_size[i], bidir=bidir, learn_pos=learn_pos, out_dis=False,
                                     int_first=int_first))

        self.register_parameter(name='w', param=torch.nn.Parameter(torch.tensor([0.6, 0.3, 0.1])))

        vecshape = [int(sh / int_downsize) for sh in inshape]
        self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)
        self.int_first = int_first
        ndims = len(inshape)
        self.int_downsize = int_downsize
        self.fullsize = nn.Upsample(scale_factor=int_downsize, mode='bilinear')

    def forward(self, x, y, maskx=None, masky=None):
        # x source
        # y target
        # velocity
        preint_flow = 0
        for num in range(self.num_scale):
            preint_flow = preint_flow + self.w[num] * self.scale_layers[num](x,y)

        pos_flow = preint_flow
        neg_flow = -preint_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.int_first:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None
            # rescale the flow
            if self.int_downsize > 1:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        else:
            if self.int_downsize > 1:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

        # warp image with flow field
        trans_x = self.transformer(x, pos_flow)
        trans_y = self.transformer(y, neg_flow) if self.bidir else None

        if maskx is not None:
            trans_maskx = self.transformer(maskx, pos_flow)
            trans_masky = self.transformer(masky, neg_flow) if self.bidir else None
        # return non-integrated flow field if training

        if maskx is None:
            return (trans_x, trans_y, pos_flow, neg_flow) if self.bidir else (trans_x, pos_flow)
        else:
            return (trans_x, trans_y, trans_maskx, trans_masky, pos_flow, neg_flow) if self.bidir else (
                trans_x, trans_maskx, pos_flow)
