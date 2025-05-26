# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Optional
from models.experimental.DETR3D.reference.model_utils import (
    QueryAndGroup,
    SharedMLP,
    GatherOperation,
    FurthestPointSampling,
    shift_scale_points,
)
import copy
import math
import numpy as np


def get_nested_list_shape(lst):
    shape = []
    while isinstance(lst, (list, tuple)):
        shape.append(len(lst))
        if len(lst) == 0:
            break
        lst = lst[0]
    return tuple(shape)


def truncate_nested_list(lst, max_items=3):
    if isinstance(lst, (list, tuple)):
        if len(lst) > max_items:
            return lst[:max_items] + ["..."]
        return [truncate_nested_list(item) for item in lst]
    return lst


# 1: PointnetSAModuleVotes
class PointnetSAModuleVotes(nn.Module):
    """Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes"""

    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: int = None,
        radius: float = None,
        nsample: int = None,
        bn: bool = True,
        use_xyz: bool = True,
        pooling: str = "max",
        sigma: float = None,  # for RBF pooling
        normalize_xyz: bool = False,  # noramlize local XYZ with radius
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ):
        super().__init__()
        # params = {k: v for k, v in locals().items() if k != 'self'}
        print("ref PointnetSAModuleVotes init is called with params:")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(
                radius,
                nsample,
                use_xyz=use_xyz,
                ret_grouped_xyz=True,
                normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly,
                ret_unique_cnt=ret_unique_cnt,
            )
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)
        self.gather_operation = GatherOperation()
        self.furthest_point_sample = FurthestPointSampling()

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        # params = {k: v for k, v in locals().items() if k != 'self'}
        print(f"ref PointnetSAModuleVotes forward called with params:")
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")

        # xyz: torch.Tensor,features: torch.Tensor ,inds: torch.Tensor",xyz.shape,features.shape,inds.shape
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = self.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = (
            self.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        )

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
        if self.pooling == "max":
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "avg":
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "rbf":
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(
                -1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma**2) / 2
            )  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(
                self.nsample
            )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


# 2: Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        dropout_attn=None,
        activation="relu",
        normalize_before=True,
        norm_name="ln",
        use_ffn=True,
        ffn_use_bias=True,
    ):
        super().__init__()
        # params = {k: v for k, v in locals().items() if k != 'self'}
        print("TransformerEncoderLayer init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            # self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        print("TransformerEncoderLayer forwardpost is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            elif isinstance(v, (int, float, bool)):
                print(f"  {k}: {v}")
            elif v is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: {type(v).__name__}")
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2  # self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.activation(self.linear1(src)))
            src = src + src2  # self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        return_attn_weights: Optional[Tensor] = False,
    ):
        print("TransformerEncoderLayer forwardpre is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(
            q, k, value=value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + src2  # self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.activation(self.linear1(src2)))
            src = src + src2  # self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        return_attn_weights: Optional[Tensor] = False,
    ):
        print("TransformerEncoderLayer forwardddd is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st


# 3: Transformer Encoder init
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        print("TransformerEncoder init is called")
        for k, v in params.items():
            print(f"  {k}: {v}")
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = nn.init.xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                func(p)


# 4:MaskedTransformerEncoder


class MaskedTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
        weight_init_name="xavier_uniform",
    ):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        params = {k: v for k, v in locals().items() if k != "self"}
        print("MaskedTransformerEncoder init is called")
        for k, v in params.items():
            print(f"  {k}: {v}")
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        xyz: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
    ):
        print("MaskedTransformerEncoder forward is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            elif isinstance(v, (int, float, bool)):
                print(f"  {k}: {v}")
            elif v is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: {type(v).__name__}")
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                # output is npoints x batch x channel. make batch x channel x npoints
                output = output.permute(1, 2, 0)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                # swap back
                output = output.permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds


# 5: Generic MLP (ENCODER TO DECODER PROJ)


class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        print("GenericMLP init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        activation = nn.ReLU
        norm = None
        if norm_fn_name is not None:
            norm = nn.BatchNorm1d
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = None
        for _, param in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        print("GenericMLP forward is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        output = self.layers(x)
        return output


# 6: PositionEmbeddingCoordsSine


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        print("PositionEmbeddingCoordsSine init is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        for k, v in params.items():
            print(f"  {k}: {v}")
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2
        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        # print("PositionEmbeddingCoordsSine forward is called",input_range)
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     elif isinstance(v, (list, tuple)):
        #         try:
        #             nested_shape = get_nested_list_shape(v)
        #             print(f"  {k}: {type(v).__name__}(shape={nested_shape}, values={truncate_nested_list(v)})")
        #         except Exception as e:
        #             print(f"  {k}: {type(v).__name__}, could not get shape or values ({e})")
        #     else:
        # print(f"  {k}: {type(v).__name__}")
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st


# 7 : Decoder Layer


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        dropout_attn=None,
        activation="relu",
        normalize_before=True,
        norm_fn_name="ln",
    ):
        super().__init__()
        print("TransformerDecoderLayer init is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        print("TransformerDecoderLayer forward post is called")
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        print("TransformerDecoderLayer forward pre is called")
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        print("TransformerDecoderLayer forwarddd is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                return_attn_weights,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            return_attn_weights,
        )


# 8: Transformer Decoder


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm_fn_name="ln", return_intermediate=False, weight_init_name="xavier_uniform"
    ):
        super().__init__()
        print("TransformerDecoder init is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        for k, v in params.items():
            print(f"  {k}: {v}")
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = nn.LayerNorm(self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = nn.init.xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
        return_attn_weights: Optional[bool] = False,
    ):
        print("TransformerDecoder forward is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            elif isinstance(v, (int, float, bool)):
                print(f"  {k}: {v}")
            elif v is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: {type(v).__name__}")
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1)  # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                return_attn_weights=return_attn_weights,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns
