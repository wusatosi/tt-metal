# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import List, Tuple


def query_ball_point(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    b, m, _ = new_xyz.shape
    _, n, _ = xyz.shape
    device = new_xyz.device
    radius2 = radius * radius

    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist2 = torch.sum(diff**2, dim=3)
    mask = dist2 < radius2

    idx = torch.zeros((b, m, nsample), dtype=torch.int32, device=device)

    arange_n = torch.arange(n, device=device).view(1, 1, n).expand(b, m, n)
    arange_n_masked = torch.where(mask, arange_n, torch.full_like(arange_n, n + 1))

    sorted_indices, _ = torch.sort(arange_n_masked, dim=2)
    first_nsample = sorted_indices[:, :, :nsample]

    invalid_mask = first_nsample == (n + 1)
    first_valid = first_nsample[:, :, 0].unsqueeze(2).expand_as(first_nsample)
    first_nsample[invalid_mask] = first_valid[invalid_mask]

    return first_nsample.to(torch.int32)


class BallQuery(nn.Module):
    def __init__(self, radius, nsample):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

    def __call__(self, xyz, new_xyz):
        return query_ball_point(new_xyz, xyz, self.radius, self.nsample)


class GroupingOperation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, idx):
        """
        points: (B, C, N)
        idx: (B, npoint, nsample)
        """
        B, C, N = points.shape
        _, npoint, nsample = idx.shape

        idx = idx.to(torch.int64)
        # Flatten points and idx for gather operation
        points_flat = points.view(B * C, N)
        idx_flat = idx.view(B, npoint * nsample)
        # Expand idx to match points channels dimension
        idx_expand = idx_flat.unsqueeze(1).expand(-1, C, -1).contiguous().view(B * C, npoint * nsample)
        # Gather the points
        out_flat = torch.gather(points_flat, 1, idx_expand)
        # Reshape to (B, C, npoint, nsample)
        output = out_flat.view(B, C, npoint, nsample)
        return output


def furthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    B, N, _ = points.shape
    device = points.device
    centroids = torch.zeros((B, n_samples), dtype=torch.long, device=device)
    distance = torch.ones((B, N), dtype=points.dtype, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=2)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=1)[1]

    return centroids


class FurthestPointSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, npoint):
        """
        Args:
            xyz: (B, N, 3) input points
            npoint: int, number of points to sample

        Returns:
            fps_inds: (B, npoint) sampled indices
        """
        return furthest_point_sampling(xyz, npoint)


def gather_points(points, idx):
    B, C, N = points.shape
    M = idx.shape[1]
    idx_expand = idx.long().unsqueeze(1).expand(B, C, M)
    out = torch.gather(points, 2, idx_expand)
    return out


class GatherOperation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, idx):
        """
        Args:
            features: (B, C, N) tensor
            idx: (B, S) or (B, S, K) tensor of indices to gather from `features`

        Returns:
            gathered: (B, C, S) or (B, C, S, K) tensor depending on idx shape
        """
        return gather_points(features, idx)


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self,
        radius,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        self.ball_query = BallQuery(radius=self.radius, nsample=self.nsample)
        self.grouping_operation = GroupingOperation()
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = self.ball_query(xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = xyz.transpose(1, 2).contiguous()

        grouped_xyz = self.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = self.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation=nn.ReLU(inplace=True),
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
        )


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation=nn.ReLU(inplace=True),
        preact: bool = False,
        first: bool = False,
        name: str = "",
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                ),
            )


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff) + dst_range[0][:, None, :]
    return prop_xyz
