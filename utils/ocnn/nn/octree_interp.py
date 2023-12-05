# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.sparse
from typing import Optional

from utils import ocnn
from utils.ocnn.octree import Octree


def octree_nearest_pts(data: torch.Tensor, octree: Octree, depth: int,
                       pts: torch.Tensor, nempty: bool = False,
                       bound_check: bool = False):
  ''' The nearest-neighbor interpolatation with input points.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The octree to interpolate.
    depth (int): The depth of the data.
    pts (torch.Tensor): The coordinates of the points with shape :obj:`(N, 4)`,
        i.e. :obj:`N x (x, y, z, batch)`.
    nempty (bool): If true, the :attr:`data` only contains features of non-empty 
        octree nodes
    bound_check (bool): If true, check whether the point is in :obj:`[0, 2^depth)`.

  .. note::
    The :attr:`pts` MUST be scaled into :obj:`[0, 2^depth)`.
  '''

  nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
  assert data.shape[0] == nnum, 'The shape of input data is wrong.'

  idx = octree.search_xyzb(pts, depth, nempty)
  valid = idx > -1   # valid indices
  if bound_check:
    bound = torch.logical_and(pts[:, :3] >= 0, pts[:, :3] < 2**depth).all(1)
    valid = torch.logical_and(valid, bound)

  size = (pts.shape[0], data.shape[1])
  out = torch.zeros(size, device=data.device, dtype=data.dtype)
  out[valid] = data.index_select(0, idx[valid])
  return out


def octree_linear_pts(data: torch.Tensor, octree: Octree, depth: int,
                      pts: torch.Tensor, nempty: bool = False,
                      bound_check: bool = False):
  ''' Linear interpolatation with input points.

  Refer to :func:`octree_nearest_pts` for the meaning of the arguments.
  '''

  nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
  assert data.shape[0] == nnum, 'The shape of input data is wrong.'

  device = data.device
  grid = torch.tensor(
      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
       [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=device)

  # 1. Neighborhood searching
  xyzf = pts[:, :3] - 0.5   # the value is defined on the center of each voxel
  xyzi = xyzf.floor()       # the integer part  (N, 3)
  frac = xyzf - xyzi        # the fraction part (N, 3)

  xyzn = (xyzi.unsqueeze(1) + grid).view(-1, 3)
  batch = pts[:, 3].unsqueeze(1).repeat(1, 8).view(-1, 1)
  idx = octree.search_xyzb(torch.cat([xyzn, batch], dim=1), depth, nempty)
  valid = idx > -1  # valid indices
  if bound_check:
    bound = torch.logical_and(xyzn >= 0, xyzn < 2**depth).all(1)
    valid = torch.logical_and(valid, bound)
  idx = idx[valid]

  # 2. Build the sparse matrix
  npt = pts.shape[0]
  ids = torch.arange(npt, device=idx.device)
  ids = ids.unsqueeze(1).repeat(1, 8).view(-1)
  ids = ids[valid]
  indices = torch.stack([ids, idx], dim=0).long()

  frac = (1.0 - grid) - frac.unsqueeze(dim=1)  # (8, 3) - (N, 1, 3) -> (N, 8, 3)
  weight = frac.prod(dim=2).abs().view(-1)     # (8*N,)
  weight = weight[valid]

  h = data.shape[0]
  mat = torch.sparse_coo_tensor(indices, weight, [npt, h], device=device)

  # 3. Interpolatation
  output = torch.sparse.mm(mat, data)
  ones = torch.ones(h, 1, dtype=data.dtype, device=device)
  norm = torch.sparse.mm(mat, ones)
  output = torch.div(output, norm + 1e-12)
  return output


class OctreeInterp(torch.nn.Module):
  r''' Interpolates the points with an octree feature.

  Refer to :func:`octree_nearest_pts` for a description of arguments.
  '''

  def __init__(self, method: str = 'linear', nempty: bool = False,
               bound_check: bool = False, rescale_pts: bool = True):
    super().__init__()
    self.method = method
    self.nempty = nempty
    self.bound_check = bound_check
    self.rescale_pts = rescale_pts
    self.func = octree_linear_pts if method == 'linear' else octree_nearest_pts

  def forward(self, data: torch.Tensor, octree: Octree, depth: int,
              pts: torch.Tensor):
    r''''''

    # rescale points from [-1, 1] to [0, 2^depth]
    if self.rescale_pts:
      scale = 2 ** (depth - 1)
      pts[:, :3] = (pts[:, :3] + 1.0) * scale

    return self.func(data, octree, depth, pts, self.nempty, self.bound_check)

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('method={}, nempty={}, bound_check={}, rescale_pts={}').format(
            self.method, self.nempty, self.bound_check, self.rescale_pts)  # noqa


def octree_nearest_upsample(data: torch.Tensor, octree: Octree, depth: int,
                            nempty: bool = False):
  r''' Upsamples the octree node features from :attr:`depth` to :attr:`(depth+1)`
  with the nearest-neighbor interpolation.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The octree to interpolate.
    depth (int): The depth of the data.
    nempty (bool): If true, the :attr:`data` only contains features of non-empty 
        octree nodes.
  '''

  nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
  assert data.shape[0] == nnum, 'The shape of input data is wrong.'

  out = data
  if not nempty:
    out = ocnn.nn.octree_depad(out, octree, depth)
  out = out.unsqueeze(1).repeat(1, 8, 1).flatten(end_dim=1)
  if nempty:
    out = ocnn.nn.octree_depad(out, octree, depth + 1)  # !!! depth+1
  return out


class OctreeUpsample(torch.nn.Module):
  r'''  Upsamples the octree node features from :attr:`depth` to
  :attr:`(target_depth)`.

  Refer to :class:`octree_nearest_pts` for details.
  '''

  def __init__(self, method: str = 'linear', nempty: bool = False):
    super().__init__()
    self.method = method
    self.nempty = nempty
    self.func = octree_linear_pts if method == 'linear' else octree_nearest_pts

  def forward(self, data: torch.Tensor, octree: Octree, depth: int,
              target_depth: Optional[int] = None):
    r''''''

    if target_depth is None:
      target_depth = depth + 1
    if target_depth == depth:
      return data   # return, do nothing
    assert target_depth >= depth, 'target_depth must be larger than depth'

    if target_depth == depth + 1 and self.method == 'nearest':
      return octree_nearest_upsample(data, octree, depth, self.nempty)

    xyzb = octree.xyzb(target_depth, self.nempty)
    pts = torch.stack(xyzb, dim=1).float()
    pts[:, :3] = (pts[:, :3] + 0.5) * (2**(depth - target_depth))  # !!! rescale
    return self.func(data, octree, depth, pts, self.nempty)

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('method={}, nempty={}').format(self.method, self.nempty)
