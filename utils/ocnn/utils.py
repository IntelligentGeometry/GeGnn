# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
import torch
from typing import Optional

__all__ = ['trunc_div', 'meshgrid', 'cumsum', 'scatter_add', 'xavier_uniform_',
           'resize_with_last_val', 'list2str']
classes = __all__


def trunc_div(input, other):
  r''' Wraps :func:`torch.div` for compatibility. It rounds the results of the
  division towards zero and is equivalent to C-style integer  division.
  '''

  version = torch.__version__.split('.')
  larger_than_170 = int(version[0]) > 0 and int(version[1]) > 7

  if larger_than_170:
    return torch.div(input, other, rounding_mode='trunc')
  else:
    return torch.floor_divide(input, other)


def meshgrid(*tensors, indexing: Optional[str] = None):
  r''' Wraps :func:`torch.meshgrid` for compatibility.
  '''

  version = torch.__version__.split('.')
  larger_than_190 = int(version[0]) > 0 and int(version[1]) > 9

  if larger_than_190:
    return torch.meshgrid(*tensors, indexing=indexing)
  else:
    return torch.meshgrid(*tensors)


def cumsum(data: torch.Tensor, dim: int, exclusive: bool = False):
  r''' Extends :func:`torch.cumsum` with the input argument :attr:`exclusive`.

  Args:
    data (torch.Tensor): The input data.
    dim (int): The dimension to do the operation over.
    exclusive (bool): If false, the behavior is the same as :func:`torch.cumsum`;
        if true, returns the cumulative sum exclusively. Note that if ture,
        the shape of output tensor is larger by 1 than :attr:`data` in the
        dimension where the computation occurs.
  '''

  out = torch.cumsum(data, dim)

  if exclusive:
    size = list(data.size())
    size[dim] = 1
    zeros = out.new_zeros(size)
    out = torch.cat([zeros, out], dim)
  return out


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
  r''' Broadcast :attr:`src` according to :attr:`other`, originally from the 
  library `pytorch_scatter`.
  '''

  if dim < 0:
    dim = other.dim() + dim

  if src.dim() == 1:
    for _ in range(0, dim):
      src = src.unsqueeze(0)
  for _ in range(src.dim(), other.dim()):
    src = src.unsqueeze(-1)

  src = src.expand_as(other)
  return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,) -> torch.Tensor:
  r''' Reduces all values from the :attr:`src` tensor into :attr:`out` at the
  indices specified in the :attr:`index` tensor along a given axis :attr:`dim`.
  This is just a wrapper of :func:`torch.scatter` in a boardcasting fashion.

  Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The indices of elements to scatter.
    dim (torch.Tensor): The axis along which to index, (default: :obj:`-1`).
    out (torch.Tensor or None): The destination tensor.
    dim_size (int or None): If :attr:`out` is not given, automatically create
        output with size :attr:`dim_size` at dimension :attr:`dim`. If
        :attr:`dim_size` is not given, a minimal sized output tensor according
        to :obj:`index.max() + 1` is returned.
    '''

  index = broadcast(index, src, dim)

  if out is None:
    size = list(src.size())
    if dim_size is not None:
      size[dim] = dim_size
    elif index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

  return out.scatter_add_(dim, index, src)


def xavier_uniform_(weights: torch.Tensor):
  r''' Initialize convolution weights with the same method as
  :obj:`torch.nn.init.xavier_uniform_`.

  :obj:`torch.nn.init.xavier_uniform_` initialize a tensor with shape
  :obj:`(out_c, in_c, kdim)`. It can not be used in :class:`ocnn.nn.OctreeConv`
  since the the shape of :attr:`OctreeConv.weights` is :obj:`(kdim, in_c,
  out_c)`.
  '''

  shape = weights.shape     # (kernel_dim, in_conv, out_conv)
  fan_in = shape[0] * shape[1]
  fan_out = shape[0] * shape[2]
  std = math.sqrt(2.0 / float(fan_in + fan_out))
  a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

  torch.nn.init.uniform_(weights, -a, a)


def resize_with_last_val(list_in: list, num: int = 3):
  r''' Resizes the number of elements of :attr:`list_in` to :attr:`num` with
  the last element of :attr:`list_in` if its number of elements is smaller
  than :attr:`num`.
  '''

  assert (type(list_in) is list and len(list_in) < num + 1)
  for i in range(len(list_in), num):
    list_in.append(list_in[-1])
  return list_in


def list2str(list_in: list):
  r''' Returns a string representation of :attr:`list_in`
  '''

  out = [str(x) for x in list_in]
  return ''.join(out)
