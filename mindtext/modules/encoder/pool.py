# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pool class for Encoder"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class MaxPool(nn.Cell):
    """
        Max-pooling Module
    """

    def __init__(self, kernel_size=None, stride=1, dimension=1, pad_mode="valid"):
        """
        param:
            kernel_size: size of max pooling, default tensor.shape[-1]
            stride: default 1
            dimension: dimension of MaxPool, supported dimension [1,2] ,default 1
            pad_mode: 1.same 2.valid , default "valid"
        """
        super(MaxPool, self).__init__()
        if dimension not in [1, 2]:
            raise AssertionError('Now we only support 1d or 2d Pooling')
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode

    def construct(self, x):
        """
        Args:
            x: (mindspore.Tensor): [N, L, C] initialize tensor

        Returns:
            x: (mindspore.Tensor): [N, C]
        """

        if self.dimension == 1:
            x = x.transpose((0, 2, 1))  # [N,L,C] -> [N,C,L]
            pooling = nn.MaxPool1d(
                kernel_size=self.kernel_size if self.kernel_size is not None else x.shape[-1],
                stride=self.stride, pad_mode=self.pad_mode)
        else:
            pooling = nn.MaxPool2d(
                kernel_size=self.kernel_size if self.kernel_size is not None else (x.shape[-2], x.shape[-1]),
                stride=self.stride, pad_mode=self.pad_mode)
        x = pooling(x)
        return x.squeeze(axis=-1)


class MaxPoolWithMask(nn.Cell):
    """
    max pooling with mask, while max-pooling without considering zero in the mask
    """

    def __init__(self):
        super(MaxPoolWithMask, self).__init__()
        self.inf = 10e12

    def construct(self, tensor, mask, axis=1):
        """
        inputs:
            tensor : (mindspore.Tensor): [batch_size, seq_len, channels] initialize tensor
            mask : (mindspore.Tensor): [batch_size, seq_len] 0/1 mask
            axis : (int): dimension when max pooling

        outputs:
            tensor , after max-pooling with mask
        """

        masks = mask.view(mask.shape[0], mask.shape[1], -1)
        shape = (-1, -1, tensor.shape[-1])
        broadcast_to = ops.BroadcastTo(shape)
        masks = broadcast_to(masks).astype(mindspore.float32)
        masks = (masks <= 0.5).astype(mindspore.float32)
        return (ops.ArgMaxWithValue(axis=axis)(tensor + masks * -self.inf))[1]


class KMaxPool(nn.Cell):
    """K max-pooling module."""

    def __init__(self, k=1):
        super(KMaxPool, self).__init__()
        self.k = k

    def construct(self, x):
        """
        inputs:
            x : (mindspore.Tensor): [N, L, C] initialize tensor

        outputs:
            x : (mindspore.Tensor): [N, C*k]  result of k-max pool
        """
        x = x.transpose((0, 2, 1))  # [N, L, C] -> [N, C, L]
        topk = ops.TopK()
        x = topk(x, self.k)[0]
        x = x.reshape((x.shape[0], -1))
        return x


class AvgPool(nn.Cell):
    """
    input tensor : [batch_size, max_len, hidden_size], avg pooling at the last dimension.
    output : [batch_size, hidden_size]
    """

    def __init__(self, stride=1, pad_mode="valid"):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.pad_mode = pad_mode

    def construct(self, x):
        """
        inputs:
            x : (mindspore.Tensor): [N, L, C] initialize tensor

        outputs:
            x : (mindspore.Tensor): [N, C]  result of avg pool
        """
        # [N,L,C] -> [N,C,L]
        x = x.transpose((0, 2, 1))
        kernel_size = x.shape[2]
        pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=self.stride, pad_mode=self.pad_mode)
        x = pooling(x)
        return x.squeeze(axis=-1)


class AvgPoolWithMask(nn.Cell):
    """
    input tensor : [batch_size, max_len, hidden_size],avg pooling at the last dimension.
    output : [batch_size, hidden_size], only consider whether the position of mask is 1 when pooling
    """

    def __init__(self):
        super(AvgPoolWithMask, self).__init__()
        self.inf = 10e12

    def construct(self, tensor, mask, axis=1):
        """
        inputs:
            tensor : (mindspore.Tensor): [batch_size, seq_len, channels] initialize tensor
            mask : (mindspore.Tensor): [batch_size, seq_len] 0/1 mask
            axis : (int): dimension when max pooling

        outputs:
            tensor : after AvgPooling with mask
        """

        masks = mask.view(mask.shape[0], mask.shape[1], -1).astype(mindspore.float32)
        reducesum = ops.ReduceSum()
        return reducesum(tensor * masks, axis) / reducesum(masks, axis)
