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
# ==============================================================================
"""softmax cross entropy between logits and labels"""

import mindspore.nn as nn
from mindspore.nn.loss.loss import _Loss

from mindtext.common.utils.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class CrossEntropy(_Loss):
    r"""
    Computes softmax cross entropy between logits and labels.

    Measures the distribution error between the probabilities of the input (computed with softmax
    function) and the target where the classes are mutually exclusive (only one class is positive)
    using cross entropy loss.

    Typical input into this function is unnormalized scores denoted as x whose shape is (N, C), and
    the corresponding targets.

    Parameters:
        sparse (bool) - Specifies whether labels use sparse format or not. Default: False.
        reduction (str) - Type of reduction to be applied to loss. The optional values are "mean",
        "sum", and "none". If "none", do not perform reduction. Default: "none".

    Inputs:
        logits (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
        labels (Tensor) - Tensor of shape (N, ). If sparse is True, The type of labels is int32 or
        int64.
        Otherwise, the type of labels is the same as the type of logits.

    Outputs:
        Tensor, a tensor of the same shape and type as logits with the component-wise logistic
        losses.
    """
    def __init__(self, sparse=False, reduction='none'):
        super(CrossEntropy, self).__init__()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction=reduction)

    def construct(self, logits, labels):
        """CrossEntropy construct"""
        loss = self.ce(logits, labels)
        return loss
