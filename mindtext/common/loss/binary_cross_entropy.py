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
"""binary cross entropy between the true labels and predicted labels"""

import mindspore.nn as nn
from mindspore.nn.loss.loss import _Loss

from mindtext.common.utils.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class BinaryCrossEntropy(_Loss):
    r"""
    BCELoss creates a criterion to measure the binary cross entropy between the true labels and
    predicted labels.

    Parameters:
        weight (Tensor, optional) - A rescaling weight applied to the loss of each batch element.
        And it must have same shape and data type as inputs. Default: None
        reduction (str) - Specifies the reduction to be applied to the output. Its value must be one
        of 'none', 'mean', 'sum'. Default: 'none'.

    Inputs:
        logits (Tensor) - The input Tensor with shape (N,*) where * means, any number of additional
        dimensions. The data type must be float16 or float32.
        labels (Tensor) - The label Tensor with shape (N,*), same shape and data type as logits.

    Outputs:
        Tensor or Scalar, if reduction is 'none', then output is a tensor and has the same shape as
        logits.
        Otherwise, the output is a scalar.
    """
    def __init__(self, weight=None, reduction='none'):
        super(BinaryCrossEntropy, self).__init__()
        self.bce = nn.BCELoss(weight=weight, reduction=reduction)

    def construct(self, logits, labels):
        """BinaryCrossEntropy construct"""
        loss = self.bce(logits, labels)
        return loss
