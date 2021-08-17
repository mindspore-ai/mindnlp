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
"""
 Produce the loss
"""
import mindspore.nn as nn


def create_loss(config):
    """
    create loss
    """
    if config.TRAIN.loss_function == "SoftmaxCrossEntropyWithLogits":
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    elif config.TRAIN.loss_function == "BCEWithLogitsLoss":
        loss = nn.BCEWithLogitsLoss()
    return loss