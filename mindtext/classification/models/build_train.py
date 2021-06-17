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
Build model and trainer
"""
from mindspore.train import Model

from .backbones.fasttext import FastText, FastTextTrainOneStep
from .classifiers import BaseClassifier

MODEL_LIST = {'FastText': FastTextTrainOneStep}


class Trainer:
    def __init__(self, net, loss, optimizer):
        if net.backbone.__class__.__name__ not in MODEL_LIST.keys():
            raise ValueError("model not found in {}".format(MODEL_LIST.keys()))
        MODEL_TRAIN = MODEL_LIST[net.backbone.__class__.__name__]
        self.train_one_step = MODEL_TRAIN(net, loss, optimizer)
        self.train_one_step.set_train(True)
        self.Model = Model(self.train_one_step)

    def train(self, epoch, train_dataset, callbacks, dataset_sink_mode=False):
        self.Model.train(epoch, train_dataset, callbacks, dataset_sink_mode)


def build_model(config):
    model_params = config.MODEL_PARAMETERS
    if config.model_name == "fasttext":
        net = BaseClassifier(backbone=FastText(num_class=model_params.num_class, vocab_size=model_params.vocab_size,
                                               embedding_dims=model_params.embedding_dims), neck=None)
        return net
    print("dont have this model")
    return None