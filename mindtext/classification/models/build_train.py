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
from tqdm import tqdm
from mindspore.train import Model as MM
from mindspore import Tensor
from mindspore import dtype as mstype

from sklearn.metrics import accuracy_score
import numpy as np

from .backbones.fasttext import FastText, FastTextTrainOneStep, FastTextInferCell
from .classifiers import BaseClassifier

MODEL_LIST = {'FastText': (FastTextTrainOneStep, FastTextInferCell)}


class Model:
    def __init__(self, net, loss, optimizer, metrics=None):
        if net.backbone.__class__.__name__ not in MODEL_LIST.keys():
            raise ValueError("model not found in {}".format(MODEL_LIST.keys()))
        MODEL_TRAIN, MODEL_INFER = MODEL_LIST[net.backbone.__class__.__name__]
        self.train_one_step = MODEL_TRAIN(net, loss, optimizer)
        self.train_one_step.set_train(True)
        self.Train_Model = MM(self.train_one_step)
        self.Infer_Model = MODEL_INFER(net)
        self.metrics = metrics

    def train(self, epoch, train_dataset, callbacks, dataset_sink_mode=False):
        self.Train_Model.train(epoch, train_dataset, callbacks, dataset_sink_mode)

    def eval(self, dataset):
        predictions = []
        target_sens = []
        inputs = {}
        inputs_name = []
        label_name = None
        for batch in tqdm(dataset.create_dict_iterator(output_numpy=True, num_epochs=1),
                          total=dataset.get_dataset_size()):
            if len(inputs_name) == 0:
                inputs_name = list(batch.keys())[0:-1]
                label_name = list(batch.keys())[-1]
            target_sens.append(batch[label_name])
            for i in inputs_name:
                inputs[i] = Tensor(batch[i], mstype.int32)
            predicted_idx = self.Infer_Model(**inputs)
            predictions.append(predicted_idx.asnumpy())
            inputs = {}
        target_sens = np.array(target_sens).flatten()
        merge_target_sens = []
        for target_sen in target_sens:
            merge_target_sens.extend(target_sen)
        target_sens = merge_target_sens
        predictions = np.array(predictions).flatten()
        merge_predictions = []
        for prediction in predictions:
            merge_predictions.extend(prediction)
        predictions = merge_predictions
        acc = accuracy_score(target_sens, predictions)
        return acc, target_sens, predictions


def build_model(config):
    model_params = config.MODEL_PARAMETERS
    if config.model_name == "fasttext":
        net = BaseClassifier(backbone=FastText(num_class=model_params.num_class, vocab_size=model_params.vocab_size,
                                               embedding_dims=model_params.embedding_dims), neck=None)
        return net
    print("dont have this model")
    return None
