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
"""Bert Embedding."""
import logging
from typing import Tuple
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ..modules.encoder.bert import BertModel, BertConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BertEmbedding(nn.Cell):
    '''
    This is a class that loads pre-trained weight files into the model.
    '''


    def from_pretrain(self, ckpt_file: str, bert_config: BertConfig, is_training: bool = False) -> None:
        '''
        Instantiate a pretrained  model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
        train the model, you should first set it back in training mode with ``model.train()``.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in bert` means that the layer XXX is not used by bert, therefore those
        weights are discarded.

        :param ckpt_file: the path of ckpt file.
        :param bert_config:the Bert_config class which contain the parameter for bert.
        :param is_training:if is_training is false,
        the drop out of bert will be 0,else the drop out define by config file.

        '''
        bert = BertModel(bert_config, is_training)
        param_dict = load_checkpoint(ckpt_file)
        p_load = load_param_into_net(bert, param_dict)
        self.model_dict = {"bert": bert}
        not_init = param_dict - self.model_dict['bert'].parameters_dict().keys()
        if not_init:
            for name in not_init:
                logger.warning("Weights from %s not initialized from pretrained model . ", name)
            logger.warning("It is up to you to train those weights with a downstream fine-tuning task.")
        if p_load:
            for name in p_load:
                logger.warning("Weights from %s not used in bert. ", name)
            logger.warning("those  weights are discarded.")
    def construct(self, input_ids: Tensor, token_type_ids: Tensor, input_mask: Tensor)-> Tuple[Tensor, Tensor]:
        '''
        Returns the result of the model after loading the pre-training weights

        Args:
            input_ids:A vector containing the transformation of characters into corresponding ids.
            token_type_ids:A vector containing segemnt ids.
            input_mask:the mask for input_ids.

        Returns:
            sequence_output:the sequence output .
            pooled_output:the pooled output of first token:cls..
        '''
        sequence_output, pooled_output, _ = self.model_dict['bert'](input_ids, token_type_ids, input_mask)
        return sequence_output, pooled_output
