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
    DPCNN eval script.
"""
import os
from sklearn.metrics import classification_report

import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Accuracy

from mindtext.embeddings.static_embedding import StaticEmbedding
from mindtext.classification.models.dpcnn import DPCNN
from mindtext.common.utils.config import parse_args, Config
from mindtext.classification.models.build_train import Model
from mindtext.dataset.builder import build_dataset


def eval_main(pargs):
    """
    eval function
    """
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)
    dataset = build_dataset(config.dataset)
    dataloader = dataset()
    test_dataloader = dataloader['test']
    embedding = StaticEmbedding(dataset.vocab, model_dir_or_name=config.model.embedding,
                                embedding_path=config.model.embedding_path)

    # set network, loss, optimizer
    network = DPCNN(embedding, config)
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    network_opt = nn.Adam(network.trainable_params(), 1e-3, beta1=0.9, beta2=0.999)

    print(f'[Start eval `{config.model_name}`]')
    print("=" * 80)
    ckpt_files = os.listdir(config.train.save_ckpt_dir)
    ckpt_files.sort()
    ckpt_files.pop()
    for ckpt_file in ckpt_files:
        # load pretrain model
        param_dict = load_checkpoint(os.path.join(config.train.save_ckpt_dir, ckpt_file))
        load_param_into_net(network, param_dict)

        # init the whole Model
        model = Model(network, network_loss, network_opt, metrics={"Accuracy": Accuracy()})

        # begin to eval
        acc, target_sens, predictions = model.eval(test_dataloader)
        result_report = classification_report(target_sens, predictions,
                                              target_names=[str(i) for i in
                                                            range(int(config.model.decoder.num_classes))])
        print("Epoch: ", ckpt_file.split('_')[0].split('-')[1])
        print("********Accuracy: ", acc)
        print(result_report)
    print(f'[End of eval `{config.model_name}`]')


if __name__ == '__main__':
    args = parse_args()
    eval_main(args)
