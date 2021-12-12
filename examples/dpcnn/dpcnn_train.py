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
    DPCNN train script
"""
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from mindtext.embeddings.static_embedding import StaticEmbedding
from mindtext.common.utils.config import parse_args, Config
from mindtext.classification.models.dpcnn import DPCNN
from mindtext.classification.models.build_train import Model
from mindtext.dataset.builder import build_dataset


set_seed(5)


def train_main(pargs):
    """
        Train function

        Args:
            config: Yaml config
    """
    config = Config(pargs.config)
    context.set_context(**config.context)

    # initialize dataset
    dataset = build_dataset(config.dataset)
    dataloader = dataset()
    train_dataloader = dataloader['train']

    # initialize embedding
    embedding = StaticEmbedding(dataset.vocab, model_dir_or_name=config.model.embedding,
                                embedding_path=config.model.embedding_path)

    # set network, loss, and optimizer
    network = DPCNN(embedding, config)
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    network_opt = nn.Adam(network.trainable_params(), 1e-3, beta1=0.9, beta2=0.999)

    # set callbacks for the network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.train.save_checkpoint_steps,
                                   keep_checkpoint_max=config.train.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=config.train.save_ckpt_dir,
                                    config=ckpt_config)
    loss_monitor = LossMonitor()
    time_monitor = TimeMonitor(data_size=train_dataloader.get_dataset_size())
    callbacks = [loss_monitor, time_monitor, ckpt_callback]
    model = Model(network, network_loss, network_opt)
    # begin to train
    print(f'[Start training `{config.model_name}`]')
    print("=" * 80)
    model.train(config.train.epochs, train_dataset=train_dataloader, callbacks=callbacks)
    print(f'[End of training `{config.model_name}`]')


if __name__ == '__main__':
    args = parse_args()
    train_main(args)
