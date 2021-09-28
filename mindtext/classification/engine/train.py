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
"""train classification model."""
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor

from ...embeddings.static_embedding import StaticEmbedding
from ...dataset.classification.yelp import YelpFullDataset
from ...common.utils.config import parse_args, Config
from ...classification.models.dpcnn import DPCNN
from ...classification.models import Model
from ...common.loss.builder import build_loss
from ...common.optimizer.builder import build_optimizer


def main(pargs):
    config = Config(pargs.config)
    context.set_context(**config.context)
    yelp_full = YelpFullDataset(tokenizer='spacy', lang='en')
    dataloader = yelp_full()
    embedding = StaticEmbedding(yelp_full.vocab, model_dir_or_name=config.model.embedding)
    # set network.
    network = DPCNN(embedding, config)

    # set loss.
    network_loss = build_loss(config.loss)

    # set optimizer.
    network_opt = build_optimizer(config.optimizer)

    # set callbacks for the network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.train.save_checkpoint_steps,
                                   keep_checkpoint_max=config.train.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=config.train.save_ckpt_dir,
                                    config=ckpt_config)
    loss_monitor = LossMonitor()
    time_monitor = TimeMonitor(data_size=dataloader['train'].get_dataset_size())
    callbacks = [time_monitor, loss_monitor, ckpt_callback]

    model = Model(network, network_loss, network_opt, metrics={'Accuracy': Accuracy})

    # begin to train
    print(f'[Start training `{config.model_name}`]')
    print("=" * 80)
    model.train(config.train.epochs, dataloader['train'], callbacks=callbacks)


if __name__ == '__main__':
    args = parse_args()
    main(args)
