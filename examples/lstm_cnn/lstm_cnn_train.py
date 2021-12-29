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
    Lstm_cnn train script
"""
import os
import time
from mindspore import context, save_checkpoint
from mindspore.train.callback import LossMonitor
from mindspore.nn.optim import Adam
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindtext.dataset.builder import build_dataset
from mindtext.common.utils.config import Config, parse_args
from mindtext.tagging.models.lstm_cnn import LstmCnn, LstmCnnConfig, LstmCnnWithLoss, LstmCnnTrainOneStep


def main(config):
    """
    Train function

    Args:
        config: Yaml config
    """
    context.set_context(**config.context)  # jingtaitu mode
    if not os.path.exists(config.model.result_path):
        os.makedirs(config.model.result_path)

    # 初始化dataset
    dataset = build_dataset(config.train.dataset)
    dataloader = dataset()

    lstm_cnn_config = LstmCnnConfig(vocab=dataset.vocab)
    network = LstmCnn(lstm_cnn_config).set_train(True)

    # set loss.
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    network_loss = LstmCnnWithLoss(network, loss)

    # set optimizer.
    optimizer = Adam(network_loss.trainable_params(), learning_rate=5e-3)
    netwithgrads = LstmCnnTrainOneStep(network_loss, optimizer)

    model = Model(netwithgrads)
    loss_monitor = LossMonitor()
    callbacks = [loss_monitor]

    epochs = config.epochs
    # begin to train
    train_begin = time.time()
    print('[Start training]')
    print("=" * 80)
    model.train(epochs, dataloader['train'], callbacks=callbacks, dataset_sink_mode=False)
    train_end = time.time()
    print("Train one step time:", (train_end - train_begin) / dataloader['train'].get_dataset_size() / epochs)
    save_checkpoint(model.train_network.network, config.model.save_path)


if __name__ == '__main__':
    args = parse_args()
    main(Config(args.config))
