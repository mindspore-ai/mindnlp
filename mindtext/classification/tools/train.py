# Copyright 2020 Huawei Technologies Co., Ltd
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
""" MindText Classification training script. """
from mindspore import context

from ..utils import get_config, parse_args
from ..dataset import create_dataset
from ..models import build_model, create_loss, create_optimizer
from ..models import Trainer
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import TimeMonitor, LossMonitor


def main(pargs):
    '''

    :param pargs:
    :return:
    '''

    # set config context
    config = get_config(pargs.config_path, overrides=pargs.override)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)

    dataset_train = create_dataset(config)

    # set network, loss, optimizer
    network = build_model(config)
    network_loss = create_loss(config)
    network_opt = create_optimizer(network.trainable_params(), config)

    # set callbacks for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config.TRAIN.save_ckpt_steps,
        keep_checkpoint_max=config.TRAIN.keep_ckpt_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=config.TRAIN.save_ckpt_dir,
                                    config=ckpt_config)
    loss_monitor = LossMonitor(per_print_times=2000)
    time_monitor = TimeMonitor(data_size=dataset_train.get_dataset_size)
    callbacks = [time_monitor, loss_monitor, ckpt_callback]

    # init the Trainer
    model = Trainer(network,
                    network_loss,
                    network_opt)

    print(f'[Start training `{config.model_name}`]')
    print("=" * 80)

    model.train(config.TRAIN.epochs,
                train_dataset=dataset_train,
                callbacks=callbacks)
    print(f'[End of training `{config.model_name}`]')

    '''
    distribute train ....
    pass
    '''


if __name__ == "__main__":
    args = parse_args()
    main(args)
