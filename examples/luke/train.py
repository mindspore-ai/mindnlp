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
"""LUKE train script"""
import time

from examples.luke.utils.squad_luke import SquadLuke
from mindtext.common.utils.config import parse_args, Config
from mindtext.common.lr_schedule.bert_lr import BertLearningRate
from mindtext.modules.encoder.luke import LukeConfig
from mindtext.tagging.models.luke import LukeForReadingComprehension, LukeForReadingComprehensionWithLoss, LukeSquadCell
from mindspore import save_checkpoint
from mindspore import context
import mindspore
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell



class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_ids=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_ids
        self.time_stamp_first = get_ms_timestamp()

    def step_end(self, run_context):
        """Monitor the loss in training."""
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - self.time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                time_stamp_current - self.time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy())))
            f.write('\n')


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


def main(pargs):
    config = Config(pargs.config)
    context.set_context(**config.context)
    context.set_context(enable_graph_kernel=True)
    squad = SquadLuke()
    train_dataloader = squad(evaluate=False, config=config.dataset)
    luke_config = LukeConfig()
    epoch = config.epoch
    luke_model = LukeForReadingComprehension(luke_config)
    param_dict = load_checkpoint(config.model.ckpt_path)
    load_param_into_net(luke_model, param_dict)

    # update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)

    luke_squad = LukeForReadingComprehensionWithLoss(luke_model)

    lr_schedule = BertLearningRate(learning_rate=15e-6,
                                   end_learning_rate=15e-6 * 0,
                                   warmup_steps=int(train_dataloader.get_dataset_size() * epoch * 0.1),
                                   decay_steps=train_dataloader.get_dataset_size() * epoch,
                                   power=1.0)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)

    params = luke_squad.trainable_params()

    optimizer = mindspore.nn.AdamWeightDecay(params,
                                             learning_rate=lr_schedule,
                                             beta1=0.9,
                                             beta2=0.98,
                                             eps=1e-06)

    netwithgrads = LukeSquadCell(luke_squad, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)

    loss_monitor = LossCallBack()

    model.train(epoch, train_dataloader, callbacks=[loss_monitor], dataset_sink_mode=False)
    save_checkpoint(model.train_network.network.net, config.model.save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
