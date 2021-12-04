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
    bert train script
"""
import os
import time
from mindspore.train.callback import Callback
from mindspore.train.callback import TimeMonitor
from mindspore.train import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
from mindtext.modules.encoder.bert import BertConfig, BertModel
from mindtext.classification.models.bert import BertforSequenceClassification, BertLearningRate, BertFinetuneCell
from mindtext.embeddings.bert_embedding import BertEmbedding
from mindtext.common.utils.config import Config, parse_args
from mindtext.dataset.builder import build_dataset


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


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
                str(cb_params.net_outputs.asnumpy())))
            f.write('\n')


def train_main(config):
    """
        Train function

        Args:
            config: Yaml config
        """
    # 设置图模式（默认为静态图）
    context.set_context(**config.context)
    if not os.path.exists(config.model.result_path):
        os.makedirs(config.model.result_path)

    # 初始化dataset
    dataset = build_dataset(config.dataset)
    dataloader = dataset()
    train_dataloader = dataloader['train']

    # 初始化bert
    bert_config = BertConfig.from_yaml_file(config.model.config_path)
    model = BertModel(bert_config, True)

    # 载入权重
    param_dict = load_checkpoint(config.model.ckpt_path)
    load_param_into_net(model, param_dict)
    bert = BertforSequenceClassification(bert_config, True, num_labels=config.num_labels,
                                         dropout_prob=config.drop_prob, multi_sample_dropout=config.num_drop)
    embedding = BertEmbedding(bert_config, True)
    embedding.init_bertmodel(model)
    bert.init_embedding(embedding)

    # 初始化epochs、learning_rate
    epochs = config.epochs
    lr_schedule = BertLearningRate(learning_rate=2e-5,
                                   end_learning_rate=2e-5 * 0,
                                   warmup_steps=int(train_dataloader.get_dataset_size() * epochs * 0.1),
                                   decay_steps=train_dataloader.get_dataset_size() * epochs,
                                   power=1.0)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)

    # 初始化优化器
    params = bert.trainable_params()
    optimizer = AdamWeightDecay(params, lr_schedule, eps=1e-8)
    netwithgrads = BertFinetuneCell(bert, optimizer=optimizer, scale_update_cell=update_cell)
    train_model = Model(netwithgrads)

    # 定义回调函数
    callbacks = [TimeMonitor(train_dataloader.get_dataset_size()), LossCallBack(train_dataloader.get_dataset_size())]

    # 开始训练
    start_time = time.time()
    train_model.train(epochs, train_dataloader, callbacks=callbacks, dataset_sink_mode=False)
    end_time = time.time()

    # 保存权重
    save_path = config.model.save_path
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    save_checkpoint(train_model.train_network.network, save_path)

    one_step_time = (end_time - start_time) / train_dataloader.get_dataset_size() / epochs
    with open(os.path.join(config.model.result_path, 'train_one_step_time.txt'), "w") as f:
        f.write(str(one_step_time))
    print("Train one step time:", one_step_time)


if __name__ == "__main__":
    args = parse_args()
    train_main(Config(args.config))
