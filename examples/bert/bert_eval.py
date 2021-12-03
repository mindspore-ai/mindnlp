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
    bert eval script
"""
import os
from tqdm import tqdm
from mindspore import context
from mindspore.nn import Accuracy, Softmax
from mindspore.ops import Squeeze
from mindtext.modules.encoder.bert import BertConfig
from mindtext.classification.models.bert import BertforSequenceClassification
from mindtext.common.utils.config import Config, parse_args
from mindtext.dataset.builder import build_dataset


def eval_main(config):
    """
    Eval function

    Args:
        config: Yaml config
    """
    # 设置图模式（默认为静态图）
    context.set_context(**config.context)
    # 初始化模型、加载权重
    bert_config = BertConfig.from_yaml_file(config.model.config_path)
    model = BertforSequenceClassification(bert_config, False, num_labels=config.num_labels,
                                          dropout_prob=0, multi_sample_dropout=1)
    model.from_pretrain(config.model.save_path)

    # 初始化数据集
    dataset = build_dataset(config.dataset)
    dataloader = dataset.from_cache(columns_list=config.dataset.columns_list,
                                    test_columns_list=config.dataset.test_columns_list,
                                    batch_size=config.dataset.batch_size)
    dev_dataloader = dataloader['dev']

    # 开始评估
    if not os.path.exists(config.model.result_path):
        os.makedirs(config.model.result_path)
    metirc = Accuracy('classification')
    metirc.clear()
    squeeze = Squeeze(1)
    for batch in tqdm(dev_dataloader.create_dict_iterator(num_epochs=1), total=dev_dataloader.get_dataset_size()):
        input_ids = batch['input_ids']
        token_type_id = batch['token_type_ids']
        input_mask = batch['attention_mask']
        label_ids = batch['label']
        inputs = {"input_ids": input_ids,
                  "input_mask": input_mask,
                  "token_type_id": token_type_id
                  }
        output = model(**inputs)
        sm = Softmax(axis=-1)
        output = sm(output)
        metirc.update(output, squeeze(label_ids))

    result = str(metirc.eval())
    with open(os.path.join(config.model.result_path, 'result.txt'), "w") as f:
        f.write(result)
    print('result:', result)


if __name__ == "__main__":
    args = parse_args()
    eval_main(Config(args.config))
