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
    Bart eval script
"""
import os
from tqdm import tqdm
from rouge import Rouge
from mindspore import load_checkpoint, load_param_into_net
from mindtext.modules.encoder.bart import BartConfig, BartModel, BartForConditionalGeneration
from mindtext.common.utils.config import Config, parse_args
from mindtext.dataset.builder import build_dataset
from transformers import BartTokenizer


def eval_main(config):
    """
    Eval function

    Args:
        config: Yaml config
    """
    # 初始化模型、加载权重
    model_config = BartConfig.from_json_file(config.model.config_path)
    model = BartModel(model_config)
    model = BartForConditionalGeneration(model, model_config)
    param_dict = load_checkpoint(config.model.save_path)
    load_param_into_net(model, param_dict, strict_load=True)
    model.set_train(mode=False)

    # 初始化数据集
    dataset = build_dataset(config.dataset)
    dataloader = dataset.from_cache(columns_list=config.dataset.columns_list,
                                    test_columns_list=config.dataset.test_columns_list,
                                    batch_size=config.dataset.test_batch_size)
    dev_dataloader = dataloader['dev']

    # 开始评估
    if not os.path.exists(config.model.result_path):
        os.makedirs(config.model.result_path)
    rouge_eval_str = []
    rouge = Rouge()

    for batch in tqdm(dev_dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        depict_output = model.generation(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         sequence_len=config.dataset.max_length)[:, 1:].asnumpy().tolist()
        labels = labels.asnumpy().tolist()
        tokenizer = BartTokenizer.from_pretrained(config.tokenizer)
        for seq, label in zip(depict_output, labels):
            if 2 in seq:
                seq_len = seq.index(2) + 1
            else:
                seq_len = len(seq)
            label_len = label.index(2) + 1
            single_tokens = tokenizer.convert_ids_to_tokens(seq[:seq_len])
            single_string = tokenizer.convert_tokens_to_string(single_tokens)
            single_label_tokens = tokenizer.convert_ids_to_tokens(label[:label_len])
            single_label_string = tokenizer.convert_tokens_to_string(single_label_tokens)
            rouge_eval_str.append([single_string, single_label_string])

    hyps, refs = map(list, zip(*rouge_eval_str))
    scores = rouge.get_scores(hyps, refs, avg=True)

    with open(os.path.join(config.model.result_path, 'result.txt'), "w") as f:
        f.write(scores)
    print('result:', scores)


if __name__ == "__main__":
    args = parse_args()
    eval_main(Config(args.config))
