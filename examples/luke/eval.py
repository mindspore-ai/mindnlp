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
"""LUKE eval script"""
import collections
import json
import os

import numpy as np
from tqdm import tqdm
from mindspore import context, Model, load_checkpoint, load_param_into_net

from mindtext.common.utils.config import parse_args, Config
from mindtext.modules.encoder.luke import LukeConfig
from mindtext.tagging.models.luke import LukeForReadingComprehension
from utils.dataset import SquadV1Processor
from utils.squad_luke import SquadLuke
from utils.result_writer import write_predictions
from utils.squad_eval import EvalOpts as SQUAD_EVAL_OPTS
from utils.squad_eval import main as evaluate_on_squad


def do_eval(dataset=None, load_checkpoint_path=""):
    """

    Args:
        dataset:
        load_checkpoint_path:

    Returns:

    """
    config = LukeConfig()
    luke_model = LukeForReadingComprehension(config)
    luke_model.set_train(False)
    model = Model(luke_model)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(luke_model, param_dict)
    output = []

    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["unique_id", "word_ids", "word_segment_ids", "word_attention_mask", "entity_ids",
                    "entity_position_ids", "entity_segment_ids", "entity_attention_mask"]

    for data in tqdm(dataset.create_dict_iterator(num_epochs=1)):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])

        unique_id, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, \
            entity_segment_ids, entity_attention_mask = input_data
        logits = model.predict(word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                               entity_segment_ids, entity_attention_mask)
        ids = unique_id.asnumpy()
        start = logits[0].asnumpy()
        end = logits[1].asnumpy()
        for i in range(len(ids)):  # eval_batch_size
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return output


def main(pargs):
    config = Config(pargs.config)
    context.set_context(**config.context)
    squad = SquadLuke()
    dev_dataloader = squad(evaluate=True, config=config.dataset)

    outputs = do_eval(dataset=dev_dataloader, load_checkpoint_path=config.model.save_path)
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

    all_results = []
    for item in outputs:
        unique_id = int(item[0])
        start_logits = item[1]
        end_logits = item[2]
        all_results.append(RawResult(unique_id, start_logits, end_logits))

    processor = SquadV1Processor()
    examples = processor.get_dev_examples(config.dataset.path)

    features = squad.process(config=config.dataset, evaluate=True)
    list_dict = []
    for item in features:
        dict_temp = json.loads(item)
        list_dict.append(dict_temp)
    features = np.array(list_dict)
    #
    output_prediction_file = os.path.join("./result", "predictions_{}.json".format(""))
    output_nbest_file = os.path.join("./result", "nbest_predictions_{}.json".format(""))
    output_null_log_odds_file = None

    write_predictions(
        examples,
        features,
        all_results,
        20,
        30,
        True,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        0.0,
    )

    evaluate_on_squad(
        SQUAD_EVAL_OPTS(
            "./dataset/dev-v1.1.json",
            pred_file=output_prediction_file,
            na_prob_file=output_null_log_odds_file,
        )
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
