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
"""LSTM_CNN eval script"""
import os
import mindspore.nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindtext.common.utils.config import Config, parse_args
from mindtext.dataset.builder import build_dataset
from mindtext.tagging.models.lstm_cnn import LstmCnn, LstmCnnConfig, LstmCnnWithLoss
from tqdm import tqdm


def compute_precision(guessed_sentences, correct_sentences):
    """
    NER eval function

    Args:
        guessed_sentences: predict labels
        correct_sentences: correct labels
    """
    all_correct_count = 0
    all_count = 0

    for sentence_idx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentence_idx]
        correct = correct_sentences[sentence_idx]
        correct_count, count = compare(guessed, correct)
        all_correct_count += correct_count
        all_count += count

    # if user find there is error, please consider if variable count==0, it means that no 'B' label.
    return float(all_correct_count) / all_count


def compare(guessed, correct):
    """
    NER compare function

    Args:
        guessed: each sentence predict labels
        correct: each sentence correct labels
    """
    correct_count, count = 0, 0
    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:
                idx += 1
                correctly_found = True

                while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctly_found = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':  # The chunk in correct was longer
                        correctly_found = False

                if correctly_found:
                    correct_count += 1
            else:
                idx += 1
        else:
            idx += 1
    return correct_count, count


def main(config):
    """
    Eval function

    Args:
        config: Yaml config
    """
    context.set_context(**config.context)  # jingtaitu mode

    # 初始化数据集
    dataset = build_dataset(config.eval.dataset)
    dataloader = dataset()

    lstmcnn_config = LstmCnnConfig(vocab=dataset.vocab)

    network = LstmCnn(lstmcnn_config)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    model = LstmCnnWithLoss(network, loss).set_train(False)
    param_dict = load_checkpoint(config.model.save_path)
    load_param_into_net(model, param_dict)

    ner_dict = {0: "O",
                1: "B-PER",
                2: "I-PER",
                3: "B-ORG",
                4: "I-ORG",
                5: "B-LOC",
                6: "I-LOC",
                7: "B-MISC",
                8: "I-MISC"}
    # begin to eval
    print('[Start eval]')
    print("=" * 80)
    sm = mindspore.nn.Softmax(axis=-1)
    predict_list = []
    label_list = []
    for batch in tqdm(dataloader['test'].create_dict_iterator(num_epochs=1),
                      total=dataloader['test'].get_dataset_size()):
        label = batch['ner_tags']
        length = batch['input_length'].asnumpy()
        input_mask = batch['input_mask']
        predict = model(batch['input_ids'], input_mask, label)
        label = label.asnumpy()

        predict = sm(predict).argmax(axis=-1)
        predict = predict.asnumpy()

        for i in range(label.shape[0]):
            predict_list.append([ner_dict[p] for p in predict[i][:length[i][0]].tolist()])
            label_list.append([ner_dict[l] for l in label[i][:length[i][0]].tolist()])

    prec = compute_precision(predict_list, label_list)
    rec = compute_precision(label_list, predict_list)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    with open(os.path.join(config.model.result_path, 'result.txt'), "w") as f:
        f.write(str(f1))
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (prec, rec, f1))


if __name__ == '__main__':
    args = parse_args()
    main(Config(args.config))
