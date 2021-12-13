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
"""The tool of trans the form of the data from raw to which mindtext.dataset can deal"""
import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


_SPLIT = '$%$'
_SCRIPT_PATH = './utils/conll2003.py'

parser = argparse.ArgumentParser("MindVison classification script.")
parser.add_argument('-s',
                    '--source',
                    type=str,
                    default="",
                    help='Dataset source directory path.')
parser.add_argument('-t',
                    '--target',
                    type=str,
                    default="",
                    help='Target directory path.')


def conll_convert_pd(source_path, target_path):
    """
    Args:
        source_path: A file of saving the raw conll2003 data, form:  .txt
        target_path: A file of saving the conll2003 data after processing, form:  .csv

    Returns:
        None
    """
    if source_path[-1] != '/':
        source_path = source_path + '/'
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    with open(_SCRIPT_PATH, 'r', encoding='utf-8') as f:
        r_lines = f.readlines()
    changed = False
    with open(_SCRIPT_PATH, 'w', encoding='utf-8') as f:
        url = '_URL = "' + source_path + '"\n'
        for line in r_lines:
            if not changed and line.find('_URL') != -1:
                f.write('%s' % url)
                changed = True
            else:
                f.write('%s' % line)
    conll = load_dataset(_SCRIPT_PATH)
    for dataset_type in conll.keys():
        data_dict = {key: [] for key in conll[dataset_type].column_names}
        for inst in tqdm(conll[dataset_type]):
            data_dict['id'].append(inst['id'])
            data_dict['tokens'].append(_SPLIT.join(inst['tokens']))
            data_dict['pos_tags'].append(_SPLIT.join([str(i) for i in inst['pos_tags']]))
            data_dict['chunk_tags'].append(_SPLIT.join([str(i) for i in inst['chunk_tags']]))
            data_dict['ner_tags'].append(_SPLIT.join([str(i) for i in inst['ner_tags']]))
        dataset = pd.DataFrame(data_dict)
        if dataset_type == 'validation':
            dataset.to_csv(os.path.join(target_path, './dev.csv'), index=False)
            continue
        dataset.to_csv(os.path.join(target_path, './' + dataset_type + '.csv'), index=False)
        if dataset_type == 'train':
            pos_tags_dict = {'tags': []}
            chunk_tags_dict = {'tags': []}
            ner_tags_dict = {'tags': []}
            for i in conll[dataset_type].features['pos_tags'].feature.names:
                pos_tags_dict['tags'].append(i)
            for i in conll[dataset_type].features['chunk_tags'].feature.names:
                chunk_tags_dict['tags'].append(i)
            for i in conll[dataset_type].features['ner_tags'].feature.names:
                ner_tags_dict['tags'].append(i)
            pos = pd.DataFrame(pos_tags_dict)
            chunk = pd.DataFrame(chunk_tags_dict)
            ner = pd.DataFrame(ner_tags_dict)
            pos.to_csv(os.path.join(target_path, './pos_tags.csv'), index=False)
            chunk.to_csv(os.path.join(target_path, './chunk_tags.csv'), index=False)
            ner.to_csv(os.path.join(target_path, './ner_tags.csv'), index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    conll_convert_pd(args.source, args.target)
