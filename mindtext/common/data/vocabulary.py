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
"""vocabulary class"""
import re
import spacy


class Vocabulary():
    """
        vocabulary class

        """

    def __init__(self, data_list=None,
                 max_length=None,
                 is_eng=True,
                 is_train=True):
        self.data_list = data_list
        self.max_length = max_length
        self.text_greater = '>'
        self.text_less = '<'
        self.word2idx = dict()
        self.idx2words = dict()
        self.non_str = '\\'
        self.end_string = ['.', '?', '!']
        self.word2idx['PAD'] = 0
        self.idx2words[0] = 'PAD'
        self.word2idx['UNK'] = 1
        self.idx2words[1] = 'UNK'
        self.str_html = re.compile(r'<[^>]+>')
        self.is_eng = is_eng
        self.is_train = is_train

    def text2tokens(self, src_text, is_train=False):
        """src text to token """
        if self.is_eng:
            spacy_nlp = spacy.load('load_core_web_lg', disable=['parser', 'tagger', 'ner'])
            spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
            doc = spacy_nlp(src_text)
            bows_token = [token.text for token in doc]
            if is_train is True:
                for ngms in bows_token:
                    idx = self.word2idx.get(ngms)
                    if idx is None:
                        idx = len(self.word2idx)
                        self.word2idx[ngms] = idx
                        self.idx2words[idx] = ngms
            processed_out = [self.word2idx[ng] if ng in self.word2idx else self.word2idx['UNK'] for ng in bows_token]

        return processed_out

    def update(self, data_list, is_train=True):
        """update vocab"""
        for item in data_list:
            self.text2tokens(item, is_train=is_train)

    def vocab_to_txt(self, path):
        """write vocab.txt"""
        with open(path, "w") as f:
            for k, v in self.word2idx.items():
                f.write(k + "\t" + str(v) + "\n")

    def read_vocab_txt(self, path):
        """read vocab.txt"""
        with open(path, "r") as f:
            lines = f.readlines()
            for i, word in enumerate(lines):
                s = word.split("\t")
                self.word2idx[s[0]] = i


if __name__ == "__main__":
    vocab = Vocabulary()
