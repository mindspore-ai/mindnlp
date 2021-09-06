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
Tokenization.
"""

import unicodedata
import collections

def convert_to_unicode(text: str) -> str:
    """
    Convert text into unicode type.
    Args:
        text: input str.

    Returns:
        input str in unicode.
    """
    ret = text
    if isinstance(text, str):
        ret = text
    elif isinstance(text, bytes):
        ret = text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return ret


def vocab_to_dict_key_token(vocab_file: str) -> dict:
    """Loads a vocab file into a dict, key is token.
       Args:
           vocab_file:the path of voacb file.
       Returns:
            vocab:A dictionary containing tokens and their corresponding ids.
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='UTF-8') as reader:
        line = reader.readline()
        while line:
            token = convert_to_unicode(line)
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
            line = reader.readline()
    return vocab


def vocab_to_dict_key_id(vocab_file: str) ->dict:
    """Loads a vocab file into a dict, key is id.
       Args:
           vocab_file:the path of voacb file.
       Returns:
            vocab:a dictionary containing ids and their corresponding tokens.
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        line = reader.readline()
        while line:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[index] = token
            index += 1
            line = reader.readline()
    return vocab


def whitespace_tokenize(text: str) -> list:
    """Runs basic whitespace cleaning and splitting on a piece of text.
       Args:
            text:text to be processed.
       Returns:
            tokens: cleaning and splitting on a piece of text.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer():
    """
    Full tokenizer.
    Args:
        vocab_file:the path of vocab file.
        do_lower_case:Whether to lower case the input text. Should be True for uncased.
    """
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab_dict = vocab_to_dict_key_token(vocab_file)
        self.do_lower_case = do_lower_case
        self.basic_tokenize = BasicTokenizer(do_lower_case)
        self.wordpiece_tokenize = WordpieceTokenizer(self.vocab_dict)
        self.id_dict = vocab_to_dict_key_id(vocab_file)

    def convert_tokens_to_ids(self, tokens: list) -> list:
        """
        Convert tokens to ids.
        Args:
            vocab_dict:  vocab dict.
            tokens: list of tokens.

        Returns:
            list of ids.
        """

        output = []
        for token in tokens:
            output.append(self.vocab_dict[token])
        return output

    def convert_ids_to_tokens(self, ids: list) -> list:
        """
        Convert ids to tokens.
        Args:
            vocab_file: path to vocab.txt.
            ids: list of ids.

        Returns:
            list of tokens.
        """

        output = []
        for i in ids:
            output.append(self.id_dict[i])
        return output

    def tokenize(self, text: str) -> list:
        """
        Do full tokenization.
        Args:
            text: str of text.

        Returns:
            list of tokens.
        """
        tokens_ret = []
        text = convert_to_unicode(text)
        for tokens in self.basic_tokenize.tokenize(text):
            wordpiece_tokens = self.wordpiece_tokenize.tokenize(tokens)
            tokens_ret.extend(wordpiece_tokens)
        return tokens_ret


class BasicTokenizer():
    """
    Basic tokenizer.
    Args:
        do_lower_case:Whether to lower case the input text. Should be True for uncased.
    """
    def __init__(self, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> list:
        """
        Do basic tokenization.
        Args:
            text: text in unicode.

        Returns:
            a list of tokens split from text.
        """
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            aaa = self._run_split_on_punc(token)
            split_tokens.extend(aaa)

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str) ->str:
        """Strips accents from a piece of text.
        Args:
            text:text to be processed."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text: str) -> list:
        """Splits punctuation on a piece of text.
        Args:
            text:text to be processed.
        """
        i = 0
        start_new_word = True
        output = []
        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _clean_text(self, text: str) ->str:
        """Performs invalid character removal and whitespace cleanup on text.
        Args:
            text:text to be processed."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text: str) -> str:
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character.
        Args:
            cp:whether CP is the codepoint of a CJK character.
        Return:
            if CP is the codepoint of a CJK character,return True.
            else,Flase.
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False


class WordpieceTokenizer():
    """
    Wordpiece tokenizer.
    """
    def __init__(self, vocab: dict):
        self.vocab_dict = vocab

    def tokenize(self, tokens: str) -> list:
        """
        Do word-piece tokenization.
        Args:
            tokens: a word.

        Returns:
            a list of tokens that can be found in vocab dict.
        """
        output_tokens = []
        tokens = convert_to_unicode(tokens)
        for token in whitespace_tokenize(tokens):
            chars = list(token)
            len_chars = len(chars)
            start = 0
            end = len_chars
            while start < len_chars:
                while start < end:
                    substr = "".join(token[start:end])
                    if start != 0:
                        substr = "##" + substr
                    if substr in self.vocab_dict:
                        output_tokens.append(substr)
                        start = end
                        end = len_chars
                    else:
                        end = end - 1
                if start == end and start != len_chars:
                    output_tokens.append("[UNK]")
                    break
        return output_tokens


def _is_whitespace(char: str) -> bool:
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    whitespace_char = [" ", "\t", "\n", "\r"]
    if char in whitespace_char:
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char: str) -> bool:
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    control_char = ["\t", "\n", "\r"]
    if char in control_char:
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char: str) -> bool:
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
