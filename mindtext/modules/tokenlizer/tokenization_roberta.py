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
     Tokenization classes for RoBERTa.
"""

import json
import itertools
import logging
from functools import lru_cache
import regex as re

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_ROBERTA_MODEL_DIR = {
    'en': 'roberta-base.zip',
    'en-large': 'roberta-large.zip'
}

PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}

PATTERN = \
    re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@lru_cache()
def bytes_to_unicode() -> dict:
    """
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at such like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.

    Returns: list of utf-8 byte and a mapping to unicode strings.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: tuple) -> set:
    """
    Return set of symbol pairs in a word.

    Args:
        word:input a tuple of symbols.

    Returns:
        pairs:symbols being variable-length strings.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def roberta_tokenize(text: str, add_prefix_space=True) -> list:
    """
    the base roberta tokenize.

    Args:
        text(str):input text.
        add_prefix_space(bool): While True can make sure same as Roberta in training.

    Returns:
        tokens:the token's list.
    """
    if text == '':
        return []
    if add_prefix_space:
        text = ' ' + text
    tokens = []
    for token in re.findall(PATTERN, text):
        tokens.append(token)
    return tokens


class RobertaTokenizer:
    """
    Derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.
    Due to the particularity of Chinese, it is not suitable to adopt byte level coding,
    So most open-source ChineseRoberta pre-training models still use a single word thesaurus,
    So they can be read directly by bert_tokenizer without roberta_tokenizer

    Args:
        _token:the special token.
        vocab_file:The path of vocab file.
        merges_file:The path of merges file.All tokens obtained by the merge file.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "sep_token",
    ]

    padding_side = "right"

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            **kwargs
    ):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.max_len = int(1e12)
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and \
                           all(isinstance(t, str) for t in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

        self.max_len_single_sentence = self.max_len - 2
        self.max_len_sentences_pair = self.max_len - 4

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

    def _reinit_on_new_vocab(self, vocab: dict):
        self.encoder = {k: v for k, v in vocab.items()}
        self.decoder = {v: k for k, v in vocab.items()}
        self.cache = {}

    @property
    def bos_token(self) -> str:
        """Beginning of sentence token (string). Log an error if used while not having been set."""
        if self._bos_token is None:
            logging.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self) -> str:
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logging.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self) -> str:
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logging.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def pad_token(self) -> str:
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logging.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self) -> str:
        """ Classification token (string). E.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model.
            Log an error if used while not having been set. """
        if self._cls_token is None:
            logging.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def sep_token(self) -> str:
        """ Separation token (string). E.g. to be the end of a sequence
            or separation two sentence.
            Log an error if used while not having been set. """
        if self._sep_token is None:
            logging.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def mask_token(self) -> str:
        """ Mask token (string). E.g. when training a model with masked-language modeling.
            Log an error if used while not having been set. """
        if self._mask_token is None:
            logging.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @property
    def bos_index(self) -> list:
        """Id of the beginning of sentence token in the vocabulary.
           Log an error if used while not having been set."""
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def sep_index(self) -> list:
        """Id of the separation in the sequence.
           Log an error if used while not having been set."""
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def eos_index(self) -> list:
        """Id of the end of sentence token in the vocabulary.
           Log an error if used while not having been set."""
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_index(self) -> list:
        """ Id of the unknown token in the vocabulary.
            Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def pad_index(self) -> list:
        """ Id of the padding token in the vocabulary.
            Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> list:
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_index(self) -> list:
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of
            an input sequence leveraging self-attention along the full depth of the model.
            Log an error if used while not having been set."""
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_index(self) -> list:
        """ Id of the mask token in the vocabulary. E.g. when training a model with
            masked-language modeling. Log an error if used while not having been set."""
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def bpe(self, token: list) -> str:
        """
        Byte-level Byte-Pair-Encoding.
        Requires a space to start the input string => the encoding and tokenize methods.
        Should be called with the ``add_prefix_space`` flag set to ``True``.
        Otherwise, this tokenizer's 'encode', 'decode', 'tokenize' methods will not conserve.
        Example:
            the spaces at the beginning of a string:
            tokenizer.decode(tokenizer.encode(" Hello")) = "Hello"

        Args:
            token:input token.

        Returns:
            word:a tuple of tokens.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:

            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text: str, add_prefix_space=False) -> list:
        """
        Tokenize a string.

        Args:
            text(str):input text.
            add_prefix_space (boolean, default False):Begin the sentence with
            at least one space to get invariance to word order in RoBERTa tokenizers.

        Returns:
            bpe_tokens:tokens of the BPE.
        """
        bpe_tokens = []
        for token in roberta_tokenize(text, add_prefix_space=add_prefix_space):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controller tokens of the BPE.
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token: list) -> list:
        """
        Converts a token (str) in an id using the vocab.

        Args:
            token:input token.

        Returns:
            list of ids.
        """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: list) -> list:
        """
        Converts an index (integer) in a token (str) using the vocab.

        Args:
            index: list of ids.

        Returns:
            list of tokens.
        """
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens: list) -> str:
        """
        Converts a sequence of tokens (string) in a single string.

        Args:
            tokens: list of token.

        Returns:
            text:input tokens in string.
        """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.

        Returns:
            a number of size.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def tokenize(self, text: str, add_prefix_space=True) -> list:
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        Split in words for word-based vocabulary or sub-words for sub-word-based
        vocabularies (BPE/SentencePieces/WordPieces).
        Take care of added tokens.

        Args:
            text: The sequence to be encoded.
            add_prefix_space (boolean, default True):Begin the sentence with
            at least one space to get invariance to word order in RoBERTa tokenizers.

        Returns:
            tokenized_text:text in tokenized.
        """
        all_special_tokens = self.all_special_tokens

        def lowercase_text(t: str) -> str:
            """
            Convert non-special tokens to lowercase.

            Args:
                t:input text.

            Returns:
                to lower case the input text. Should be True for uncased.
            """
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r'(' + r'|'.join(escaped_special_toks) + r')|' + \
                      r'(.+?)'
            return re.sub(
                pattern,
                lambda m: m.groups()[0] or m.groups()[1].lower(),
                t)

        if self.init_kwargs.get('do_lower_case', False):
            text = lowercase_text(text)

        def split_on_token(tok: str, text: str) -> list:
            """
            Do basic tokenization.

            Args:
                tok: input token.
                text: text in unicode.

            Returns:
                a list of tokens split from text.
            """
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list: list, text: str) -> list:
            """
            Do basic tokenization.

            Args:
                tok_list: input a list of token.
                text: text in unicode.

            Returns:
                a list of tokens split from text.
            """
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text, add_prefix_space=add_prefix_space)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (self._tokenize(token, add_prefix_space=add_prefix_space)
                     if token not in self.added_tokens_encoder and token not in all_special_tokens
                     else [token] for token in tokenized_text)))

        added_tokens = list(self.added_tokens_encoder.keys()) + all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def convert_tokens_to_ids(self, tokens: list) -> list:
        """
        Converts a single token, or a sequence of tokens, (str) in a single integer id
        (resp. a sequence of ids), using the vocabulary.

        Args:
            tokens:a list of token.

        Returns:
            a list of id.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: list) -> list:
        """
        The dictionary you added converts the token into an ID.

        Args:
            token:input token.

        Returns:
            token in id.
        """
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def convert_ids_to_tokens(self, ids: list, skip_special_tokens=False) -> list:
        """
        Converts a single index or a sequence of indices (integers) in a token "
        (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

        Args:
            skip_special_tokens: Don't decode special tokens (self.all_special_tokens).

        Returns:
            a list of token.
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_id_to_tokens(self, token_ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=True) -> list:
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        To avoid mixing byte-level and unicode for byte-level BPT
        we need to build string separately for added tokens and byte-level tokens

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the
                                            `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.

        Returns:
            a list of tokens.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids,
                                                     skip_special_tokens=skip_special_tokens)
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text

    @property
    def special_tokens_map(self) -> dict:
        """
        A dictionary mapping special token class attribute (cls_token, unk_token...) to their
        values ('<unk>', '<cls>'...).

        Returns:
            a dictionary mapping.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> list:
        """
        List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
        (cls_token, unk_token...).

        Returns:
            a list of all special tokens.
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value)
                                   if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> list:
        """
        List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
        class attributes (cls_token, unk_token...).

        Returns:
            a list of all special ids.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like
        spaces before punctuations and abbreviated forms.

        Args:
            out_string:The text does not remove spaces.

        Returns:
            the text have removed spaces.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def encode(self, text: str, add_special_tokens=False, add_prefix_space=True) -> list:
        """
        Given the text input, encode the data in the form of index.
        Example::
             from mindtext.modules.tokenlizer import tokenization_roberta
             roberta_tokenizer = tokenization_roberta.from_pretrained('en')
             print(tokenization_roberta.encode('from'))
             print(tokenization_roberta.encode("This is a demo sentence"))
             print(tokenization_roberta.encode(["This", "is", 'a']))

        Args:
            text(List[str],str): An entry entered is considered a sentence.

            add_special_tokens(bool): Whether to ensure that the beginning
            and end of the sentence are CLS and Sep.

            add_prefix_space(bool):(boolean, default True):Begin the sentence with
            at least one space to get invariance to word order in RoBERTa tokenizers.

        Returns:
             word_pieces:the data in the form of index encoded by text.
        """
        if isinstance(text, str):
            words = text.split()
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("Only support str or List[str]")

        word_pieces = []
        for word in words:
            tokens = self.tokenize(word, add_prefix_space=add_prefix_space)
            word_piece_ids = self.convert_tokens_to_ids(tokens)
            word_pieces.extend(word_piece_ids)
        if add_special_tokens:
            if self._cls_token is not None and word_pieces[0] != self.cls_index:
                word_pieces.insert(0, self.cls_index)
            if self._sep_token is not None and word_pieces[-1] != self.sep_index:
                word_pieces.append(self.eos_index)
        return word_pieces

    def get_used_merge_pair_vocab(self, token: list) -> (tuple, dict):
        """
        If the token is not found, it will be split into letters and returned.

        Examples:
            If the word is "abcd"，it will  split into((a,b), (b,c), (c, d), (e,f))

        Args:
            token:input token.

        Returns:
            word:the word after split.
            used_pairs:The most common pair.
        """
        used_pairs = {}
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token, used_pairs

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            used_pairs[bigram] = self.bpe_ranks[bigram]
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        return word, used_pairs
