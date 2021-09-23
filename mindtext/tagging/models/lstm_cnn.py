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
"""LSTM-CNNs Model for sequence tagging."""
from typing import Union, List, Optional

import mindspore
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindtext.common.data import Vocabulary
from mindtext.embeddings.static_embedding import StaticEmbedding
from mindtext.embeddings.char_embedding import CNNCharEmbedding
from mindtext.modules.decoder.norm_decoder import NormalDecoder


class LstmCnnConfig(nn.Cell):

    """
    Config of LSTM-CNN model.

    Args:
        vocab (Vocabulary): The vocabulary for word embedding and char embedding.
        model_dir_or_name (Union[str, None]): The model file path for pretrained embedding, such as Google Glove.
        word_embed_size (int): The embedding size of word embedding. Default: 100.
        char_embed_size (int): The embedding size of char embedding. Default: 100.
        embed_dropout (float): The dropout rate of the embedding. Default: 0.1.
        filter_nums (List[int]): The number of filters. The length needs to be consistent with the kernels.
                                Default: [40, 30, 20].
        conv_kernel_sizes (List[int]): The size of kernel. Default: [5, 3, 1].
        pool_method (str): The pool method used when synthesizing the representation of the character into a
                            representation, supports 'avg' and 'max'. Default: max.
        conv_char_embed_activation (str): The activation method used after CNN, supports 'relu','sigmoid','tanh' or
                                            custom functions.
        min_char_freq (int): The minimum frequency of occurrence of a character. Default: 2.
        num_layers (int): The number of layers of bi_lstm cell. Default: 2.
        hidden_size (int): The size of hidden layers of bi_lstm cell. Default: 100.
        output_size (int): The size of output of LSTM-CNNs model. Default: 100.
        hidden_activation (str): The activation function of hidden layer (linear decoder). Default: 'relu'.
        hidden_dropout (float): The dropout rate of bi_lstm and linear decoder. Default: 0.1.
        embed_requires_grad (bool): Whether to update the weight.
        pre_train_char_embed (str): There are two ways to call the pre-trained character embedding: the first is to pass
                                    in the embedding folder (there should be only one file with .txt as the suffix) or
                                    the file path. The second is to pass in the name of the embedding. In the second
                                    case, it will automatically check whether the model exists in the cache, if not,
                                    it will be downloaded automatically. If the input is None, use the dimension of
                                    embedding_dim to randomly initialize an embedding.
        include_word_start_end (bool): Whether to add special marking symbols before and ending the character at the
                                        beginning and end of each word.

    """

    def __init__(self,
                 vocab: Vocabulary,
                 model_dir_or_name: Union[str, None] = None,
                 word_embed_size: int = 100,
                 char_embed_size: int = 100,
                 embed_dropout: float = 0.1,
                 filter_nums: List[int] = (40, 30, 20),
                 conv_kernel_sizes: List[int] = (5, 3, 1),
                 pool_method: str = 'max',
                 conv_char_embed_activation: str = 'relu',
                 min_char_freq: int = 2,
                 num_layers: int = 2,
                 hidden_size: int = 100,
                 output_size: int = 100,
                 hidden_activation: str = 'relu',
                 hidden_dropout: float = 0.1,
                 embed_requires_grad: bool = True,
                 pre_train_char_embed: Optional[str] = None,
                 include_word_start_end: bool = True
                 ):
        super(LstmCnnConfig, self).__init__()
        self.vocab = vocab
        self.model_dir_or_name = model_dir_or_name
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.embed_dropout = embed_dropout
        self.embed_requires_grad = embed_requires_grad
        self.filter_nums = filter_nums
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_method = pool_method
        self.conv_char_embed_activation = conv_char_embed_activation
        self.min_char_freq = min_char_freq
        self.pre_train_char_embed = pre_train_char_embed
        self.include_word_start_end = include_word_start_end
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = word_embed_size + char_embed_size
        self.hidden_dropout = hidden_dropout
        self.output_size = output_size
        self.hidden_activation = hidden_activation


class LstmCnn(nn.Cell):

    """
    LSTM-CNNs model.

    Args:
        config (LstmCnnConfig): The config object for LSTM-CNNs model.
    """

    def __init__(self, config: LstmCnnConfig):
        super(LstmCnn, self).__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        # Word embedding.
        self.word_embedding = StaticEmbedding(vocab=config.vocab, model_dir_or_name=config.model_dir_or_name,
                                              embedding_dim=config.word_embed_size,
                                              requires_grad=config.embed_requires_grad,
                                              dropout=config.embed_dropout)

        # CNN char embedding.
        self.cnn_char_embedding = CNNCharEmbedding(vocab=config.vocab, embed_size=config.word_embed_size,
                                                   char_emb_size=config.char_embed_size,
                                                   dropout=config.embed_dropout, filter_nums=config.filter_nums,
                                                   kernel_sizes=config.conv_kernel_sizes,
                                                   pool_method=config.pool_method,
                                                   activation=config.conv_char_embed_activation)

        # bi_lstm
        self.bi_lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                               num_layers=config.num_layers, batch_first=True,
                               dropout=config.hidden_dropout, bidirectional=True)

        # Output layer (Decoder): linear + log_softmax
        # NormalDecoder is a linear decoder.
        self.liner_decoder = NormalDecoder(num_filters=config.hidden_size * 2, num_classes=config.output_size,
                                           classes_dropout=config.hidden_dropout,
                                           activation=config.hidden_activation)
        # log_softmax
        self.log_softmax = nn.LogSoftmax()

        # Utils for LSTM-CNNs.
        self.concat = P.Concat(axis=-1)
        self.zeros = P.Zeros()

    def construct(self, words: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply LSTM-CNNs model.

        Args:
            words (Tensor): The shape is (batch_size, max_len). The index of words.

        Returns:
            output (Tensor): The shape is (batch_size, max_len, output_size). The output of LSTM-CNNs model.
        """
        batch_size = words.shape[0]
        word_emb = self.word_embedding(words)
        char_emb = self.cnn_char_embedding(words)
        word_char_emb = self.concat((word_emb, char_emb))

        h0 = self.zeros((2 * self.num_layers, batch_size, self.hidden_size), mstype.float32)
        c0 = self.zeros((2 * self.num_layers, batch_size, self.hidden_size), mstype.float32)
        # word_char_emb = word_char_emb.transpose(1, 0, 2)
        output, _ = self.bi_lstm(word_char_emb, (h0, c0))
        # output = output.transpose(1, 0, 2)
        output = self.liner_decoder(output)
        output = self.log_softmax(output)
        return output
