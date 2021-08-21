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
"""Bert model."""
import copy
import math
from typing import Tuple
import six
import yaml
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


class BertConfig:
    """
    Configuration for `BertModel`.

    Args:
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the BertTransformer
                             encoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the BertTransformer
                           encoder cell. Default: 3072.
        hidden_act (str): Activation function used in the BertTransformer encoder
                    cell. Default: "gelu".
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        type_vocab_size (int): Size of token type vocab. Default: 16.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer.
                                        Default: mstype.float32.
    """

    def __init__(self,
                 seq_length: int = 128,
                 vocab_size: int = 32000,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size=3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 16,
                 initializer_range: float = 0.02,
                 dtype: mstype = mstype.float32,
                 compute_type: mstype = mstype.float32):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.dtype = dtype
        self.compute_type = compute_type

    @classmethod
    def from_dict(cls, yaml_object: dict) -> object:
        """Constructs a `BertConfig` from a yaml object of parameters.
        Args:
           yaml_object : a dict  contains the parameters in the YAMl file.
        Returns:
            config : a Bertconfig object that  contains the parameters.



        """
        config = BertConfig()
        for (key, value) in yaml_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_yaml_file(cls, yaml_file: str) -> object:
        """Constructs a `BertConfig` from a yaml file of parameters.
        Args:
           yaml_file : Path to the YAML file.
        Returns:
           cls.from_dict : a Bertconfig object that  contains the parameters.

        """
        f = open(yaml_file, 'r', encoding='utf-8')
        cont = f.read()
        yaml_dict = yaml.load(cont, Loader=yaml.FullLoader)
        return cls.from_dict(yaml_dict)

class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
                              Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 embedding_shape: int,
                 use_one_hot_embeddings: bool = False,
                 initializer_range: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer(TruncatedNormal(initializer_range),
                                                     [vocab_size, embedding_size]))
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)  # One-hot value
        self.off_value = Tensor(0.0, mstype.float32)  # Except for the one-hot position,                                          # other locations take value `off_value`.
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)

    def construct(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Get output and embeddings lookup table.

        Args:
            input_ids(Tensor vector):A vector containing the transformation of characters into corresponding ids.

        Returns:
            output(Tensor vector):input_ids are converted to high dimensional word embedding.
            embedding_table(Tensor matrix):fixed word embedding table.

        """
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # Reshape to [batch_size, seq_length, 1].
        if input_ids.shape.ndims == 2:
            input_ids = self.expand(input_ids, -1)
        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table


def assert_rank(tensor: Tensor, expected_rank: list, name: str = "input") -> None:
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for integer in expected_rank:
            expected_rank_dict[integer] = True

    actual_rank = len(tensor.shape)
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s` , the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor: Tensor, expected_rank: list = None, name: str = "input") -> list:
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      out_shape: A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as Tensor scalars.
    """

    if expected_rank:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape

    out_shape = []
    for dim in enumerate(shape):
        out_shape.append(dim)
    return out_shape


def reshape_to_matrix(input_tensor: Tensor) -> Tensor:
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).
    Args:
        input_tensor:a >= rank 2 tensor.

    Returns:
         output_tensor:a rank 2 tensor (i.e., a matrix).

    """
    ndims = len(input_tensor.shape)
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]

    output_tensor = P.Reshape()(input_tensor, [-1, width])
    return output_tensor


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: False.
        token_type_vocab_size (int): Size of token type vocab. Default: 16.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
                            Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """

    def __init__(self,
                 embedding_size: int,
                 embedding_shape: list,
                 use_token_type: bool = False,
                 token_type_vocab_size: int = 16,
                 use_one_hot_embeddings: bool = False,
                 max_position_embeddings: int = 512,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.token_type_embedding = nn.Embedding(
            vocab_size=token_type_vocab_size,
            embedding_size=embedding_size,
            use_one_hot=use_one_hot_embeddings)
        self.shape_flat = (-1,)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.gather = P.Gather()
        self.slice = P.StridedSlice()
        _, seq, _ = self.shape
        self.full_position_embedding = nn.Embedding(
            vocab_size=max_position_embeddings,
            embedding_size=embedding_size,
            use_one_hot=False)
        self.layernorm = nn.LayerNorm((embedding_size,))
        self.position_ids = Tensor(np.arange(seq).reshape(-1, seq).astype(np.int32))
        self.add = P.Add()

    def construct(self, token_type_ids: Tensor, word_embeddings: Tensor) -> Tensor:
        """Postprocessors apply positional and token type embeddings to word embeddings.
           Args: token_type_ids:The tensor vector with the segment id.
                 word_embeddings:Word embedding vector.

           Returns:
                output:A embedding vector that fuses a word embedding vector
                with a position embedding vector or segment embedding vector.

        """
        output = word_embeddings
        if self.use_token_type:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            output = self.add(output, token_type_embeddings)
        position_embeddings = self.full_position_embedding(self.position_ids)
        output = self.add(output, position_embeddings)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


class EncoderOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        dropout_prob (float): The dropout probability. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer.
                                        Default: mstype.float32.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer_range: float = 0.02,
                 dropout_prob: float = 0.1,
                 compute_type: mstype = mstype.float32):
        super().__init__()
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()

    def construct(self, hidden_status: Tensor, input_tensor: Tensor) -> Tensor:
        '''
        Args:
            hidden_status: hidden status.
            input_tensor: the input of residual computation.

        Returns:
            output:a linear computation to hidden status and a residual computation to input.
        '''
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(input_tensor, output)
        output = self.layernorm(output)
        return output


class AttentionLayer(nn.Cell):
    """ Layer for attention.

  Args:
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        from_seq_length (int): Length of from_tensor sequence.
        to_seq_length (int): Length of to_tensor sequence.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertAttention. Default: mstype.float32.


    """
    def __init__(self, from_tensor_width: int,
                 to_tensor_width: int,
                 num_attention_heads: int = 1,
                 size_per_head: int = 512,
                 has_attention_mask: bool = True,
                 query_act: str = None,
                 key_act: str = None,
                 value_act: str = None,
                 attention_probs_dropout_prob: float = 0.0,
                 initializer_range: float = 0.02,
                 do_return_2d_tensor: bool = False,
                 batch_size: int = None,
                 from_seq_length: int = None,
                 to_seq_length: int = None,
                 compute_type: mstype = mstype.float32):
        super().__init__()
        self.from_tensor_width = from_tensor_width
        self.to_tensor_width = to_tensor_width
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.BatchMatMul()
        self.multiply = P.Mul()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attention_probs_dropout_prob)
        self.multiply_data = -10000.0
        self.trans_shape = (0, 2, 1, 3)
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.shape_to = (-1, to_seq_length, num_attention_heads, size_per_head)
        units = num_attention_heads * size_per_head
        weight = TruncatedNormal(initializer_range)
        self.has_attention_mask = has_attention_mask
        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()
        if do_return_2d_tensor:
            self.shape_return = (-1, num_attention_heads * size_per_head)
        else:
            self.shape_return = (-1, from_seq_length, num_attention_heads * size_per_head)
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    weight_init=weight).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  weight_init=weight).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    weight_init=weight).to_float(compute_type)

    def transpose_for_scores(self, input_tensor: Tensor, batch_size: int, num_attention_heads: int,
                             seq_length: int, width: int) -> Tensor:
        ''' do the transpose for the attention score.

         Args:
             input_tensor:the input tensor.
             batch_size: batch size.
             num_attention_heads:the number of attention of heads.
             seq_length:sequcence length.
             width:embedding dimension.
         Returns:
               output_tensor:transposed input tensor.


         '''
        output_tensor = self.reshape(
            input_tensor, tuple((batch_size, seq_length, num_attention_heads, width)))
        input_perm = (0, 2, 1, 3)
        output_tensor = self.transpose(output_tensor, input_perm)
        return output_tensor

    def construct(self, from_tensor: Tensor, to_tensor: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''construct attention layer.
           Args:
                from_tensor:encoder sequence.
                to_tensor:decoder sequence.
                attention_mask:mask for attention.

           Returns:
                context_layer:the attention for the layer.
        '''
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (not self.batch_size  or not self.from_seq_length
                    or not self.to_seq_length):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)
        # `value_layer` = [B*T, N*H]
        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_out, self.batch_size,
                                                self.num_attention_heads, self.from_seq_length,
                                                self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_out, self.batch_size, self.num_attention_heads,
                                              self.to_seq_length, self.size_per_head)
        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores,
                                         1.0 / math.sqrt(float(self.size_per_head)))
        if not attention_mask is None:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        value_layer = self.reshape(value_out, self.shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)
        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        return context_layer


class BertEncoderCell(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 512.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
                            Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        hidden_act (str): Activation function. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int = 768,
                 seq_length: int = 512,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 attention_probs_dropout_prob: float = 0.02,
                 initializer_range: float = 0.02,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "gelu",
                 compute_type: mstype = mstype.float32):
        super().__init__()
        self.size_per_head = int(hidden_size / num_attention_heads)

        self.attention = AttentionLayer(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            has_attention_mask=True,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.attention_output = EncoderOutput(in_channels=hidden_size,
                                              out_channels=hidden_size,
                                              initializer_range=initializer_range,
                                              dropout_prob=hidden_dropout_prob,
                                              compute_type=compute_type)
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=hidden_act,
                                     weight_init=TruncatedNormal(initializer_range))\
                                                                .to_float(compute_type)
        self.output = EncoderOutput(in_channels=intermediate_size,
                                    out_channels=hidden_size,
                                    initializer_range=initializer_range,
                                    dropout_prob=hidden_dropout_prob,
                                    compute_type=compute_type)

    def construct(self, input_tensor: Tensor, attention_mask: Tensor) -> Tensor:
        '''
        Args:
            input_tensor: the input for BertEncoderCell
            attention_mask: the mask for the attention

        :return:
             output: the output of bert encoder
        '''
        input_tensor = self.reshape(input_tensor, self.shape)
        attention_score = self.attention(input_tensor, input_tensor, attention_mask)
        attention_output = self.attention_output(attention_score, input_tensor)
        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output)
        return output


class BertTransformer(nn.Cell):
    """
    Multi-layer bert transformer.

    Args:
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
                             Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer.
                                        Default: mstype.float32.
        return_all_encoders (bool): Specifies whether to return all encoders. Default: False.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 seq_length: int,
                 num_hidden_layers: int,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "gelu",
                 compute_type: mstype = mstype.float32,
                 return_all_encoders: bool = False):
        super().__init__()
        self.return_all_encoders = return_all_encoders
        layers = []
        for _ in range(num_hidden_layers):
            layer = BertEncoderCell(batch_size=batch_size, hidden_size=hidden_size,
                                    seq_length=seq_length,
                                    num_attention_heads=num_attention_heads,
                                    intermediate_size=intermediate_size,
                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                    initializer_range=initializer_range,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    hidden_act=hidden_act,
                                    compute_type=compute_type)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
        self.out_shape = (-1, seq_length, hidden_size)

    def construct(self, input_tensor: Tensor, attention_mask: Tensor) -> Tensor:
        """Multi-layer bert transformer."""
        prev_output = self.reshape(input_tensor, self.shape)
        all_encoder_layers = ()
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

            if self.return_all_encoders:
                layer_output = self.reshape(layer_output, self.out_shape)
                all_encoder_layers = all_encoder_layers +(layer_output,)

        if not self.return_all_encoders:
            prev_output = self.reshape(prev_output, self.out_shape)
            all_encoder_layers = all_encoder_layers +(prev_output,)
        return all_encoder_layers


def numbtpye2mstype(src_type: str) -> mstype:
    ''' Convert String to Mstype.
        Args:
            src_type:the String for mstype.
         Return:
             desc_type:convert String to Mstype.

    '''
    desc_type = None

    if src_type == "mstype.int32":
        desc_type = mstype.int32
    elif src_type == "mstype.int64":
        desc_type = mstype.int64
    elif src_type == "mstype.float32":
        desc_type = mstype.float32
    elif src_type == "mstype.float64":
        desc_type = mstype.float64
    return desc_type


class SecurityCast(nn.Cell):
    """
    Performs a safe saturating cast.
    This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor.
                                    Default: mstype.float32.
    """

    def __init__(self, dst_type: mstype = mstype.float32):
        super().__init__()
        np_type = float

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x: Tensor) -> Tensor:
        '''
        :param x: the input data.
        :return: casted data for preventing
         the danger that the value will overflow or underflow.
        '''
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.input_mask = None

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = (-1, 1, config.seq_length)

    def construct(self, input_mask: Tensor) -> Tensor:
        '''

        :param input_mask: the mask for input
        :return: Create attention mask according to input mask.
        '''
        attention_mask = self.cast(self.reshape(input_mask, self.shape), mstype.float32)
        return attention_mask


class BertModel(nn.Cell):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
                            Default: False.
    """

    def __init__(self,
                 config: BertConfig,
                 is_training: bool,
                 use_one_hot_embeddings: bool = False):
        super().__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None
        self.compute_type = numbtpye2mstype(config.compute_type)
        self.last_idx = self.num_hidden_layers - 1
        output_embedding_shape = [-1, self.seq_length, self.embedding_size]
        self.bert_embedding_lookup = nn.Embedding(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            use_one_hot=use_one_hot_embeddings,
            embedding_table=TruncatedNormal(config.initializer_range))

        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        self.bert_encoder = BertTransformer(
            batch_size=config.batch_size,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=self.compute_type,
            return_all_encoders=True)

        self.cast = P.Cast()
        self.dtype = numbtpye2mstype(config.dtype)
        self.cast_compute_type = SecurityCast()
        self.slice = P.StridedSlice()

        self.squeeze_1 = P.Squeeze(axis=1)
        self.dense = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(config.initializer_range))\
                                    .to_float(mstype.float32)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids: Tensor, token_type_ids: Tensor, input_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Bidirectional Encoder Representations from Transformers.
        Args:
            input_ids:A vector containing the transformation of characters into corresponding ids.
            token_type_ids:A vector containing segemnt ids.
            input_mask:the mask for input_ids.

        Returns:
            sequence_output:the sequence output .
            pooled_output:the pooled output of first token:cls.
            embedding_table:fixed embedding table.

        """
        # embedding
        embedding_tables = self.bert_embedding_lookup.embedding_table
        word_embeddings = self.bert_embedding_lookup(input_ids)
        embedding_output = self.bert_embedding_postprocessor(token_type_ids,
                                                             word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.bert_encoder(self.cast_compute_type(embedding_output),
                                           attention_mask)

        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)
        return sequence_output, pooled_output, embedding_tables
