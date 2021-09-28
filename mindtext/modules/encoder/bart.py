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
    Encoder classes for BART.
"""
from typing import Optional, Tuple
import math
import random
import numpy


import mindspore
import mindspore.nn as nn

import mindspore.ops as ops
from mindspore.ops import SoftmaxCrossEntropyWithLogits
import mindspore.numpy as np
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = ops.composite.MultitypeFuncGraph("clip_grad")


class BartConfig:
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BartModel`. It is used to
    instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BART `facebook/bart-large
    <https://huggingface.co/facebook/bart-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.
    """
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 max_sent_len=5,
                 vocab_size=50265,
                 max_position_embeddings=1024,
                 encoder_layers=12,
                 encoder_ffn_dim=4096,
                 encoder_attention_heads=16,
                 decoder_layers=12,
                 decoder_ffn_dim=4096,
                 decoder_attention_heads=16,
                 encoder_layerdrop=1.0,
                 decoder_layerdrop=1.0,
                 activation_function="gelu",
                 d_model=1024,
                 dropout=0.9,
                 attention_dropout=1.0,
                 activation_dropout=1.0,
                 init_std=0.02,
                 classifier_dropout=1.0,
                 scale_embedding=False,
                 gradient_checkpointing=False,
                 use_cache=True,
                 num_labels=3,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 is_encoder_decoder=True,
                 decoder_start_token_id=2,
                 forced_eos_token_id=2
                 ):
        super().__init__()
        self.max_sent_len = max_sent_len
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model


ACT2FN = {
    "relu": ops.ReLU(),
    "gelu": ops.GeLU(),
    "tanh": ops.Tanh(),
    "sigmoid": ops.Sigmoid(),
}


def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = ops.Zeros()(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    select = ops.Select()
    cond = shifted_input_ids == -100
    cond = not cond
    replace_mat = ops.Zeros()(input_ids.shape) + pad_token_id
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = select(cond, shifted_input_ids, replace_mat)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: mindspore.Tensor.shape, dtype: mindspore.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = mindspore.common.initializer.Constant(np.zeros([tgt_len, tgt_len]) - 1e9).value
    mask_size = mask.shape
    mask_cond = nn.Range(0, mask_size[-1], 1)()
    mask_cond_contact = (mask_cond + 1).reshape(mask_size[-1], 1)
    mask_cond = mask_cond >= mask_cond_contact
    replace = mindspore.Tensor(np.zeros(mask_cond.shape))
    select = ops.Select()
    mask = select(mask_cond, mask, replace)

    if past_key_values_length > 0:
        mask = ops.Concat(-1)((ops.Zeros()((tgt_len, past_key_values_length), dtype), mask))
    broadcast_to = ops.BroadcastTo((bsz, 1, tgt_len, tgt_len + past_key_values_length))

    return broadcast_to(mask[None, None, :, :])


def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    broadcast_to = ops.BroadcastTo((bsz, 1, tgt_len, src_len))
    expanded_mask = broadcast_to(mask[:, None, None, :]).to(dtype)
    ops.DType()
    select = ops.Select()
    cond = expanded_mask.bool()
    inverted_mask = 1 - expanded_mask
    expanded_mask = inverted_mask * -1e9
    inverted_mask = select(cond, inverted_mask, expanded_mask)
    return inverted_mask


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, input_ids_shape: mindspore.Tensor.shape, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        _, seq_len = input_ids_shape[:2]

        positions = nn.Range(
            past_key_values_length, past_key_values_length + seq_len
        )()
        return super().construct(positions + self.offset)


class BartAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super(BartAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(keep_prob=dropout)
        self.head_dim = embed_dim // num_heads
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        tensor = self.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        return self.transpose(tensor, (0, 2, 1, 3))

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            key_value_states: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.Concat(2)(past_key_value[0], key_states)
            value_states = ops.Concat(2)(past_key_value[0], value_states)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = self.reshape(query_states, proj_shape)
        key_states = self.reshape(key_states, proj_shape)
        value_states = self.reshape(value_states, proj_shape)

        src_len = key_states.shape[1]
        key_states = self.transpose(key_states, (0, 2, 1))
        attn_weights = ops.BatchMatMul()(query_states, key_states)

        assert attn_weights.shape == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is " \
           f"{attn_weights.shape}"

        if attention_mask is not None:
            assert attention_mask.shape == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
            attn_weights = self.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = self.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = self.softmax(attn_weights)

        attn_probs = self.dropout(attn_weights)

        attn_output = ops.BatchMatMul()(attn_probs, value_states)

        assert attn_output.shape == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is " \
           f"{attn_output.shape}"

        attn_output = self.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = self.transpose(attn_output, (0, 2, 1, 3))
        attn_output = self.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value


class BartEncoderLayer(nn.Cell):
    """ EncoderLayer """

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(self,
                  hidden_states: mindspore.Tensor,
                  attention_mask: mindspore.Tensor):
        """
        Args:
            hidden_states (:obj:`mindspore.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states
        hidden_states, _ = self.self_attn.construct(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = nn.Dropout(keep_prob=self.dropout)(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.Dropout(keep_prob=self.activation_dropout)(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.Dropout(keep_prob=self.dropout)(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == mindspore.float16 and (
                np.isinf(hidden_states.asnumpy()).any() or np.isnan(hidden_states.asnumpy()).any()
        ):
            clamp_value = numpy.finfo(hidden_states.asnumpy().dtype).max - 1000
            clamp_value = mindspore.Tensor(clamp_value)
            hidden_states = ops.clip_by_value(hidden_states, clip_value_min=-clamp_value, clip_value_max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class BartDecoderLayer(nn.Cell):
    """BartDecoderLayer"""

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout_rate = config.dropout
        self.dropout = nn.Dropout(keep_prob=self.dropout_rate)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(self,
                  hidden_states: mindspore.Tensor,
                  attention_mask: Optional[mindspore.Tensor] = None,
                  encoder_hidden_states: Optional[mindspore.Tensor] = None,
                  encoder_attention_mask: Optional[mindspore.Tensor] = None,
                  past_key_value: Optional[Tuple[mindspore.Tensor]] = None
                  ):
        """
        Args:
            hidden_states (:obj:`mindspore.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative
                values.
            encoder_hidden_states (:obj:`mindspore.Tensor`): cross attention input to the layer of shape
                `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`mindspore.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative
                values.
            past_key_value (:obj:`Tuple(mindspore.Tensor)`): cached past key and value projection states
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, present_key_value = self.self_attn.construct(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        #cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_present_key_value = self.encoder_attn.construct(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs


class BartClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""

    def __init__(self,
                 input_dim: int,
                 inner_dim: int,
                 num_classes: int,
                 pooler_dropout: float,
                 ):
        super().__init__()
        self.dense = nn.Dense(input_dim, inner_dim)
        self.dropout = nn.Dropout(keep_prob=pooler_dropout)
        self.out_proj = nn.Dense(inner_dim, num_classes)

    def construct(self, hidden_states: mindspore.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPretrainedModel(nn.Cell):
    """BartPretrainedModel"""
    config_class = BartConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Dense):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = mindspore.Tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": ops.not_equal(input_ids, pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class BartEncoder(nn.Cell):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (mindspore.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, padding_idx=self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.CellList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([embed_dim])

    def construct(self,
                  input_ids=None,
                  attention_mask=None):
        r"""
        Args:
            input_ids (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

            attention_mask (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        """

        shape = ops.Shape()
        reshape = ops.Reshape()
        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            input_shape = shape(input_ids)
            input_ids = reshape(input_ids, (-1, input_shape[-1]))
        else:
            raise ValueError("You have to specify input_ids")

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions.construct(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.Dropout(keep_prob=self.dropout)(hidden_states)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        for encoder_layer in self.layers:
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer.construct(
                    hidden_states,
                    attention_mask,
                )
                hidden_states = layer_outputs[0]

        return (hidden_states,)


class BartDecoder(nn.Cell):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (mindspore.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.CellList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([config.d_model])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """ attention_mask """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  encoder_hidden_states=None,
                  encoder_attention_mask=None,
                  past_key_values=None
                  ):
        r"""
        Args:
            input_ids (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, encoder_sequence_length,
                                    hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, encoder_sequence_length)`,
                `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[mindspore.Tensor]]` of length :obj:`config.n_layers` with each tuple
                having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1,
                embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
        """
        #use_cache = self.config.use_cache
        shape = ops.Shape()
        reshape = ops.Reshape()
        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            input_shape = shape(input_ids)
            input_ids = reshape(input_ids, (-1, input_shape[-1]))
        else:
            raise ValueError("You have to specify either decoder_input_ids")

        # past_key_values_length
        past_key_values_length = shape(past_key_values[0][0])[2] if past_key_values is not None else 0

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.Dropout(keep_prob=self.dropout)(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer.construct(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value
            )

            hidden_states = layer_outputs[0]

        return (hidden_states,)


class BartModel(nn.Cell):
    """BartModel"""

    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx=padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  decoder_input_ids=None,
                  decoder_attention_mask=None,
                  past_key_values=None
                  ):
        """ method """

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        encoder_outputs = self.encoder.construct(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder.construct(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values
        )

        return decoder_outputs + encoder_outputs


class BartForConditionalGeneration(nn.Cell):
    """BartForConditionalGeneration"""
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias",
                             mindspore.ops.Zeros()((1, self.model.shared.num_embeddings), mindspore.float32))
        self.lm_head = nn.Dense(config.d_model, self.model.shared.vocab_size)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.Zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.Concat(1)((self.final_logits_bias, extra_bias))
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  labels=None,
                  decoder_attention_mask=None,
                  past_key_values=None
                  ):
        """
        labels (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        """

        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = SoftmaxCrossEntropyWithLogits()
            reshape = ops.Reshape()
            masked_lm_loss = loss_fct(reshape(lm_logits, (-1, self.config.vocab_size)), reshape(labels, (-1)))

        output = (lm_logits,) + outputs[1:]
        return masked_lm_loss if masked_lm_loss is not None else output

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      past=None,
                                      attention_mask=None,
                                      encoder_outputs=None,
                                      ):
        """prepare inputs"""
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state[beam_idx] for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class BartForConditionalGenerationOneStep(nn.Cell):
    """
    BartForConditionalGenerationOneStep
    """
    def __init__(self, net, optimizer, sens=1.0):
        super(BartForConditionalGenerationOneStep).__init__()
        self.network = net
        self.network.init_parameters_data()
        self.optimizer = optimizer
        self.weights = mindspore.ParameterTuple(self.network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.parallel_mode = mindspore.context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in context.ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = ops.composite.HyperMap()
        self.cast = ops.operations.Cast()

    def construct(self,
                  input_ids,
                  attention_mask=None,
                  labels=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            attention_mask,
                            labels)
        grads = self.grad(self.network, weights)(input_ids,
                                                 attention_mask,
                                                 labels,
                                                 self.cast(ops.functional.tuple_to_array((self.sens,)),
                                                           mindspore.dtype.float32))
        grads = self.hyper_map(ops.functional.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return ops.functional.depend(loss, succ)
