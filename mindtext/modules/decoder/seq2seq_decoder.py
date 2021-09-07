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
"""Seq2seq decoder class."""

from typing import Optional

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from ..encoder.attention import SelfAttention, LayerPreprocess
from ..encoder.seq2seq_encoder import FeedForward

__all__ = [
    "TransformerDecoder",
]


class DecoderCell(nn.Cell):
    """
    Decoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the Transformer decoder layers. Default: 1024.
        num_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attn_dropout_prob (float): The dropout probability for self-attention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        has_key_padding_mask (bool): Specifies whether to use key padding mask. Default: False.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size: int,
                 hidden_size: int = 1024,
                 num_heads: int = 12,
                 intermediate_size: int = 4096,
                 attn_dropout_prob: float = 0.02,
                 use_one_hot_embeddings: bool = False,
                 has_key_padding_mask: bool = False,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "relu",
                 compute_type: mindspore.dtype = mstype.float32):
        super(DecoderCell, self).__init__()
        self.self_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_dropout_prob=attn_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            has_key_padding_mask=has_key_padding_mask,
            is_encdec_attn=False,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.cross_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_dropout_prob=attn_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            has_key_padding_mask=has_key_padding_mask,
            is_encdec_attn=True,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states: mindspore.Tensor, attn_mask: mindspore.Tensor,
                  enc_states: mindspore.Tensor, enc_attn_mask: mindspore.Tensor,
                  key_padding_mask: Optional[mindspore.Tensor] = None,
                  enc_key_padding_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Apply DecoderCell.

        Args:
            hidden_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of decoder layer.
            attn_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix(2D
                                or 3D) of hidden_states for self-attention, the values should be [0/1] or [True/False].
            enc_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): Similar to attn_mask, but it is used for enc_states.
            key_padding_mask (Tensor): The shape is (batch_size, from_seq_len). Used to indicate which positions of
                                        hidden_states are padding, and these padding positions will be ignored during
                                        attention calculation. The values should be [0/1] or [True/False].
                                        Default: None.
            enc_key_padding_mask (Tensor): Similar to key_padding_mask, but it is used for enc_states. Default: None.

        return:
            output (Tensor): The output of DecoderCell.
        """

        # Self-attention with layerNorm, residual.
        attn_output = self.self_attention(hidden_states, hidden_states, attn_mask, key_padding_mask)
        # Cross-attention with layerNorm, residual.
        attn_output = self.cross_attention(attn_output, enc_states, enc_attn_mask, enc_key_padding_mask)
        # Feed forward with layerNorm, residual.
        output = self.feedforward(attn_output)
        return output


class TransformerDecoder(nn.Cell):
    """
    Multi-layer transformer decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_len (int): The length of input sequence.
        num_layers (int): Number of layers(decoder cells) in Transformer Decoder.
        num_heads (int): Number of attention heads in encoder cells. Default: 8.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 2048.
        attn_dropout_prob (float): The dropout probability for self-attention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        has_key_padding_mask (bool): Specifies whether to use key padding mask. Default: False.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 seq_len: int,
                 num_layers: int,
                 num_heads: int = 16,
                 intermediate_size: int = 4096,
                 attn_dropout_prob: float = 0.1,
                 use_one_hot_embeddings: bool = False,
                 has_key_padding_mask: bool = False,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "gelu",
                 compute_type: mindspore.dtype = mstype.float32):
        super(TransformerDecoder, self).__init__()

        layers = []
        for _ in range(num_layers):
            layer = DecoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                intermediate_size=intermediate_size,
                                attn_dropout_prob=attn_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                has_key_padding_mask=has_key_padding_mask,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.shape = (-1, hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def construct(self, input_tensor: mindspore.Tensor, attn_mask: mindspore.Tensor,
                  enc_states: mindspore.Tensor, enc_attn_mask: mindspore.Tensor,
                  key_padding_mask: Optional[mindspore.Tensor] = None,
                  enc_key_padding_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Apply TransformerDecoder.

        Args:
            input_tensor (Tensor): The shape is (batch_size, seq_len, hidden_size). The input sequence tensor of
                                Transformer Decoder.
            attn_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix(2D
                                or 3D) of input_tensor for self-attention, the values should be [0/1] or [True/False].
            enc_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): Similar to attn_mask, but it is used for enc_states.
            key_padding_mask (Tensor): The shape is (batch_size, from_seq_len). Used to indicate which positions of
                                        input_tensor are padding, and these padding positions will be ignored during
                                        attention calculation. The values should be [0/1] or [True/False].
                                        Default: None.
            enc_key_padding_mask (Tensor): Similar to key_padding_mask, but it is used for enc_states. Default: None.

        return:
            output (Tensor): The output of TransformerDecoder.
        """
        out_shape = (self.batch_size, self.seq_len, self.hidden_size)
        prev_output = input_tensor

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attn_mask, enc_states, enc_attn_mask,
                                        key_padding_mask, enc_key_padding_mask)
            prev_output = layer_output

        prev_output = prev_output.reshape(self.shape)
        prev_output = self.layer_preprocess(prev_output)
        output = prev_output.reshape(out_shape)
        return output
