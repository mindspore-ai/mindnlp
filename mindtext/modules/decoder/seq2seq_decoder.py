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
import numpy as np

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindtext.modules.encoder.attention import SelfAttention, LayerPreprocess
from mindtext.modules.encoder.seq2seq_encoder import FeedForward
from mindtext.embeddings.transformer_embedding import EmbeddingLookup, EmbeddingPostprocessor
from mindtext.modules.encoder.attention import CastWrapper

__all__ = [
    "TransformerDecoder",
    "TransformerDecoderStep",
    "BeamSearchDecoder"
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
            enc_states (Tensor): The shape is (batch_size, enc_seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): Similar to attn_mask, but it is used for enc_states.
            key_padding_mask (Tensor): The shape is (batch_size, seq_len). Used to indicate which positions of
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
            input_tensor (Tensor): The shape is (batch_size, input_len, hidden_size). The input sequence tensor of
                                Transformer Decoder.
            attn_mask (Tensor): The shape is (input_len, input_len) or (batch_size, input_len, input_len). The mask
                                matrix(2D or 3D) of `input_tensor` for self-attention, the values should be [0/1] or
                                [True/False].
            enc_states (Tensor): The shape is (batch_size, enc_states_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): Similar to attn_mask, but it is used for enc_states.
            key_padding_mask (Tensor): The shape is (batch_size, input_len). Used to indicate which positions of
                                        `input_tensor` are padding, and these padding positions will be ignored during
                                        attention calculation. The values should be [0/1] or [True/False].
                                        Default: None.
            enc_key_padding_mask (Tensor): Similar to key_padding_mask, but it is used for enc_states. Default: None.

        return:
            output (Tensor): The output of TransformerDecoder.
        """
        out_shape = (self.batch_size, -1, self.hidden_size)
        prev_output = input_tensor

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attn_mask, enc_states, enc_attn_mask,
                                        key_padding_mask, enc_key_padding_mask)
            prev_output = layer_output

        prev_output = prev_output.reshape(self.shape)
        prev_output = self.layer_preprocess(prev_output)
        output = prev_output.reshape(out_shape)
        return output


class CreateAttentionMask(nn.Cell):
    """Create Attention Mask Matrix for Self-Attention."""

    def __init__(self):
        super(CreateAttentionMask, self).__init__()
        self.ones = P.Ones()
        self.tril = nn.Tril()
        self.tile = P.Tile()
        self.eye = P.Eye()

    def construct(self, batch_size: int, seq_len: int):
        """
        Apply Create Attention Mask.

        Args:
            batch_size (int): Batch size of input dataset.
            seq_len (int): Length of input sequence.

        return:
            attn_mask (Tensor): The shape is (batch_size, seq_len, seq_len).
                                The attention mask tensor for self-attention.
        """
        attn_mask = ~self.tril(self.ones((seq_len, seq_len), mindspore.bool_))
        attn_mask = attn_mask.reshape(1, seq_len, seq_len)
        attn_mask = self.tile(attn_mask, (batch_size, 1, 1))
        return attn_mask


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        batch_size (int): Batch size.
        hidden_size (int): Hidden size.
        compute_type (class:`mindspore.dtype`): Compute type. Default: mstype.float32.
        dtype (class:`mindspore.dtype`): Compute type to compute log_softmax. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 compute_type: mindspore.dtype = mstype.float32,
                 dtype: mindspore.dtype = mstype.float32):
        super(PredLogProbs, self).__init__()
        self.batch_size = batch_size
        self.width = hidden_size
        self.compute_type = compute_type
        self.dtype = dtype

        self.reshape = P.Reshape()
        self.matmul = P.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = P.Cast()

    def construct(self,
                  input_tensor: mindspore.Tensor,
                  output_weights: mindspore.Tensor,
                  seq_len: int) -> mindspore.Tensor:
        """
        Get log probs.

        Args:
            input_tensor (Tensor): The output of decoder in transformer model.
            output_weights (Tensor): The embedding_table of EmbeddingLookup layer in transformer model.
            seq_len: The length of input sequence.

        returns:
            Tensor: The log_probs.
        """
        shape_flat_sequence_tensor = (self.batch_size * seq_len, self.width)

        input_tensor = self.reshape(input_tensor, shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class TransformerDecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_len (int): The length of input sequence.
        max_decode_length (int): Max decode length.
        num_layers (int): Number of hidden layers in encoder cells.
        num_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attn_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (class:`mindspore.dtype`): Compute type. Default: mstype.float32.
        embedding_lookup (class:`EmbeddingLookup`): Embedding lookup module.
        embedding_processor (class:`EmbeddingPostprocessor`) Embedding postprocessor module.
        projection (class:`PredLogProbs`): PredLogProbs module
    """
    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 seq_len: int,
                 max_decode_length: int,
                 num_layers: int,
                 num_heads: int = 16,
                 intermediate_size: int = 4096,
                 attn_dropout_prob: float = 0.3,
                 use_one_hot_embeddings: bool = False,
                 hidden_dropout_prob: float = 0.3,
                 hidden_act: str = "relu",
                 compute_type: mindspore.dtype = mstype.float32,
                 embedding_lookup: Optional[EmbeddingLookup] = None,
                 embedding_processor: Optional[EmbeddingPostprocessor] = None,
                 projection: Optional[PredLogProbs] = None):
        super(TransformerDecoderStep, self).__init__(auto_prefix=False)
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.tfm_embedding_lookup = embedding_lookup
        self.tfm_embedding_processor = embedding_processor
        self.projection = projection

        self.tfm_decoder = TransformerDecoder(
            batch_size=batch_size,
            hidden_size=hidden_size,
            seq_len=seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type)

        self.shape = P.Shape()

        self._create_attention_mask = CreateAttentionMask()
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()

        ones = np.ones(shape=(max_decode_length, max_decode_length))
        self.future_mask = mindspore.Tensor(np.tril(ones), dtype=mstype.float32)

        self.cast_compute_type = CastWrapper(dst_type=compute_type)

    def construct(self,
                  input_ids: mindspore.Tensor,
                  enc_states: mindspore.Tensor,
                  enc_attn_mask: mindspore.Tensor) -> mindspore.Tensor:
        """
        Multi-layer transformer decoder step.

        Args:
            input_ids: The shape is [batch_size, beam_width]. The ids of input sequence tokens.
            enc_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix
                                    (2D or 3D) of `enc_states` for self-attention, the values should be [0/1] or
                                    [True/False].

        Returns:
            Tensor: The log_probs.
        """
        # Process embedding.
        input_embedding, embedding_tables = self.tfm_embedding_lookup(input_ids)
        input_embedding = self.tfm_embedding_processor(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]

        input_mask = self._create_attention_mask(self.batch_size, input_len)
        input_mask = self.cast_compute_type(input_mask)

        enc_attn_mask = enc_attn_mask[::, 0:input_len:1, ::]

        # Call TransformerDecoder.
        decoder_output = self.tfm_decoder(input_embedding, input_mask, enc_states, enc_attn_mask)

        # Take the last step.
        decoder_output = decoder_output[::, input_len-1:input_len:1, ::]

        # Projection and log_prob.
        log_probs = self.projection(decoder_output, embedding_tables, 1)

        return log_probs


class LengthPenalty(nn.Cell):
    """
    Normalize scores of translations according to their length.

    Args:
        weight (float): Weight of length penalty. Default: 1.0.
        compute_type (class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """
    def __init__(self,
                 weight=1.0,
                 compute_type: mindspore.dtype = mstype.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.compute_type = compute_type
        self.add = P.Add()
        self.pow = P.Pow()
        self.div = P.RealDiv()
        self.cast = P.Cast()
        self.five = mindspore.Tensor(5.0, compute_type)
        self.six = mindspore.Tensor(6.0, compute_type)

    def construct(self, length_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply LengthPenalty.

        Args:
            length_tensor (Tensor): The length of hidden states.

        Returns:
            output (Tensor): The results of length penalty.
        """
        length_tensor = self.cast(length_tensor, self.compute_type)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class TileBeam(nn.Cell):
    """
    TileBeam.

    Args:
        beam_width (int): Beam width setting. Default: 4.
    """
    def __init__(self,
                 beam_width: int):
        super(TileBeam, self).__init__()
        self.beam_width = beam_width
        self.expand = P.ExpandDims()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_tensor):
        """
        Apply TileBeam.

        Args:
            input_tensor (Tensor): The shape is (batch, dim1, dim2).

        Returns:
            output (Tensor): The shape is (batch*beam, dim1, dim2).
        """
        shape = self.shape(input_tensor)
        input_tensor = self.expand(input_tensor, 1)
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape)-1):
            tile_shape = tile_shape + (1,)
        output = self.tile(input_tensor, tile_shape)
        out_shape = (shape[0]*self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)
        return output


class Mod(nn.Cell):
    """
    Mod function.

    """
    def __init__(self):
        super(Mod, self).__init__()
        self.floor_div = P.FloorDiv()
        self.sub = P.Sub()
        self.multiply = P.Mul()

    def construct(self, input_x: mindspore.Tensor, input_y: mindspore.Tensor):
        """
        Apply Mod function.

        Args:
            input_x (Tensor): The input_x tensor of Mod function.
            input_y (Tensor): The input_y tensor of Mod function.

        Returns:
            Tensor, the results of Mod functions.
        """
        x = self.floor_div(input_x, input_y)
        x = self.multiply(x, input_y)
        x = self.sub(input_x, x)
        return x


class BeamSearchDecoder(nn.Cell):
    """
    Beam search decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_len (int): Length of input sequence.
        vocab_size (int): Size of vocabulary.
        decoder (:class:`TransformerDecoderStep`): Decoder module.
        beam_width (int): beam width setting. Default: 4.
        length_penalty_weight (float): Weight of length penalty. Default: 1.0.
        max_decode_length (int): max decode length. Default: 128.
        sos_id (int): Id of sequence start token. Default: 1.
        eos_id (int): Id of sequence end token. Default: 2.
    """
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 vocab_size: int,
                 decoder: TransformerDecoderStep,
                 beam_width: int = 4,
                 length_penalty_weight: float = 1.0,
                 max_decode_length: int = 128,
                 sos_id: int = 1,
                 eos_id: int = 2):
        super(BeamSearchDecoder, self).__init__(auto_prefix=False)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.decoder = decoder

        self.add = P.Add()
        self.expand = P.ExpandDims()
        self.reshape = P.Reshape()
        self.shape_flat = (-1,)
        self.shape = P.Shape()

        self.zero_tensor = mindspore.Tensor(np.zeros([batch_size, beam_width]), mstype.float32)
        self.ninf_tensor = mindspore.Tensor(np.full([batch_size, beam_width], -1. * 1e9), mstype.float32)

        self.select = P.Select()
        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.topk = P.TopK(sorted=True)
        self.floor_div = P.FloorDiv()
        self.vocab_size_tensor = mindspore.Tensor(self.vocab_size, mstype.int32)
        self.real_div = P.RealDiv()
        self.mod = Mod()
        self.equal = P.Equal()
        self.eos_ids = mindspore.Tensor(np.full([batch_size, beam_width], eos_id), mstype.int32)

        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = mindspore.Tensor(beam_ids, mstype.int32)
        batch_ids = np.arange(batch_size * beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = mindspore.Tensor(batch_ids, mstype.int32)
        self.concat = P.Concat(axis=-1)
        self.gather_nd = P.GatherNd()

        self.greater_equal = P.GreaterEqual()
        self.sub = P.Sub()
        self.cast = P.Cast()
        self.zeroslike = P.ZerosLike()

        # Init inputs and states
        self.start_ids = mindspore.Tensor(np.full([batch_size * beam_width, 1], sos_id), mstype.int32)
        self.init_seq = mindspore.Tensor(np.full([batch_size, beam_width, 1], sos_id), mstype.int32)
        init_scores = np.tile(np.array([[0.] + [-1. * 1e9] * (beam_width-1)]), [batch_size, 1])
        self.init_scores = mindspore.Tensor(init_scores, mstype.float32)
        self.init_finished = mindspore.Tensor(np.zeros([batch_size, beam_width], dtype=np.bool_))
        self.init_length = mindspore.Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
        self.length_penalty = LengthPenalty(weight=length_penalty_weight)
        self.one = mindspore.Tensor(1, mstype.int32)

    def one_step(self,
                 cur_input_ids: mindspore.Tensor,
                 enc_states: mindspore.Tensor,
                 enc_attn_mask: mindspore.Tensor,
                 state_log_probs: mindspore.Tensor,
                 state_seq: mindspore.Tensor,
                 state_finished: mindspore.Tensor,
                 state_length: mindspore.Tensor):
        """
        One step for decode.

        Args:
            cur_input_ids (Tensor): The ids of input sequence tokens.
            enc_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix
                                    (2D or 3D) of `enc_states` for self-attention, the values should be [0/1] or
                                    [True/False].
            state_log_probs (Tensor): The log_probs of sequence state.
            state_seq (Tensor): The sequence state.
            state_finished (Tensor): The finished flags of sequence state.
            state_length (Tensor): The length of sequence state.
        """
        log_probs = self.decoder(cur_input_ids, enc_states, enc_attn_mask)
        log_probs = self.reshape(log_probs, (self.batch_size, self.beam_width, self.vocab_size))

        # Select topk indices
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1))

        # Mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))

        # Reshape scores to [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)
        # Select topk
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = self.cast(self.greater_equal(temp, 0), mstype.int32)
            beam_indices = beam_indices + res
        word_indices = topk_indices - beam_indices * self.vocab_size_tensor

        # Mask finished indices
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        # Put finished sequences to the end.
        # Sort according to scores with -inf for finished beams.
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)
        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # Update.
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        beam_indices = self.gather_nd(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd(topk_scores, tmp_gather_indices)

        # Generate new beam_search states.
        # Gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # Length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd(state_length, gather_indices)

        # Concat seq
        seq = self.gather_nd(state_seq, gather_indices)
        state_seq = self.concat((seq, self.expand(word_indices, -1)))

        # New finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores

        # Generate new inputs and decoder states
        cur_input_ids = self.reshape(state_seq, (self.batch_size*self.beam_width, -1))
        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length

    def construct(self, enc_states: mindspore.Tensor, enc_attn_mask: mindspore.Tensor) -> mindspore.Tensor:
        """
        Get beam search result.

        Args:
            enc_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            enc_attn_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix
                                    (2D or 3D) of `enc_states` for self-attention, the values should be [0/1] or
                                    [True/False].

        Returns:
            predicted_ids (Tensor): The predict results.
        """
        cur_input_ids = self.start_ids
        # Beam search states
        state_log_probs = self.init_scores
        state_seq = self.init_seq
        state_finished = self.init_finished
        state_length = self.init_length

        for _ in range(self.max_decode_length):
            # Run one step decoder to get outputs of the current step.
            # Shape [batch*beam, 1, vocab].
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length = self.one_step(
                cur_input_ids, enc_states, enc_attn_mask, state_log_probs, state_seq, state_finished, state_length)

        # Add length penalty scores.
        penalty_len = self.length_penalty(state_length)
        # Get penalty length.
        log_probs = self.real_div(state_log_probs, penalty_len)

        # Sort according to scores.
        _, top_beam_indices = self.topk(log_probs, self.beam_width)
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(top_beam_indices, -1)))
        # Sort sequence.
        predicted_ids = self.gather_nd(state_seq, gather_indices)
        # Take the first one.
        predicted_ids = predicted_ids[::, 0:1:1, ::]
        return predicted_ids
