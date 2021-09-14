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
"""Transformer Class."""

from typing import Optional
import copy
import numpy as np

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore.ops.primitive import constexpr
from mindtext.embeddings.transformer_embedding import EmbeddingLookup, EmbeddingPostprocessor
from mindtext.modules.encoder.attention import CastWrapper
from mindtext.modules.encoder.seq2seq_encoder import TransformerEncoder
from mindtext.modules.decoder.seq2seq_decoder import TransformerDecoder, TransformerDecoderStep
from mindtext.modules.decoder.seq2seq_decoder import TileBeam, PredLogProbs, BeamSearchDecoder, CreateAttentionMask


class TransformerConfig:
    """
    Configuration for `Transformer`.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_len (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 36560.
        hidden_size (int): Size of the layers. Default: 1024.
        num_layers (int): Number of hidden layers in the Transformer encoder/decoder cell. Default: 6.
        num_heads (int): Number of attention heads in the Transformer encoder/decoder cell. Default: 16.
        intermediate_size (int): Size of intermediate layer in the Transformer encoder/decoder cell. Default: 4096.
        hidden_act (str): Activation function used in the Transformer encoder/decoder cell. Default: "relu".
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.3.
        attn_dropout_prob (float): The dropout probability for multi-attention. Default: 0.3.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        max_position_embeddings (int): Maximum length of sequences used in this model. Default: 128.
        label_smoothing (float): label smoothing setting. Default: 0.1
        beam_width (int): beam width setting. Default: 4
        max_decode_length (int): max decode length in evaluation. Default: 80
        length_penalty_weight (float): normalize scores of translations according to their length. Default: 1.0
        dtype (class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size: int,
                 seq_len: int = 128,
                 vocab_size: int = 36560,
                 hidden_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 16,
                 intermediate_size: int = 4096,
                 hidden_act: str = "relu",
                 hidden_dropout_prob: float = 0.3,
                 attn_dropout_prob: float = 0.3,
                 use_one_hot_embeddings: bool = False,
                 max_position_embeddings: int = 128,
                 label_smoothing: float = 0.1,
                 beam_width: int = 4,
                 max_decode_length: int = 80,
                 length_penalty_weight: float = 1.0,
                 dtype: mindspore.dtype = mstype.float32,
                 compute_type: mindspore.dtype = mstype.float32):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_type = compute_type


@constexpr
def convert_np_to_tensor_encoder(seq_len: int) -> mindspore.Tensor:
    """
    Convert numpy.array to mindspore.tensor for encoder.

    Args:
        seq_len (int): The length of input sequence.
    """
    ones = np.ones(shape=(seq_len, seq_len))
    return mindspore.Tensor(np.tril(ones), dtype=mstype.float32)


class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.

    Args:
        config (Class): Configuration for Transformer.
        is_training (bool): True for training mode. False for eval mode.
    """
    def __init__(self,
                 config: TransformerConfig,
                 is_training: bool):
        super(TransformerModel, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attn_dropout_prob = 0.0

        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.seq_len
        self.num_layers = config.num_layers
        self.embedding_size = config.hidden_size

        self.last_idx = self.num_layers - 1
        self.beam_width = config.beam_width
        self.max_decode_length = config.max_decode_length

        self.tfm_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=config.use_one_hot_embeddings)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        self.tfm_embedding_postprocessor_for_decoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        self.tfm_encoder = TransformerEncoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            num_heads=config.num_heads,
            num_layers=self.num_layers,
            intermediate_size=config.intermediate_size,
            attn_dropout_prob=config.attn_dropout_prob,
            use_one_hot_embeddings=config.use_one_hot_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)

        if is_training:
            self.projection = PredLogProbs(
                batch_size=self.batch_size,
                hidden_size=self.hidden_size,
                compute_type=config.compute_type,
                dtype=config.dtype)
            self.tfm_decoder = TransformerDecoder(
                batch_size=self.batch_size,
                hidden_size=self.hidden_size,
                seq_len=self.seq_len,
                num_heads=config.num_heads,
                num_layers=self.num_layers,
                intermediate_size=config.intermediate_size,
                attn_dropout_prob=config.attn_dropout_prob,
                use_one_hot_embeddings=config.use_one_hot_embeddings,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_act=config.hidden_act,
                compute_type=config.compute_type)
        else:
            self.projection = PredLogProbs(
                batch_size=self.batch_size * config.beam_width,
                hidden_size=self.hidden_size,
                compute_type=config.compute_type,
                dtype=config.dtype)
            self.tfm_decoder = TransformerDecoderStep(
                batch_size=self.batch_size * config.beam_width,
                hidden_size=self.hidden_size,
                seq_len=self.seq_len,
                max_decode_length=config.max_decode_length,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                attn_dropout_prob=config.attn_dropout_prob,
                use_one_hot_embeddings=False,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_act=config.hidden_act,
                compute_type=config.compute_type,
                embedding_lookup=self.tfm_embedding_lookup,
                embedding_processor=self.tfm_embedding_postprocessor_for_decoder,
                projection=self.projection)
            self.tfm_decoder = BeamSearchDecoder(
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                vocab_size=config.vocab_size,
                decoder=self.tfm_decoder,
                beam_width=config.beam_width,
                length_penalty_weight=config.length_penalty_weight,
                max_decode_length=config.max_decode_length)

            self.tfm_decoder.add_flags(loop_can_unroll=True)
            self.tile_beam = TileBeam(beam_width=self.beam_width)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()
        self.shape = P.Shape()

        self._create_attention_mask = CreateAttentionMask()

    def construct(self,
                  source_ids: mindspore.Tensor,
                  target_ids: Optional[mindspore.Tensor] = None):
        """
        Transformer with encoder and decoder.

        Args:
            source_ids (Tensor): The ids of source sequence tokens.
            target_ids (Tensor): The ids of target sequence tokens.

        Returns:
            ret (Tensor): The output results of TransformerModel.

        """
        batch_size, seq_len = self.shape(source_ids)

        # Process source sentence.
        src_word_embeddings, embedding_tables = self.tfm_embedding_lookup(source_ids)
        src_embedding_output = self.tfm_embedding_postprocessor_for_encoder(src_word_embeddings)
        # Attention mask [batch_size, seq_len, seq_len].
        enc_attn_mask = self._create_attention_mask(batch_size, seq_len)
        # Transformer encoder.
        encoder_output = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                          self.cast_compute_type(enc_attn_mask))

        if self.is_training:
            # Process target sentence.
            tgt_word_embeddings, _ = self.tfm_embedding_lookup(target_ids)
            tgt_embedding_output = self.tfm_embedding_postprocessor_for_decoder(tgt_word_embeddings)
            # Attention mask [batch_size, seq_len, seq_len].
            tgt_attn_mask = self._create_attention_mask(tgt_embedding_output.shape[0], tgt_embedding_output.shape[1])
            # Transformer decoder.
            decoder_output = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                              self.cast_compute_type(tgt_attn_mask),
                                              encoder_output, enc_attn_mask)
            # Calculate logits and log_probs.
            log_probs = self.projection(decoder_output, embedding_tables, seq_len)
            ret = log_probs
        else:
            beam_encoder_output = self.tile_beam(encoder_output)

            beam_enc_attention_mask = self.tile_beam(enc_attn_mask)
            beam_enc_attention_mask = self.cast_compute_type(beam_enc_attention_mask)
            predicted_ids = self.tfm_decoder(beam_encoder_output, beam_enc_attention_mask)
            ret = predicted_ids
        return ret
