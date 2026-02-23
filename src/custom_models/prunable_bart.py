from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartModel,
    BartConfig,
    BartForConditionalGeneration,
)
from transformers.pytorch_utils import (
    prune_linear_layer,
    find_pruneable_heads_and_indices,
)
import torch
from torch import nn

from typing import Optional, Tuple


class PrunableBartAttention(BartAttention):
    """BartAttention но с возможностью прунить головы"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_head_size = self.embed_dim // self.num_heads
        # self.all_head_size = self.embed_dim
        self.pruned_heads = set()

    def prune_heads(self, heads):

        if len(heads) == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.attention_head_size, self.pruned_heads
        )

        self.q_proj = prune_linear_layer(self.q_proj, index)
        self.k_proj = prune_linear_layer(self.k_proj, index)
        self.v_proj = prune_linear_layer(self.v_proj, index)

        self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)

        self.num_heads = self.num_heads - len(heads)
        # self.all_head_size = self.attention_head_size * self.num_heads
        self.embed_dim = self.attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

        self.head_size = self.attention_head_size

    # # HACK: это некрасиво, но варианта сделать иначе нет.
    # #       это полная копия форварда из оригинального BartAttention но с измененным решейпом в конце
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     key_value_states: Optional[torch.Tensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     layer_head_mask: Optional[torch.Tensor] = None,
    #     output_attentions: bool = False,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     """Input shape: Batch x Time x Channel"""

    #     # if key_value_states are provided this layer is used as a cross-attention layer
    #     # for the decoder
    #     is_cross_attention = key_value_states is not None

    #     bsz, tgt_len, _ = hidden_states.size()

    #     # get query proj
    #     query_states = self.q_proj(hidden_states) * self.scaling
    #     # get key, value proj
    #     # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    #     # is checking that the `sequence_length` of the `past_key_value` is the same as
    #     # the provided `key_value_states` to support prefix tuning
    #     if (
    #         is_cross_attention
    #         and past_key_value is not None
    #         and past_key_value[0].shape[2] == key_value_states.shape[1]
    #     ):
    #         # reuse k,v, cross_attentions
    #         key_states = past_key_value[0]
    #         value_states = past_key_value[1]
    #     elif is_cross_attention:
    #         # cross_attentions
    #         key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    #         value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    #     elif past_key_value is not None:
    #         # reuse k, v, self_attention
    #         key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    #         value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     else:
    #         # self_attention
    #         key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    #         value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    #     if self.is_decoder:
    #         # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
    #         # Further calls to cross_attention layer can then reuse all cross-attention
    #         # key/value_states (first "if" case)
    #         # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
    #         # all previous decoder key/value_states. Further calls to uni-directional self-attention
    #         # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
    #         # if encoder bi-directional self-attention `past_key_value` is always `None`
    #         past_key_value = (key_states, value_states)

    #     proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    #     query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    #     key_states = key_states.reshape(*proj_shape)
    #     value_states = value_states.reshape(*proj_shape)

    #     src_len = key_states.size(1)
    #     attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    #     if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #             f" {attn_weights.size()}"
    #         )

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    #     if layer_head_mask is not None:
    #         if layer_head_mask.size() != (self.num_heads,):
    #             raise ValueError(
    #                 f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
    #                 f" {layer_head_mask.size()}"
    #             )
    #         attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     if output_attentions:
    #         # this operation is a bit awkward, but it's required to
    #         # make sure that attn_weights keeps its gradient.
    #         # In order to do so, attn_weights have to be reshaped
    #         # twice and have to be reused in the following
    #         attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #         attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    #     else:
    #         attn_weights_reshaped = None

    #     attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    #     attn_output = torch.bmm(attn_probs, value_states)

    #     if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #         raise ValueError(
    #             f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
    #             f" {attn_output.size()}"
    #         )

    #     attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    #     attn_output = attn_output.transpose(1, 2)

    #     # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    #     # partitioned across GPUs when using tensor-parallelism.
    #     attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)  # <- !! Изменено тут !!

    #     attn_output = self.out_proj(attn_output)

    #     return attn_output, attn_weights_reshaped, past_key_value


class PrunableBartModel(BartModel):
    """BartModel но с возможностью прунить головы"""

    def __init__(self, config: BartConfig):
        super().__init__(config)

        for layer in self.encoder.layers:
            layer.self_attn = self._mk_attn_prunable(
                layer.self_attn,
            )

        for layer in self.decoder.layers:
            layer.self_attn = self._mk_attn_prunable(
                layer.self_attn,
            )
            layer.encoder_attn = self._mk_attn_prunable(
                layer.encoder_attn,
            )

    def _mk_attn_prunable(self, attn: BartAttention):

        new_attn = PrunableBartAttention(
            embed_dim=attn.embed_dim,
            num_heads=attn.num_heads,
            dropout=attn.dropout,
            is_decoder=attn.is_decoder,
            bias=attn.q_proj.bias is not None,
        )

        new_attn.load_state_dict(attn.state_dict())

        return new_attn

    def _prune_heads(self, heads_to_prune):

        assert (
            len(
                set(heads_to_prune)
                - {"encoder_self_heads", "decoder_self_heads", "decoder_cross_heads"}
            )
            == 0
        )

        encoder_self_heads_to_prune = heads_to_prune.get("encoder_self_heads", dict())
        decoder_self_heads_to_prune = heads_to_prune.get("decoder_self_heads", dict())
        decoder_cross_heads_to_prune = heads_to_prune.get("decoder_cross_heads", dict())

        for layer_id, heads in encoder_self_heads_to_prune.items():
            self.encoder.layers[layer_id].self_attn.prune_heads(heads)

        for layer_id, heads in decoder_self_heads_to_prune.items():
            self.decoder.layers[layer_id].self_attn.prune_heads(heads)

        for layer_id, heads in decoder_cross_heads_to_prune.items():
            if self.decoder.layers[layer_id].encoder_attn is not None:
                self.decoder.layers[layer_id].encoder_attn.prune_heads(heads)


class PrunableBartForConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = PrunableBartModel(config)

    def _prune_heads(self, heads_to_prune):
        self.model._prune_heads(heads_to_prune)
