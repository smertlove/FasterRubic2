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
import pathlib


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

    # NOTE: у Барта нет отдельного фид форвард слоя, поэтому приходится делать вот так
    def _prune_ffn_layer(self, bart_layer, neuron_indices):

        if len(neuron_indices) == 0:
            return

        current_size = bart_layer.fc1.out_features

        keep_indices = [i for i in range(current_size) if i not in neuron_indices]

        keep_indices = torch.tensor(
            keep_indices,
            dtype=torch.long,
            device=bart_layer.fc1.weight.device
        )

        bart_layer.fc1 = prune_linear_layer(bart_layer.fc1, keep_indices, dim=0)
        bart_layer.fc2 = prune_linear_layer(bart_layer.fc2, keep_indices, dim=1)


    def _prune_ffn(self, neurons_to_prune):

        assert (
            len(
                set(neurons_to_prune)
                - {"encoder_neurons", "decoder_neurons",}
            )
            == 0
        )

        encoder_neurons_to_prune = neurons_to_prune.get("encoder_neurons", dict())
        decoder_neurons_to_prune = neurons_to_prune.get("decoder_neurons", dict())

        for layer_id, neurons in encoder_neurons_to_prune.items():
            self._prune_ffn_layer(
                self.encoder.layers[layer_id],
                neurons,
            )

        for layer_id, neurons in decoder_neurons_to_prune.items():
            self._prune_ffn_layer(
                self.decoder.layers[layer_id],
                neurons,
            )


class PrunableBartForConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = PrunableBartModel(config)

    def prune_heads(self, heads_to_prune):
        self.model._prune_heads(heads_to_prune)

    def prune_ffn(self, neurons_to_prune):
        self.model._prune_ffn(neurons_to_prune)


def load_model(path: pathlib.Path):
    return torch.load(path / "model.pt", weights_only=False)


def save_model(model, path: pathlib.Path):
    if not path.exists():
        path.mkdir()
    return torch.save(model, path / "model.pt")
