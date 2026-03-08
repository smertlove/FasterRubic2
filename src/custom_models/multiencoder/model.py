import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


class FeatsEncoder(nn.Module):
    def __init__(self, n_feats: int, hidden_dim: int, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WordFormEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, embed_tokens: nn.Embedding | None = None, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        if embed_tokens is not None:
            self.embedding.weight = embed_tokens.weight

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lstm.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop(x)
        x, _ = self.lstm(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class MultiEncoder(nn.Module):
    def __init__(self, n_feats: int, vocab_size: int, embed_tokens, hidden_dim: int):
        super().__init__()
        self.feats_encoder = FeatsEncoder(n_feats, hidden_dim)
        self.word_form_encoder = WordFormEncoder(vocab_size, hidden_dim, embed_tokens, )

    def forward(self, tokens, feats):
        tokens_encoding = self.word_form_encoder(tokens)
        feats_encoding = self.feats_encoder(feats.float())
        return tokens_encoding, feats_encoding


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def SwiGLU(self, x):
        return self.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.drop(self.w2(self.SwiGLU(x)))


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wa.weight)
        nn.init.xavier_uniform_(self.Ua.weight)
        nn.init.xavier_uniform_(self.Va.weight)

    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query).unsqueeze(1) + self.Ua(keys)))

        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys).squeeze(1)

        return context


class DecoderLayer(nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout_rate = 0.2):
        super().__init__()

        self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        self.attn = BahdanauAttention(hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x, state, encoded_wordform):

        prev_h, prev_c = state if state is not None else (None, None)

        # RNN
        x_norm = self.lstm_norm(x)
        new_h, new_c = self.cell(x_norm, (prev_h, prev_c))
        new_h = x + self.drop(new_h)

        # ATTN
        h_norm = self.attn_norm(new_h)
        context = self.attn(
            query=h_norm,
            keys=encoded_wordform,
        )
        attn_h = h_norm + context

        # FFN
        attn_h = self.ffn_norm(attn_h)
        ffd = self.ffn(attn_h)
        out = new_h + self.drop(ffd)

        return out, (new_h, new_c)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        ffn_dim: int,
        num_layers: int,
        bos: int,
        eos: int,
        pad: int,
        embed_tokens: nn.Embedding | None = None,
        dropout_rate=0.2,
        max_length=32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.bos = bos
        self.eos = eos
        self.pad = pad

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        if embed_tokens is not None:
            self.embedding.weight = embed_tokens.weight

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, ffn_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(
        self,
        wordform_encoding,
        feats_encoding,
        target_tensor,
    ):

        bs = wordform_encoding.size(0)
        device = wordform_encoding.device

        decoder_input = torch.full((bs,), self.bos, dtype=torch.long, device=device)

        # initial decoder states -- wordform's morphological features
        h = feats_encoding
        c = feats_encoding
        decoder_states = [(h, c) for _ in range(self.num_layers)]

        decoder_outputs = []
        max_len = target_tensor.size(1)


        for t in range(max_len):
            current_input = decoder_input

            current_states = decoder_states

            for layer_idx, layer in enumerate(self.layers):

                if layer_idx == 0:  # Only embed at first layer
                    embedded = self.dropout(self.embedding(current_input))
                else:
                    embedded = current_input

                layer_output, (new_h, new_c) = layer(
                    x=embedded,
                    state=current_states[layer_idx],
                    encoded_wordform=wordform_encoding
                )

                current_states[layer_idx] = (new_h, new_c)
                current_input = layer_output

            decoder_step_output = current_input

            output_logits = self.output_projection(decoder_step_output)

            decoder_outputs.append(output_logits)

            # Teacher forcing
            if t < max_len - 1:
                decoder_input = target_tensor[:, t+1]
            else:
                break

        # Stack outputs
        decoder_outputs = torch.stack(decoder_outputs, dim=1)

        decoder_outputs = F.softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_states

    def forward_step(self, input_token, states, encoded_wordform):
        embedded = self.dropout(self.embedding(input_token))
        current_input = embedded.squeeze(1)
        current_states = states

        for layer_idx, layer in enumerate(self.layers):

            layer_output, (new_h, new_c) = layer(
                x=current_input,
                state=current_states[layer_idx],
                encoded_wordform=encoded_wordform
            )

            current_states[layer_idx] = (new_h, new_c)
            current_input = layer_output

        output_logits = self.output_projection(current_input)
        output = F.log_softmax(output_logits, dim=-1)

        return output, current_states


class EncoderDecoderLemmatizer(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer, n_feats: int, hidden_dim: int, n_decoder_layers: int):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.encoder = MultiEncoder(
            n_feats,
            self.vocab_size,
            self.embedding,
            hidden_dim
        )
        self.decoder = Decoder(
            self.vocab_size,
            hidden_dim,
            hidden_dim*2,
            n_decoder_layers,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            self.embedding
        )

    def forward(self, tokens, feats, tgt):
        tokens_encoding, feats_encoding = self.encoder(tokens, feats)
        tgt[tgt == -100] = 0
        preds = self.decoder(tokens_encoding, feats_encoding, tgt)
        return preds
