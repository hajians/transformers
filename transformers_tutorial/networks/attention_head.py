from collections import OrderedDict
from math import sqrt

import torch
from torch import nn


class AttentionHead(nn.Module):

    def __init__(self, emb_dim, hidden_dim, is_decoder):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.q = nn.Linear(emb_dim, hidden_dim)
        self.k = nn.Linear(emb_dim, hidden_dim)
        self.v = nn.Linear(emb_dim, hidden_dim)

        self.is_decoder = is_decoder

    @staticmethod
    def compute_weights(q, k, attention_mask=None, is_decoder=False):
        dim_ = q.size(-1)
        attention_score = torch.bmm(q, k.transpose(1, 2)) / sqrt(dim_)
        if attention_mask is not None:
            padding_mask = 1 - attention_mask
            attention_mask_mat = attention_mask.unsqueeze(
                -1
            ) * attention_mask.unsqueeze(1)
            padding_mask_mat = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)

            if is_decoder:
                attention_mask_mat = torch.tril(attention_mask_mat)

            coupling_mask_mat = 1 - attention_mask_mat - padding_mask_mat

            attention_score = attention_score + coupling_mask_mat * (-1.0e5)
        return torch.softmax(attention_score, dim=-1)

    @staticmethod
    def scaled_dot_product(q, k, v, attention_mask=None, is_decoder=False):
        weights = AttentionHead.compute_weights(q, k, attention_mask, is_decoder)
        return torch.bmm(weights, v)

    def forward(self, hidden_state, attention_masks=None):
        return self.scaled_dot_product(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state),
            attention_mask=attention_masks,
            is_decoder=self.is_decoder,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads, is_decoder):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.hidden_dim_per_head = self.hidden_dim // self.n_heads

        self.is_decoder = is_decoder

        self.heads = nn.ModuleList(
            [
                AttentionHead(self.emb_dim, self.hidden_dim_per_head, is_decoder=is_decoder)
                for _ in range(n_heads)
            ]
        )

        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, hidden_state, attention_masks=None):
        output_att = torch.cat(
            [head(hidden_state, attention_masks) for head in self.heads], dim=-1
        )
        return self.dense(output_att)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, p_dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.p_dropout = p_dropout

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("layer_1", nn.Linear(self.hidden_dim, self.intermediate_dim)),
                    ("gelu", nn.GELU()),
                    ("layer_2", nn.Linear(self.intermediate_dim, self.hidden_dim)),
                    ("dropout", nn.Dropout(self.p_dropout)),
                ]
            )
        )

    def forward(self, hidden_state):
        return self.layers(hidden_state)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads, intermediate_dim, p_dropout=0.2):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            emb_dim=emb_dim, hidden_dim=hidden_dim, n_heads=n_heads, is_decoder=False,
        )
        self.ff = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            p_dropout=p_dropout,
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask=0):
        x = self.multi_head_attention(x, attention_mask)
        x = x + self.layer_norm_1(x)
        x = x + self.ff(x)
        return self.layer_norm_2(x)
