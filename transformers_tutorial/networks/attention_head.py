from math import sqrt

import torch
from torch import nn


class AttentionHead(nn.Module):

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.q = nn.Linear(emb_dim, hidden_dim)
        self.k = nn.Linear(emb_dim, hidden_dim)
        self.v = nn.Linear(emb_dim, hidden_dim)

    @staticmethod
    def compute_weights(q, k, attention_mask=None):
        dim_ = q.size(-1)
        attention_score = torch.bmm(q, k.transpose(1, 2)) / sqrt(dim_)
        if attention_mask is not None:
            padding_mask = 1 - attention_mask
            attention_mask_mat = attention_mask.unsqueeze(
                -1
            ) * attention_mask.unsqueeze(1)
            padding_mask_mat = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
            coupling_mask_mat = 1 - attention_mask_mat - padding_mask_mat

            attention_score = attention_score + coupling_mask_mat * (-1.0e5)
        return torch.softmax(attention_score, dim=-1)

    @staticmethod
    def scaled_dot_product(q, k, v, attention_mask=None):
        weights = AttentionHead.compute_weights(q, k, attention_mask)
        return torch.bmm(weights, v)

    def forward(self, hidden_state, attention_masks=None):
        return self.scaled_dot_product(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state),
            attention_mask=attention_masks,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.hidden_dim_per_head = self.hidden_dim // self.n_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(self.emb_dim, self.hidden_dim_per_head)
                for _ in range(n_heads)
            ]
        )

        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, hidden_state, attention_masks=None):
        output_att = torch.cat(
            [head(hidden_state, attention_masks) for head in self.heads], dim=-1
        )
        return self.dense(output_att)
