import torch
import math

import torch.nn as nn
import torch.nn.functional as F

max_seq_length = 256


class MHA(nn.Module):
    def __init__(self, d_embed, n_head=8, dropout=0.0) -> None:
        super().__init__()
        assert d_embed % n_head == 0
        self.d_embed = d_embed
        self.n_head = n_head
        self.head_dim = d_embed // n_head

        self.q = nn.Linear(d_embed, d_embed)
        self.k = nn.Linear(d_embed, d_embed)
        self.v = nn.Linear(d_embed, d_embed)

        self.output = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_length, max_seq_length)))

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.d_embed
        H, D = self.n_head, self.head_dim
        assert H * D == C

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()  # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()  # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()  # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)

        wei = Q @ K.transpose(-2, -1) / math.sqrt(D)  # (B, H, T, D) @ (B, H, D, T) = (B, H, T, T)
        wei = wei.masked_fill(self.causal_mask[:T, :T] == 0, -float("inf"))  # (B, H, T, T)

        att = F.softmax(wei, dim=-1)  # (B, H, T, T)
        att = self.dropout(att)

        out = att @ V  # (B, H, T, T) @ (B, H, T, D) = (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = self.output(out)  # (B, H, C)
        out = self.dropout(out)  # (B, H, C)

        return out


class MLP(nn.Module):
    def __init__(self, d_embed):
        super().__init__()

        self.l1 = nn.Linear(d_embed, 4 * d_embed)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x):
        return self.l2(self.gelu(self.l1(x)))


class Transformer(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embed)
        self.mha = MHA(d_embed)
        self.ln2 = nn.LayerNorm(d_embed)
        self.mlp = MLP(d_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_embed, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.pos_embedding = nn.Embedding(max_seq_length, d_embed)
        self.layers = nn.ModuleList([Transformer(d_embed) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        token_embed = self.token_embedding(idx)  # (B, T) -> (B, T, C)
        positions = torch.arange(0, T, device=idx.device)  # (T, )
        pos_embed = self.pos_embedding(positions)  # (T, ) -> (T, C)

        x = token_embed + pos_embed  # (B, T, C) + (T, C) -> (B, T, C)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.linear(x)
        return logits
