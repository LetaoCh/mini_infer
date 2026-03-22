import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1000
    max_seq_length: int = 64
    d_embed: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.d_embed % config.n_heads == 0

        self.d_embed = config.d_embed
        self.n_heads = config.n_heads
        self.head_dim = config.d_embed // config.n_heads
        self.max_seq_length = config.max_seq_length

        self.q_proj = nn.Linear(config.d_embed, config.d_embed)
        self.k_proj = nn.Linear(config.d_embed, config.d_embed)
        self.v_proj = nn.Linear(config.d_embed, config.d_embed)
        self.out_proj = nn.Linear(config.d_embed, config.d_embed)

        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)
        att_scores = att_scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)  # (B, H, T, T)
        att_weights = self.dropout(att_weights)

        out = att_weights @ v  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_embed, 4 * config.d_embed)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_embed, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_embed)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_embed)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.config.max_seq_length, "sequence too long"

        positions = torch.arange(T, device=idx.device)  # (T,)
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.pos_embedding(positions)  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            _, _, V = logits.shape
            shifted_logits = logits[:, :-1, :].reshape(B * (T - 1), V)
            shifted_targets = targets[:, 1:].reshape(B * (T - 1))
            loss = self.loss_fn(shifted_logits, shifted_targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_length :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]  # (B, V)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)

        return idx


def main():
    config = GPTConfig(
        vocab_size=1000,
        max_seq_length=64,
        d_embed=128,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 32, 64
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)

    model.train()
    logits, loss = model(idx, idx)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("logits shape:", logits.shape)
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
