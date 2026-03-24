import math
import random
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


@dataclass
class GenerationConfig:
    max_new_tokens: int = 10
    crop_context: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 64
    steps: int = 1000
    lr: float = 1e-3
    log_every: int = 100


@dataclass
class PrefillOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class DecodeOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # x: (B, T, C)
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        q_len = q.size(2)
        k_len = k.size(2)
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, q_len, k_len)

        # During cached decode q_len is typically 1, and the query corresponds to the newest token,
        # so it may attend to all cached keys and itself. For prefill/full-sequence forward, use causal mask.
        if q_len > 1:
            mask = self.causal_mask[k_len - q_len : k_len, :k_len]  # (q_len, k_len)
            att_scores = att_scores.masked_fill(mask == 0, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights)

        out = att_weights @ v  # (B, H, q_len, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, present_kv


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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


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
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.config.max_seq_length, "sequence too long"

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        past_length = 0
        if past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(2)

        assert past_length + T <= self.config.max_seq_length, "cache + input exceeds max_seq_length"

        positions = torch.arange(past_length, past_length + T, device=idx.device)
        tok_emb = self.token_embedding(idx)        # (B, T, C)
        pos_emb = self.pos_embedding(positions)    # (T, C)
        x = tok_emb + pos_emb                      # (B, T, C)

        present_key_values = []
        for i, block in enumerate(self.blocks):
            x, present_kv = block(x, past_kv=past_key_values[i], use_cache=use_cache)
            present_key_values.append(present_kv)

        x = self.final_ln(x)
        logits = self.lm_head(x)                  # (B, T, V)

        loss = None
        if targets is not None:
            _, _, V = logits.shape
            shifted_logits = logits[:, :-1, :].reshape(B * (T - 1), V)
            shifted_targets = targets[:, 1:].reshape(B * (T - 1))
            loss = self.loss_fn(shifted_logits, shifted_targets)

        return logits, loss, present_key_values

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor) -> PrefillOutput:
        logits, _, past_key_values = self(idx, past_key_values=None, use_cache=True)
        return PrefillOutput(logits=logits, past_key_values=past_key_values)  # type: ignore[arg-type]

    @torch.no_grad()
    def decode(
        self,
        idx: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> DecodeOutput:
        assert idx.shape[1] == 1, "decode expects exactly one token"
        logits, _, new_past_key_values = self(idx, past_key_values=past_key_values, use_cache=True)
        return DecodeOutput(logits=logits, past_key_values=new_past_key_values)  # type: ignore[arg-type]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, crop_context: bool = True) -> torch.Tensor:
        self.eval()
        assert idx.shape[1] <= self.config.max_seq_length, "prompt too long"
        if max_new_tokens <= 0:
            return idx

        prefill_output = self.prefill(idx)
        next_token_logits = prefill_output.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
        past_key_values = prefill_output.past_key_values

        for _ in range(max_new_tokens - 1):
            decode_output = self.decode(next_token, past_key_values)
            next_token_logits = decode_output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            past_key_values = decode_output.past_key_values

            if past_key_values[0][0].size(2) >= self.config.max_seq_length:
                if not crop_context:
                    raise ValueError("generation exceeded max_seq_length cache capacity")
                # Keep the full generated sequence for display, but trim the cached context window.
                idx_window = idx[:, -self.config.max_seq_length :]
                prefill_output = self.prefill(idx_window)
                past_key_values = prefill_output.past_key_values

        return idx

    @torch.no_grad()
    def generate_naive(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_length :]
            logits, _, _ = self(idx_cond, past_key_values=None, use_cache=False)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # data: (N,)
    ix = torch.randint(0, len(data) - seq_len, (batch_size,), device=device)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = x.clone()
    return x, y


def train(
    data: torch.Tensor,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_config: TrainingConfig,
) -> list[float]:
    device = next(model.parameters()).device
    model.train()
    losses = []

    for step in range(train_config.steps):
        x, y = get_batch(data, train_config.batch_size, train_config.seq_len, device)
        _, loss, _ = model(x, y)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if step % train_config.log_every == 0:
            print(f"step={step:04d} loss={loss_value:.6f}")

    return losses


@torch.no_grad()
def run_inference_checks(model: GPT) -> None:
    model.eval()
    device = next(model.parameters()).device

    prompts = [
        torch.tensor([[1, 2, 3]], device=device),
        torch.tensor([[2, 3, 4]], device=device),
        torch.tensor([[5, 1, 2]], device=device),
    ]

    for prompt in prompts:
        for max_new_tokens in (1, 3, 10):
            out_cached = model.generate(prompt.clone(), max_new_tokens=max_new_tokens)
            out_naive = model.generate_naive(prompt.clone(), max_new_tokens=max_new_tokens)
            assert torch.equal(out_cached, out_naive), (
                f"cached and naive generation mismatch for prompt={prompt.tolist()} "
                f"max_new_tokens={max_new_tokens}"
            )
            print(f"prompt={prompt.tolist()} max_new_tokens={max_new_tokens} -> {out_cached.tolist()}")


def main() -> None:
    set_seed(0)

    config = GPTConfig(
        vocab_size=1000,
        max_seq_length=64,
        d_embed=128,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
    )
    train_config = TrainingConfig(
        batch_size=32,
        seq_len=64,
        steps=1000,
        lr=1e-3,
        log_every=100,
    )
    generation_config = GenerationConfig(max_new_tokens=10, crop_context=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    data = torch.tensor([1, 2, 3, 4, 5] * 1000, dtype=torch.long, device=device)
    train(data, model, optimizer, train_config)

    prompt = torch.tensor([[1, 2, 3]], device=device)
    sample = model.generate(prompt, max_new_tokens=generation_config.max_new_tokens, crop_context=generation_config.crop_context)
    print("sample:", sample.tolist())

    run_inference_checks(model)


if __name__ == "__main__":
    main()
