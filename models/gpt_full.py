from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MAX_SEQ_LENGTH = 256


@dataclass(frozen=True)
class ToyGPTConfig:
    vocab_size: int
    d_embed: int = 256
    n_layers: int = 4
    n_head: int = 8
    dropout: float = 0.0
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH


@dataclass
class ToyCausalLMOutput:
    logits: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None


@dataclass
class ToyModelBundle:
    tokenizer: "ToyTokenizer"
    model: "ToyGPTForCausalLM"
    device: torch.device
    model_name: str


class ToyTokenizer:
    def __init__(self, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH):
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        base_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        self.tokens = special_tokens + base_chars
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.max_seq_length = max_seq_length

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):
        lines = []
        for message in messages:
            role = message.get("role", "user").strip().capitalize()
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("Assistant:")
        text = "\n".join(lines)

        if tokenize:
            return self._encode_text(text)
        return text

    def _encode_text(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.unk_token_id) for ch in text]

    def encode(self, text: str) -> list[int]:
        return self._encode_text(text)

    def __call__(
        self,
        texts,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for text in texts:
            token_ids = self._encode_text(text)
            if truncation:
                token_ids = token_ids[: self.max_seq_length]
            encoded.append(token_ids)

        if not encoded:
            raise ValueError("expected at least one text input")

        target_len = max(len(token_ids) for token_ids in encoded)
        if not padding and len(texts) == 1:
            target_len = len(encoded[0])

        input_ids = []
        attention_mask = []
        for token_ids in encoded:
            pad_len = target_len - len(token_ids)
            input_ids.append(token_ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(token_ids) + [0] * pad_len)

        if return_tensors != "pt":
            raise ValueError(f"unsupported return_tensors: {return_tensors}")

        return {
            # input_ids / attention_mask: [batch, seq_len]
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().view(-1).tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        pieces = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in {
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            }:
                continue
            pieces.append(token)
        return "".join(pieces)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed: int, n_head: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if d_embed % n_head != 0:
            raise ValueError("d_embed must be divisible by n_head")

        self.d_embed = d_embed
        self.n_head = n_head
        self.head_dim = d_embed // n_head
        self.q = nn.Linear(d_embed, d_embed)
        self.k = nn.Linear(d_embed, d_embed)
        self.v = nn.Linear(d_embed, d_embed)
        self.output = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        # [B, T, C] -> [B, H, T, D]
        return x.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        # [B, H, T, D] -> [B, T, C]
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_embed)

    def _build_causal_mask(self, query_len: int, total_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        if query_len == 1:
            return torch.ones(1, total_len, dtype=torch.bool, device=device)

        key_positions = torch.arange(total_len, device=device)
        query_positions = past_len + torch.arange(query_len, device=device)
        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # x: [B, T_new, C]
        # layer_past: each tensor [B, H, T_cache, D]
        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))

        past_len = 0
        if layer_past is not None:
            past_k, past_v = layer_past
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            # k / v are now [B, H, T_cache + T_new, D]

        # scores: [B, H, T_new, T_total]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = self._build_causal_mask(
            query_len=q.size(2),
            total_len=k.size(2),
            past_len=past_len,
            device=x.device,
        )
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.size(1) != k.size(2):
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} does not match key length {k.size(2)}"
                )
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)
        out = att @ v  # [B, H, T_new, D]
        out = self._merge_heads(out)
        out = self.output(out)
        out = self.dropout(out)

        present = (k, v) if use_cache else None
        return out, present


class FeedForward(nn.Module):
    def __init__(self, d_embed: int, dropout: float = 0.0):
        super().__init__()
        self.l1 = nn.Linear(d_embed, 4 * d_embed)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4 * d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.l2(self.gelu(self.l1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_embed: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embed)
        self.attn = MultiHeadAttention(d_embed, n_head=n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_embed)
        self.ffn = FeedForward(d_embed, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attn_out, present = self.attn(
            self.ln1(x),
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present


class ToyGPTForCausalLM(nn.Module):
    def __init__(self, config: ToyGPTConfig):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=config.vocab_size,
            hidden_size=config.d_embed,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_head,
            max_position_embeddings=config.max_seq_length,
        )
        self.max_seq_length = config.max_seq_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_embed)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_embed,
                    n_head=config.n_head,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.linear.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_past_key_values(self, past_key_values):
        if past_key_values is None:
            return [None] * len(self.layers)

        normalized = []
        for layer in past_key_values:
            if len(layer) < 2:
                raise ValueError(f"unexpected cache layer format: {type(layer)}")
            normalized.append((layer[0], layer[1]))
        return normalized

    def _infer_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values,
    ) -> torch.Tensor:
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(dim=1) - 1
            position_ids = position_ids.clamp_min(0)
            return position_ids[:, -input_ids.size(1) :]

        past_len = 0
        if past_key_values:
            first_layer = past_key_values[0]
            if first_layer is not None:
                past_len = first_layer[0].size(2)

        positions = torch.arange(
            past_len,
            past_len + input_ids.size(1),
            device=input_ids.device,
        )
        return positions.unsqueeze(0).expand(input_ids.size(0), -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> ToyCausalLMOutput:
        normalized_past = self._normalize_past_key_values(past_key_values)

        if position_ids is None:
            position_ids = self._infer_position_ids(input_ids, attention_mask, normalized_past)

        if position_ids.max().item() >= self.max_seq_length:
            raise ValueError(
                f"position_ids exceed max_seq_length={self.max_seq_length}; "
                "increase max_seq_length for longer prompts"
            )

        # input_ids:   [B, T_new]
        # position_ids:[B, T_new]
        token_embed = self.token_embedding(input_ids)   # [B, T_new, C]
        pos_embed = self.pos_embedding(position_ids)    # [B, T_new, C]
        x = token_embed + pos_embed                     # [B, T_new, C]

        presents = []
        for layer, layer_past in zip(self.layers, normalized_past):
            x, present = layer(
                x,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present is not None:
                presents.append(present)

        x = self.ln(x)
        logits = self.linear(x)  # [B, T_new, vocab_size]

        return ToyCausalLMOutput(
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
        )


def build_toy_model_bundle(
    *,
    d_embed: int = 256,
    n_layers: int = 4,
    n_head: int = 8,
    dropout: float = 0.0,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    device: torch.device | None = None,
) -> ToyModelBundle:
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    tokenizer = ToyTokenizer(max_seq_length=max_seq_length)
    config = ToyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_embed=d_embed,
        n_layers=n_layers,
        n_head=n_head,
        dropout=dropout,
        max_seq_length=max_seq_length,
    )

    model = ToyGPTForCausalLM(config).to(device)
    model.eval()

    return ToyModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name="toy-gpt",
    )
