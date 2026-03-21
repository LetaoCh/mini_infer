import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Prefill
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    print("output shape:", outputs.logits.shape)

    # Get the logits for the LAST position only.
    next_token_logits = outputs.logits[:, -1, :]  # (B, 1, C) -> (B, C)
    print("next_token_logits shape:", next_token_logits.shape)

    # Greedy decode: pick argmax token
    next_token_id = torch.argmax(next_token_logits, dim=-1)  # (1)
    print("next_token_id shape:", next_token_id.shape)

    generated_ids = [next_token_id.item()]
    past_key_values = outputs.past_key_values

    # Decode 3 more tokens
    for step in range(3):
        # Build input_ids for decode step using ONLY the newest token.
        decode_input_ids = next_token_id.unsqueeze(0)  # (1, 1)
        print(f"Step {step} decode_input_ids shape:", decode_input_ids.shape)

        with torch.no_grad():
            outputs = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        print(f"Step {step} output shape:", outputs.logits.shape)
        # Again, read only the last-position logits.
        next_token_logits = outputs.logits[:, -1, :]
        print(f"Step {step} next_token_logits shape:", next_token_logits.shape)

        # Greedy pick the next token.
        next_token_id = torch.argmax(next_token_logits, dim=-1)  # (1)
        print(f"Step {step} next_token_id shape:", next_token_id.shape)

        generated_ids.append(next_token_id.item())
        past_key_values = outputs.past_key_values

    full_ids = torch.cat(
        [
            input_ids[0],
            torch.tensor(generated_ids, device=device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
