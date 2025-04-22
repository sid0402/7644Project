# eed.py

import torch
import torch.nn.functional as F
from decoder import DecoderOnlyTransformer, load_wikitext_data
from tokenizer import SimpleTokenizer


def early_exit_generate(model, input_ids, max_length=50, threshold=0.9, eos_token_id=None, device='cuda'):
    model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_length):
            all_layer_outputs = model(input_ids, return_all=True)  # list of [B, T, V]
            exit_layer = len(all_layer_outputs) - 1

            for i, logits in enumerate(all_layer_outputs):
                probs = F.softmax(logits[:, -1, :], dim=-1)
                max_prob, _ = probs.max(dim=-1)
                if max_prob.item() >= threshold:
                    exit_layer = i
                    break

            selected_logits = all_layer_outputs[exit_layer][:, -1, :]
            next_token = selected_logits.argmax(dim=-1).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return input_ids


def ssd_generate(full_model, draft_model, input_ids, max_length=50, k=4, eos_token_id=None, device='cuda'):
    full_model.eval()
    draft_model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_length // k):
            draft_ids = early_exit_generate(draft_model, input_ids, max_length=k, device=device)
            new_tokens = draft_ids[:, input_ids.shape[1]:]

            for i in range(new_tokens.shape[1]):
                current_input = torch.cat([input_ids, new_tokens[:, :i]], dim=1)
                verified_logits = full_model(current_input, return_all=False)
                verified_token = verified_logits[:, -1, :].argmax(dim=-1)

                if verified_token != new_tokens[:, i]:
                    input_ids = torch.cat([input_ids, verified_token.unsqueeze(1)], dim=1)
                    break
                else:
                    input_ids = torch.cat([input_ids, new_tokens[:, i].unsqueeze(1)], dim=1)

                if eos_token_id is not None and input_ids[0, -1].item() == eos_token_id:
                    return input_ids

    return input_ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and vocab
    train_loader, val_loader, vocab_size, vocab = load_wikitext_data(seq_length=100, batch_size=32)
    tokenizer = SimpleTokenizer(vocab)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=6,
        num_heads=8,
        d_ff=512,
        max_seq_len=100,
        dropout=0.1,
        max_skip_prob=0.1,
    ).to(device)

    checkpoint = torch.load("best_decoder_d128_h8_l6_ff512_dp0.1_ls0_e1_val6.4181.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)  # device will already be 'cpu'

    prompt = "The history of language models"
    input_ids = tokenizer.encode(prompt)

    output_ids = ssd_generate(model, model, input_ids, max_length=50, k=4, eos_token_id=tokenizer.eos_token_id, device=device)
    print(tokenizer.decode(output_ids[0]))
