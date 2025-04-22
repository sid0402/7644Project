def ssd_generate(full_model, draft_model, tokenizer, prompt_ids, max_length=50, k=4):
    full_model.eval()
    draft_model.eval()
    device = next(full_model.parameters()).device
    input_ids = prompt_ids.to(device)

    with torch.no_grad():
        for _ in range(max_length // k):
            # === Draft k tokens using early exit model ===
            draft_ids = early_exit_generate(draft_model, tokenizer, input_ids, max_length=k)
            new_tokens = draft_ids[:, input_ids.shape[1]:]  # Only get newly drafted ones

            # === Verify with full model ===
            full_out = full_model(input_ids, return_all=False)  # [B, T, V]
            for i in range(new_tokens.shape[1]):
                token_pos = input_ids.shape[1] + i
                logits = full_model(input_ids[:, :token_pos], return_all=False)[:, -1, :]  # [B, V]
                pred_token = torch.argmax(logits, dim=-1)

                if pred_token != new_tokens[:, i]:
                    input_ids = torch.cat([input_ids, pred_token.unsqueeze(1)], dim=1)
                    break
                else:
                    input_ids = torch.cat([input_ids, new_tokens[:, i].unsqueeze(1)], dim=1)

            if input_ids[0, -1] == tokenizer.eos_token_id:
                break

    return input_ids
