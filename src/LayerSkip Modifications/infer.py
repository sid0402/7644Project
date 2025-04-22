"""Fast inference for your LayerSkip decoder with
 • Dynamic‑Early‑Exit  (--mode dee)
 • Self‑Speculative    (--mode ssd)
using the SAME vocab as training (rebuilt via decoder.build_vocab).
"""

import argparse, math, torch, torch.nn as nn
from pathlib import Path
from datasets import load_dataset
import json
import time

# your model + helpers live here
import decoder
from decoder import DecoderOnlyTransformer, build_vocab

# ───────────── hyper‑parameters (must match checkpoint) ───────────────
hyper = dict(
    vocab_size   = 66651,
    d_model      = 128,
    num_layers   = 6,
    num_heads    = 8,
    d_ff         = 512,
    max_seq_len  = 100,
    dropout      = 0.0,
    max_skip_prob= 0.0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def attach_exit_heads(model: nn.Module):
    if hasattr(model, "exit_heads") and len(model.exit_heads):
        return model
    heads = nn.ModuleList()
    for _ in model.layers:
        h = nn.Linear(model.fc_out.in_features,
                      model.fc_out.out_features, bias=False)
        h.weight = model.fc_out.weight
        heads.append(h)
    model.exit_heads = heads
    return model

def generate_dee_trace(model, ids, conf=0.9, max_new=120, eos_id=2):
    start = time.time()
    exit_layers = []
    L = len(model.layers)

    for _ in range(max_new):
        x = model.pos_encoder(model.embedding(ids) * math.sqrt(model.embedding.embedding_dim))
        exited = False

        for l, blk in enumerate(model.layers):
            x = blk(x)
            logits = model.exit_heads[l](model.norm(x))[:, -1]

            if logits.softmax(-1).max().item() >= conf:
                nxt_id = logits.argmax(-1)
                ids = torch.cat([ids, nxt_id.view(1, 1)], 1)
                exit_layers.append(l)
                exited = True
                break

        if not exited:
            logits = model.fc_out(model.norm(x))[:, -1]
            nxt_id = logits.argmax(-1)
            ids = torch.cat([ids, nxt_id.view(1, 1)], 1)
            exit_layers.append(L)

        if nxt_id.item() == eos_id:
            break

    end = time.time()
    return ids.squeeze(), exit_layers, end - start

@torch.no_grad()
def generate_baseline(model, ids, max_new=120, eos_id=2):
    start = time.time()
    for _ in range(max_new):
        x = model.pos_encoder(model.embedding(ids) * math.sqrt(model.embedding.embedding_dim))
        for blk in model.layers:
            x = blk(x)
        logits = model.fc_out(model.norm(x))[:, -1]
        nxt_id = logits.argmax(-1)
        ids = torch.cat([ids, nxt_id.view(1, 1)], 1)
        if nxt_id.item() == eos_id:
            break
    end = time.time()
    return ids.squeeze(), end - start

@torch.no_grad()
def generate_ssd_with_dee(model,
                          ids,
                          draft_layer=3,
                          k=4,
                          conf=0.9,
                          lam=None,
                          max_new=120,
                          eos_id=2):

    L = len(model.layers)
    def thr(l): return lam[l] if lam else conf

    while ids.size(1) < max_new + prompt_ids.size(1):
        draft_tokens = []
        for _ in range(k):
            x = model.pos_encoder(model.embedding(ids) *
                                  math.sqrt(model.embedding.embedding_dim))
            exited = False
            for l, blk in enumerate(model.layers[:draft_layer]):
                x = blk(x)
                logits = model.exit_heads[l](model.norm(x))[:, -1]
                if logits.softmax(-1).max().item() >= thr(l):
                    nxt = logits.argmax(-1)
                    exited = True
                    break
            if not exited:
                logits = model.exit_heads[draft_layer-1](model.norm(x))[:, -1]
                nxt = logits.argmax(-1)

            draft_tokens.append(nxt)
            ids = torch.cat([ids, nxt.view(1, 1)], 1)
            if nxt.item() == eos_id:
                break

        span = ids[:, -len(draft_tokens)-1:]
        x = model.pos_encoder(model.embedding(span) *
                              math.sqrt(model.embedding.embedding_dim))
        for blk in model.layers:
            x = blk(x)
        verified = model.fc_out(model.norm(x)) \
                       [:, -len(draft_tokens):].argmax(-1)

        agree = (verified == torch.stack(draft_tokens, 1))
        if agree.all():
            continue
        bad = (~agree).nonzero(as_tuple=False)[0, 1]
        ids = torch.cat([ids[:, :-len(draft_tokens)+bad],
                         verified[:, bad:bad+1]], 1)
        if ids[0, -1].item() == eos_id:
            break

    return ids.squeeze()

# ───────────── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt",    required=True)
    parser.add_argument("--mode",      choices=["dee","ssd"], default="dee")
    parser.add_argument("--conf",      type=float, default=0.9,
                        help="DEE confidence threshold")
    parser.add_argument("--draft_layer", type=int, default=3,
                        help="SSD cut‑point (ignored by DEE)")
    parser.add_argument("--k", type=int, default=4,
                    help="How many tokens to draft before each verify step (SSD)")
    parser.add_argument("--lam_file", type=str, default="",
                        help="JSON file with per‑layer λ thresholds; overrides --conf (DEE only)")

    args = parser.parse_args()

    print("→ rebuilding vocab (WikiText‑2 train split)")
    ds     = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab  = build_vocab(ds)
    inv    = {v:k for k,v in vocab.items()}

    prompt_ids = torch.tensor(
        [[vocab.get(tok, vocab["<unk>"]) for tok in args.prompt.lower().split()]],
        device=device
    )

    model = DecoderOnlyTransformer(**hyper).to(device)
    attach_exit_heads(model)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    if args.mode == "dee":
        out_ids_dee, exit_layers, time_dee = generate_dee_trace(model, prompt_ids.clone(), conf=args.conf)
        out_ids_baseline, time_base = generate_baseline(model, prompt_ids.clone())

        print("\n=== DEE Output ===")
        print("→", " ".join(inv.get(int(t), "<unk>") for t in out_ids_dee.tolist()))
        print(f"DEE Generation Time: {time_dee:.4f} seconds")
        print("per‑token exit layer:", exit_layers)

        print("\n=== Baseline Output ===")
        print("→", " ".join(inv.get(int(t), "<unk>") for t in out_ids_baseline.tolist()))
        print(f"Baseline Generation Time: {time_base:.4f} seconds")

    elif args.mode == "ssd":
        lam = json.load(open(args.lam_file)) if getattr(args, "lam_file", None) else None
        out_ids = generate_ssd_with_dee(model,
                                        prompt_ids,
                                        draft_layer=args.draft_layer,
                                        k=args.k,
                                        conf=args.conf,
                                        lam=lam)
        print("\n=== SSD Output ===")
        print("→", " ".join(inv.get(int(t), "<unk>") for t in out_ids.tolist()))
