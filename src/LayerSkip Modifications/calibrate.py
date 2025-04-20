# calibrate.py ──────────────────────────────────────────────────────────
"""
Pick a per‑layer CALM threshold λ  such that P(disagree) ≤ δ (1 % by default)
using the *margin score*  (p_top − p_2nd).

Outputs  calm_thresholds.json  in the same directory.
"""

import math, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import decoder                                      # your model + build_vocab
from decoder import DecoderOnlyTransformer, build_vocab

# ───────────────────────── hyper‑parameters MUST match checkpoint ─────
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

CKPT         = "mymodel_earlyexit_noLS.pt"
OUT_JSON     = "calm_thresholds_noLS.json"
BATCH_SIZE   = 16
MAX_VAL_TOKS = 300_000          # speed – feel free to raise
DELTA        = 0.01            # target disagreement rate (1 %)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────── util: attach heads (same as retrofit) ──────
def attach_exit_heads(model):
    if hasattr(model, "exit_heads") and len(model.exit_heads):
        return
    heads = torch.nn.ModuleList()
    for _ in model.layers:
        h = torch.nn.Linear(model.fc_out.in_features,
                            model.fc_out.out_features, bias=False)
        h.weight = model.fc_out.weight
        heads.append(h)
    model.exit_heads = heads

# ───────────────────────── CALM margin score ──────────────────────────
def margin_score(logits):                          # shape (B, V)
    probs = logits.softmax(-1)
    top2  = probs.topk(2, -1).values
    return (top2[..., 0] - top2[..., 1])           # (B,)

def calm_max(logits):                 # softmax max
    return logits.softmax(-1).max(-1).values


# ───────────────────────── build vocab & data loader ──────────────────
print("→ rebuilding vocab (WikiText‑2 train)")
ds    = load_dataset("wikitext", "wikitext-2-raw-v1")
vocab = build_vocab(ds)
pad   = vocab["<pad>"]

def encode(example):
    example["ids"] = [vocab.get(t, vocab["<unk>"])
                      for t in example["text"].lower().split()]
    return example

ds_val = ds["validation"].map(encode, remove_columns=["text"])
val_toks = []
for ex in ds_val:
    val_toks.extend(ex["ids"])
val_toks = val_toks[:MAX_VAL_TOKS]
seq_len  = hyper["max_seq_len"]

def chunks(lst, n):             # yield lists of length n
    for i in range(0, len(lst)-n, n):
        yield lst[i:i+n]

val_data = list(chunks(val_toks, seq_len+1))
def collate(batch):
    x = torch.tensor([b[:-1] for b in batch], dtype=torch.long)
    y = torch.tensor([b[1:]  for b in batch], dtype=torch.long)
    return x, y
loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                    shuffle=False, collate_fn=collate)

# ───────────────────────── load model & collect stats ─────────────────
print("→ loading model")
model = DecoderOnlyTransformer(**hyper).to(device)
attach_exit_heads(model)
state = torch.load(CKPT, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state, strict=True)
model.eval()

L = hyper["num_layers"]
score_bank = [[] for _ in range(L)]
agree_bank = [[] for _ in range(L)]

print("→ scanning validation set")
with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)

        # ----- forward manually so we can store every layer's output -----
        h = model.embedding(x) * math.sqrt(model.embedding.embedding_dim)
        h = model.pos_encoder(h)

        hiddens = []
        for layer in model.layers:
            h = layer(h)            # (B, T, d_model)
            hiddens.append(h)
        h_final = model.norm(h)
        logits  = model.fc_out(h_final)          # (B, T, V)
        final_pred = logits.argmax(-1)

        # ----- collect scores & agreement --------------------------------
        for l, h_l in enumerate(hiddens):
            out = model.exit_heads[l](h_l)       # (B, T, V)
            #sc  = margin_score(out)              # (B, T)
            sc = calm_max(out)
            score_bank[l].append(sc.flatten())
            agree_bank[l].append((out.argmax(-1) == final_pred).flatten())

# ───────────────────────── choose λ per layer ────────────────────────
print("→ selecting λ per layer (δ = 1%)")
lam = []
for l in range(L):
    s = torch.cat(score_bank[l]); a = torch.cat(agree_bank[l])
    grid = torch.linspace(0, 1, 201, device=s.device)
    lam_l = 0.0
    for t in grid:
        if a[s >= t].float().mean() > 1 - DELTA:
            lam_l = t.item()
    lam.append(lam_l)
    print(f"  layer {l}: λ = {lam_l:.3f}")

json.dump(lam, open(OUT_JSON, "w"))
print(f"✓ wrote {OUT_JSON}")
