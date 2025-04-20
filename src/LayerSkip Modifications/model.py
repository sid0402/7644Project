# retrofit.py ----------------------------------------------------------
"""
Add one weight‑tied LM head to every layer of your pretrained decoder.
NO training – just load weights, add heads, save a new checkpoint.
"""

import torch, torch.nn as nn
from decoder import DecoderOnlyTransformer


hyper = dict(
    vocab_size = 66651,
    d_model    = 128,
    num_layers = 6,
    num_heads  = 8,
    d_ff       = 512,
    max_seq_len= 100,
    dropout    = 0.0,        
    max_skip_prob = 0.0        
)

CKPT_IN  = "best_decoder_d128_h8_l6_ff512_dp0.1_ls0_e1_val6.4181.pt" 
CKPT_OUT = "mymodel_earlyexit_noLS.pt"

print("→ loading backbone")
model = DecoderOnlyTransformer(**hyper)
ckpt = torch.load(CKPT_IN, map_location="cpu")

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]

model.load_state_dict(ckpt, strict=True)

print("→ attaching per‑layer exit heads")
heads = nn.ModuleList()
for _ in range(hyper["num_layers"]):
    head = nn.Linear(hyper["d_model"], hyper["vocab_size"], bias=False)
    head.weight = model.fc_out.weight
    heads.append(head)
model.exit_heads = heads           

torch.save(model.state_dict(), CKPT_OUT)
print(f"✓ wrote {CKPT_OUT} (added {len(heads)} exit heads)")
