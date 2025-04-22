import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

# -------------------------------
# Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Masked Multi-Head Self Attention
# -------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)

# -------------------------------
# Feed Forward (SwiGLU)
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

# -------------------------------
# Decoder Block
# -------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, skip_prob=0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.skip_prob = skip_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.skip_prob:
            return x
        identity = x
        x = self.norm1(x)
        x = identity + self.dropout1(self.attn(x))

        identity = x
        x = self.norm2(x)
        x = identity + self.dropout2(self.ff(x))
        return x

# -------------------------------
# Full Decoder-Only Transformer
# -------------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1, max_skip_prob=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, max_skip_prob * i / max(1, num_layers - 1))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None, return_all=False):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)

        all_logits = []
        for layer in self.layers:
            x = layer(x)
            if return_all:
                logits = self.fc_out(self.norm(x))
                all_logits.append(logits)

        x = self.norm(x)
        final_logits = self.fc_out(x)
        all_logits.append(final_logits)

        if not return_all:
            return final_logits

        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            all_losses = [loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)) for logits in all_logits]
            return all_logits, all_losses
        return all_logits

# -------------------------------
# Data Processing for WikiText-2
# -------------------------------
def build_vocab(dataset, tokenizer=lambda x: x.lower().split(), min_freq=1):
    counter = Counter()
    for example in dataset['train']:
        counter.update(tokenizer(example['text']))
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def tokenize_text(example, vocab, tokenizer=lambda x: x.lower().split()):
    example['input_ids'] = [vocab.get(token, vocab['<unk>']) for token in tokenizer(example['text'])]
    return example

class WikiTextDataset(Dataset):
    def __init__(self, dataset_split, seq_length):
        self.data = []
        for example in dataset_split:
            tokens = example['input_ids']
            if len(tokens) < seq_length + 1:
                continue
            for i in range(0, len(tokens) - seq_length):
                self.data.append((tokens[i:i+seq_length], tokens[i+1:i+seq_length+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_wikitext_data(seq_length=100, batch_size=32):
    print("Preparing Data")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab = build_vocab(dataset)
    dataset = dataset.map(lambda x: tokenize_text(x, vocab), batched=False)
    train_dataset = WikiTextDataset(dataset['train'], seq_length)
    val_dataset = WikiTextDataset(dataset['validation'], seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("Data Preperation Complete!")
    return train_loader, val_loader, len(vocab), dataset, vocab

if __name__ == "__main__":
    train_loader, val_loader, vocab_size, dataset, vocab = load_wikitext_data()
    id_to_token = {idx: token for token, idx in vocab.items()}
    x, y = next(iter(val_loader))
    x, y = x[:1], y[:1]  # reduce to batch size 1 for fair timing
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # Baseline model inference
    # -----------------------------
    base_model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        max_seq_len=100,
        dropout=0.1,
        max_skip_prob=0.0,
    ).eval()

    with torch.no_grad():
        start = time.time()
        logits = base_model(x)
        time_base = time.time() - start
        loss_base = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        ppl_base = math.exp(loss_base)

    print("=== Baseline ===")
    print(f"Time: {time_base:.4f} sec | Perplexity: {ppl_base:.2f}")

    # -----------------------------
    # DEE model (return_all = True)
    # -----------------------------
    dee_model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        max_seq_len=100,
        dropout=0.1,
        max_skip_prob=0.3  # simulate training with layer skip
    ).eval()

    with torch.no_grad():
        start = time.time()
        _, losses_dee = dee_model(x, targets=y, return_all=True)
        time_dee = time.time() - start
        loss_dee = losses_dee[-1].item()
        ppl_dee = math.exp(loss_dee)

    print("=== Dynamic Early Exit (DEE) ===")
    print(f"Time: {time_dee:.4f} sec | Perplexity: {ppl_dee:.2f}")

    # -----------------------------
    # Self-Speculative Decoding (SSD)
    # -----------------------------
    def self_speculative_decode(model, x, draft_layer=3, k=4):
        with torch.no_grad():
            x_embed = model.embedding(x) * math.sqrt(model.embedding.embedding_dim)
            x_embed = model.pos_encoder(x_embed)
            for i, layer in enumerate(model.layers[:draft_layer]):
                x_embed = layer(x_embed)
            draft_logits = model.fc_out(model.norm(x_embed))
            topk = torch.topk(draft_logits[:, -1], k, dim=-1).indices
            return topk  # simplified SSD mock

    ssd_model = dee_model  # same weights as DEE

    start = time.time()
    ssd_tokens = self_speculative_decode(ssd_model, x)
    time_ssd = time.time() - start
    # Not decoding here, just speed test
    print("=== Self-Speculative Decoding (SSD) ===")
    print(f"Time: {time_ssd:.4f} sec | Top-K Next Tokens: {ssd_tokens.tolist()}")

