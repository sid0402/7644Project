import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter


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
        T = x.size(1)
        if T > self.pe.size(1):                       # need a longer table
            extra = T - self.pe.size(1)
            pos     = torch.arange(self.pe.size(1), T, dtype=torch.float,
                                device=x.device).unsqueeze(1)
            d_model = self.pe.size(2)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float()
                                * (-math.log(10000.0) / d_model))
            pe_new = torch.zeros(1, extra, d_model, device=x.device)
            pe_new[0, :, 0::2] = torch.sin(pos * div_term)
            pe_new[0, :, 1::2] = torch.cos(pos * div_term)
            self.pe = torch.cat([self.pe, pe_new], dim=1)
        return x + self.pe[:, :T]



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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


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
            return x  # skip entire block
        identity = x
        x = self.norm1(x)
        x = identity + self.dropout1(self.attn(x))

        identity = x
        x = self.norm2(x)
        x = identity + self.dropout2(self.ff(x))
        return x


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

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc_out(x)


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
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab = build_vocab(dataset)
    dataset = dataset.map(lambda x: tokenize_text(x, vocab), batched=False)
    train_dataset = WikiTextDataset(dataset['train'], seq_length)
    val_dataset = WikiTextDataset(dataset['validation'], seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, len(vocab)