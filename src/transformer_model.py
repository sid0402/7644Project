import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Multi-Head Self-Attention Module
# -------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(attn_output)

# -------------------------------
# Feed-Forward Network Module
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

# -------------------------------
# Transformer Encoder Layer with Layer Dropout (Stochastic Depth)
# -------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, skip_prob=0.0):
        """
        :param skip_prob: Probability of skipping this layer during training.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.skip_prob = skip_prob

    def forward(self, src, src_mask=None):
        identity = src

        attn_output = self.self_attn(src, mask=src_mask)
        x = self.norm1(src + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        candidate = x

        if self.training:
            if torch.rand(1).item() < self.skip_prob:
                return identity
            else:
                return identity + (candidate - identity) / (1 - self.skip_prob)
        else:
            return candidate

# -------------------------------
# Transformer Encoder (Stack of Layers)
# -------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, max_skip_prob=0.0):
        """
        :param max_skip_prob: Maximum skip probability for the deepest layer.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            skip_prob = max_skip_prob * float(i) / (num_layers - 1) if num_layers > 1 else 0.0
            self.layers.append(TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, skip_prob))

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# -------------------------------
# Complete Transformer Model
# -------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length,
                 dropout=0.1, max_skip_prob=0.0):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout, max_skip_prob)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_skip_prob = max_skip_prob
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.encoder(x, src_mask)
        return self.fc_out(x)

# -------------------------------
# Data Loading and Processing for WikiText-2
# -------------------------------
def build_vocab(dataset, tokenizer=lambda x: x.split(), min_freq=1):
    counter = Counter()
    for example in dataset["train"]:
        tokens = tokenizer(example["text"])
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def tokenize_text(example, vocab, tokenizer=lambda x: x.split()):
    example["input_ids"] = [vocab.get(token, vocab["<unk>"]) for token in tokenizer(example["text"])]
    return example

class WikiTextDataset(Dataset):
    def __init__(self, dataset_split, seq_length):
        self.data = []
        for example in dataset_split:
            tokens = example["input_ids"]
            if len(tokens) < seq_length:
                continue
            for i in range(0, len(tokens) - seq_length + 1, seq_length):
                self.data.append(tokens[i:i+seq_length])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

def load_wikitext_data(seq_length=100, batch_size=32):
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab = build_vocab(dataset)
    dataset = dataset.map(lambda x: tokenize_text(x, vocab), batched=False)
    train_dataset = WikiTextDataset(dataset["train"], seq_length)
    val_dataset = WikiTextDataset(dataset["validation"], seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, len(vocab)
