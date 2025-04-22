# tokenizer.py

import torch

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab.copy()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.eos_token_id = self.vocab.get(self.eos_token, len(self.vocab))

        if self.eos_token not in self.vocab:
            self.vocab[self.eos_token] = self.eos_token_id
            self.inv_vocab[self.eos_token_id] = self.eos_token

    def encode(self, text):
        tokens = text.lower().split()
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        ids.append(self.eos_token_id)
        return torch.tensor(ids).unsqueeze(0)  # [1, T]

    def decode(self, ids, skip_special_tokens=True):
        words = [self.inv_vocab.get(id, self.unk_token) for id in ids]
        if skip_special_tokens:
            words = [w for w in words if w not in [self.pad_token, self.unk_token, self.eos_token]]
        return ' '.join(words)
