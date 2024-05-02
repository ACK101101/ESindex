from IndexDataset import IndexDataset
import torch
from torch import nn
import pandas as pd

class BaselineEmbed(nn.Module):
    def __init__(self, data: IndexDataset, token_len: int, embed_size: int):
        super(BaselineEmbed, self).__init__()
        
        self.data = data
        self.token_len = token_len
        self.embed_size = embed_size
        
        self.vocab, self.max_len = self.create_vocab()
        self.embed = nn.Embedding(len(self.vocab.keys()), self.embed_size)
    
    def create_vocab(self):
        vocab = {}
        idx, max_len = 0, 0
        
        for string in self.data.get_series():
            max_len = max(max_len, len(string))
            for i in range(len(string) - self.token_len + 1):
                token = string[i: i + self.token_len]
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        
        return vocab, max_len


    def forward(self, keys: tuple):
        embeds = []
        for key in keys:
            embed = torch.zeros(self.max_len, self.embed_size)
            token_idxs = []
            for i in range(len(key) - self.token_len + 1):
                token = key[i: i + self.token_len]
                token_idxs.append(self.vocab[token])

            token_idxs = torch.LongTensor(token_idxs)
            embed[:len(token_idxs), :] = self.embed(token_idxs)
            embed = embed.view(embed.size(0), -1)
            embeds.append(embed)
        
        embeds = torch.stack(embeds, dim=0)
        return embeds.view(embeds.size(0), -1)