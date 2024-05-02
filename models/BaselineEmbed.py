from IndexDataset import IndexDataset
import torch
from torch import nn
import pandas as pd

class BaselineEmbed:
    def __init__(self, data: IndexDataset, token_len: int, embed_size: int, 
                 model: nn.Module):
        '''
        data: string information
        hash: function to hash data
        '''
        self.data = data
        self.token_len = token_len
        self.embed_size = embed_size
        
        self.vocab, self.max_len = self.create_vocab()
        self.embed = nn.Embedding(len(self.vocab.keys()), self.embed_size)
        
        self.model = model
        self.sort('embed')
    
    def create_vocab(self):
        vocab = {}
        idx, max_len = 0, 0
        
        for string in self.data.get_series:
            max_len = max(max_len, len(string))
            for i in range(len(string) - self.token_len + 1):
                token = string[i: i + self.token_len]
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        
        return vocab, max_len

    def predict(self, key: str):
        embeds = torch.zeros(self.max_len, self.embed_size)
        token_idxs = []
        for i in range(len(key) - self.token_len + 1):
            token = key[i: i + self.token_len]
            token_idxs.append(self.vocab[token])

        embeds[:len(token_idxs), :] = self.embed[token_idxs]
        embeds = embeds.view(1, self.max_len, -1)
        prediction = self.model(embeds)
        
        return prediction