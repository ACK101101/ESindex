from IndexDataset import IndexDataset

import mmh3
import pandas as pd
from torch import nn, no_grad, Tensor

'''
Baseline Linear Model
- Take in data
- Hash data
- Sort data by hash
- Linear model to predict index of randomized digests
'''

class BaselineHash:
    def __init__(self, data: IndexDataset, hash: callable):
        '''
        data: string information
        hash: function to hash data
        '''
        self.data = data
        self.hash = hash
        self.min_val, self.max_val = float("inf"), float("-inf")
        self.sort('digest')

    def generate_digests(self, col: str):
        series = self.data.get_series()
        digest_series = pd.Series([''] * len(series))

        for i in range(len(series)):
            digest = self.hash(series[i])
            norm_digest = digest % self.data.__len__()
            digest_series[i] = norm_digest
            self.min_val = min(self.min_val, norm_digest)
            self.max_val = max(self.max_val, norm_digest)
        
        self.data.add_series(col, digest_series)

    def sort(self, col: str):
        self.generate_digests(col)
        self.data.sort_by_series(col)

    def forward(self, keys: tuple):
        digests = [0] * len(keys)
        for i, key in enumerate(keys):
            digest = self.hash(key.encode())
            norm_digest = digest % self.data.__len__()
            # norm_digest = (digest - self.min_val) / self.max_val
            digests[i] = norm_digest
        
        return Tensor(digests).unsqueeze(1)
    
# class HierarchicalModel():
#     pass
        
def hash(key: str):
    return mmh3.hash(key)