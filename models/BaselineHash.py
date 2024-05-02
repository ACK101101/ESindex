from IndexDataset import IndexDataset

import mmh3
import pandas as pd
from torch import nn, no_grad

'''
Baseline Linear Model
- Take in data
- Hash data
- Sort data by hash
- Linear model to predict index of randomized digests
'''

class BaselineHash:
    def __init__(self, data: IndexDataset, hash: callable, model: nn.Module):
        '''
        data: string information
        hash: function to hash data
        '''
        self.data = data
        self.hash = hash
        self.model = model
        self.sort('digest')

    def generate_digests(self, col: str):
        series = self.data.get_series()
        digest_series = pd.Series([''] * len(series))

        for i in range(len(series)):
            digest_series[i] = self.hash(series[i])
        
        self.data.add_series(col, digest_series)

    def sort(self, col: str):
        self.generate_digests(col)
        self.data.sort_by_series(col)

    def predict(self, key: str):
        digest = self.hash(key)
        prediction = self.model(nn.Tensor(digest))
        
        return prediction
    
# class HierarchicalModel():
#     pass
        
def hash(key: str):
    return mmh3.hash(key)