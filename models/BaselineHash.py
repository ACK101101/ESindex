from IndexDataset import IndexDataset
from IndexPredictor import IndexPredictor

import mmh3
import pandas as pd
from torch import nn

'''
Baseline Linear Model
- Take in data
- Hash data
- Sort data by hash
- Linear model to predict index of randomized digests
'''

class BaselineHash(IndexPredictor):
    def __init__(self, data: IndexDataset, model: nn.Module, hash: callable):
        '''
        data: string information
        hash: function to hash data
        '''
        super(data, model)
        self.hash = hash
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

    # Quaterny Search Implementation
    def last_mile_search(self, predicition: int):
        
        pass

    def predict(self, key: str) -> int:
        digest = self.hash(key)
        prediction = self.model(nn.Tensor(digest)).item()
        
        return prediction
    
# class HierarchicalModel():
#     pass
        
def hash(key: str):
    return mmh3.hash(key)