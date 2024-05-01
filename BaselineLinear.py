import mmh3
from DataArray import DataArray
import pandas as pd

'''
Baseline Linear Model
- Take in data
- Hash data
- Sort data by hash
- Linear model to predict index
'''

class BaselineLinear:
    def __init__(self, data: DataArray, hash):
        '''
        data: string information
        hash: function to hash data
        '''
        self.data = data
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
        self.data.sort_by(col)
        
def hash(self, key: str):
    return mmh3.hash(key)