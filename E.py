from IndexDataset import IndexDataset
from torch import nn
import pandas as pd

class BaselineEmbed:
    def __init__(self, data: IndexDataset, embed_size: int, model: nn.Module):
        '''
        data: string information
        hash: function to hash data
        '''
        self.data = data
        self.embed = nn.Embedding(data.__len__, embed_size)
        self.model = model
        self.sort('embed')

    def generate_embeds(self, col: str):
        series = self.data.get_series()
        embed_series = pd.Series([''] * len(series))

        for i in range(len(series)):
            embed_series[i] = self.embed(i)
        
        self.data.add_series(col, embed_series)

    def sort(self, col: str):
        self.generate_embeds(col)
        self.data.sort_by_series(col)

    def predict(self, key: str):
        digest = self.hash(key)
        prediction = self.model(nn.Tensor(digest))
        
        return prediction