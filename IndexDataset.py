import numpy as np
from typing import Optional
import pandas as pd

class IndexDataset:
    def __init__(self, df: pd.DataFrame, col: str) -> None:
        self.df = df
        self.main = col
        self.sort_by_series(self.main)

    def __len__(self):
        return len(self.df[self.main])
    
    def __getitem__(self, idx):
        return self.df[self.main][idx], idx
    
    def sort_by_series(self, col: str):
        self.df.sort_values(by=col, inplace=True)
        self.df.drop_duplicates(subset=col, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
    
    def get_series(self, col: str = None):
        if col is None:
            col = self.main
        return self.df[col]
            
    def add_series(self, col: str, series: Optional[pd.Series] = np.nan):
        self.df[col] = series

    def remove_series(self):
        self.df.drop(columns=[self.col], inplace=True)
