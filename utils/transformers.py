import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 columns,
                 fillna=False):
        self.columns = columns
        self.fillna = fillna
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            raise KeyError('input must be a pandas dataframe')
        
        columns = X.columns.tolist()
        for c in self.columns:
            if c not in columns:
                raise KeyError(f'{c} not in {columns}')
            if self.fillna:
                X[c] = X[c].fillna('unk')
                
        return X[self.columns]

class ConcatTexts(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns,
                 out_col,
                 delimiter=','):
        self.columns = columns
        self.out_col = out_col
        self.delimiter = delimiter
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.out_col] = ''
        columns = X.columns.tolist()
        for c in self.columns:
            if c not in columns:
                raise KeyError(f'{c} not in {columns}')
            X[c] = X[c].fillna('')
            X[self.out_col] = X[self.out_col] + self.delimiter + X[c]
        return X[self.out_col]

class FrequencyEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, min_cnt=10):
        self.min_cnt = min_cnt
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise KeyError('input X must be a dataframe or series')
        self.column = X.columns[0]
        X = X.fillna('unk')
        freq_cnt = X.value_counts()
        index = freq_cnt.index.to_numpy()
        mask = freq_cnt < self.min_cnt
        invalid_index = index[mask.tolist()]
        for c in invalid_index:
            X[X==c] = 'rare'
        
        self.freq_dict = {}
        for k, v in X.value_counts().reset_index().to_numpy():
            self.freq_dict[k] = v
        return self
    
    def _get_freq(self, x):
        return self.freq_dict.get(x[self.column], 0)
    
    def transform(self, X):
        fe = X.apply(self._get_freq, axis=1)
        if len(fe.shape) == 1:
            return fe[:, None]
        return fe

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe.fit(X)
        return self
    
    def transform(self, X):
        X = X.copy()
        ohe_data = self.ohe.transform(X)
        return ohe_data

#%%