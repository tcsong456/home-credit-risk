import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.pipeline import make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from utils.transformers import ColumnSelector, FrequencyEncoding, OneHotEncoding
from utils.target_encoding import target_encoding_train, target_encoding_inference

class BaseFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, *args):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['credit_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['annuity_income'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['annuity_credit'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_is_missing'] = (X['DAYS_LAST_PHONE_CHANGE'] == 0).apply(int)
        X['employment_status'] = (X['DAYS_EMPLOYED'] == 365243).apply(int)
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        
        days_normaliztion_col = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']
        for col in days_normaliztion_col:
            X[col.lower()] = abs(X[col] / 365)
        
        X['employed_ratio'] = X['days_employed'] / X['days_birth']
        X['reg_interaction_ratio'] = X['days_registration'] / X['days_birth']
        X['id_iteraction_ratio'] = X['days_id_publish'] / X['days_birth']
        
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        X['ext_source_missing'] =  X[ext_cols].isnull().sum(axis=1)
        X['ext_mean'] = X[ext_cols].mean(axis=1)
        X['ext_median'] = X[ext_cols].median(axis=1)
        X['ext_min'] = X[ext_cols].min(axis=1)
        X['ext_max'] = X[ext_cols].max(axis=1)
        X['ext_std'] = X[ext_cols].std(axis=1)
        X['ext_range'] = X['ext_max'] - X['ext_min']
        X['ext_var'] = X[ext_cols].var(axis=1)
        for c in ext_cols:
            X[f'{c.lower()}_rank'] = X[c].rank(pct=True)
        X['ext12_diff'] = X['EXT_SOURCE_1'] - X['EXT_SOURCE_2']
        X['ext23_diff'] = X['EXT_SOURCE_2'] - X['EXT_SOURCE_3']
        X['ext13_diff'] = X['EXT_SOURCE_1'] - X['EXT_SOURCE_3']
        X['ext12_prod'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2']
        X['ext23_prod'] = X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
        X['ext13_prod'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_3']
        X['ext_mean_bin'] = pd.qcut(X['ext_mean'], 20, labels=False)
        X['ext_mean_income'] = X['ext_mean'] / X['AMT_INCOME_TOTAL']
        X['ext_mean_credit'] = X['ext_mean'] / X['AMT_CREDIT']
        
        X['down_payment_ratio'] = 1 - X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['goods_income'] = X['AMT_GOODS_PRICE'] / X['AMT_INCOME_TOTAL']
        
        X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
        X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
        X['own_car_realty'] = X['FLAG_OWN_CAR'] * 2 + X['FLAG_OWN_REALTY']
        
        ext_feat_cols = [col for col in X.columns if 'ext_' in col and col not in ext_cols]
        new_features = ['credit_income', 'annuity_income', 'annuity_credit', 'phone_is_missing', 'employment_status',
                        'days_birth', 'days_employed', 'days_registration', 'days_id_publish', 'days_last_phone_change',
                        'employed_ratio', 'reg_interaction_ratio', 'id_iteraction_ratio', 'down_payment_ratio', 'goods_income',
                        'own_car_realty']
        new_features += ext_feat_cols
        return X[new_features]

class GroupPercentileFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,
                 group_col,
                 ):
        self.group_col = group_col
    
    def fit(self, X, *args):
        X = X.copy()
        X['credit_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['annuity_income'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['annuity_credit'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['employment_length'] = abs(X['DAYS_EMPLOYED'] / 365)
        X['age'] = abs(X['DAYS_BIRTH'] / 365)
        
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        X['ext_mean'] = X[ext_cols].mean(axis=1)
        X['ext_mean_income'] = X['ext_mean'] / X['AMT_INCOME_TOTAL']
        X['ext_mean_credit'] = X['ext_mean'] / X['AMT_CREDIT']
        
        self.group_sorted, self.global_sorted = {}, {}
        self.num_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'credit_income', 'annuity_income', 'annuity_credit',
                         'employment_length', 'age', 'ext_mean_income', 'ext_mean_credit']
        for col in self.num_cols:
            v = X[col]
            mask = ~pd.isnull(v)
            self.global_sorted[col] = np.sort(v.loc[mask])
            
            feat_map = {}
            for grp, idx in X.loc[mask].groupby(self.group_col).groups.items():
                group_v = v.loc[idx]
                feat_map[grp] = np.sort(group_v)
            self.group_sorted[col] = feat_map
        return self
    
    def transform(self, X):
        X = X.copy()
        X['credit_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['annuity_income'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['annuity_credit'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['employment_length'] = abs(X['DAYS_EMPLOYED'] / 365)
        X['age'] = abs(X['DAYS_BIRTH'] / 365)
        
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        X['ext_mean'] = X[ext_cols].mean(axis=1)
        X['ext_mean_income'] = X['ext_mean'] / X['AMT_INCOME_TOTAL']
        X['ext_mean_credit'] = X['ext_mean'] / X['AMT_CREDIT']
        
        output = []
        for col in self.num_cols:
            v = X[col]
            pct = np.full(v.shape[0], np.nan)
            for grp, idx in X.groupby(self.group_col).groups.items():
                val = v.loc[idx]
                dist = self.group_sorted.get(col, {}).get(grp, None)
                if dist is None or dist.size == 0:
                    dist = self.global_sorted.get(col, np.array([]))
                if dist.size == 0:
                    continue
                
                mask = pd.isnull(val)
                if np.all(mask):
                    continue
                
                vv = val.loc[~mask]
                rank = np.searchsorted(dist, vv, side="right")
                p = rank / dist.shape[0]
                pct[idx[~mask]] = p
            output.append(pct)
        output = np.stack(output, axis=1)
        return output

def cur_app_features():
    pp = make_union(
        BaseFeatures(),
        GroupPercentileFeatures('OCCUPATION_TYPE'),
        GroupPercentileFeatures('NAME_EDUCATION_TYPE'),
        make_pipeline(
                ColumnSelector(['NAME_TYPE_SUITE']),
                FrequencyEncoding(min_cnt=50)
            ),
        make_pipeline(
                ColumnSelector(['NAME_INCOME_TYPE']),
                FrequencyEncoding(min_cnt=20)
            ),
        make_pipeline(
                ColumnSelector(['NAME_FAMILY_STATUS']),
                FrequencyEncoding(min_cnt=100)
            ),
        make_pipeline(
                ColumnSelector(['NAME_HOUSING_TYPE']),
                FrequencyEncoding(min_cnt=100)
            ),
        make_pipeline(
                ColumnSelector(['OCCUPATION_TYPE']),
                FrequencyEncoding(min_cnt=100)
            ),
        make_pipeline(
                ColumnSelector(['ORGANIZATION_TYPE']),
                FrequencyEncoding(min_cnt=20)
            ),
        make_pipeline(
             ColumnSelector(columns=['CODE_GENDER']),
             OneHotEncoding()
            ),
        make_pipeline(
             ColumnSelector(columns=['FLAG_OWN_CAR']),
             OneHotEncoding()
            ),
        make_pipeline(
             ColumnSelector(columns=['FLAG_OWN_REALTY']),
             OneHotEncoding()
            ),
        make_pipeline(
             ColumnSelector(columns=['NAME_CONTRACT_TYPE']),
             OneHotEncoding()
            )
        )
    return pp

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    val_currs = np.load('artifacts/val_sk_currs.npy')
    X_tr = train[~train['SK_ID_CURR'].isin(val_currs)].reset_index(drop=True)
    X_val = train[train['SK_ID_CURR'].isin(val_currs)].reset_index(drop=True)
    X_te = test.copy()
    
    
    pipeline = cur_app_features()
    x_train = pipeline.fit_transform(X_tr)
    x_val = pipeline.transform(X_val)
    x_test = pipeline.transform(X_te)
    
    X_tr['OCCUPATION_TYPE'] = X_tr['OCCUPATION_TYPE'].fillna('unk')
    te1_tr = target_encoding_train(X_tr, columns=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], alpha=50)
    te2_tr = target_encoding_train(X_tr, columns=['CODE_GENDER', 'NAME_FAMILY_STATUS'], alpha=50)
    te3_tr = target_encoding_train(X_tr, columns=['CODE_GENDER', 'OCCUPATION_TYPE'], alpha=50)
    x_train = np.concatenate([x_train, te1_tr, te2_tr, te3_tr], axis=1)
    
    X_val['OCCUPATION_TYPE'] = X_val['OCCUPATION_TYPE'].fillna('unk')
    te1_val = target_encoding_inference(X_tr, X_val, columns=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], alpha=50)
    te2_val = target_encoding_inference(X_tr, X_val, columns=['CODE_GENDER', 'NAME_FAMILY_STATUS'], alpha=50)
    te3_val = target_encoding_inference(X_tr, X_val, columns=['CODE_GENDER', 'OCCUPATION_TYPE'], alpha=50)
    x_val = np.concatenate([x_val, te1_val, te2_val, te3_val], axis=1)
    
    X_te['OCCUPATION_TYPE'] = X_te['OCCUPATION_TYPE'].fillna('unk')
    te1_te = target_encoding_inference(train, X_te, columns=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], alpha=50)
    te2_te = target_encoding_inference(train, X_te, columns=['CODE_GENDER', 'NAME_FAMILY_STATUS'], alpha=50)
    te3_te = target_encoding_inference(train, X_te, columns=['CODE_GENDER', 'OCCUPATION_TYPE'], alpha=50)
    x_test = np.concatenate([x_test, te1_te, te2_te, te3_te], axis=1)
    
    

#%%
train['CNT_CHILDREN'].unique()


    
    
    