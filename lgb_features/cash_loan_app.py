import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.pipeline import make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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
        
        ext_feat_cols = [col for col in X.columns if 'ext_' in col and col not in ext_cols]
        new_features = ['credit_income', 'annuity_income', 'annuity_credit', 'phone_is_missing', 'employment_status',
                        'days_birth', 'days_employed', 'days_registration', 'days_id_publish', 'days_last_phone_change',
                        'employed_ratio', 'reg_interaction_ratio', 'id_iteraction_ratio', 'down_payment_ratio', 'goods_income']
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
        
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        X['ext_mean'] = X[ext_cols].mean(axis=1)
        X['ext_mean_income'] = X['ext_mean'] / X['AMT_INCOME_TOTAL']
        X['ext_mean_credit'] = X['ext_mean'] / X['AMT_CREDIT']
        
        self.group_sorted, self.global_sorted = {}, {}
        self.num_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'credit_income', 'annuity_income', 'annuity_credit',
                         'ext_mean_income', 'ext_mean_credit']
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
        GroupPercentileFeatures('NAME_INCOME_TYPE'),
        )
    return pp

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    # test = pd.read_csv('data/application_test.csv')
    # pos_cash = pd.read_csv('data/POS_CASH_balance.csv', encoding="latin1")
    # cash_loan_train = train[train['NAME_CONTRACT_TYPE']=='Cash loans'].reset_index(drop=True)
    
    # base_feats = base_features(cash_loan_train)
    # base_feats = BaseFeatures().fit_transform(cash_loan_train)
    # pipeline = cur_app_features()
    # x = pipeline.fit_transform(cash_loan_train)

#%%
# v = cash_loan_train['AMT_INCOME_TOTAL']
# for grp, idx in cash_loan_train.groupby('NAME_INCOME_TYPE').groups.items():
#     print(np.sort(v.loc[idx]))
# z = cash_loan_train[cash_loan_train['NAME_INCOME_TYPE']=='Pensioner']
train[train['NAME_EDUCATION_TYPE'] == 'Academic degree']
    
    
    
    