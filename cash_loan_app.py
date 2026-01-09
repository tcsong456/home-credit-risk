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
        X['ext_min_credit'] = X['ext_min'] / X['AMT_CREDIT']
        
        X['down_payment_ratio'] = 1 - X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['goods_income'] = X['AMT_GOODS_PRICE'] / X['AMT_INCOME_TOTAL']
        
        ext_feat_cols = [col for col in X.columns if 'ext_' in col and col not in ext_cols]
        new_features = ['credit_income', 'annuity_income', 'annuity_credit', 'phone_is_missing', 'employment_status',
                        'days_birth', 'days_employed', 'days_registration', 'days_id_publish', 'days_last_phone_change',
                        'employed_ratio', 'reg_interaction_ratio', 'id_iteraction_ratio', 'down_payment_ratio', 'goods_income']
        new_features += ext_feat_cols
        return X[new_features]

def cur_app_features():
    return make_pipeline(make_union(BaseFeatures()))

if __name__ == '__main__':
    # train = pd.read_csv('data/application_train.csv')
    # test = pd.read_csv('data/application_test.csv')
    # pos_cash = pd.read_csv('data/POS_CASH_balance.csv', encoding="latin1")
    # cash_loan_train = train[train['NAME_CONTRACT_TYPE']=='Cash loans']
    
    # base_feats = base_features(cash_loan_train)
    # base_feats = BaseFeatures().fit_transform(cash_loan_train)
    x = cur_app_features()

#%%
x