import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve
from sklearn.pipeline import make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from utils.target_encoding import target_encoding_train, target_encoding_inference
from utils.transformers import ColumnSelector, FrequencyEncoding, OneHotEncoding, ConvertType

def map_cnt_children_bin(x):
    if x == 0:
        return 'low'
    elif x == 1:
        return 'mid'
    else:
        return 'high'

def emp_stats(x):
    if x < 2.8:
        return 'short'
    elif x >= 2.8 and x < 7:
        return 'medium'
    elif x >= 7 and x < 50:
        return 'long'
    else:
        return 'retired'

class BaseFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, *args):
        return self
    
    def _map_cnt_children_bin(self, x):
        if x == 0:
            return 0
        elif x == 1:
            return 1
        else:
            return 2
    
    def transform(self, X):
        X = X.copy()
        X['credit_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['annuity_income'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['annuity_credit'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_is_missing'] = (X['DAYS_LAST_PHONE_CHANGE'] == 0).apply(int)
        X['employment_status'] = (X['DAYS_EMPLOYED'] == 365243).apply(int)
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['DAYS_DIFF_EMP_REG'] = X['DAYS_REGISTRATION'] - X['DAYS_EMPLOYED']
        X['DAYS_DIFF_EMP_ID'] = X['DAYS_ID_PUBLISH'] - X['DAYS_EMPLOYED']
        
        days_normaliztion_col = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE',
                                 'DAYS_DIFF_EMP_REG', 'DAYS_DIFF_EMP_ID']
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
        X['cnt_children_bin'] = X['CNT_CHILDREN'].map(self._map_cnt_children_bin)
        
        X['disposable_income_avg'] = X['AMT_INCOME_TOTAL'] / (X['CNT_CHILDREN'] + 1)
        X['pressure'] = X['AMT_CREDIT'] / (X['disposable_income_avg'] + 1e-6)
        
        ext_feat_cols = [col for col in X.columns if 'ext_' in col and col not in ext_cols]
        new_features = ['credit_income', 'annuity_income', 'annuity_credit', 'phone_is_missing', 'employment_status',
                        'days_birth', 'days_employed', 'days_registration', 'days_id_publish', 'days_last_phone_change',
                        'days_diff_emp_reg', 'days_diff_emp_id',
                        'employed_ratio', 'reg_interaction_ratio', 'id_iteraction_ratio', 'down_payment_ratio', 'goods_income',
                        'own_car_realty', 'cnt_children_bin', 'OWN_CAR_AGE', 'disposable_income_avg', 'pressure',
                        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_CHILDREN', 'REGION_POPULATION_RELATIVE']
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

class CntPayment(BaseEstimator, TransformerMixin):
    def __init__(self, apr):
        self.apr = apr
    
    def _calculate_cnt_payment(self, amt_credit, amt_annuity, apr):
        if np.isnan(amt_annuity):
            return np.nan
        r = apr / 100 / 12

        min_payment = amt_credit * r / (1 + r)
        if amt_annuity <= min_payment:
            return np.nan

        def equation(n):
            return amt_annuity - (amt_credit * r * (1 + r)**n) / ((1 + r)**n - 1)

        initial_guess = 12
        n_solution = fsolve(equation, initial_guess)[0]

        return int(np.round(n_solution))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cnt_payments = []
        for a, b in tqdm(X[['AMT_CREDIT', 'AMT_ANNUITY']].to_numpy(), total=X.shape[0],
                         desc='calculating total cnt_payments'):
            r = self._calculate_cnt_payment(a, b, self.apr)
            cnt_payments.append(r)
        cnt_payments = np.array(cnt_payments)[:, None]
        return cnt_payments

def cur_app_features():
    pp = make_pipeline(
            make_union(
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
                    ColumnSelector(['NAME_EDUCATION_TYPE']),
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
                ),
            make_pipeline(
                  ColumnSelector(columns=['employ_years']),
                  OneHotEncoding()
                ),
            make_pipeline(
                  ColumnSelector(columns=['AMT_CREDIT', 'AMT_ANNUITY']),
                  CntPayment(apr=3.5)
                ),
            ),
            ConvertType(dtype='float32')
    )
    return pp

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    train['OCCUPATION_TYPE'] = train['OCCUPATION_TYPE'].fillna('unk')
    train['cnt_chilren_bin'] = train['CNT_CHILDREN'].map(map_cnt_children_bin)
    employ_years = abs(train.loc[train['DAYS_EMPLOYED'] != 365243, 'DAYS_EMPLOYED'] / 365)
    train.loc[employ_years.index, 'employ_years'] = employ_years
    employ_years = train['employ_years'].fillna(51)
    train['employ_years'] = employ_years.map(emp_stats)
    
    test['OCCUPATION_TYPE'] = test['OCCUPATION_TYPE'].fillna('unk')
    test['cnt_chilren_bin'] = test['CNT_CHILDREN'].map(map_cnt_children_bin)
    employ_years = abs(test.loc[test['DAYS_EMPLOYED'] != 365243, 'DAYS_EMPLOYED'] / 365)
    test.loc[employ_years.index, 'employ_years'] = employ_years
    employ_years = test['employ_years'].fillna(51)
    test['employ_years'] = employ_years.map(emp_stats)
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    X_tr = train[~train['SK_ID_CURR'].isin(val_currs)].reset_index(drop=True)
    X_val = train[train['SK_ID_CURR'].isin(val_currs)].reset_index(drop=True)
    X_te = test.copy()
    
    pipeline = cur_app_features()
    x_train = pipeline.fit_transform(X_tr)
    x_val = pipeline.transform(X_val)
    x_test = pipeline.transform(X_te)
    
    x_train = np.concatenate([X_tr[['SK_ID_CURR']].to_numpy().astype(np.float32), 
                              X_tr[['TARGET']].to_numpy().astype(np.float32), x_train], axis=1)
    
    x_val = np.concatenate([X_val[['SK_ID_CURR']].to_numpy().astype(np.float32), 
                            X_val[['TARGET']].to_numpy().astype(np.float32), x_val], axis=1)
    
    np.save('artifacts/train/app_features.npy', x_train)
    np.save('artifacts/validation/app_features.npy', x_val)
    np.save('artifacts/test/app_features.npy', x_test)

#%%
column_desc = pd.read_csv('data/HomeCredit_columns_description.csv', encoding="latin1")

#%%
(train['AMT_CREDIT'] / train['AMT_ANNUITY']).median()
    
    