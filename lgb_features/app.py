import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve
from sklearn.pipeline import make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# from utils.target_encoding import target_encoding_train, target_encoding_inference
from utils.transformers import ColumnSelector, FrequencyEncoding, OneHotEncoding, ConvertType, ConcatTexts

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
        
        contact_columns = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                           'FLAG_PHONE', 'FLAG_EMAIL']
        X['contact_mean'] = X[contact_columns].mean(axis=1)
        X['contact_sum'] = X[contact_columns].sum(axis=1)
        X['contact_std'] = X[contact_columns].std(axis=1)
        X['region_rating'] = X['REGION_RATING_CLIENT'] * X['REGION_RATING_CLIENT_W_CITY']
        
        reg_word_city_columns = ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                                 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']
        X['rwc_mean'] = X[reg_word_city_columns].mean(axis=1)
        X['rwc_sum'] = X[reg_word_city_columns].sum(axis=1)
        X['rwc_std'] = X[reg_word_city_columns].std(axis=1)
        
        avg_columns = [col for col in X.columns if col.endswith('_AVG')]
        X['avg_min'] = np.nanmin(X[avg_columns], axis=1, keepdims=True)
        X['avg_max'] = np.nanmax(X[avg_columns], axis=1, keepdims=True)
        X['avg_mean'] = np.nanmean(X[avg_columns], axis=1, keepdims=True)
        X['avg_median'] = np.nanmedian(X[avg_columns], axis=1, keepdims=True)
        X['avg_std'] = np.nanstd(X[avg_columns], axis=1, keepdims=True)
        X['avg_nan_ratio'] = pd.isnull(X[avg_columns]).sum(axis=1) / len(avg_columns)
        
        mode_columns = [col for col in X.columns if col.endswith('_MODE') if col not in ['FONDKAPREMONT_MODE',
                        'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']]
        X['mode_min'] = np.nanmin(X[mode_columns], axis=1, keepdims=True)
        X['mode_max'] = np.nanmax(X[mode_columns], axis=1, keepdims=True)
        X['mode_mean'] = np.nanmean(X[mode_columns], axis=1, keepdims=True)
        X['mode_median'] = np.nanmedian(X[mode_columns], axis=1, keepdims=True)
        X['mode_std'] = np.nanstd(X[mode_columns], axis=1, keepdims=True)
        X['mode_nan_ratio'] = pd.isnull(X[mode_columns]).sum(axis=1) / len(mode_columns)
        
        medi_columns = [col for col in X.columns if col.endswith('_MEDI')]
        X['medi_min'] = np.nanmin(X[medi_columns], axis=1, keepdims=True)
        X['medi_max'] = np.nanmax(X[medi_columns], axis=1, keepdims=True)
        X['medi_mean'] = np.nanmean(X[medi_columns], axis=1, keepdims=True)
        X['medi_median'] = np.nanmedian(X[medi_columns], axis=1, keepdims=True)
        X['medi_std'] = np.nanstd(X[medi_columns], axis=1, keepdims=True)
        X['medi_nan_ratio'] = pd.isnull(X[medi_columns]).sum(axis=1) / len(medi_columns)
        
        X['default_raio_friends30'] = X['DEF_30_CNT_SOCIAL_CIRCLE'] / (X['OBS_30_CNT_SOCIAL_CIRCLE'] + 1e-6)
        X['default_raio_friends60'] = X['DEF_60_CNT_SOCIAL_CIRCLE'] / (X['OBS_60_CNT_SOCIAL_CIRCLE'] + 1e-6)
        X['obs_trend'] = abs(X['OBS_60_CNT_SOCIAL_CIRCLE'] - X['OBS_30_CNT_SOCIAL_CIRCLE'])
        X['default_trend'] = abs(X['DEF_60_CNT_SOCIAL_CIRCLE'] - X['DEF_30_CNT_SOCIAL_CIRCLE'])
        X['obs_trend_ratio'] = abs(X['OBS_60_CNT_SOCIAL_CIRCLE'] - X['OBS_30_CNT_SOCIAL_CIRCLE']) / (X['OBS_30_CNT_SOCIAL_CIRCLE'] + 1e-6)
        X['default_trend_ratio'] = abs(X['DEF_60_CNT_SOCIAL_CIRCLE'] - X['DEF_30_CNT_SOCIAL_CIRCLE']) / (X['DEF_30_CNT_SOCIAL_CIRCLE'] + 1e-6)
        
        doc_columns = [col for col in X.columns if 'document' in col]
        X['doc_mean'] = X[doc_columns].mean(axis=1)
        X['doc_sum'] = X[doc_columns].sum(axis=1)
        X['doc_std'] = X[doc_columns].std(axis=1)
        
        bureau_columns = [col for col in X.columns if 'BURUEAU' in col]
        X['bureau_sum'] = X[bureau_columns].sum(axis=1)
        X['bureau_dist'] = X[bureau_columns] / X[['bureau_sum']]
        
        ext_feat_cols = [col for col in X.columns if 'ext_' in col and col not in ext_cols]
        new_features = ['credit_income', 'annuity_income', 'annuity_credit', 'phone_is_missing', 'employment_status',
                        'days_birth', 'days_employed', 'days_registration', 'days_id_publish', 'days_last_phone_change',
                        'days_diff_emp_reg', 'days_diff_emp_id', 'CNT_FAM_MEMBERS', 'rwc_mean', 'rwc_sum', 'rwc_std',
                        'employed_ratio', 'reg_interaction_ratio', 'id_iteraction_ratio', 'down_payment_ratio', 'goods_income',
                        'own_car_realty', 'cnt_children_bin', 'OWN_CAR_AGE', 'disposable_income_avg', 'pressure',
                        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_CHILDREN', 'REGION_POPULATION_RELATIVE',
                        'contact_mean', 'contact_sum', 'contact_std', 'region_rating', 'avg_min', 'avg_max', 'avg_mean',
                        'avg_median', 'avg_std', 'avg_nan_ratio', 'mode_min', 'mode_max', 'mode_mean', 'mode_median',
                        'mode_std', 'mode_nan_ratio', 'medi_min', 'medi_max', 'medi_mean', 'medi_median', 'obs_trend', 'default_trend',
                        'medi_std', 'medi_nan_ratio', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'default_raio_friends30', 'default_raio_friends60',
                        'obs_trend_ratio', 'default_trend_ratio', 'doc_mean', 'doc_sum', 'doc_std', 'bureau_sum', 'bureau_dist']
        new_features += ext_feat_cols
        new_features += bureau_columns
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

class HourBins(BaseEstimator, TransformerMixin):
    def _hour_bins(self, x):
        if x >= 0 and x < 6:
            return 'midnight'
        elif x >= 6 and x < 12:
            return 'morning'
        elif x >= 12 and x < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['hour_bins'] = X['HOUR_APPR_PROCESS_START'].map(self._hour_bins)
        return X

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
                  ColumnSelector(columns=['AMT_CREDIT', 'AMT_ANNUITY']),
                  CntPayment(apr=3.5)
                ),
            make_pipeline(
                    ColumnSelector(columns=['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']),
                    HourBins(),
                    ConcatTexts(['WEEKDAY_APPR_PROCESS_START', 'hour_bins'], out_col='concat_text', delimiter=' '),
                    FrequencyEncoding(min_cnt=100)
                )
            ),
            ConvertType(dtype='float32')
    )
    return pp

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    train['OCCUPATION_TYPE'] = train['OCCUPATION_TYPE'].fillna('unk')    
    test['OCCUPATION_TYPE'] = test['OCCUPATION_TYPE'].fillna('unk')
    
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