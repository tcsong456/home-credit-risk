import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def build_cur_app_features(df):
    features = pd.DataFrame()
    features['credit_income'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    features['annuity_income'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    features['annuity_credit'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    df['phone_is_missing'] = (df['DAYS_LAST_PHONE_CHANGE'] == 0).apply(int)
    df['employment_status'] = (df['DAYS_EMPLOYED'] == 365243).apply(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    days_normaliztion_col = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']
    for col in days_normaliztion_col:
        features[col.lower()] = abs(df[col] / 365)
    features['employed_ratio'] = features['days_employed'] / features['days_birth']
    features['reg_interaction_ratio'] = features['days_registration'] / features['days_birth']
    features['id_iteraction_ratio'] = features['days_id_publish'] / features['days_birth']
    features['employment_status'] = df['employment_status']
    features['phone_is_missing'] = df['phone_is_missing']
    
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['ext_source_missing'] =  df[ext_cols].isnull().sum(axis=1)
    df['ext_mean'] = df[ext_cols].mean(axis=1)
    df['ext_median'] = df[ext_cols].median(axis=1)
    df['ext_min'] = df[ext_cols].min(axis=1)
    df['ext_max'] = df[ext_cols].max(axis=1)
    df['ext_std'] = df[ext_cols].std(axis=1)
    df['ext_range'] = df['ext_max'] - df['ext_min']
    df['ext_var'] = df[ext_cols].var(axis=1)
    for c in ext_cols:
        df[f'{c.lower()}_rank'] = df[c].rank(pct=True)
    df['ext12_diff'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
    df['ext23_diff'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
    df['ext13_diff'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_3']
    df['ext12_prod'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['ext23_prod'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['ext13_prod'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['ext_mean_bin'] = pd.qcut(df['ext_mean'], 20, labels=False)
    df['ext_mean_income'] = df['ext_mean'] / df['AMT_INCOME_TOTAL']
    df['ext_min_credit'] = df['ext_min'] / df['AMT_CREDIT']
    ext_feat_cols = [col for col in df.columns if 'ext_' in col and col not in ext_cols]
    features[ext_feat_cols] = df[ext_feat_cols]
    
    df['down_payment_ratio'] = 1 - df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['goods_income'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    features['down_payment_ratio'] = df['down_payment_ratio']
    features['goods_income'] = df['goods_income']
    
    return features

if __name__ == '__main__':
    # train = pd.read_csv('data/application_train.csv')
    # test = pd.read_csv('data/application_test.csv')
    # pos_cash = pd.read_csv('data/POS_CASH_balance.csv', encoding="latin1")
    # cash_loan_train = train[train['NAME_CONTRACT_TYPE']=='Cash loans']
    
    feats = build_cur_app_features(cash_loan_train)

#%%
# desc_column = pd.read_csv('data/HomeCredit_columns_description.csv', encoding="latin1")
# prev_app = prev_app[prev_app['NAME_CONTRACT_TYPE']=='Cash loans']
# train['NAME_CONTRACT_TYPE'].isnull().sum()
# np.nanmax(interest_rates)
# for col in feats.columns:
#     print(f'{col} nan: {np.isinf(feats[col]).sum()}')
train['NAME_TYPE_SUITE'].isnull().sum()