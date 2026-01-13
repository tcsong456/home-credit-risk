import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def build_features(df):
    active_status = df.groupby('SK_ID_CURR')['NAME_CONTRACT_TYPE'].count()
    active_days_per_loan = (df.groupby('SK_ID_CURR')['DAYS_DECISION'].min().map(abs) + 1) / (active_status + 1)
    last_loan = df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].max().map(abs)
    
    contract_status_0 = df[df['seq']==0].groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().fillna(0)
    contract_status_3 = df[df['seq']<3].groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])['ref'].sum().unstack().fillna(0)
    contract_status_3 = contract_status_3 / contract_status_3.sum(axis=1)[:, None]
    contract_status = df.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])['ref'].sum().unstack().fillna(0)
    contract_status = contract_status / contract_status.sum(axis=1)[:, None]
    
    app_credit_0 = df[df['seq']==0][['SK_ID_CURR','amt_credit_diff', 'amt_credit_ratio']].set_index('SK_ID_CURR')
    app_credit_3 = df[df['seq']<3].groupby(['SK_ID_CURR'])['amt_credit_diff'].agg(['min', 'max', 'mean', 'median','std'])
    app_credit = df.groupby(['SK_ID_CURR'])['amt_credit_diff'].agg(['min', 'max', 'mean', 'median','std'])
    app_credit_ratio_3 = df[df['seq']<3].groupby(['SK_ID_CURR'])['amt_credit_ratio'].agg(['min', 'max', 'mean', 'median','std'])
    app_credit_ratio = df.groupby(['SK_ID_CURR'])['amt_credit_ratio'].agg(['min', 'max', 'mean', 'median','std'])
    
    x = pd.concat([active_status, active_days_per_loan, last_loan, contract_status_0, contract_status_3, contract_status,
                   app_credit_0, app_credit_3, app_credit, app_credit_ratio_3, app_credit_ratio], axis=1)
    return x

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    prev_app = prev_app.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    sorted_prev = prev_app.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, True])
    sorted_prev = sorted_prev.groupby("SK_ID_CURR").cumcount() + 1
    prev_app['max_order'] = prev_app.groupby(['SK_ID_CURR'])['SK_ID_CURR'].transform('size')
    prev_app = pd.concat([prev_app, sorted_prev], axis=1).rename(columns={0: 'seq_order'})
    prev_app['seq'] = prev_app['max_order'] - prev_app['seq_order']
    del prev_app['seq_order'], prev_app['max_order']
    
    prev_app['ref'] = 1
    prev_app['amt_credit_diff'] = prev_app['AMT_CREDIT'] - prev_app['AMT_APPLICATION']
    prev_app['amt_credit_ratio'] = prev_app['amt_credit_diff'] / prev_app['AMT_APPLICATION']
    prev_app['amt_credit_diff'] = np.log1p(prev_app['amt_credit_diff'])
    
    x = build_features(prev_app)
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = x[x.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = x[x.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = x[x.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/prev_app_features.npy', x_train)
    np.save('artifacts/validation/prev_app_features.npy', x_val)
    np.save('artifacts/test/prev_app_features.npy', x_test)


#%%
# len(set(prev_app['SK_ID_CURR']) & set(test['SK_ID_CURR']))
# prev_app.groupby('SK_ID_CURR')['NAME_CONTRACT_TYPE'].count().std()
# prev_app['NAME_CONTRACT_TYPE'].isnull().sum()
# z['total_interest'] = z['AMT_ANNUITY'] * z['CNT_PAYMENT'] - z['AMT_CREDIT']
# z['interest_per_month'] = z['total_interest'] / z['CNT_PAYMENT']
# z['interest_rate'] = z['interest_per_month'] / z['AMT_ANNUITY']
# 32696.1 * 48






#%%
# from tqdm import tqdm
# from scipy.optimize import fsolve
# def solve_interest_rate(P, A, n):
#     if P <= 0 or A <= 0 or n <= 0:
#         return np.nan
    
#     def f(r):
#         return P * (r * (1 + r)**n) / ((1 + r)**n - 1) - A
    
#     try:
#         r = fsolve(f, 0.03)[0]
#         return r
#     except:
#         return np.nan
# interest_rates = []
# z = prev_app[~(prev_app["AMT_ANNUITY"] * prev_app["CNT_PAYMENT"] < prev_app["AMT_CREDIT"])]
# z = z[['AMT_ANNUITY', 'AMT_CREDIT', 'CNT_PAYMENT']].to_numpy()
# for P, A, n in tqdm(z, total=z.shape[0], desc='calculating interest rate'):
#     interest_rates.append(solve_interest_rate(P, A, n))




