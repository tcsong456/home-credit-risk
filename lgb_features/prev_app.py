import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve

def solve_interest_rate(P, A, n):
    if P <= 0 or A <= 0 or n <= 0:
        return np.nan
    
    def f(r):
        return P * (r * (1 + r)**n) / ((1 + r)**n - 1) - A
    
    try:
        r = fsolve(f, 0.03)[0]
        return r
    except:
        return np.nan

def aggregation(df, col):
    columns = ['SK_ID_CURR', col]
    d_0 = df[df['seq']==0][columns].set_index('SK_ID_CURR')
    d_3 = df[df['seq']<3].groupby(['SK_ID_CURR'])[col].agg(['min', 'max', 'mean', 'median','std'])
    d = df.groupby(['SK_ID_CURR'])[col].agg(['min', 'max', 'mean', 'median','std'])
    d = pd.concat([d_0, d_3, d], axis=1)
    return d

def build_features(df):
    active_status = df.groupby('SK_ID_CURR')['NAME_CONTRACT_TYPE'].count()
    active_days = df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].min() - df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].max()
    active_days = active_days.map(abs)
    active_days_per_loan = active_days / active_status
    first_loan = df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].min().map(abs)
    last_loan = df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].max().map(abs)
    z = df[df['NAME_CONTRACT_STATUS']=='Approved']
    days_since_last_approval = z.groupby(['SK_ID_CURR'])['DAYS_DECISION'].max()
    z = df[df['NAME_CONTRACT_STATUS']=='Refused']
    days_since_last_refusal = z.groupby(['SK_ID_CURR'])['DAYS_DECISION'].max()
    
    contract_status_0 = df[df['seq']==0].groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().fillna(0)
    contract_status_3 = df[df['seq']<3].groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])['ref'].sum().unstack().fillna(0)
    contract_status_3 = contract_status_3 / contract_status_3.sum(axis=1)[:, None]
    contract_status = df.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])['ref'].sum().unstack().fillna(0)
    contract_status = contract_status / contract_status.sum(axis=1)[:, None]
    
    approve = df[df['NAME_CONTRACT_STATUS'].isin(['Approved'])]
    appr_credit_app = approve.groupby(['SK_ID_CURR'])['credit_app_ratio'].agg(['min', 'max', 'mean', 'median','std'])
    appr_credit_income = approve.groupby(['SK_ID_CURR'])['credit_income_ratio'].agg(['min', 'max', 'mean', 'median','std'])
    ref = df[df['NAME_CONTRACT_STATUS'].isin(['Refused'])]
    ref_credit_app = ref.groupby(['SK_ID_CURR'])['credit_app_ratio'].agg(['min', 'max', 'mean', 'median','std'])
    ref_credit_income = ref.groupby(['SK_ID_CURR'])['credit_income_ratio'].agg(['min', 'max', 'mean', 'median','std'])

    f1 = prev_app.groupby(['SK_ID_CURR'])['RATE_INTEREST_PRIMARY'].mean()
    f2 = prev_app.groupby(['SK_ID_CURR'])['RATE_INTEREST_PRIVILEGED'].mean()
    f3 = prev_app.groupby(['SK_ID_CURR'])['f1'].mean()
    f4 = prev_app.groupby(['SK_ID_CURR'])['f2'].mean()
    
    non_revolving_loan = df[df['NAME_CONTRACT_TYPE'].isin(['Cash loans', 'Consumer loans'])]
    credit_sum = non_revolving_loan.groupby(['SK_ID_CURR'])['AMT_CREDIT'].sum()
    non_revolving_loan['credit_pct'] = non_revolving_loan["AMT_CREDIT"].rank(pct=True)
    credit_pct = non_revolving_loan.groupby(['SK_ID_CURR'])['credit_pct'].agg(['min', 'max', 'mean', 'median','std'])
    non_revolving_loan['credit_income_pct'] = non_revolving_loan["credit_income_ratio"].rank(pct=True)
    non_revolving_loan['credit_app_pct'] = non_revolving_loan["credit_app_ratio"].rank(pct=True)
    credit_income_pct = non_revolving_loan.groupby(['SK_ID_CURR'])['credit_income_pct'].agg(['min', 'max', 'mean', 'median','std'])
    credit_app_pct = non_revolving_loan.groupby(['SK_ID_CURR'])['credit_app_pct'].agg(['min', 'max', 'mean', 'median','std'])
    
    apply_freq = df.groupby('SK_ID_CURR')['apply_interval'].agg(['min', 'max', 'mean', 'median','std'])
    loan_type = df.groupby(['SK_ID_CURR', 'NAME_CONTRACT_TYPE'])['ref'].sum().unstack().fillna(0)
    loan_purpose = df.groupby(['SK_ID_CURR', 'NAME_CASH_LOAN_PURPOSE'])['ref'].sum().unstack().fillna(0)
    reject_reason = df.groupby(['SK_ID_CURR', 'CODE_REJECT_REASON'])['ref'].sum().unstack().fillna(0)
    
    app_credit = aggregation(df, 'amt_credit_diff')
    app_credit_ratio = aggregation(df, 'amt_credit_ratio')
    interest_portion = aggregation(df, 'interest_ratio')
    interest_rate = aggregation(df, 'interest_rate')
    down_payment_ratio = aggregation(df, 'down_payment_ratio')
    annuity_income_ratio = aggregation(df, 'annuity_income_ratio')
    credit_income_ratio = aggregation(df, 'credit_income_ratio')
    
    x = pd.concat([active_status, active_days, active_days_per_loan, last_loan, contract_status_0, contract_status_3, contract_status,
                   app_credit, app_credit_ratio, interest_portion, interest_rate, down_payment_ratio, annuity_income_ratio,
                   first_loan, credit_income_ratio, f1, f2, f3, f4, days_since_last_approval, days_since_last_refusal,
                   appr_credit_app, ref_credit_app, appr_credit_income, ref_credit_income, credit_sum, credit_pct, credit_income_pct,
                   credit_app_pct, apply_freq, loan_type, loan_purpose, reject_reason], axis=1)
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
    
    prev_app['interest_ratio'] = (prev_app['AMT_ANNUITY'] * prev_app['CNT_PAYMENT'] - prev_app['AMT_CREDIT']) / prev_app['AMT_CREDIT']
    prev_app.loc[prev_app['CNT_PAYMENT']==0, 'interest_ratio'] = np.nan
    prev_app['down_payment_ratio'] = prev_app['AMT_DOWN_PAYMENT'] / prev_app['AMT_GOODS_PRICE']
    
    interest_rates = []
    z = prev_app[['AMT_ANNUITY', 'AMT_CREDIT', 'CNT_PAYMENT']].to_numpy()
    for P, A, n in tqdm(z, total=z.shape[0], desc='calculating interest rate'):
        interest_rates.append(solve_interest_rate(P, A, n))
    prev_app['interest_rate'] = interest_rates
    prev_app.loc[prev_app['CNT_PAYMENT']==0, 'interest_rate'] = np.nan
    
    prev_app = prev_app.merge(train[['SK_ID_CURR', 'AMT_INCOME_TOTAL']], how='left', on=['SK_ID_CURR'])
    prev_app['annuity_income_ratio'] = prev_app['AMT_ANNUITY'] / prev_app['AMT_INCOME_TOTAL']
    prev_app['credit_income_ratio'] = prev_app['AMT_CREDIT'] / prev_app['AMT_INCOME_TOTAL']
    prev_app['credit_app_ratio'] = prev_app['AMT_CREDIT'] / prev_app['AMT_APPLICATION']
    
    prev_app['f1'] = prev_app['RATE_INTEREST_PRIVILEGED'] - prev_app['RATE_INTEREST_PRIMARY']
    prev_app['f2'] = prev_app['RATE_INTEREST_PRIVILEGED'] / prev_app['RATE_INTEREST_PRIMARY']
    
    prev_app['apply_interval'] = prev_app.groupby(['SK_ID_CURR'])['DAYS_DECISION'].diff()
    
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


b = prev_app[~pd.isnull(prev_app['RATE_INTEREST_PRIMARY'])]



#%%
# z = prev_app[prev_app['NAME_CONTRACT_TYPE'].isin(['Cash loans', 'Consumer loans'])]
prev_app['CODE_REJECT_REASON'].value_counts()


    




