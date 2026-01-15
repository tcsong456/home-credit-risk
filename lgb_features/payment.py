import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def build_features(data, group_feature, prefix=''):
    features = []
    missing_pay_cnt = data[~data['is_delay']].groupby([group_feature])['missing_pay'].sum()
    missing_pay_ratio = missing_pay_cnt / data[~data['is_delay']].groupby([group_feature]).size()
    features += [missing_pay_cnt, missing_pay_ratio]
    
    amt_instal = data.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['AMT_INSTALMENT'].max().reset_index()
    toatl_instal = amt_instal.groupby(group_feature)['AMT_INSTALMENT'].sum()
    total_instal_cnt = data[~data['is_delay']].groupby(group_feature)['delay_money'].sum()
    total_instal_ratio =  total_instal_cnt/ toatl_instal
    features += [total_instal_cnt, total_instal_ratio]
    
    missing_payment_stats = data[(data['missing_pay']) & (~data['is_delay'])].groupby(group_feature)['delay_money'].agg(['min', 'max', 'mean', 'median', 'std'])
    features += [missing_payment_stats]
    
    actual_paid = data.groupby(group_feature)['AMT_PAYMENT'].sum()
    final_balance = actual_paid - toatl_instal
    final_balance[np.abs(final_balance) < 0.01] = 0
    features += [actual_paid, final_balance]
    
    data['breach_both'] = data['is_delay'] & data['missing_pay']
    severe_breach_cnt = data.groupby(group_feature)['breach_both'].sum()
    severe_breach_ratio = severe_breach_cnt / data.groupby([group_feature]).size()
    features += [severe_breach_cnt, severe_breach_ratio]
    
    cnt_payment = amt_instal.groupby(group_feature).size()
    features += [cnt_payment]
    
    columns = ['missing_pay_cnt', 'missing_pay_ratio', 'total_instal_cnt', 'total_instal_ratio']
    agg_col = ['min', 'max', 'mean', 'median', 'std']
    agg_col = [prefix+'_'+col for col in agg_col]
    columns += agg_col
    columns += ['total_paid', 'final_balance', 'severe_breach_cnt', 'severe_breach_ratio', 'payment_terms']
    
    features = pd.concat(features, axis=1)
    features.columns = columns
    return features, columns

if __name__ == '__main__':
    instal = pd.read_csv('data/installments_payments.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    
    instal = instal.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=(True, True, True))
    instal['delay_days'] = instal['DAYS_ENTRY_PAYMENT'] - instal['DAYS_INSTALMENT']
    instal['delay_money'] = instal['AMT_PAYMENT'] - instal['AMT_INSTALMENT']
    instal['is_delay'] = instal['delay_days'] > 0
    instal['missing_pay'] = instal['delay_money'] < 0
    instal['delay_money'] = instal['delay_money'].map(abs)
    
    x_prev, cols = build_features(instal, 'SK_ID_PREV', prefix='prev_')
    x_curr, _ = build_features(instal, 'SK_ID_CURR', prefix='curr_')
    
    x = prev_app[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'DAYS_DECISION', 'AMT_CREDIT', 'CNT_PAYMENT']].merge(x_prev, how='left', on=['SK_ID_PREV'])
    x.loc[x['NAME_CONTRACT_TYPE']=='Revolving loans', ['total_paid', 'AMT_CREDIT']] = np.nan
    x['interest_portion'] = (x['total_paid'] - x['AMT_CREDIT']) / x['AMT_CREDIT']
    x['interest_cnt']  = x['total_paid'] - x['AMT_CREDIT']
    
    cols += ['interest_portion', 'interest_cnt']
    prev_0 = x[x['DAYS_DECISION']>=-400].groupby('SK_ID_CURR')[cols].agg(['mean'])
    prev_1 = x.groupby('SK_ID_CURR')[cols].agg(['mean'])
    prev_agg = pd.concat([prev_0, prev_1, x_curr], axis=1)
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = prev_agg[prev_agg.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = prev_agg[prev_agg.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = prev_agg[prev_agg.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/payment_features.npy', x_train)
    np.save('artifacts/validation/payment_features.npy', x_val)
    np.save('artifacts/test/payment_features.npy', x_test)
    
    

#%%







