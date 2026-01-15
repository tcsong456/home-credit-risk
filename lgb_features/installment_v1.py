import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def installment_features(data):
    single_data = data.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['delay_days'].max().reset_index()
    single_data['is_delay'] = single_data['delay_days'] > 0
    single_data['delay_delta'] = single_data['delay_days'].map(abs)
    
    rep_data = data.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).size().reset_index()
    rep_data = rep_data.rename(columns={0: 'cnt'})
    rep_data['rep'] = rep_data['cnt'] > 1
    
    prev_len = data.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER']).size().reset_index()
    prev_len = prev_len.rename(columns={0: 'cnt'})
    prev_len['cnt'] -= 1
    prev_len = data.groupby(['SK_ID_CURR']).size() - prev_len.groupby(['SK_ID_CURR'])['cnt'].sum()
    prev_len = prev_len.reset_index().rename(columns={0: 'cnt'})
    prev_len = prev_len[['SK_ID_CURR', 'cnt']].set_index('SK_ID_CURR')
    
    features = []
    real_delay_cnt = data.groupby('SK_ID_CURR')['is_delay'].sum()
    real_delay_ratio = real_delay_cnt / data.groupby('SK_ID_CURR').size()
    fake_delay_cnt = single_data.groupby('SK_ID_CURR')['is_delay'].sum() 
    fake_delay_ratio = fake_delay_cnt / single_data.groupby('SK_ID_CURR').size()
    features += [real_delay_cnt, real_delay_ratio, fake_delay_cnt, fake_delay_ratio]
    
    prompt_pay = data[~data['is_delay']].groupby('SK_ID_CURR')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    late_pay = single_data[single_data['is_delay']].groupby('SK_ID_CURR')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    features += [prompt_pay, late_pay]
    
    data.loc[data['delay_days']==0, 'last_day_pay'] = 1
    last_day_pay_cnt = data.groupby('SK_ID_CURR')['last_day_pay'].sum().to_frame()
    last_day_pay_cnt = last_day_pay_cnt.rename(columns={'last_day_pay': 'cnt'})
    last_day_pay_ratio = last_day_pay_cnt / prev_len
    features += [last_day_pay_cnt, last_day_pay_ratio]
    
    rep_cnt = rep_data.groupby('SK_ID_CURR')['rep'].sum()
    rep_ratio = rep_cnt / rep_data.groupby('SK_ID_CURR').size()
    features += [rep_cnt, rep_ratio]
    
    x = pd.concat(features, axis=1)
    columns = ['real_delay_cnt', 'real_delay_ratio', 'fake_delay_cnt', 'fake_delay_ratio']
    agg_cols = [ 'min', 'max', 'mean', 'median', 'std']
    prompt_agg = ['prompt_pay_' + c for c in agg_cols]
    late_agg = ['late_pay_' + c for c in agg_cols]
    columns += prompt_agg
    columns += late_agg
    columns += ['last_day_pay_cnt', 'last_day_pay_ratio']
    columns += ['rep_cnt', 'rep_ratio']
    x.columns = columns
    
    return x

if __name__ == '__main__':
    instal = pd.read_csv('data/installments_payments.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    
    instal = instal.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=(True, True, True))
    instal['delay_days'] = instal['DAYS_ENTRY_PAYMENT'] - instal['DAYS_INSTALMENT']
    instal['delay_money'] = instal['AMT_PAYMENT'] - instal['AMT_INSTALMENT']
    instal['is_delay'] = instal['delay_days'] > 0
    instal['delay_delta'] = instal['delay_days'].map(abs)

    x_0 = installment_features(instal[instal['DAYS_INSTALMENT']>=-60])
    x_1 = installment_features(instal[instal['DAYS_INSTALMENT']>=-180])
    x_2 = installment_features(instal[instal['DAYS_INSTALMENT']>=-365])
    x_3 = installment_features(instal)
    x = pd.concat([x_0, x_1, x_2, x_3], axis=1)
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = x[x.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = x[x.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = x[x.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/instal_features.npy', x_train)
    np.save('artifacts/validation/instal_features.npy', x_val)
    np.save('artifacts/test/instal_features.npy', x_test)



#%%
z = instal.groupby(['SK_ID_PREV'])['DAYS_ENTRY_PAYMENT'].diff().reset_index()
