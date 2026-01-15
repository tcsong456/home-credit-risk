import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

if __name__ == '__main__':
    instal = pd.read_csv('data/installments_payments.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    
    prev_features, curr_features = [], []
    instal = instal.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=(True, True, True))
    instal['delay_days'] = instal['DAYS_ENTRY_PAYMENT'] - instal['DAYS_INSTALMENT']
    instal['delay_money'] = instal['AMT_PAYMENT'] - instal['AMT_INSTALMENT']
    instal['is_delay'] = instal['delay_days'] > 0
    
    prev_len = instal.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER']).size().reset_index()
    prev_len = prev_len.rename(columns={0: 'cnt'})
    prev_len['cnt'] -= 1
    prev_len = instal.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size() - prev_len.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['cnt'].sum()
    prev_len = prev_len.reset_index().rename(columns={0: 'cnt'})
    sk_prev_len = prev_len[['SK_ID_PREV', 'cnt']].set_index('SK_ID_PREV')
    sk_curr_len = prev_len.groupby('SK_ID_CURR')['cnt'].sum()
    
    single_delay = instal.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['delay_days'].max().reset_index()
    single_delay['is_delay'] = single_delay['delay_days'] > 0
    single_delay['delay_delta'] = single_delay['delay_days'].map(abs)
    
    repetition = instal.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).size().reset_index()
    repetition = repetition.rename(columns={0: 'cnt'})
    repetition['rep'] = repetition['cnt'] > 1
    
    real_delay_cnt = instal.groupby('SK_ID_PREV')['is_delay'].sum()
    real_delay_ratio = real_delay_cnt / instal.groupby('SK_ID_PREV').size()
    fake_delay_cnt = single_delay.groupby('SK_ID_PREV')['is_delay'].sum() 
    fake_delay_ratio = fake_delay_cnt / single_delay.groupby('SK_ID_PREV').size()
    prev_features += [real_delay_cnt, real_delay_ratio, fake_delay_cnt, fake_delay_ratio]
    
    instal['delay_delta'] = instal['delay_days'].map(abs)
    prompt_pay = instal[~instal['is_delay']].groupby('SK_ID_PREV')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    late_pay = single_delay[single_delay['is_delay']].groupby('SK_ID_PREV')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    prev_features += [prompt_pay, late_pay]
    payment_duration = instal.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].min() - instal.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].max()
    prev_features += [payment_duration]
    
    instal.loc[instal['delay_days']==0, 'last_day_pay'] = 1
    last_day_pay_cnt = instal.groupby('SK_ID_PREV')['last_day_pay'].sum().to_frame()
    last_day_pay_cnt = last_day_pay_cnt.rename(columns={'last_day_pay': 'cnt'})
    last_day_pay_ratio = last_day_pay_cnt / sk_prev_len
    prev_features += [last_day_pay_cnt, last_day_pay_ratio]
    
    rep_cnt = repetition.groupby('SK_ID_PREV')['rep'].sum()
    rep_ratio = rep_cnt / repetition.groupby('SK_ID_PREV').size()
    prev_features += [rep_cnt, rep_ratio]
    
    x_prev = pd.concat(prev_features, axis=1)
    
    columns = ['real_delay_cnt', 'real_delay_ratio', 'fake_delay_cnt', 'fake_delay_ratio']
    agg_cols = [ 'min', 'max', 'mean', 'median', 'std']
    prompt_agg = ['prompt_pay_' + c for c in agg_cols]
    late_agg = ['late_pay_' + c for c in agg_cols]
    columns += prompt_agg
    columns += late_agg
    columns += ['days_duration', 'last_day_pay_cnt', 'last_day_pay_ratio']
    columns += ['rep_cnt', 'rep_ratio']
    x_prev.columns = columns
    
    real_delay_cnt_curr = instal.groupby('SK_ID_CURR')['is_delay'].sum()
    real_delay_ratio_curr = real_delay_cnt_curr / instal.groupby('SK_ID_CURR').size()
    fake_delay_cnt_curr = single_delay.groupby('SK_ID_CURR')['is_delay'].sum()
    max_intal_num = single_delay.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().reset_index()
    max_intal_num = max_intal_num.rename(columns={0: 'cnt'})
    fake_denom = max_intal_num.groupby('SK_ID_CURR')['cnt'].sum()
    fake_delay_ratio_curr = fake_delay_cnt_curr / fake_denom
    curr_features += [real_delay_cnt_curr, real_delay_ratio_curr, fake_delay_cnt_curr, fake_delay_ratio_curr]
    
    prompt_pay_curr = instal[~instal['is_delay']].groupby('SK_ID_CURR')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    late_pay_curr = single_delay[single_delay['is_delay']].groupby('SK_ID_CURR')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    curr_features += [prompt_pay_curr, late_pay_curr]
    
    last_day_pay_cnt_curr = instal.groupby('SK_ID_CURR')['last_day_pay'].sum()
    last_day_pay_ratio = last_day_pay_cnt_curr / sk_curr_len
    curr_features += [last_day_pay_cnt_curr, last_day_pay_ratio]
    
    rep_cnt_curr = repetition.groupby('SK_ID_CURR')['rep'].sum()
    rep_ratio_curr = rep_cnt_curr / repetition.groupby('SK_ID_CURR').size()
    curr_features += [rep_cnt_curr, rep_ratio_curr]
    
    x_curr = pd.concat(curr_features, axis=1)
    columns.remove('days_duration')
    x_curr.columns = columns
    
    prev = prev_app[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION']].merge(x_prev, how='left', on=['SK_ID_PREV'])
    cols = [col for col in prev.columns if col not in ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION']]
    prev_agg_0 = prev[prev['DAYS_DECISION']>=-400].groupby('SK_ID_CURR')[cols].agg('mean')
    prev_agg_1 = prev.groupby(['SK_ID_CURR'])[cols].agg('mean')
    prev_agg = pd.concat([prev_agg_0, prev_agg_1, x_curr], axis=1).reset_index()
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = prev_agg[prev_agg.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = prev_agg[prev_agg.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = prev_agg[prev_agg.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/instal_features.npy', x_train)
    np.save('artifacts/validation/instal_features.npy', x_val)
    np.save('artifacts/test/instal_features.npy', x_test)

#%%


#%%








