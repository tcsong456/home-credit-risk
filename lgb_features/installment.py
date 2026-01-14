import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

# instal = instal.merge(prev_app[['SK_ID_PREV', 'AMT_ANNUITY']], how='left', on=['SK_ID_PREV'])
if __name__ == '__main__':
    instal = pd.read_csv('data/installments_payments.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    
    prev_features, curr_features = [], []
    instal = instal.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=(True, True, True))
    instal['delay_days'] = instal['DAYS_ENTRY_PAYMENT'] - instal['DAYS_INSTALMENT']
    instal['delay_money'] = instal['AMT_PAYMENT'] - instal['AMT_INSTALMENT']
    instal['is_delay'] = instal['delay_days'] > 0
    
    real_delay_cnt = instal.groupby('SK_ID_PREV')['is_delay'].sum()
    real_delay_ratio = real_delay_cnt / instal.groupby('SK_ID_PREV').size()
    single_delay = instal.groupby(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['delay_days'].max().reset_index()
    single_delay['is_delay'] = single_delay['delay_days'] > 0
    single_delay['delay_delta'] = single_delay['delay_days'].map(abs)
    fake_deley_cnt = single_delay.groupby('SK_ID_PREV')['is_delay'].sum() 
    fake_deley_ratio = fake_deley_cnt / instal.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].max()
    prev_features += [real_delay_cnt, real_delay_ratio, fake_deley_cnt, fake_deley_ratio]
    
    instal['delay_delta'] = instal['delay_days'].map(abs)
    prompt_pay = instal[~instal['is_delay']].groupby('SK_ID_PREV')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    late_pay = single_delay[single_delay['is_delay']].groupby('SK_ID_PREV')['delay_delta'].agg(['min', 'max', 'mean', 'median', 'std'])
    prev_features += [prompt_pay, late_pay]
    payment_duration = instal.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].min() - instal.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].max()
    prev_features += [payment_duration]
    
    instal.loc[instal['delay_days']==0, 'last_day_pay'] = 1
    last_day_pay_cnt = instal.groupby('SK_ID_PREV')['last_day_pay'].sum()
    last_day_pay_ratio = last_day_pay_cnt / instal.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].max()
    prev_features += [last_day_pay_cnt, last_day_pay_ratio]
    
    x_prev = pd.concat(prev_features, axis=1)
    
    prev_columns = ['real_delay_cnt', 'real_delay_ratio', 'fake_deley_cnt', 'fake_deley_ratio']
    agg_cols = [ 'min', 'max', 'mean', 'median', 'std']
    prompt_agg = ['prompt_pay_' + c for c in agg_cols]
    late_agg = ['late_pay_' + c for c in agg_cols]
    prev_columns += prompt_agg
    prev_columns += late_agg
    prev_columns += ['days_duration', 'last_day_pay_cnt', 'last_day_pay_ratio']
    x_prev.columns = prev_columns
    
    real_delay_cnt_curr = instal.groupby('SK_ID_CURR')['is_delay'].sum()
    real_delay_ratio_curr = real_delay_cnt_curr / instal.groupby('SK_ID_CURR').size()
    fake_delay_cnt_curr = single_delay.groupby('SK_ID_CURR')['is_delay'].sum()
    fake_delay_ratio_curr = fake_delay_cnt_curr / single_delay.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max()
    curr_features += [real_delay_cnt_curr, real_delay_ratio_curr, fake_delay_cnt_curr, fake_delay_ratio_curr]

#%%
f = instal.drop_duplicates(['SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'DAYS_INSTALMENT']).groupby(['SK_ID_PREV'])['AMT_INSTALMENT'].sum()
z = instal.groupby(['SK_ID_PREV'])['AMT_PAYMENT'].sum()
k = pd.concat([f, z], axis=1)
y = k[~np.isclose(k['AMT_INSTALMENT'], k['AMT_PAYMENT'], rtol=0.5, atol=0.5)]

#%%

single_delay = instal.groupby(['SK_ID_CURR', 'SK_ID_PREV','NUM_INSTALMENT_NUMBER'])['delay_days'].max().reset_index()
single_delay['is_delay'] = single_delay['delay_days'] > 0
single_delay['delay_delta'] = single_delay['delay_days'].map(abs)

instal.groupby('SK_ID_CURR')['is_delay'].sum() / instal.groupby('SK_ID_CURR').size(),
single_delay.groupby('SK_ID_CURR')['is_delay'].sum() / single_delay.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max()

