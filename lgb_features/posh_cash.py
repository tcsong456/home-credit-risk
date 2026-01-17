import numpy as np
import pandas as pd

def max_consecutive_true(mask):
    a = mask.to_numpy(dtype=np.int8)
    if a.size == 0:
        return 0

    diff = np.diff(np.r_[0, a, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return int((ends - starts).max()) if starts.size else 0

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
    pos_cash = pos_cash.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=True)
    pos_cash['dpd_breach'] = pos_cash['SK_DPD'] > 0
    pos_cash['dpd_def_breach'] = pos_cash['SK_DPD_DEF'] > 0
    
    final_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last().reset_index()
    total = pos_cash.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()

    features = []
    for status in ['Completed', 'Active', 'Canceled', 'Demand', 'Signed', 'Returned to the store']:
        cnt = (final_status['NAME_CONTRACT_STATUS'] == status).groupby(final_status['SK_ID_CURR']).sum()
        features += [cnt, cnt / total]
        
    status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].apply(set).reset_index()
    status['ever_demand'] = status['NAME_CONTRACT_STATUS'].map(lambda x: 1 if 'Demand' in x else 0)
    status['ever_canceled'] = status['NAME_CONTRACT_STATUS'].map(lambda x: 1 if 'Canceled' in x else 0)
    status['ever_returned'] = status['NAME_CONTRACT_STATUS'].map(lambda x: 1 if 'Returned to the store' in x else 0)
    status['ever_completed'] = status['NAME_CONTRACT_STATUS'].map(lambda x: 1 if 'Completed' in x else 0)
    status['unique_status'] = status['NAME_CONTRACT_STATUS'].map(len)
    del status['NAME_CONTRACT_STATUS']

    def transition(seq):
        return (seq != seq.shift()).sum() - 1
    
    seq_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS']
    seq_status = seq_status.apply(transition).reset_index(name='num_transitions')
    status = status.merge(seq_status, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])
    prev_len = pos_cash.groupby('SK_ID_PREV').size().reset_index(name='prev_len')
    status['volatility'] = status['num_transitions'] / prev_len['prev_len']
    cust_feats = status.groupby('SK_ID_CURR').agg(
        pos_n_contracts=('SK_ID_PREV', 'count'),
        pos_ever_demand_ratio=('ever_demand', 'mean'),
        pos_ever_canceled_ratio=('ever_canceled', 'mean'),
        pos_ever_returned_ratio=('ever_returned', 'mean'),
        pos_ever_completed_ratio=('ever_completed', 'mean'),
        pos_mean_volatility=('volatility', 'mean'),
        pos_max_volatility=('volatility', 'max'),
    )
    features += [cust_feats]

    dpd_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['dpd_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR').size()
    dpd_def_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['dpd_def_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR').size()
    dpd_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['dpd_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR').size()
    dpd_def_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['dpd_def_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR').size()
    dpd_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['dpd_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR').size()
    dpd_def_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['dpd_def_breach'].sum() / pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR').size()
    dpd = pos_cash.groupby('SK_ID_CURR')['dpd_breach'].sum() / pos_cash.groupby('SK_ID_CURR').size()
    dpd_def = pos_cash.groupby('SK_ID_CURR')['dpd_def_breach'].sum() / pos_cash.groupby('SK_ID_CURR').size()
    dpd_seq = pd.concat([dpd_3m, dpd_def_3m, dpd_6m, dpd_def_6m, dpd_12m, dpd_def_12m, dpd, dpd_def], axis=1)
    features += [dpd_seq]
    
    dpd_max_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['SK_DPD'].max()
    dpd_def_max_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    dpd_mean_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['SK_DPD'].mean()
    dpd_def_mean_3m = pos_cash[pos_cash['MONTHS_BALANCE']>=-3].groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()
    dpd_max_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['SK_DPD'].max()
    dpd_def_max_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    dpd_mean_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['SK_DPD'].mean()
    dpd_def_mean_6m = pos_cash[pos_cash['MONTHS_BALANCE']>=-6].groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()
    dpd_max_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['SK_DPD'].max()
    dpd_def_max_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    dpd_mean_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['SK_DPD'].mean()
    dpd_def_mean_12m = pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()
    dpd_max = pos_cash.groupby('SK_ID_CURR')['SK_DPD'].max()
    dpd_def_max = pos_cash.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    dpd_mean = pos_cash.groupby('SK_ID_CURR')['SK_DPD'].mean()
    dpd_def_mean = pos_cash.groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()
    dpd_severity = pd.concat([dpd_max_3m, dpd_def_max_3m, dpd_mean_3m, dpd_def_mean_3m, dpd_max_6m, dpd_def_max_6m, dpd_mean_6m,
                              dpd_def_mean_6m, dpd_max_12m, dpd_def_max_12m, dpd_mean_12m, dpd_def_mean_12m, dpd_max, dpd_def_max,
                              dpd_mean, dpd_def_mean], axis=1)
    features += [dpd_severity]
    
    # streak_prev = (
    #     pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby(['SK_ID_CURR', 'SK_ID_PREV'])['SK_DPD']
    #         .apply(lambda s: max_consecutive_true(s.gt(0)))
    #         .reset_index(name='max_consecutive_dpd_12m_prev')
    # )
    # streak_prev_12m = streak_prev.groupby('SK_ID_CURR')['max_consecutive_dpd_12m_prev'].max()

    streak_prev = (
        pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['SK_DPD']
            .apply(lambda s: max_consecutive_true(s.gt(0)))
            .reset_index(name='max_consecutive_dpd_prev')
    )
    streak_prev = streak_prev.groupby('SK_ID_CURR')['max_consecutive_dpd_prev'].max()

    dpd_rows = pos_cash[pos_cash['SK_DPD'] > 0]
    dpd_def_rows = pos_cash[pos_cash['SK_DPD_DEF'] > 0]
    last_dpd_mb = dpd_rows.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()
    last_dpd_def_mb = dpd_def_rows.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()
    features += [streak_prev, last_dpd_mb, last_dpd_def_mb]
    
    latest_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last().reset_index()
    latest_status['is_active'] = (latest_status['NAME_CONTRACT_STATUS'] == 'Active').astype(int)
    active_now_cnt = latest_status.groupby('SK_ID_CURR')['is_active'].sum()
    active_now_ratio = active_now_cnt / latest_status.groupby('SK_ID_CURR').size()
    features += [active_now_cnt, active_now_ratio]
    
    latest = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    active = latest[latest['NAME_CONTRACT_STATUS'] == 'Active']
    remaining_instals = active.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum()
    features += [remaining_instals]
    
    # duration = (pos_cash.groupby(['SK_ID_CURR' ,'SK_ID_PREV'])['MONTHS_BALANCE'].max() - pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].min()).reset_index()
    # duration = duration.groupby('SK_ID_CURR')['MONTHS_BALANCE'].agg(['min', 'max', 'mean'])
    # features += [duration]
    
    x = pd.concat(features, axis=1)
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = x[x.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = x[x.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = x[x.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/pos_cash_features.npy', x_train)
    np.save('artifacts/validation/pos_cash_features.npy', x_val)
    np.save('artifacts/test/pos_cash_features.npy', x_test)
    

#%%
df["STATUS"] = df["NAME_CONTRACT_STATUS"].astype("category")
df["STATUS_CODE"] = df["STATUS"].cat.codes.astype(np.int16)





#%%
pos_cash['NAME_CONTRACT_STATUS'].astype("category").cat.codes.astype(np.int16)






#%%

