import numpy as np
import pandas as pd

def transition(seq):
    return (seq != seq.shift()).sum() - 1

def max_consecutive_true(mask):
    a = mask.to_numpy(dtype=np.int8)
    if a.size == 0:
        return 0

    diff = np.diff(np.r_[0, a, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return int((ends - starts).max()) if starts.size else 0

def clean_status(status_set):
    call = {'Active', 'Completed'}.issubset(status_set) and \
        not bool({'Returned to the store', 'Demand', 'Amortized debt', 'Canceled'} & status_set)
    if call:
        return 1
    else:
        return 0
    
def dpd_violation(data):
    dpd = data.groupby('SK_ID_CURR')['dpd_breach'].sum() / data.groupby('SK_ID_CURR').size()
    dpd_def = data.groupby('SK_ID_CURR')['dpd_def_breach'].sum() / data.groupby('SK_ID_CURR').size()
    return pd.concat([dpd, dpd_def], axis=1)

def dpd_breach(data):
    dpd_max = data.groupby('SK_ID_CURR')['SK_DPD'].max()
    dpd_def_max = data.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    dpd_mean = data.groupby('SK_ID_CURR')['SK_DPD'].mean()
    dpd_def_mean = data.groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()
    return pd.concat([dpd_max, dpd_def_max, dpd_mean, dpd_def_mean], axis=1)

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
    pos_cash = pos_cash.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=True)
    pos_cash['dpd_breach'] = pos_cash['SK_DPD'] > 0
    pos_cash['dpd_def_breach'] = pos_cash['SK_DPD_DEF'] > 0
    
    features = []
    last_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last().reset_index(name='status')
    for status in ['Completed', 'Active', 'Demand', 'Signed', 'Returned to the store', 'Amortized debt']:
        status_cnt = (last_status['status'] == status).groupby(last_status['SK_ID_CURR']).sum()
        status_ratio = status_cnt / last_status.groupby('SK_ID_CURR').size()
        features += [status_cnt, status_ratio]
    
    combo = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].apply(set).reset_index(name='status_combo')
    combo['is_clean'] = combo['status_combo'].map(clean_status)
    combo = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].apply(set).reset_index(name='status_combo')
    combo['is_clean'] = combo['status_combo'].map(clean_status)
    combo = combo.merge(prev_app[['SK_ID_PREV', 'DAYS_DECISION']], how='left', on=['SK_ID_PREV'])
    combo_cnt_12m = combo[combo['DAYS_DECISION']>=-365].groupby('SK_ID_CURR')['is_clean'].sum()
    combo_ratio_12m = combo[combo['DAYS_DECISION']>=-365].groupby('SK_ID_CURR')['is_clean'].mean()
    combo_cnt = combo.groupby('SK_ID_CURR')['is_clean'].sum()
    combo_ratio = combo.groupby('SK_ID_CURR')['is_clean'].mean()
    features += [combo_cnt_12m, combo_ratio_12m, combo_cnt, combo_ratio]
    
    for status in ['Demand', 'Canceled', 'Returned to the store', 'Amortized debt']:
        head = status.split()[0].lower()
        combo[f'ever_{head}'] = combo['status_combo'].map(lambda x: 1 if status in x else 0)
    seq_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS']
    seq_status = seq_status.apply(transition).reset_index(name='num_transitions')
    status = combo.merge(seq_status, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])
    prev_len = pos_cash.groupby('SK_ID_PREV').size().reset_index(name='prev_len')
    status['volatility'] = status['num_transitions'] / prev_len['prev_len']
    cust_feats_12m = status[status['DAYS_DECISION']>=-365].groupby('SK_ID_CURR').agg(
        pos_ever_demand_ratio=('ever_demand', 'mean'),
        pos_ever_canceled_ratio=('ever_canceled', 'mean'),
        pos_ever_returned_ratio=('ever_returned', 'mean'),
        pos_ever_amortized_ratio=('ever_amortized', 'mean'),
        pos_mean_volatility=('volatility', 'mean'),
        pos_max_volatility=('volatility', 'max'),
    )
    
    cust_feats = status.groupby('SK_ID_CURR').agg(
        pos_n_contracts=('SK_ID_PREV', 'count'),
        
        pos_ever_demand=('ever_demand', 'sum'),
        pos_ever_canceled=('ever_canceled', 'sum'),
        pos_ever_returned=('ever_returned', 'sum'),
        pos_ever_amortized=('ever_amortized', 'sum'),
        
        pos_ever_demand_ratio=('ever_demand', 'mean'),
        pos_ever_canceled_ratio=('ever_canceled', 'mean'),
        pos_ever_returned_ratio=('ever_returned', 'mean'),
        pos_ever_amortized_ratio=('ever_amortized', 'mean'),
        pos_mean_volatility=('volatility', 'mean'),
        pos_max_volatility=('volatility', 'max'),
    )
    features += [cust_feats_12m, cust_feats]
    
    streak_prev = (
        pos_cash[pos_cash['MONTHS_BALANCE']>=-12].groupby(['SK_ID_CURR', 'SK_ID_PREV'])['SK_DPD']
            .apply(lambda s: max_consecutive_true(s.gt(0)))
            .reset_index(name='max_consecutive_dpd_12m_prev')
    )
    streak_prev_12m = streak_prev.groupby('SK_ID_CURR')['max_consecutive_dpd_12m_prev'].max()

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
    features += [streak_prev_12m, streak_prev, last_dpd_mb, last_dpd_def_mb]
    
    latest_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last().reset_index()
    latest_status['is_active'] = (latest_status['NAME_CONTRACT_STATUS'] == 'Active').astype(int)
    active_now_cnt = latest_status.groupby('SK_ID_CURR')['is_active'].sum()
    active_now_ratio = active_now_cnt / latest_status.groupby('SK_ID_CURR').size()
    features += [active_now_cnt, active_now_ratio]
    
    latest = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    active = latest[latest['NAME_CONTRACT_STATUS'] == 'Active']
    remaining_instals = active.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum()
    features += [remaining_instals]
    
    duration = (pos_cash.groupby(['SK_ID_CURR' ,'SK_ID_PREV'])['MONTHS_BALANCE'].max() - pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].min()).reset_index()
    duration = duration.groupby('SK_ID_CURR')['MONTHS_BALANCE'].agg(['min', 'max', 'mean'])
    features += [duration]
    
    for i in [-3, -6, -9, -12, -100]:
        d = pos_cash[pos_cash['MONTHS_BALANCE']>=i]
        dpd = dpd_violation(d)
        features += [dpd]
    
    for i in [-3, -6, -9, -12, -100]:
        d = pos_cash[pos_cash['MONTHS_BALANCE']>=i]
        dpd = dpd_breach(d)
        features += [dpd]
    
    x = pd.concat(features, axis=1)
    
    val_currs = np.load('artifacts/val_sk_currs.npy')
    train = train[~train['SK_ID_CURR'].isin(val_currs)]
    x_train = x[x.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    x_val = x[x.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    x_test = x[x.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    np.save('artifacts/train/pos_cash_features.npy', x_train)
    np.save('artifacts/validation/pos_cash_features.npy', x_val)
    np.save('artifacts/test/pos_cash_features.npy', x_test)
    


