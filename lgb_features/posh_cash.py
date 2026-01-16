import numpy as np
import pandas as pd

def build_features(data):
    dpd_breach_cnt = data.groupby('SK_ID_CURR')['dpd_breach'].sum()
    dpd_breach_ratio = dpd_breach_cnt / data.groupby('SK_ID_CURR').size()
    dpd_def_breach_cnt = data.groupby('SK_ID_CURR')['dpd_def_breach'].sum()
    dpd_def_breach_ratio = dpd_def_breach_cnt / data.groupby('SK_ID_CURR').size()
    
    x = pd.concat([dpd_breach_cnt, dpd_breach_ratio, dpd_def_breach_cnt, dpd_def_breach_ratio], axis=1)
    return x

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    test = pd.read_csv('data/application_test.csv')
    prev_app = pd.read_csv('data/previous_application.csv')
    pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
    pos_cash = pos_cash.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=True)
    
    final_status = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last().reset_index()
    total = pos_cash.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()

    features = []
    for status in ['Completed', 'Active', 'Canceled', 'Demand', 'Signed', 'Returned to the store']:
        cnt = (final_status['NAME_CONTRACT_STATUS'] == status).groupby(final_status['SK_ID_CURR']).sum()
        features.append(cnt / total)
        
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
    
    
    # val_currs = np.load('artifacts/val_sk_currs.npy')
    # train = train[~train['SK_ID_CURR'].isin(val_currs)]
    # x_train = x[x.index.isin(train['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    # x_val = x[x.index.isin(val_currs)].reset_index().to_numpy().astype(np.float32)
    # x_test = x[x.index.isin(test['SK_ID_CURR'])].reset_index().to_numpy().astype(np.float32)
    
    # np.save('artifacts/train/pos_cash_features.npy', x_train)
    # np.save('artifacts/validation/pos_cash_features.npy', x_val)
    # np.save('artifacts/test/pos_cash_features.npy', x_test)
    

#%%


#%%


#%%

