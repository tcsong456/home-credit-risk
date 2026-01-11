import numpy as np
import pandas as pd

train = pd.read_csv('data/application_train.csv')
test = pd.read_csv('data/application_test.csv')

test_len = test.shape[0]
train_len = train.shape[0]
val_ratio = test_len / (test_len + train_len)
val_len = int(train.shape[0] * val_ratio)

test_revolve_loan_len = test[test['NAME_CONTRACT_TYPE']=='Revolving loans'].shape[0]
revolve_loan_ratio = test_revolve_loan_len / test.shape[0]
val_revolve_loan_len = int(val_len * revolve_loan_ratio)
val_cash_loan_len = val_len - val_revolve_loan_len

np.random.seed(601)
cash_loan_currs = train[train['NAME_CONTRACT_TYPE']=='Cash loans']['SK_ID_CURR'].to_numpy()
revolve_loan_currs = train[train['NAME_CONTRACT_TYPE']=='Revolving loans']['SK_ID_CURR'].to_numpy()
cash_loan_currs = np.random.choice(cash_loan_currs, val_cash_loan_len, replace=False)
revolve_loan_currs = np.random.choice(revolve_loan_currs, val_revolve_loan_len, replace=False)
val_currs = np.concatenate([cash_loan_currs, revolve_loan_currs]).astype(np.int32)
    
np.save('artifacts/val_sk_currs.npy', val_currs)
