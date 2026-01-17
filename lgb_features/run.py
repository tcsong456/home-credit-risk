import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

train_data, val_data, test_data = [], [], []
ft_name = ['app_features', 'prev_app_features', 'instal_features', 'payment_features', 'pos_cash_features']
for f in ft_name:
    x_train = np.load(f'artifacts/train/{f}.npy')
    if f == 'app_features':
        y_train = x_train[:, 1]
        x_train = np.delete(x_train, 1, axis=1)
    x_train = pd.DataFrame(x_train, columns=['SK_ID_CURR']+[f'feature_{i}' for i in range(x_train.shape[1]-1)]).set_index('SK_ID_CURR')
    train_data.append(x_train)
    
    x_val = np.load(f'artifacts/validation/{f}.npy')
    if f == 'app_features':
        y_val = x_val[:, 1]
        x_val = np.delete(x_val, 1, axis=1)
    x_val = pd.DataFrame(x_val, columns=['SK_ID_CURR']+[f'feature_{i}' for i in range(x_val.shape[1]-1)]).set_index('SK_ID_CURR')
    val_data.append(x_val)
    
    x_te = np.load(f'artifacts/test/{f}.npy')
    x_te = pd.DataFrame(x_te, columns=['SK_ID_CURR']+[f'feature_{i}' for i in range(x_te.shape[1]-1)]).set_index('SK_ID_CURR')
    test_data.append(x_te)

x_train = pd.concat(train_data, axis=1).to_numpy()
x_val = pd.concat(val_data, axis=1).to_numpy()
x_test = pd.concat(test_data, axis=1).to_numpy()

params = {
    'num_leaves': 100,
    'metric': 'auc',   
    'objective': 'binary',
    'min_data_in_leaf': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.25,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
}
dtrain = lgb.Dataset(x_train, label=y_train)
dvalid = lgb.Dataset(x_val, label=y_val, reference=dtrain)
model_meta = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ],
)
p_val = model_meta.predict(x_val, num_iteration=model_meta.best_iteration)
auc = roc_auc_score(y_val, p_val)
p_test = model_meta.predict(x_test, num_iteration=model_meta.best_iteration)
ss = pd.read_csv('data/sample_submission.csv')
ss['TARGET'] = p_test
ss.to_csv('artifacts/submission.csv', index=False)
#%%