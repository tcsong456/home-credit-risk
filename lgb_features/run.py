import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ft_name = ['app_features']
for f in ft_name:
    x_train = np.load(f'artifacts/train/{f}.npy')
    if f == 'app_features':
        y_train = x_train[:, 1]
        x_train = x_train[:, 2:]
    
    x_val = np.load(f'artifacts/validation/{f}.npy')
    if f == 'app_features':
        y_val = x_val[:, 1]
        x_val = x_val[:, 2:]
    
    x_te = np.load(f'artifacts/test/{f}.npy')
    # if f == 'app_features':
    #     x_te = x_te[:, 1:]

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
p_test = model_meta.predict(x_te, num_iteration=model_meta.best_iteration)
ss = pd.read_csv('data/sample_submission.csv')
ss['TARGET'] = p_test
ss.to_csv('artifacts/submission.csv', index=False)
#%%
    

