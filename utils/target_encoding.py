import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def target_encoding_train(df, columns, alpha=50):
    df['concat_text'] = ''
    for c in columns:
        df['concat_text'] = df['concat_text'] + ' ' + df[c]
    
    y = df['TARGET']
    data = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(df, y):
        x_train = df.iloc[train_idx]
        x_val = df.iloc[val_idx]
        
        cnt = x_train['concat_text'].value_counts().to_dict()
        cat_avg = x_train.groupby(['concat_text'])['TARGET'].mean().to_dict()
        global_mean = x_train['TARGET'].mean()
        
        te = {}
        for k, n in cnt.items():
            miu = cat_avg[k] 
            t = (n * miu + alpha * global_mean) / (n + alpha)
            te[k] = t
        
        x_val['concat_text'] = x_val['concat_text'].map(te)
        data.append(x_val[['SK_ID_CURR', 'concat_text']])
    data = pd.concat(data, axis=0)
    data = df[['SK_ID_CURR']].merge(data, how='left', on=['SK_ID_CURR'])
    return data[['concat_text']].to_numpy().astype(np.float32)

def target_encoding_inference(df_train, df_val, columns, alpha=30):
    df_train['concat_text'] = ''
    df_val['concat_text'] = ''
    for c in ['CODE_GENDER', 'NAME_EDUCATION_TYPE']:
        df_train['concat_text'] = df_train['concat_text'] + ' ' + df_train[c]
        df_val['concat_text'] = df_val['concat_text'] + ' ' + df_val[c]
    cnt = df_train['concat_text'].value_counts().to_dict()
    cat_avg = df_train.groupby(['concat_text'])['TARGET'].mean().to_dict()
    global_mean = df_train['TARGET'].mean()
    te = {}
    for k, n in cnt.items():
        miu = cat_avg[k] 
        t = (n * miu + alpha * global_mean) / (n + alpha)
        te[k] = t
    df_val['concat_text'] = df_val['concat_text'].map(te)
    return df_val[['concat_text']].to_numpy().astype(np.float32)

if __name__ == '__main__':
    train = pd.read_csv('data/application_train.csv')
    cash_loan_train = train[train['NAME_CONTRACT_TYPE']=='Cash loans'].reset_index(drop=True)
    cash_loan_train['OCCUPATION_TYPE'] = cash_loan_train['OCCUPATION_TYPE'].fillna('unk')
    x = target_encoding_train(cash_loan_train, columns=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], alpha=50)
    z = target_encoding_inference(cash_loan_train, cash_loan_train, columns=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], alpha=50)