#
#Author: zephyr
#Date: 2020-12-25 09:22:45
#LastEditors: zephyr
#LastEditTime: 2021-01-03 18:48:40
#FilePath: \MovieRatingPrediction\LGBRegression.py
#
import pickle
import os
import re
import copy
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm

import matplotlib.pyplot as plt
import seaborn as sns

data_df = pd.read_pickle('data/processed/data_df.pkl')
usr_df = pd.read_pickle('data/processed/usr_df.pkl')
item_df = pd.read_pickle('data/processed/item_df.pkl')

occupation_lst = [i.strip() for i in open('data/raw_data/u.occupation','r',encoding='utf-8').readlines()]

usr_feature_df = pd.DataFrame()
usr_feature_df['usr_id'] = usr_df['usr_id'].copy()

def scale_and_norm(df,colname):
    '''连续数值型特征归一化和正则化
    '''
    df = df.copy()
    max_ = max(df[colname])
    min_ = min(df[colname])
    df[colname] = df[colname].apply(lambda x:(x-min_)/(max_-min_))

    mean_scaled = np.mean(df[colname])
    std_scaled = np.std(df[colname])

    df[colname] = df[colname].apply(lambda x:(x-mean_scaled)/std_scaled)
    return df

def target_encoding(df,colname,cate_lst=None):
    '''
    此处使用target encoding方法 将类别在总体占比作为其特征值(cate_lst基本无用)
    '''
    df = df.copy()
    col_value = df[colname].tolist()
    col_counter = Counter(col_value)

    col_frac = {k:col_counter[k]/sum(col_counter.values()) for k in col_counter}

    df[colname] = df[colname].apply(lambda x:col_frac[x])
    return df

def onehot_encoding(df,colname,cate_lst=None):
    '''离散型特征分桶类别化(categorization)
    cate_lst为对应类别的既有顺序 默认为None 如果有的话 类别化时参照lst中的顺序进行编码
    '''
    df = df.copy()
    col_value = df[colname].tolist()
    if cate_lst is None:
        cate_lst = list(set(col_value))
    df[colname] = df[colname].apply(lambda x:cate_lst.index(x))
    return df

usr_feature_df['age'] = [int(i) for i in usr_df['age']]
usr_feature_df['gender'] = usr_df['gender']
usr_feature_df['occupation'] = usr_df['occupation']

usr_feature_df = scale_and_norm(usr_feature_df,'age')
usr_feature_df = target_encoding(usr_feature_df,'gender')
usr_feature_df = target_encoding(usr_feature_df,'occupation')

movie_type = [i.strip().split('|')[0] for i in open('data/raw_data/u.genre').readlines()]
movie_type = movie_type[:-1]# 去掉最后的‘’

item_feature_df = pd.DataFrame()
item_feature_df['item_id'] = item_df['item_id']
item_feature_df['movie_title'] = item_df['movie_title'].apply(lambda x:x.split('(')[0])

item_feature_df['release_year'] = item_df['release_date'].apply(lambda x:x.split('-')[-1])
item_feature_df = target_encoding(item_feature_df,'release_year')

item_type_matrix = item_df[movie_type].to_numpy()
item_type_matrix = item_type_matrix.astype(int)

item_feature_df['type_feature'] = [i for i in item_type_matrix]

# 计算 TF & IDF
# from sklearn.feature_extraction.text import TfidfVectorizer

# movie_title_text = item_feature_df['item_id'].tolist()

# tfidf_obj = TfidfVectorizer(max_features=50)

# item_tfidf_feature = tfidf_obj.fit_transform(movie_title_text)

# item_tfidf_feature = item_tfidf_feature.toarray()

# item_feature_df['tfidf_feature'] = [i for i in item_tfidf_feature]

usr_feautre = usr_feature_df[['age','gender','occupation']].copy().to_numpy()
usr_feature_df = pd.DataFrame()
usr_feature_df['usr_id'] = usr_df['usr_id'].copy()
usr_feature_df['features'] = [i for i in usr_feautre]

item_feature = []
for num,_ in enumerate(item_feature_df.itertuples()):
    year = item_feature_df.loc[num,'release_year']
    type_feature = item_feature_df.loc[num,'type_feature']
    feature = np.hstack([np.array([year]),type_feature])
    item_feature.append(feature)

item_feature_df = pd.DataFrame()
item_feature_df['item_id'] = item_df['item_id'].copy()
item_feature_df['features'] = [i for i in item_feature]

def get_Xy(df,
           usr_feautre_df=usr_feature_df,
           item_feature_df=item_feature_df):
    '''给定数据df 按照顺序输出特征和对应标签
    拼接之后的数据
    '''
    df = df.copy()
    uid_lst = [int(i) for i in df['usr_id'].tolist()]
    iid_lst = [int(i) for i in df['item_id'].tolist()]

    u_feature = [usr_feature_df.loc[i-1,'features'] for i in uid_lst]

    u_feature = np.array(u_feature)

    i_feature = [item_feature_df.loc[i-1,'features'] for i in iid_lst]

    i_feature = np.array(i_feature)

    X = np.hstack([u_feature,i_feature])
    y = [int(i) for i in df['rating'].tolist()]
    y = np.array(y)
    return X,y

item_feature_df.to_pickle('data/processed/item_feature_df.pkl')
usr_feature_df.to_pickle('data/processed/usr_feature_df.pkl')



cv_df_lst = []
for i in range(5):
    df_fname = 'data/processed/cv_{0}_df.pkl'.format(i+1)
    df = pd.read_pickle(df_fname)
    cv_df_lst.append(df)

best_param = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'regression',
          'nthread': 8, # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.1,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'scale_pos_weight': 1,
          'num_class' : 1,}

model = lightgbm.LGBMRegressor(**best_param)
mse_lst = []
for test_idx in range(5):
    print('----------------')
    print('Test on cv_{0}_df'.format(test_idx+1))
    train_idx_lst = [i for i in range(5) if i!=test_idx]
    # 训练过程 其余四份test集合作为训练集 当前作为测试
    for train_idx in train_idx_lst:
        df = cv_df_lst[train_idx]
        train_df = df[df['type']=='train']
        train_X,train_y = get_Xy(train_df)
        model.fit(train_X,train_y)
    # 测试过程
    df = cv_df_lst[test_idx]
    test_df = df[df['type']=='test']
    test_X,test_true_y = get_Xy(test_df)
    test_pred_y = model.predict(test_X)
    mse = mean_squared_error(test_true_y,test_pred_y)
    mse_lst.append(mse)
    print('MSE on cv_{0}_df:{1}'.format(i+1,mse))
print('Average MSE:{0}'.format(np.average(mse_lst)))