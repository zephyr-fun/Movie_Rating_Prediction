#
#Author: zephyr
#Date: 2020-12-24 17:25:10
#LastEditors: zephyr
#LastEditTime: 2021-01-03 20:39:55
#FilePath: \MovieRatingPrediction\pre_process.py
#
import pickle
import os
import re
import copy
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# %matplotlib inline 可以在Ipython编译器里直接使用
# 功能是可以内嵌绘图，并且可以省略掉plt.show()这一步

if not os.path.isdir('data/processed'):
    os.mkdir('data/processed')

def mk_df(_dir, tag = None):
    columns = ['usr_id','item_id','rating','timestamp']
    data = [i.strip().split('\t') for i in open(_dir, 'r', encoding='utf-8').readlines()]
    if tag is not None:
        columns += ['type']
        data = [i+[tag] for i in data]
    df = pd.DataFrame(data=data, columns=columns)
    return df

data_df = mk_df('data/raw_data/u.data')
data_df.to_pickle('data/processed/data_df.pkl')

# 'usr_id','item_id','rating','timestamp','tag'
# data/processed/cv_i_df.pkl
for i in range(0,5):
    print('Fold:{0} Processing...'.format(i+1))
    train_name = 'data/raw_data/u{0}.base'.format(i+1)
    test_name = 'data/raw_data/u{0}.test'.format(i+1)
    train_df = mk_df(train_name, 'train')
    test_df = mk_df(test_name, 'test')
    df = pd.concat([train_df, test_df])
    name = 'data/processed/cv_{0}_df.pkl'.format(i+1)
    df.to_pickle(name)

for tag in ['a','b']:
    print('Type:{0} Processing...'.format(tag))
    train_name = 'data/raw_data/u{0}.base'.format(tag)
    test_name = 'data/raw_data/u{0}.test'.format(tag)
    train_df = mk_df(train_name,'train')
    test_df = mk_df(test_name,'test')
    df = pd.concat([train_df,test_df])
    name = 'data/processed/type_{0}_df.pkl'.format(tag)
    df.to_pickle(name)
# user & item raw data
usr_data = [i.strip().split('|') for i in open('data/raw_data/u.user', 'r', encoding='utf-8').readlines()]
usr_columns = ['usr_id', 'age', 'gender', 'occupation', 'zip_code']
usr_df = pd.DataFrame(data=usr_data, columns=usr_columns)

item_data = [i.strip().split('|') for i in open('data/raw_data/u.item', 'r', encoding='utf-8').readlines()]
item_columns = '''item_id | movie_title | release_date | video_release_date |
              IMDb_URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |'''
item_columns = [i.strip() for i in item_columns.split('|')][:-1]
item_df = pd.DataFrame(data=item_data, columns=item_columns)

usr_df.to_pickle('data/processed/usr_df.pkl')
item_df.to_pickle('data/processed/item_df.pkl')

print('data_df : {0}'.format(data_df.shape))
print('usr_df  : {0}'.format(usr_df.shape))
print('item_df : {0}'.format(item_df.shape))

# data_df['rating'] = data_df['rating'].apply(int)
data_df.rating.hist()
plt.show()
# user_id & item_id 
usr_id_lst = usr_df.usr_id.tolist()
item_id_lst = item_df.item_id.tolist()
# interaction
usr_item_dict = {}
for uid in usr_id_lst:
    item_dict = {i:0 for i in item_id_lst}
    usr_item_dict[uid] = item_dict

for num,_ in enumerate(data_df.itertuples()):
    uid = data_df.loc[num,'usr_id']
    iid = data_df.loc[num,'item_id']
    rating = data_df.loc[num,'rating']
    usr_item_dict[uid][iid] = int(rating)

inter_df = pd.DataFrame(data=usr_item_dict)

inter_df = inter_df.transpose()

inter_df.fillna(value=0)

sns.heatmap(inter_df,cmap="YlGnBu")
plt.show()

def analyze(a_df,b_df):
    '''给定df 输出其中usrid和itemid占总体的比率
    非100%则表明存在缺省情况
    '''
    full_usr_id = set(a_df.usr_id)
    full_item_id = set(a_df.item_id)
    
    df_usr_lst = set(b_df.usr_id.tolist())
    df_item_lst = set(b_df.item_id.tolist())
    
    usr_frac = len(full_usr_id & df_usr_lst)/len(full_usr_id)
    item_frac = len(full_item_id & df_item_lst)/len(full_item_id)
    return usr_frac,item_frac

for i in range(0,5):
#     print('Fold:{0} Processing...'.format(i+1))
    name = 'data/processed/cv_{0}_df.pkl'.format(i+1)
    temp_df = pd.read_pickle(name)
    
    train_df = temp_df[temp_df['type']=='train'].copy()
    test_df = temp_df[temp_df['type']=='test'].copy()
    
    train_u,train_i = analyze(data_df,train_df)
    test_u,test_i = analyze(data_df,test_df)
    print('*******************')
    print('Compare with full data, Train<usr:{0:0.4} item:{1:0.4}>;Test<usr:{1:0.4} item:{2:0.4}>'.format(
        train_u,train_i,test_u,test_i))
    print('-------------------')
    u,i = analyze(train_df,test_df)
    print('Compare between train and test data, Usr:{0:0.4} Item:{1:0.4}'.format(u,i))

for i in ['a','b']:
#     print('Fold:{0} Processing...'.format(i+1))
    name = 'data/processed/type_{0}_df.pkl'.format(i)
    temp_df = pd.read_pickle(name)
    
    train_df = temp_df[temp_df['type']=='train'].copy()
    test_df = temp_df[temp_df['type']=='test'].copy()
    
    train_u,train_i = analyze(data_df,train_df)
    test_u,test_i = analyze(data_df,test_df)
    print('*******************')
    print('Compare with full data, Train:usr{0:0.4} item{1:0.4};Test:usr{1:0.4} item{2:0.4}'.format(train_u,train_i,test_u,test_i))
    print('-------------------')
    u,i = analyze(train_df,test_df)
    print('Compare between train and test data, Usr:{0:0.4} Item:{1:0.4}'.format(u,i))