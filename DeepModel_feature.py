#
#Author: zephyr
#Date: 2020-12-26 15:22:07
#LastEditors: zephyr
#LastEditTime: 2021-01-04 17:10:49
#FilePath: \MovieRatingPrediction\DeepModel_feature.py
#
import pickle
import os
import re
import copy
from collections import Counter
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim
from sklearn.metrics import mean_squared_error
from visdom import Visdom

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

data_df = pd.read_pickle('data/processed/data_df.pkl')

total_usr = len(set(data_df.usr_id.tolist()))
total_item = len(set(data_df.item_id.tolist()))

usr_feature_df = pd.read_pickle('data/processed/usr_feature_df.pkl')
item_feature_df = pd.read_pickle('data/processed/item_feature_df.pkl')

u_feature_dim = usr_feature_df['features'][0].shape[0]
i_feature_dim = item_feature_df['features'][0].shape[0]

# 获取用户和物品特征
def get_UI_feature(uid,iid,
           usr_feautre_df=usr_feature_df,
           item_feature_df=item_feature_df):
    '''给定用户和物品id进行特征查找 此处id是处理后的id即-1后的id
    '''
    
    u_feature = usr_feature_df.loc[uid,'features']

    u_feature = np.array(u_feature)

    i_feature = item_feature_df.loc[iid,'features']

    i_feature = np.array(i_feature)

    return u_feature,i_feature

# 深度模型 修改加入特征处理部分
class RecManModel(torch.nn.Module):
    def __init__(self,
                 usr_num:int,
                 item_num:int,
                 emb_usr_size:int,
                 emb_item_size:int,
                 usr_feature_dim:int,
                 item_feature_dim:int,
                 feature_hidden_size:int,
                 interact_hidden_size:int):
        '''
        usr_num 和 item_num 为对应用户和物品的总数
        emb_size设定用户和物品的隐变量维度
        feature_dim 为对应用户和物品的特征维度 处理后拼接
        feature_hidden_size 为输入特征映射的维度
        interact_hidden_size 为原来交互时特征维度
        '''
        super(RecManModel,self).__init__()
        self.usr_num = usr_num
        self.item_num = item_num
        self.emb_usr_size = emb_usr_size
        self.emb_item_size = emb_item_size
        self.usr_feature_dim = usr_feature_dim
        self.item_feature_dim = item_feature_dim
        self.feature_hidden_size = feature_hidden_size
        self.interact_hidden_size = interact_hidden_size
        
        # 查看显卡设备是否可用 
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
            
        self.UserEmbeddingLayer = torch.nn.Embedding(num_embeddings=self.usr_num,
                                                     embedding_dim=self.emb_usr_size)
        self.ItemEmbeddingLayer = torch.nn.Embedding(num_embeddings=self.item_num,
                                                     embedding_dim=self.emb_item_size)
        
        self.UserFeatureLayer = torch.nn.Linear(in_features=self.emb_usr_size,
                                                out_features=self.interact_hidden_size)
        self.ItemFeatureLayer = torch.nn.Linear(in_features=self.emb_item_size,
                                               out_features=self.interact_hidden_size)
        
        self.UserMannualFeatureLayer = torch.nn.Linear(in_features=self.usr_feature_dim,
                                                       out_features=self.feature_hidden_size)
        self.ItemMannualFeatureLayer = torch.nn.Linear(in_features=self.item_feature_dim,
                                                       out_features=self.feature_hidden_size)
        
        
        # to device
        self.UserEmbeddingLayer.to(self.device)
        self.UserFeatureLayer.to(self.device)
        self.ItemEmbeddingLayer.to(self.device)
        self.ItemFeatureLayer.to(self.device)
        self.UserMannualFeatureLayer.to(self.device)
        self.ItemMannualFeatureLayer.to(self.device)
        
        
    def forward(self,uid_batch,iid_batch,u_man_feature_batch,i_man_feature_batch):
        '''输入一个batch的usr和item进行交互
        '''
        # 血的教训,不要在forward函数里处理数据格式,会导致后边jit保存模型后无法显示网络结构
        # if not torch.is_tensor(uid_batch):
        #     u_batch_tensor = torch.tensor(uid_batch)
        # else:
        #     u_batch_tensor = uid_batch
        # if not torch.is_tensor(iid_batch):
        #     i_batch_tensor = torch.tensor(iid_batch)
        # else:
        #     i_batch_tensor = iid_batch
        
        # if not torch.is_tensor(u_man_feature_batch):
        #     u_man_feature_batch = torch.tensor(u_man_feature_batch)
        # if not torch.is_tensor(i_man_feature_batch):
        #     i_man_feature_batch = torch.tensor(i_man_feature_batch)
        
        u_batch_tensor = uid_batch
        i_batch_tensor = iid_batch
        u_man_feature_batch = u_man_feature_batch
        i_man_feature_batch = i_man_feature_batch

        # 装入设备
        u_batch_tensor = u_batch_tensor.to(self.device)
        i_batch_tensor = i_batch_tensor.to(self.device)
        u_man_feature_batch = u_man_feature_batch.to(self.device)
        i_man_feature_batch = i_man_feature_batch.to(self.device)
        
        # 数据类型转换
        u_man_feature_batch = u_man_feature_batch.to(torch.float)
        i_man_feature_batch = i_man_feature_batch.to(torch.float)
        
        # 嵌入 向量化
        
        u_emb_tensor = self.UserEmbeddingLayer(u_batch_tensor)
        i_emb_tensor = self.ItemEmbeddingLayer(i_batch_tensor)
        
        # 特征抽取 和 非线性化
        u_feature = self.UserFeatureLayer(u_emb_tensor)
        i_feature = self.ItemFeatureLayer(i_emb_tensor)
        
        u_feature = torch.relu(u_feature)
        i_feature = torch.relu(i_feature)
        
        # 外部特征映射
        
        u_mannual_feature = self.UserMannualFeatureLayer(u_man_feature_batch)
        i_mannual_feature = self.ItemMannualFeatureLayer(i_man_feature_batch)
        
        u_mannual_feature = torch.relu(u_mannual_feature)
        i_mannual_feature = torch.relu(i_mannual_feature)
        
        # 隐式显式特征拼接
        u_final_feature = torch.cat([u_feature,u_mannual_feature],dim=1)
        i_final_feature = torch.cat([i_feature,i_mannual_feature],dim=1)
        
        batch_size = u_feature.shape[0]
        interact_hidden_size = self.feature_hidden_size + self.interact_hidden_size
        u_final_feature = u_final_feature.reshape(batch_size,1,interact_hidden_size)
        i_final_feature = i_final_feature.reshape(batch_size,interact_hidden_size,1)
        
        output = torch.bmm(u_final_feature,i_final_feature)
        output = torch.squeeze(output)

        return output
    
nn_sample = RecManModel(usr_num=total_usr,item_num=total_item,
                emb_usr_size=50,emb_item_size=150,
                usr_feature_dim=u_feature_dim,item_feature_dim=i_feature_dim,
                feature_hidden_size=10,interact_hidden_size=25)

# 首次小样本验证
# sample_1st = data_df.sample(n=32)# 随机选择32个条目

# usr_batch = [int(i) for i in sample_1st.usr_id.tolist()]
# item_batch = [int(i) for i in sample_1st.item_id.tolist()]
# sample_1st =sample_1st.reset_index(drop=True)# 重设索引

# usr_feature,item_feature = [],[]
# for num,i in enumerate(sample_1st.itertuples()):
    
#     usr_id = int(sample_1st.loc[num,'usr_id'])-1
#     item_id = int(sample_1st.loc[num,'item_id'])-1
#     u,i = get_UI_feature(usr_id,item_id)
#     usr_feature.append(u)
#     item_feature.append(i)
# usr_feature = np.stack(usr_feature)
# item_feature = np.stack(item_feature)
# nn_sample(usr_batch,item_batch,usr_feature,item_feature)

class MLManDataSet(Dataset):
    def __init__(self,df,usr_feature_df=usr_feature_df,item_feature_df=item_feature_df):
        '''
        输入df 构造dataset
        输出 样本编号和标注
        '''
        self.df = df.copy()
        # 注意 原始数据用户id和物品id从1开始的，但是在embedding过程中是从0算的，因此此处减一
        self.df['usr_id'] = df['usr_id'].apply(lambda x:int(x)-1)
        self.df['item_id'] = df['item_id'].apply(lambda x:int(x)-1)
        self.df['rating'] = df['rating'].apply(lambda x:int(x))
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_df = self.df.iloc[idx]
        usr_feature,item_feature = get_UI_feature(sample_df.usr_id,sample_df.item_id)
        sample = (sample_df.usr_id,sample_df.item_id,usr_feature,item_feature,sample_df.rating)
        return sample
# 完整训练过程
cv_df_lst = []
for i in range(5):
    df_fname = 'data/processed/cv_{0}_df.pkl'.format(i+1)
    df = pd.read_pickle(df_fname)
    cv_df_lst.append(df)

model = RecManModel(usr_num=total_usr,item_num=total_item,
                 emb_usr_size=50,emb_item_size=150,
                 usr_feature_dim=u_feature_dim,item_feature_dim=i_feature_dim,
                 feature_hidden_size=10,interact_hidden_size=25)
optimizer = torch.optim.SGD([
        {'params': model.parameters()},
                ], lr=0.005,momentum=0.9)
BATCH_SIZE = 1024

mse_lst = []
vis_train = Visdom(env='Deep_Feature_Model_Train')
vis_test = Visdom(env='Deep_Feature_Model_Test')
for epoch in range(50):
    for test_idx in range(5):
        print('----------------')
        print('Running epoch {0} Test on cv_{1}_df Time:{2}'.format(epoch,test_idx+1,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
        train_idx_lst = [i for i in range(5) if i!=test_idx]
        # 训练过程
        train_df_lst = []
        for train_idx in train_idx_lst:
            df = cv_df_lst[train_idx]
            train_df_lst.append(df[df['type']=='train'])
        train_df = pd.concat(train_df_lst)
        train_dataset = MLManDataSet(train_df)
        train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
        

        model.train()
        
        print('----------------')
        print('Starting training porcess...')
        for num,(uid_batch,iid_batch,usr_feature,item_feature,true_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            criterion = torch.nn.MSELoss()
            # 提前处理数据格式
            u_batch_tensor = torch.as_tensor(uid_batch)
            i_batch_tensor = torch.as_tensor(iid_batch)
            u_man_feature_batch = torch.as_tensor(usr_feature)
            i_man_feature_batch = torch.as_tensor(item_feature)
            pred_y = model(uid_batch,iid_batch,usr_feature,item_feature)
            true_y = true_y.to(torch.float).to(DEVICE)
            loss = criterion(pred_y,true_y)
            loss.requires_grad_(True)
            loss.backward()
            vis_train.line(X=torch.tensor([epoch*313*5+test_idx*313+num]), Y=torch.tensor([loss.item()]), win='train_loss', update='append', opts=dict(title='train_loss'))
            print('Epoch:{0} not in Trainset:{1} Batch:{2} Loss:{3}'.format(epoch,test_idx+1,num,loss))
            optimizer.step()
        
        # 测试过程
        model.eval()
        df = cv_df_lst[test_idx]
        test_df = df[df['type']=='test']

        test_dataset = MLManDataSet(test_df)
        test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
        
        temp_mse_lst = []
        print('----------------')
        print('Starting testing porcess...')
        # 分batch进行MSE计算 最后平均
        for num,(test_uid_batch,test_iid_batch,test_usr_feature,test_item_feature,test_true_y) in enumerate(test_dataloader):
        
            test_pred_y = model(test_uid_batch,test_iid_batch,test_usr_feature,test_item_feature)
            test_pred_y = test_pred_y.cpu().detach().numpy()
            
            mse_batch = mean_squared_error(test_true_y,test_pred_y)
            temp_mse_lst.append(mse_batch)
            vis_test.line(X=torch.tensor([epoch*20*5+test_idx*20+num]), Y=torch.tensor([mse_batch.item()]), win='test_loss', update='append',opts=dict(title='test_loss'))
            print('Epoch:{0} Testset:{1} Batch:{2} Loss:{3}'.format(epoch,test_idx+1,num,mse_batch))
            
        mse = np.average(temp_mse_lst)
        mse_lst.append(mse)
        print('Epoch:{0} MSE on cv_{1}_df:{2} Time:{3}'.format(epoch,test_idx+1,mse,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
    print('Epoch:{0} Average MSE:{1} Time:{2}'.format(epoch,np.average(mse_lst),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))

torch.save(model.state_dict,'Deep_Feature_Model/Deep_Feature_Model_Weights.bin')
print('torch save model successfully to Deep_Feature_Model/Deep_Feature_Model_Weights.bin')

scripted_model = torch.jit.script(model)
scripted_model.save('Deep_Feature_Model/Deep_Feature_Model_1.pt')
print('torch_jit save model successfully to Deep_Feature_Model/Deep_Feature_Model.pt')

# loaded_model = torch.jit.load('./RecModel.pt')

# for i in model.state_dict().keys():
#     print(i)

# loaded_model.state_dict()['UserEmbeddingLayer.weight']
# model.state_dict()['UserEmbeddingLayer.weight']
# torch.jit.trace_module()

# torch.onnx open neural network exchange
# sample = data_df.sample(n=32)

# usr_batch = [int(i) for i in sample.usr_id.tolist()]
# item_batch = [int(i) for i in sample.item_id.tolist()]

# usr_batch = torch.tensor(usr_batch)
# item_batch = torch.tensor(item_batch)

# usr_feature,item_feature = get_UI_feature(sample.usr_id,sample.item_id)

# dummpy_input = (usr_batch,item_batch,usr_feature,item_feature)

# torch.onnx.export(model=model,args=dummpy_input,f='Deep_Feature_Model/Deep_Feature_Model_ONNX.onx')
# print('torch_onnx save model successfully to Deep_Feature_Model/Deep_Feature_Model_ONNX.onx')
# import onnx

# # Load the ONNX model
# model = onnx.load("Rec_OONX_model.onx")

# # Check that the IR is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)