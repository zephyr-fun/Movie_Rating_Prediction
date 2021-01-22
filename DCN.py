#
#Author: zephyr
#Date: 2020-12-29 10:18:21
#LastEditors: zephyr
#LastEditTime: 2021-01-04 21:39:24
#FilePath: \MovieRatingPrediction\DCN.py
#
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F
from visdom import Visdom
import time

class MovieLens100kDataset(Dataset):
    """
    MovieLens 100k Dataset

    """
    def __init__(self, df):
        df['rating'] = df['rating'].apply(lambda x:int(x))
        self.data = df.values
        self.items = self.data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.data[index, 2]

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.addi = torch.tensor(self.offsets).unsqueeze(0).long().to('cuda:0')
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        --param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x + self.addi
        return self.embedding(x)

class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i, ww in enumerate(self.w):
            xw = ww(x)
            x = x0 * xw + self.b[i] + x
            # only for draw arc
            # x = x0 * xw + x
        return x

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        # return torch.sigmoid(p.squeeze(1))
        return p.squeeze(1)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def main(epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):

    device = torch.device(device)
    data_df = pd.read_pickle('data/processed/data_df.pkl')

    total_usr = len(set(data_df.usr_id.tolist()))
    total_item = len(set(data_df.item_id.tolist()))

    cv_df_lst = []
    for i in range(5):
        df_fname = 'data/processed/cv_{0}_df.pkl'.format(i+1)
        df = pd.read_pickle(df_fname)
        cv_df_lst.append(df)

    model = DeepCrossNetworkModel([total_usr, total_item], embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2).to(device)
    
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
                ], lr=0.005,momentum=0.9)
    
    mse_lst = []

    vis_train = Visdom(env='DCN_Train')
    vis_test = Visdom(env='DCN_Test')
 
    criterion = torch.nn.MSELoss()
    
    # early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    # for epoch_i in range(epoch):
    #     for test_idx in range(5):
    #         print('----------------')
    #         print('Running epoch {0} Test on cv_{1}_df Time:{2}'.format(epoch_i,test_idx+1,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
    #         train_idx_lst = [i for i in range(5) if i!=test_idx]
    #         # 制作训练数据集
    #         train_df_lst = []
    #         for train_idx in train_idx_lst:
    #             df = cv_df_lst[train_idx]
    #             train_df_lst.append(df[df['type']=='train'])
    #         train_df = pd.concat(train_df_lst)
    #         train_dataset = MovieLens100kDataset(train_df)
    #         train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
            
    #         print('----------------')
    #         print('Starting training porcess...')
    #         model.train()
    #         total_loss = 0
    #         # tk0 = tqdm.tqdm(train_dataloader, smoothing=0, mininterval=1.0)
    #         for num, (fields, target) in enumerate(train_dataloader):
    #             fields, target = fields.long().to(device), target.long().to(device)
    #             optimizer.zero_grad()
    #             y = model(fields)
    #             loss = criterion(y, target.float())
    #             loss.requires_grad_(True)
    #             loss.backward()
    #             optimizer.step()
    #             vis_train.line(X=torch.tensor([epoch_i*313*5+test_idx*313+num]), Y=torch.tensor([loss.item()]), win='DCN_train_loss', update='append', opts=dict(title='DCN_train_loss'))
    #             print('Epoch:{0} not in Trainset:{1} Batch:{2} Loss:{3}'.format(epoch_i,test_idx+1,num,loss))
    #             total_loss += loss.item()
    #             # if (num + 1) % 100 == 0:
    #             #     tk0.set_postfix(loss=total_loss / 100)
    #             #     total_loss = 0

    #         # 制作测试数据集
    #         df = cv_df_lst[test_idx]
    #         test_df = df[df['type']=='test']

    #         test_dataset = MovieLens100kDataset(test_df)
    #         test_dataloader = DataLoader(test_dataset,batch_size=batch_size)
            
    #         print('----------------')
    #         print('Starting testing porcess...')
    #         model.eval()
    #         temp_mse_lst = []
    #         targets, predicts = list(), list()
    #         # tk1 = tqdm.tqdm(test_dataloader, smoothing=0, mininterval=1.0)
    #         with torch.no_grad():
    #             for num, (fields, target) in enumerate(test_dataloader):
    #                 fields, target = fields.long().to(device), target.long().to(device)
    #                 y = model(fields)
    #                 mse_batch = criterion(y, target.float())
    #                 targets.extend(target.tolist())
    #                 predicts.extend(y.tolist())
    #                 temp_mse_lst.append(mse_batch.item())
    #                 vis_test.line(X=torch.tensor([epoch_i*20*5+test_idx*20+num]), Y=torch.tensor([mse_batch.item()]), win='DCN_test_loss', update='append',opts=dict(title='DCN_test_loss'))
    #                 print('Epoch:{0} Testset:{1} Batch:{2} Loss:{3}'.format(epoch_i,test_idx+1,num,mse_batch))
    #             avg_mse = np.mean(temp_mse_lst)
    #             mse_lst.append(avg_mse)
    #             print('Epoch:{0} MSE on cv_{1}_df:{2} Time:{3}'.format(epoch_i,test_idx+1,avg_mse,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
    #         print('Epoch:{0} Average MSE:{1} Time:{2}'.format(epoch_i,np.average(mse_lst),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))

    scripted_model = torch.jit.script(model)
    scripted_model.save('DCN_Model/DCN_Model.pt')
    print('torch_jit save model successfully to DCN_Model/DCN_Model.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
