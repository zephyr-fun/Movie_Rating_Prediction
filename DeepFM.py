#
#Author: zephyr
#Date: 2020-12-28 10:25:19
#LastEditors: zephyr
#LastEditTime: 2021-01-04 21:41:10
#FilePath: \MovieRatingPrediction\DeepFM.py
#
import torch
import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch.nn import Module
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

class FactorizationMachine(Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        --param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FeaturesEmbedding(Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.addi = torch.tensor(self.offsets).unsqueeze(0).long().to('cuda:0')
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # change
        # x[:,1] += 943 # self.offsets[1]
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # x = x + torch.tensor([0,943]).to('cuda:0')
        x = x + self.addi
        # x = x + x
        return self.embedding(x)

class FeaturesLinear(Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.addi = torch.tensor(self.offsets).unsqueeze(0).long().to('cuda:0')

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # change
        # x[:,1] += 943 # self.offsets[1]
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # x = x + torch.tensor([0,943]).to('cuda:0')
        x = x + self.addi
        # x = x + x
        return torch.sum(self.fc(x), dim=1) + self.bias

class MultiLayerPerceptron(Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            # layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # self.Linear1 = nn.Linear(input_dim, embed_dims[0])
        # # self.BN1 = nn.BatchNorm1d(embed_dims[0])
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout)
        # self.Linear2 = nn.Linear(embed_dims[1], embed_dims[1])
        # # self.BN2 = nn.BatchNorm1d(embed_dims[1])
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=dropout)
        # self.output_layer = nn.Linear(embed_dims[1], 1)


    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        # x = self.Linear1(x)
        # # x = self.BN1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.Linear2(x)
        # # x = self.BN2(x4)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.output_layer(x)
        return self.mlp(x)

class DeepFactorizationMachineModel(Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        # x = self.linear(x) + self.fm(embed_x)
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)

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

    model = DeepFactorizationMachineModel([total_usr, total_item], embed_dim=16, mlp_dims=(16, 16), dropout=0.2).to(device)
    print(model)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
                ], lr=0.005,momentum=0.9)
    
    mse_lst = []

    vis_train = Visdom(env='DeepFM_Train')
    vis_test = Visdom(env='DeepFM_Test')
 
    criterion = nn.MSELoss()
    
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(1):# change from epoch to 1
        for test_idx in range(5):
            print('----------------')
            print('Running epoch {0} Test on cv_{1}_df Time:{2}'.format(epoch_i,test_idx+1,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
            train_idx_lst = [i for i in range(5) if i!=test_idx]
            # 制作训练数据集
            train_df_lst = []
            for train_idx in train_idx_lst:
                df = cv_df_lst[train_idx]
                train_df_lst.append(df[df['type']=='train'])
            train_df = pd.concat(train_df_lst)
            train_dataset = MovieLens100kDataset(train_df)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
            
            print('----------------')
            print('Starting training porcess...')
            model.train()
            total_loss = 0
            # tk0 = tqdm.tqdm(train_dataloader, smoothing=0, mininterval=1.0)
            for num, (fields, target) in enumerate(train_dataloader):
                fields, target = fields.long().to(device), target.long().to(device)
                optimizer.zero_grad()
                y = model(fields)
                loss = criterion(y, target.float())
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                vis_train.line(X=torch.tensor([epoch_i*313*5+test_idx*313+num]), Y=torch.tensor([loss.item()]), win='DeepFM_train_loss', update='append', opts=dict(title='DeepFM_train_loss'))
                print('Epoch:{0} not in Trainset:{1} Batch:{2} Loss:{3}'.format(epoch_i,test_idx+1,num,loss))
                total_loss += loss.item()
                # if (num + 1) % 100 == 0:
                #     tk0.set_postfix(loss=total_loss / 100)
                #     total_loss = 0

            # 制作测试数据集
            df = cv_df_lst[test_idx]
            test_df = df[df['type']=='test']

            test_dataset = MovieLens100kDataset(test_df)
            test_dataloader = DataLoader(test_dataset,batch_size=batch_size)
            
            print('----------------')
            print('Starting testing porcess...')
            model.eval()
            temp_mse_lst = []
            targets, predicts = list(), list()
            # tk1 = tqdm.tqdm(test_dataloader, smoothing=0, mininterval=1.0)
            with torch.no_grad():
                for num, (fields, target) in enumerate(test_dataloader):
                    fields, target = fields.long().to(device), target.long().to(device)
                    y = model(fields)
                    mse_batch = criterion(y, target.float())
                    targets.extend(target.tolist())
                    predicts.extend(y.tolist())
                    temp_mse_lst.append(mse_batch.item())
                    vis_test.line(X=torch.tensor([epoch_i*20*5+test_idx*20+num]), Y=torch.tensor([mse_batch.item()]), win='DeepFM_test_loss', update='append',opts=dict(title='DeepFM_test_loss'))
                    print('Epoch:{0} Testset:{1} Batch:{2} Loss:{3}'.format(epoch_i,test_idx+1,num,mse_batch))
                avg_mse = np.mean(temp_mse_lst)
                mse_lst.append(avg_mse)
                print('Epoch:{0} MSE on cv_{1}_df:{2} Time:{3}'.format(epoch_i,test_idx+1,avg_mse,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
            print('Epoch:{0} Average MSE:{1} Time:{2}'.format(epoch_i,np.average(mse_lst),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
    # torch.save(model, 'DeepFM_Model/DeepFM_Model_save.pt')
    # fm = FactorizationMachine()
    # fe = FeaturesEmbedding([943,1682],16)
    # fl = FeaturesLinear([943,1682])
    # mlp = MultiLayerPerceptron(32,(16,16),0.2)
    # nn = DeepFactorizationMachineModel([943, 1682], embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    # print(mlp)
    scripted_model = torch.jit.script(model)
    scripted_model.save('DeepFM_Model/model.pt')
    print('torch_jit save model successfully to DeepFM_Model/DeepFM_Model.pt')
    # dummpy_input = torch.randn(1024,2).long().to(device)
    # torch.onnx.export(model=model,args=dummpy_input,f='DeepFM_Model/DeepFM_Model_ONNX.onnx')
    # print('torch_onnx save model successfully to DeepFM_Model/DeepFM_Model_ONNX.onnx')


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
