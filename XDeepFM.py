#
#Author: zephyr
#Date: 2020-12-28 14:48:57
#LastEditors: zephyr
#LastEditTime: 2021-01-04 21:39:56
#FilePath: \MovieRatingPrediction\XDeepFM.py
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

class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        Input:
        --param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = []
        x0, h = x.unsqueeze(2), x
        for i, conv in enumerate(self.conv_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            # for draw arc hide blow
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(conv(x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.addi = torch.tensor(self.offsets).unsqueeze(0).long().to('cuda:0')
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x + self.addi
        return self.embedding(x)

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.addi = torch.tensor(self.offsets).unsqueeze(0).long().to('cuda:0')

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x + self.addi
        return torch.sum(self.fc(x), dim=1) + self.bias

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

class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        # x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
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

    model = ExtremeDeepFactorizationMachineModel([total_usr, total_item], embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2).to(device)
    
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
                ], lr=0.005,momentum=0.9)
    
    mse_lst = []

    vis_train = Visdom(env='XDeepFM_Train')
    vis_test = Visdom(env='XDeepFM_Test')
 
    criterion = torch.nn.MSELoss()
    
    # early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
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
                vis_train.line(X=torch.tensor([epoch_i*313*5+test_idx*313+num]), Y=torch.tensor([loss.item()]), win='XDeepFM_train_loss', update='append', opts=dict(title='XDeepFM_train_loss'))
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
                    vis_test.line(X=torch.tensor([epoch_i*20*5+test_idx*20+num]), Y=torch.tensor([mse_batch.item()]), win='XDeepFM_test_loss', update='append',opts=dict(title='XDeepFM_test_loss'))
                    print('Epoch:{0} Testset:{1} Batch:{2} Loss:{3}'.format(epoch_i,test_idx+1,num,mse_batch))
                avg_mse = np.mean(temp_mse_lst)
                mse_lst.append(avg_mse)
                print('Epoch:{0} MSE on cv_{1}_df:{2} Time:{3}'.format(epoch_i,test_idx+1,avg_mse,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
            print('Epoch:{0} Average MSE:{1} Time:{2}'.format(epoch_i,np.average(mse_lst),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))))
    # Cin = CompressedInteractionNetwork(2, (16,16), False)
    scripted_model = torch.jit.script(model)
    scripted_model.save('XDeepFM_Model/XDeepFM_Model.pt')
    print('torch_jit save model successfully to XDeepFM_Model/XDeepFM_Model.pt')

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
