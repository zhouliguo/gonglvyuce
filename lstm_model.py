import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

import os
import csv
import numpy as np

class Power_Dataset(Dataset):
    def __init__(self, cfg, phase='train'):
        super().__init__()
        self.csv_data = []
        with open(cfg.data_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
    
            # 遍历CSV文件中的每一行
            for row in csv_reader:
                self.csv_data.append(row)
            self.csv_data = np.array(self.csv_data, np.float32)

    def __len__(self):
        return len(self.csv_data) - 24 * 12
    
    def __getitem__(self, idx):
        return self.csv_data[idx : idx + 20 * 12], self.csv_data[idx + 20 * 12 : idx + 24 * 12, 2]

class LSTMModel(nn.Module):
    def __init__(self, in_feature=3, out_features=48) -> None:
        super().__init__()
        self.in_feature = in_feature
        # self.linear0 = nn.Linear(2, in_feature)
        # self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(in_feature, 256, 2)
        self.linear1 = nn.Linear(256, out_features)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.size(0)
        x = x.view(bs, -1, self.in_feature)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear1(x[:,-1])
        
        return x
    
def train_val(cfg):
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
            device = torch.device('cuda:'+str(cfg.device))

    train_data = Power_Dataset(cfg=cfg, phase='train')
    val_data = Power_Dataset(cfg=cfg, phase='val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LSTMModel().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.MSELoss()

    for epoch_i in range(0, cfg.epochs):
        model.train()

        loss_sum = 0
        
        for train_i, (history_data, future_power) in enumerate(train_dataloader):
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            predicted_power = model(history_data)

            loss = loss_function(predicted_power, future_power)

            loss_sum = loss_sum + loss.item()
            if (train_i+1)%100 == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                print('Epoch:', epoch_i, 'Step:', train_i, 'Train Loss:', loss_sum/100, 'Learning Rate:', lr)
                loss_sum = 0

            loss.backward()
            optimizer.step()
        
        scheduler.step()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--data-path', type=str, default='data.csv', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=64, help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--num-workers', type=int, default=0, help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--lr-init', type=float, default=0.01, help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='e.g. cpu or 0 or 0,1,2,3')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    #Power_Dataset(cfg)
    train_val(cfg)
