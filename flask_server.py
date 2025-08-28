#/media/sdb4/gonglvyuce/flask_server.py

import torch
from torch import nn, Tensor
import numpy as np
import json
from datetime import date

from flask import Flask
from flask import request

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

def get_history_data(sequence_len = 240):
    year = np.ones(sequence_len, np.int32) * 2025
    month = np.ones(sequence_len, np.int32) * 8
    day = np.ones(sequence_len, np.int32) * 31
    time = np.array(range(sequence_len)).astype(np.int32)
    power = np.random.uniform(0, 10000, sequence_len).astype(np.float32)

    history_data = np.concatenate((year, month, day, time, power))
    history_data = history_data.tolist()
    return json.dumps(history_data)

# 数据预处理
def history_data_prep(history_data):
    history_data = np.array(history_data.strip('[').strip(']').split(',')).astype(np.float32)
    sequence_len = int(len(history_data)/5)
    history_data = history_data.reshape((sequence_len,-1), order='F')

    day = np.zeros(sequence_len, np.float32)
    delta_middle = (date(int(history_data[0, 0]), 7, 2) - date(int(history_data[0, 0]), 1, 1)).days
    for i in range(sequence_len):
        delta = (date(int(history_data[i, 0]), int(history_data[i, 1]), int(history_data[i, 2])) - date(int(history_data[i, 0]), 1, 1)).days
        if delta > delta_middle:
            delta = delta_middle - (delta - delta_middle)
        day[i] = delta

    day = day / 183.0

    time = history_data[:, 3]
    time[time > 144] = 144 - (time[time > 144] - 144)
    time = time / 144.0

    power = history_data[:, 4]

    history_data = np.concatenate((day, time, power)).reshape((sequence_len,-1), order='F')

    return history_data
    

# 深度神经网络(DNN)模型
def dnn_model(history_data):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = LSTMModel().eval().to(device)
    history_data = torch.from_numpy(history_data).unsqueeze(0).to(device)

    predicted_power = model(history_data)

    predicted_power = predicted_power.detach().cpu().numpy()

    return predicted_power[0]

app = Flask(__name__)
 
@app.route('/', methods=['POST'])
def power_prediction():
    print('Response!')

    history_data = request.get_json()   # 从客户端接收历史数据：日期、时间、功率……，Json格式
    print('history data json', history_data)

    history_data = history_data_prep(history_data)

    predicted_power = dnn_model(history_data)   # 将历史数据输入DNN模型，获得预测的功率数据
    print('predicted power', predicted_power)

    predicted_power = json.dumps(predicted_power.tolist())  # 将预测的功率数据转换成Json格式

    return predicted_power, 200
 
if __name__ == '__main__':
    history_data = get_history_data()
    history_data = history_data_prep(history_data)
    predicted_power = dnn_model(history_data)
    #app.run(debug=True, host='192.168.110.93', port=1080)