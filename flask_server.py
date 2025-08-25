import torch
import numpy as np
import json

from flask import Flask
from flask import request

# 深度神经网络(DNN)模型
def dnn_model(history_data):
    predicted_power = torch.rand(48, dtype=torch.float32)
    return predicted_power.numpy()

app = Flask(__name__)
 
@app.route('/', methods=['POST'])
def power_prediction():
    print('Response!')

    history_data = request.get_json()   # 从客户端接收历史数据：日期、时间、功率……，Json格式
    print('history data', history_data)

    predicted_power = dnn_model(history_data)   # 将历史数据输入DNN模型，获得预测的功率数据
    print('predicted power', predicted_power)

    predicted_power = json.dumps(predicted_power.tolist())  # 将预测的功率数据转换成Json格式

    return predicted_power, 200
 
if __name__ == '__main__':
    app.run(debug=True, port=8000)