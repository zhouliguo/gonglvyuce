import numpy as np
import json
import requests

# 功能：获取历史数据序列，序列中每一项包含：日期(年、月、日）、时间（0、1、2……287，一天中第n个5分钟）、功率值……，Json格式
# 参数：sequence_len，获取数据序列的长度
def get_history_data(sequence_len = 240):
    year = np.ones(sequence_len, np.int32) * 2025
    month = np.ones(sequence_len, np.int32) * 8
    day = np.ones(sequence_len, np.int32) * 31
    time = np.array(range(sequence_len)).astype(np.int32)
    power = np.random.uniform(0, 10000, sequence_len).astype(np.float32)

    history_data = np.concatenate((year, month, day, time, power))
    history_data = history_data.tolist()
    return json.dumps(history_data)

if __name__ == '__main__':
    history_data = get_history_data()

    address = 'http://112.65.122.121:1920/'
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(address, history_data, headers=headers)   # 向服务器传输历史数据，并获得预测的功率数据
    pred_power = response.json()

    # response = requests.get('http://localhost:8000/')
    # pred_data = response.text
    # print(pred_data)

    pred_power = np.array(pred_power, dtype=np.float32)
    print('pred power', pred_power)