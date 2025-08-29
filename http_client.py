import numpy as np
import json
import requests
import argparse

# 功能：获取历史数据序列，序列中每一项包含：日期(年、月、日）、时间（0、1、2……287，一天中第n个5分钟）、功率值……，Json格式
# 参数：sequence_len，获取数据序列的长度。默认输入序列长度为240。
# 说明：请重写此函数从数据库中获取数据。
def get_history_data(sequence_len = 240):
    year = np.ones(sequence_len, np.int32) * 2025
    month = np.ones(sequence_len, np.int32) * 8
    day = np.ones(sequence_len, np.int32) * 31
    time = np.array(range(sequence_len)).astype(np.int32)
    power = np.random.uniform(0, 10000, sequence_len).astype(np.float32)

    history_data = np.concatenate((year, month, day, time, power))
    history_data = history_data.tolist()
    return json.dumps(history_data)

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://112.65.122.134:1920/', help='URL, 服务器ip和端口号')
    return parser.parse_args()

if __name__ == '__main__':
    history_data = get_history_data()

    cfg = parse_cfg()
    url = cfg.url 

    headers = {'Content-Type': 'application/json'}
    
    # 向服务器发送历史数据，并获得预测的未来4小时间隔5分钟的功率数据，序列长度为48
    response = requests.post(url, history_data, headers=headers) 
    pred_power = response.json()
    pred_power = np.array(pred_power, dtype=np.float32)
    print('pred power', pred_power)