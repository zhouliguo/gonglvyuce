import numpy as np
import json
import requests

# 获取历史数据：日期、时间、功率……，Json格式
def get_history_data():
    history_data = np.random.uniform(0, 10000, (3,48)).astype(np.float32)
    return json.dumps(history_data.tolist())

if __name__ == '__main__':
    history_data = get_history_data()

    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:8000/', history_data, headers=headers)   # 向服务器传输历史数据，并获得预测的功率数据
    pred_data = response.json()

    # response = requests.get('http://localhost:8000/')
    # pred_data = response.text
    # print(pred_data)

    pred_data = np.array(pred_data, dtype=np.float32)
    print('pred data', pred_data)