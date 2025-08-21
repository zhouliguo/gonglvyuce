# 功率预测客户端程序

import socket   
import time
import cv2
import numpy as np
import sys

# 获取用于模型输入的历史数据：时间、天气、功率
def get_history_data():
    image = cv2.imread('image.jpg')
    return image

def socket_init(ip_address = 'localhost', port = 1080):
    # 初始化socket
    socket_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket对象
    socket_c.connect((ip_address, port))    # 建立连接
    bufsize = socket_c.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    print('bufsize:', bufsize)
    return socket_c

if __name__ == '__main__':
    ip_address = 'localhost'    # ip地址，'112.0.133.147'
    port = 1080 # 端口号

    socket_c = socket_init(ip_address, port)

    required = 1000*4   #预测未来4小时功率，每小时12个值

    while True:
        history_data = get_history_data()
        if history_data is None:
            break

        socket_c.send(history_data.tobytes())   # 向服务器发送历史数据
        pred_data = socket_c.recv(required) # 接收预测的功率

        pred_data = np.frombuffer(pred_data, np.float32)
        print(pred_data)

    socket_c.close()    # 关闭客户端连接
    sys.exit(0)
