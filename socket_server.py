import socket
import numpy as np
import time

import torch
import torchvision
from torchvision import transforms

def socket_init(ip_address = 'localhost', port = 1080, backlog = 50):
    socket_s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket对象
    socket_s.bind((ip_address, port))   # 绑定地址
    socket_s.listen(backlog)  # 建立backlog个监听
    return socket_s

def model_init(device = 'cpu'):
    model = torchvision.models.resnet50(pretrained=True).to(device)
    return model

if __name__ == '__main__':
    w = 640
    h = 640
    required = w*h*3    # 接收的字节数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    socket_s = socket_init('192.168.1.7', 1080)
    model = model_init(device)

    while True:
        conn, addr= socket_s.accept()   # 等待客户端连接
        while True:

            data=conn.recv(required)    # 从客户端接收长度为required的数据
            if len(data) <= 0:
                break
            while len(data) < required: # 若未接收到指定长度的数据，则继续接收
                data += conn.recv(required - len(data))
            
            image = np.frombuffer(data, np.uint8).reshape((h, w, 3)).transpose(2,0,1).astype(np.float32)/255
            image = torch.tensor(image).unsqueeze(0)
            image = test_transform(image).to(device)
            output = model(image).detach().cpu().numpy()

            conn.send(output.tobytes())