import socket
 
host = '131.159.60.171'  # 或者服务器的 IP 地址
port = 1080  # 服务器的端口号
 
try:
    with socket.create_connection((host, port)):
        print("连接成功！")
except ConnectionRefusedError:
    print(f"[Errno 111] Connection refused to {host} on port {port}")
except Exception as e:
    print(f"发生错误：{e}")