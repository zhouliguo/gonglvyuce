import os
import socket
import pickle as pkl
from datetime import datetime, timedelta
import joblib  
import pandas as pd
import os
import numpy as np

from preprocess_module import json_to_dataframe, preprocess_data, load_and_prepare_data,unfold_forecast

HOST, PORT = "127.0.0.1", 4323  

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("server closed")
        mv[got:got+len(chunk)] = chunk
        got += len(chunk)
    return buf


def send_to_server(df: pd.DataFrame, host=HOST, port=PORT) -> pd.DataFrame | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))

      
        payload = pkl.dumps(df, protocol=pkl.HIGHEST_PROTOCOL)   # 发：4字节长度 + 正文
        s.sendall(len(payload).to_bytes(4, "big"))
        s.sendall(payload)

      
        head = recv_exact(s, 4)   # 收：4字节长度+正文
        n = int.from_bytes(head, "big")
        body = recv_exact(s, n)

        obj = pkl.loads(body)

        if isinstance(obj, dict) and "error" in obj: 
            print("[CLIENT][SERVER ERROR]:", obj["error"]) # 服务器是否返回错误
            return None

        
        if not isinstance(obj, pd.DataFrame): # 再校验类型
            print("[CLIENT] Unexpected reply type:", type(obj))

            return None

        return obj

    finally:
        s.close()


if __name__ == "__main__":
   
    json_dir = "/home/hao/Documents/功率预测0813/data/194"  # 历史功率文件目录
    start_year, start_month, start_day = "2025", "05", "02"
    end_year, end_month, end_day   = "2025", "06", "02"

    # 读本地 
    X_test = json_to_dataframe(
        json_dir,
        start_year=start_year, start_month=start_month, start_day=start_day,
        end_year=end_year,   end_month=end_month,   end_day=end_day
    )

    scaler_pkl_path = "scalers.pkl"  # 加载训练时保存的归一化器对象
    assert os.path.exists(scaler_pkl_path), f"缺少 {scaler_pkl_path}，请在训练后 joblib.dump(scalers, 'scalers.pkl')"
    scalers = joblib.load(scaler_pkl_path)
    
    X_test_processed, _ = preprocess_data(X_test,scalers=scalers) # 预处理（与训练一致
    X_test_processed.to_csv(os.path.join("X_test_processed.csv"), index=False)

    X_test_model, feature_columns = load_and_prepare_data("X_test_processed.csv")     # 转成模型需要的输入格式（保持 DataFrame）




    forecast = send_to_server(X_test_model)
    if forecast is None:
        print("本次预测失败，已在上面打印 server 的错误原因。")
        raise SystemExit(1)

    print(f"预测范围: {X_test['dateTime'].max()} 到 {forecast['ds'].max().tz_localize('Asia/Shanghai')}")


    last_24_rows = forecast.tail(24)

    results = []
    for idx, (i, row) in enumerate(last_24_rows.iterrows(), start=1):
        col = f"yhat{idx}"   # 第 i 行取 yhat{i}
        if col in row.index:
            results.append({"ds": row["ds"], "yhat": row[col]})

    final_preds = pd.DataFrame(results)
    print(final_preds) #打印预测的24h






