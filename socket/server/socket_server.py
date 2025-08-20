import socket
import pickle as pkl
import pandas as pd
from neuralprophet import NeuralProphet
import joblib
import torch
try:
    from neuralprophet import save, load
except Exception:
    from neuralprophet.utils import save, load

import warnings, pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# 1.加载模型与归一化器

SCALERS_PATH = "scalers.pkl"

scalers = joblib.load(SCALERS_PATH)

model = load("all_neuralprophet_model.np", map_location="cpu")


def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("client disconnected")
        mv[got:got+len(chunk)] = chunk
        got += len(chunk)
    return buf


def inverse_normalize_y(forecast, scalers):  #对预测结果反归一化
    if "power_scaler" in scalers:
        # 反归一化功率
        power_scaler = scalers["power_scaler"]
        forecast["y"] = power_scaler.inverse_transform(forecast[["y"]])
        # 反归一化其他功率特征
        for col in forecast.columns:
            if col.startswith("yhat"):
                forecast[col] = power_scaler.inverse_transform(forecast[[col]])
                # print(f"归一化了{col}")
        
    else:
        raise ValueError("缺少能量归一化器")

    return forecast





#预测
config = {
    "forecast_horizon": 2 * 12,  # 预测未来24小时
    "history_window": 24 * 12,  # 使用过去48小时的数据进行预测
    "learning_rate": 1e-3,
    "loss_func":"MSE",
    "seasonality_mode": "additive",  # 可选：'additive' 或 'multiplicative'
    "quantiles": [0.1, 0.5, 0.9],  # 用于预测区间
}

def predict(model: NeuralProphet, test_df: pd.DataFrame) -> pd.DataFrame:
    print(f"开始预测，测试集大小: {len(test_df)}")
    future = model.make_future_dataframe(
        test_df, periods=config["forecast_horizon"], n_historic_predictions=True
    )

    forecast = model.predict(future)
    print("预测完成")
    return forecast


# 启动 socket server 
def start_server(host="0.0.0.0", port=4323):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"[SERVER] 正在监听 {host}:{port} ...")

    while True:
        conn, addr = server.accept()   
        print(f"[SERVER] 接收到连接：{addr}")
        try:
           
            header = recv_exact(conn, 4)   # 先收4字节长度，再收正文
            data_size = int.from_bytes(header, "big")
            data_bytes = recv_exact(conn, data_size)

            
            df = pkl.loads(data_bytes)  # 反序列化输入 DataFrame
            if not isinstance(df, pd.DataFrame):
                raise TypeError("收到的不是 pandas DataFrame")
            print(f"[SERVER] 收到 DataFrame,形状: {df.shape}")

            forecast = predict(model, df) #模型开始预测
            forecast = forecast.copy()

            forecast = inverse_normalize_y(forecast, scalers)
            forecast["ds"] = forecast["ds"] + pd.Timedelta(hours=8)
            print(f"换为北京市区之后预测结果的时间戳范围: {forecast['ds'].min()} 到 {forecast['ds'].max()}")



            forecast_bytes = pkl.dumps(forecast, protocol=pkl.HIGHEST_PROTOCOL)
            conn.sendall(len(forecast_bytes).to_bytes(4, "big"))
            conn.sendall(forecast_bytes)
            print(f"[SERVER] 返回 {len(forecast)} 行")

        except Exception as e:
            import traceback, pickle
            print("[SERVER][ERROR]", repr(e))
            traceback.print_exc()
            err = pkl.dumps({"error": repr(e)}, protocol=pkl.HIGHEST_PROTOCOL)
            try:
                conn.sendall(len(err).to_bytes(4, "big"))
                conn.sendall(err)
            except Exception:
                pass

        finally:
            conn.close()

if __name__ == "__main__":
    start_server()
