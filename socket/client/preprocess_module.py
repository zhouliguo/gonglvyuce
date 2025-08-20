# preprocess_module.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def parse_epoch(series):
    max_val = series.max()
    if max_val > 1e15:
        ts = pd.to_datetime(series, unit='us', errors='coerce', utc=True)
    elif max_val > 1e12:
        ts = pd.to_datetime(series, unit='ms', errors='coerce', utc=True)
    else:
        ts = pd.to_datetime(series, unit='s', errors='coerce', utc=True)
    return ts

def json_to_dataframe(file_path,start_year="2025",start_month="5",start_day="13",
                      end_year="2025",end_month="06",end_day="22"):
    """
        将JSON文件转换为DataFrame格式
        :param file_path: JSON文件路径
    """
    # 将start_year, start_month, start_day, end_year, end_month, end_day转换为字符串
    start_date = f"{start_year}-{start_month}-{start_day}" if start_year and start_month and start_day else ""
    end_date = f"{end_year}-{end_month}-{end_day}" if end_year and end_month and end_day else ""
    # 选取两个日期之间所有的json文件
    all_json_list = os.listdir(file_path)
    print(f"所有的json文件: {len(all_json_list)}")
    conditional_json_list = []
    for json_file in all_json_list:
        if json_file.endswith('.json'):
            json_date = json_file.split('.')[0].split('_')[1]
            # print(f"正在处理文件: {json_date}")
            # print(json_date)
            if start_date and end_date:
                if start_date <= json_date <= end_date:
                    conditional_json_list.append(json_file)
    # 对所有的json文件进行遍历
    df_list = []
    for json_file in conditional_json_list:
        # print(f"正在处理文件: {json_file}")
        if json_file.endswith('.json'):
            file_path_full = os.path.join(file_path, json_file)
            # 跳过空文件
            if os.path.getsize(file_path_full) == 0:
                print(f"跳过空文件: {json_file}")
                continue
            # 检查文件是否存在
            if not os.path.exists(file_path_full):
                continue
            # 读取JSON文件
            with open(file_path_full, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # print(f"正在处理文件: {json_file}")
            df_day = pd.DataFrame([data['stationStatisticDay']])
            df_power = pd.DataFrame(data['stationStatisticPowerList'])
            # 针对df_day 选取usevalue,buyvalue,fullpowerhours列的数据：
            df_day = df_day[['acPowerHours','fullPowerHours',"generationValue"]]
            # print(df_power)
            df_power = df_power.loc[:, ['dateTime', 'generationPower']] # generationCapacity
            empty_columns = df_day.columns[df_day.isnull().all()]
            if not empty_columns.empty:
                print(f"警告: 在文件 {json_file} 中，以下列全为空: {', '.join(empty_columns)}")
                # 如果存在空列，使用即使用df_list最后一个DataFrame的列名
                if df_list:
                    last_day_data = df_list[-1].iloc[0]
                    df_day[empty_columns] = last_day_data[empty_columns].values
                    print(f"已使用最后一个DataFrame的值填充空列: {', '.join(empty_columns)}")
                else:
                    print("没有可用的DataFrame来填充空列。")
                    #填充问题列为0
                    df_day[empty_columns] = 0
                    print(f"已将空列填充为0: {', '.join(empty_columns)}")
            # 将df_day和df_power合并,由于df_day只有一行数据，而df_power有多行数据，所以需要进行广播
            df_day = df_day.loc[df_day.index.repeat(len(df_power))].reset_index(drop=True)
            df_power = df_power.reset_index(drop=True)

            df_power = pd.concat([df_day, df_power], axis=1)
            # 将处理后的DataFrame添加到列表中
            df_list.append(df_power)
            # print(len(df_list))
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # 将dateTime列转换为日期格式
        combined_df['dateTime'] = parse_epoch(combined_df['dateTime'])
        # 按需转换到 Asia/Shanghai 时区
        combined_df['dateTime'] = combined_df['dateTime'].dt.tz_convert('Asia/Shanghai')
        # 打印合并后的DataFrame的形状
        print(f"合并后的DataFrame形状: {combined_df.shape}")
        return combined_df
    else:
        print("没有有效的JSON文件可供处理。")
        return pd.DataFrame()

def preprocess_data(df, scalers):

    # df.sort_values("dateTime", inplace=True)

    # # 2. 时间特征工程（不改变原始时间值）
    df = df.dropna(subset=['dateTime']) 
    
    print(f"数据时间范围: {df['dateTime'].min()} 到 {df['dateTime'].max()}")
    # 基本时间特征 - 保留原始值
    df["hour"] = df["dateTime"].dt.hour.astype(int)  # 保持整型
    df["day_of_week"] = df["dateTime"].dt.dayofweek.astype(int)  # 0-6
    df["month"] = df["dateTime"].dt.month.astype(int)  # 1-12

    # 高级时间特征 - 使用数值表示（归一化0-1）
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # 将dateTime转换为时间戳（秒）
    df["timestamp"] = df["dateTime"].astype(np.int64) // 10**12  # 转换为秒级时间戳

    # 3. 处理缺失值

    # print(f"缺失值统计:\n{df.isnull().sum()}")
    for col in ['generationPower', 'fullPowerHours']:
        df[col] = df[col].interpolate(method='linear')

    # 4. 分别归一化处理
    power_feats    = ['generationPower']
    energy_feats   = ['generationValue']
    duration_feats = ['fullPowerHours']
    time_feats     = ['hour', 'day_of_week', 'month']


    if 'power_scaler' not in scalers:    raise ValueError('缺少 power_scaler')
    if 'energy_scaler' not in scalers:   raise ValueError('缺少 energy_scaler')
    if 'duration_scaler' not in scalers: raise ValueError('缺少 duration_scaler')

    df[power_feats]    = scalers['power_scaler'].transform(df[power_feats])
    df[energy_feats]   = scalers['energy_scaler'].transform(df[energy_feats])
    df[duration_feats] = scalers['duration_scaler'].transform(df[duration_feats])

    for feat in time_feats:
            maxv = scalers.get(f'time_{feat}_max', {'hour':24,'day_of_week':7,'month':12}[feat])
            df[f'{feat}_sin'] = np.sin(2*np.pi*df[feat]/maxv)
            df[f'{feat}_cos'] = np.cos(2*np.pi*df[feat]/maxv)
            df.drop(columns=[feat], inplace=True)

    return df, scalers
  


def load_and_prepare_data(test_path):
    """
    加载并准备NeuralProphet所需的数据格式
    """
    test_df = pd.read_csv(test_path)

    test_df.dropna(inplace=True)
    test_df = test_df.reset_index(drop=True)
    print(f"测试集大小: {len(test_df)}")

    test_df["dateTime"]  = pd.to_datetime(test_df["dateTime"],  utc=True).dt.tz_convert("Asia/Shanghai")
 
    test_df  = test_df[test_df["dateTime"] >= "2022-01-01"]

    feature_columns = [col for col in test_df.columns if col not in ["dateTime", "generationPower"]]

    # 重命名列
    test_df = test_df.rename(columns={"dateTime": "ds", "generationPower": "y"})
    # del train_df[""]

    # 确保时间排序
    test_df = test_df.sort_values("ds").reset_index(drop=True)
    #打印训练集和测试集第一天和最后一天的时间戳
    print(f"测试集第一天: {test_df['ds'].min()}, 最后一天: {test_df['ds'].max()}")
    # 验证时间连续性
    time_diff_test = (test_df["ds"].max() - test_df["ds"].min()).days
    print(f"测试集时间跨度: {time_diff_test} 天")

    return test_df, feature_columns



