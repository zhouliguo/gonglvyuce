import json
import glob
import numpy as np
import os
from datetime import date

json_files = glob.glob('功率预测0813/data/772/*.json')

day = []    # 0~364/365
time = []   # 0~287
power = []
for json_file in json_files:
    y_m_d = os.path.basename(json_file).split('.')[0].split('_')[1].split('-')
    delta_middle = (date(int(y_m_d[0]), 7, 2) - date(int(y_m_d[0]), 1, 1)).days
    delta = (date(int(y_m_d[0]), int(y_m_d[1]), int(y_m_d[2])) - date(int(y_m_d[0]), 1, 1)).days
    if delta > delta_middle:
        delta = delta_middle - (delta - delta_middle)
    day.append(np.ones((288)) * delta)
    
    time.append(list(range(144)) + list(range(144, 0, -1)))
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        power_list = json_data["stationStatisticPowerList"]
        list_len = len(power_list) 

        power_day = np.zeros((288))
        
        for i in range(list_len):
            seq_num = power_list[i]["seq"]
            if power_list[i]["generationPower"] is not None:
                power_day[seq_num] = power_list[i]["generationPower"]
        
        #for j in range(power_list[0]["seq"], power_list[-1]["seq"]):
        #    if power_day[j] == 0 and np.sum(power_day[j:]) > 0:
        #        power_day[j] == power_day[j-1]
        
        power.append(power_day)

day = np.array(day).flatten()/183.0
time = np.array(time).flatten()/144.0
power = np.array(power).flatten()     

power_max = np.max(power)

power = power/power_max

f = open('data.csv', 'w')
for i in range(len(day)):
    f.write(str(day[i])+','+str(time[i])+','+str(power[i])+'\n')
f.close()

'''
for i in range(288):
    if power[i] == 0:
        j = 0
        while power_values[i+j] != 0:
            j = j+1
            if i+j==288:
                power_values[i] = power_values[i-1]
                break
        power_values[i] = (power_values[i-1]+power_values[i+j])/2
        
        lack_n = 0
        for i in range(288):

            power_values[i] = power_list[i-lack_n]["generationPower"]
            seq_num = power_list[i-lack_n]["seq"]
            
            if seq_num > i:
                if seq_num - i==2:
                    print(2)
                if power_list[i-lack_n-1]["generationPower"] is None:
                    power = (power_list[i-lack_n-2]["generationPower"] + power_list[i-lack_n]["generationPower"])/2
                elif power_list[i-lack_n]["generationPower"] is None:
                    power = (power_list[i-lack_n-1]["generationPower"] + power_list[i-lack_n+1]["generationPower"])/2
                else:
                    power = (power_list[i-lack_n-1]["generationPower"] + power_list[i-lack_n]["generationPower"])/2
                lack_n = lack_n + (seq_num - i)
            else:
                power = power_list[i-lack_n]["generationPower"]

'''