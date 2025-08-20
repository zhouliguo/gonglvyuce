import json
import glob
import numpy as np

json_files = glob.glob('data/*.json')

power = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        power_list = json_data["stationGroupStatisticPowerDtoList"]
        list_len = len(power_list) 

        power_day = np.zeros((288))
        
        for i in range(list_len):
            seq_num = power_list[i]["seq"]
            if power_list[i]["generationPower"] is not None:
                power_day[seq_num] = power_list[i]["generationPower"]
        
        power.append(power_day)

power = np.array(power).flatten()
        
        
for i in range(288):
    if power_values[i] == 0:
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