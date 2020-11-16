import json
import random

json_out_path = "./data_set/deal.json"
dataset_list = []  # size:n*6
character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]

for k in range(8):
    with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
        json_data = json.load(fp)
        input_list = []
        speed_list = []
        angel_list = []
        for need_key in json_data.keys():
            speed_list.extend(json_data[need_key]["speed_accel"])
            angel_list.extend(json_data[need_key]["angle_accel"])
        for i in range(len(speed_list)):
            data_part = []
            label_part = [0] * 8
            for j in range(3):
                data_part.append(speed_list[i][j])
                data_part.append(speed_list[i][j])
            label_part[k] = 1
            dataset_list.append((data_part, label_part))
random.shuffle(dataset_list)
json_dict = {}
for i in range(len(dataset_list)):
    json_dict[i] = dataset_list[i]

with open(json_out_path, 'w', encoding="UTF-8") as fp:
    json.dump(json_dict, fp)
