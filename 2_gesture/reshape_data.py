import json
import random

def wasted_treat():
    json_out_path = "data_set/data_for_DNN.json"
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
                for j in range(3):
                    data_part.append(speed_list[i][j])
                    data_part.append(speed_list[i][j])
                label_part = k
                dataset_list.append((data_part, label_part))
    random.shuffle(dataset_list)
    json_dict = {}
    for i in range(len(dataset_list)):
        json_dict[i] = dataset_list[i]

    with open(json_out_path, 'w', encoding="UTF-8") as fp:
        json.dump(json_dict, fp)

def true_treat_data(json_path):
    dataset_list = []  # size:n*random
    character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]
    for k in range(8):
        with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
            json_data = json.load(fp)
        speed_list = []
        angel_list = []
        for need_key in json_data.keys():        # example one1.txt
            speed_list = json_data[need_key]["speed_accel"]
            angel_list = json_data[need_key]["angle_accel"]
            key_list = []
            for i in range(len(speed_list)):
                mid_list = speed_list[i]
                mid_list.extend(angel_list[i])
                key_list.append(mid_list)
            key_list = key_list[0:200]
            key_list = sum(key_list, [])
            dataset_dict = {}
            dataset_dict["data"] = key_list
            dataset_dict["label"] = k
            dataset_list.append(dataset_dict)
    random.shuffle(dataset_list)
    json_dict = {}
    for i in range(len(dataset_list)):
        json_dict[i] = dataset_list[i]

    with open(json_path, 'w', encoding="UTF-8") as fp:
        json.dump(json_dict, fp)


def treat_data_for_lstm(json_path, group_size):
    dataset_list = []  # size:n * sequence_len * 1 * 6
    character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]
    for k in range(8):
        with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
            json_data = json.load(fp)
        for need_key in json_data.keys():        # example one1.txt
            speed_list = json_data[need_key]["speed_accel"]
            angel_list = json_data[need_key]["angle_accel"]
            key_list = []
            point = 0
            while point < len(speed_list):
                mid_list = []
                for i in range(group_size):
                    if point + i < len(speed_list):
                        mid_list.extend(speed_list[point+i])
                        mid_list.extend(angel_list[point+i])
                if len(mid_list) == group_size*6:
                    mid_list = [mid_list]
                    key_list.append(mid_list)         # sequence_len * 1 * (groupsize*6)
                point += group_size
            dataset_dict = {}
            dataset_dict["data"] = key_list
            dataset_dict["label"] = k
            dataset_list.append(dataset_dict)
    random.shuffle(dataset_list)
    json_dict = {}
    for i in range(len(dataset_list)):
        json_dict[i] = dataset_list[i]

    with open(json_path, 'w', encoding="UTF-8") as fp:
        json.dump(json_dict, fp)

#json_path = "./data_set/data_for_LSTM.json"
#treat_data_for_lstm(json_path, 5)

