import tensorflow as tf
import torch
from tensorflow.keras import models, layers, metrics
import json
import os
import datetime
import numpy as np

MODEL_INPUT_TIME_SIZE = 3
INPUT_DIM = 2                               # 一次同时喂入同一个时间点下的加速度和陀螺仪信息
BATCH_SIZE = 100

def get_model():
    simple_lstm_model = models.Sequential([
        layers.LSTM(20, input_shape=[MODEL_INPUT_TIME_SIZE, INPUT_DIM]),          # 输入2*1的矩阵
        layers.Dense(20),
        layers.Dense(8, activation='softmax')
        #layers.Dense(1, activation='softmax')
    ])
    #simple_lstm_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    simple_lstm_model.compile(optimizer='adam', loss='mae', metrics=['binary_accuracy'])
    simple_lstm_model.summary()
    return simple_lstm_model

def get_dataset():
    data_list = []
    label_list = []
    character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]
    for k in range(8):
        with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
            json_data = json.load(fp)
            speed_list = []
            angel_list = []
            for need_key in json_data.keys():
                speed_list.extend(json_data[need_key]["speed_accel"])
                angel_list.extend(json_data[need_key]["angle_accel"])
            for i in range(len(speed_list)):
                data_part = []
                label_part = [0]*8
                for j in range(3):
                    data_part.append([speed_list[i][j], angel_list[i][j]])
                data_list.append(data_part[:])
                label_part[k] = 1
                label_list.append(label_part[:])

    ds = tf.data.Dataset.from_tensor_slices((data_list, label_list))
    ds = ds.shuffle(10000000,reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    return ds

class LSTM():
    model = None
    train_data = None
    valid_data = None

    def __init__(self):
        tf.keras.backend.clear_session()
        self.model = get_model()

    def model_train(self):
        self.train_data = get_dataset()
        self.valid_data = get_dataset()
        self.model.fit(self.train_data, epochs=10, verbose=1, validation_data=self.valid_data)

    def model_save(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("./models/{}".format(timestamp))
        self.model.save("./models/{}/".format(timestamp))

test_LSTM = LSTM()
test_LSTM.model_train()
#test_LSTM.model_save()