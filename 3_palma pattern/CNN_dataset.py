import image_processing
import os
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class Train_Dataset(Dataset):
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1):
        self.image_dir = image_dir
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        group_idx = i // 6 + 1
        in_idx = i % 6 + 1
        group_name = str(group_idx)
        in_name = str(in_idx)
        while len(group_name) < 3:
            group_name = '0' + group_name
        image_name = group_name + '_' + in_name + '.bmp'
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height,
                             self.resize_width, normalization=False)
        #img = Image.fromarray(img)
        img = self.data_preproccess(img)
        label_list = [0]*100
        label_list[group_idx - 1] = 1
        label = group_idx - 1
        return img, label

    def __len__(self):
        return 600

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data


class Test_Dataset(Dataset):
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1):
        self.image_dir = image_dir
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        group_idx = i // 4 + 1
        in_idx = i % 4 + 1
        group_name = str(group_idx)
        in_name = str(in_idx)
        while len(group_name) < 3:
            group_name = '0' + group_name
        image_name = group_name + '_' + in_name + '.jpg'
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height,
                             self.resize_width, normalization=False)
        #img = Image.fromarray(img)
        img = self.data_preproccess(img)
        label = group_idx - 1
        return img, label

    def __len__(self):
        return 44

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data

class new_train_data():
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1):
        self.image_dir = image_dir
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        if i < 600:
            group_idx = i // 6 + 1
            in_idx = i % 6 + 1
            group_name = str(group_idx)
            in_name = str(in_idx)
            while len(group_name) < 3:
                group_name = '0' + group_name
            image_name = group_name + '_' + in_name + '.bmp'
            image_path = os.path.join(self.image_dir, image_name)
            img = self.load_data(image_path, self.resize_height,
                             self.resize_width, normalization=False)
            #img = Image.fromarray(img)
            img = self.data_preproccess(img)
            label_list = [0]*100
            label_list[group_idx - 1] = 1
            label = group_idx - 1
        else:
            group_idx = (i-600) // 4 + 1
            in_idx = (i-600) % 4 + 1
            group_name = str(group_idx)
            in_name = str(in_idx)
            while len(group_name) < 3:
                group_name = '0' + group_name
            image_name = group_name + '_' + in_name + '.jpg'
            image_path = os.path.join(self.image_dir, image_name)
            img = self.load_data(image_path, self.resize_height,
                                 self.resize_width, normalization=False)
            # img = Image.fromarray(img)
            img = self.data_preproccess(img)
            label = group_idx - 1
        return img, label

    def __len__(self):
        return 644

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data