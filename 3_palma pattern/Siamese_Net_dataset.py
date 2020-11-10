import image_processing
import os
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SN_dataset(Dataset):
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1, length=1600):
        self.image_dir = image_dir
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()
        self.length = length

    def __getitem__(self, i):
        num_1 = np.random.randint(0, 600)
        num_2 = np.random.randint(0, 600)
        group_1 = num_1 // 6 + 1
        group_2 = num_2 // 6 + 1
        in_1 = num_1 % 6 + 1
        in_2 = num_2 % 6 + 1
        group_name_1 = str(group_1)
        group_name_2 = str(group_2)
        in_name_1 = str(in_1)
        in_name_2 = str(in_2)
        while len(group_name_1) < 3:
            group_name_1 = '0' + group_name_1
        while len(group_name_2) < 3:
            group_name_2 = '0' + group_name_2
        image_name_1 = group_name_1 + '_' + in_name_1 + '.bmp'
        image_name_2 = group_name_2 + '_' + in_name_2 + '.bmp'
        image_path_1 = os.path.join(self.image_dir, image_name_1)
        image_path_2 = os.path.join(self.image_dir, image_name_2)
        img_1 = self.load_data(image_path_1, self.resize_height,
                                self.resize_width, normalization=False)
        img_2 = self.load_data(image_path_2, self.resize_height,
                                self.resize_width, normalization=False)
        img_1 = self.data_preproccess(img_1)
        img_2 = self.data_preproccess(img_2)
        if group_1 == group_2:
            label = 1
        else:
            label = 0
        if i >= self.length // 2:
            image_name_1 = group_name_1 + '_' + in_name_1 + '.bmp'
            image_path_1 = os.path.join(self.image_dir, image_name_1)
            img_1 = self.load_data(image_path_1, self.resize_height,
                                   self.resize_width, normalization=False)
            img_2 = self.load_data(image_path_1, self.resize_height,
                                   self.resize_width, normalization=False)
            img_1 = self.data_preproccess(img_1)
            img_2 = self.data_preproccess(img_2)
            label = 1
        return img_1, img_2, label


    def __len__(self):
        return self.length

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