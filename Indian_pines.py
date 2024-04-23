# coding='utf-8'
import numpy as np
import scipy.io as scio
import torch
import torch.utils.data as data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Indian_pines(data.Dataset):
    classes = ['1 - Alfalfa', '2 - Corn-notill', '3 - Corn-mintill', '4 - Corn', '5 - Grass-pasture',
               '6 - Grass-trees', '7 - Grass-pasture-mowed', '8 - Hay-windrowed', '9 - Oats', '10 - Soybean-nottill',
               '11 - Soybean-nottill', '12 - Soybean-clean', '13 - Wheat', '14 - Woods', '15 - Buildings-Grass-Trees-Drives',
               '16 - Stone-Steel-Towers']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, train='True', test_size=0.3):
        self.train = train
        data, label = self.dataLoad()
        if self.train != 'all':
            self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(
                data, label, test_size=test_size)
        if self.train == 'True':
            self.train_data, self.train_label = torch.from_numpy(
                self.train_data), torch.from_numpy(self.train_label)
        elif self.train == 'False':
            self.test_data, self.test_label = torch.from_numpy(
                self.test_data), torch.from_numpy(self.test_label)
        else:
            self.data, self.label = torch.from_numpy(data), torch.from_numpy(label)

    def __getitem__(self, idx):
        if self.train == 'True':
            img, target = self.train_data[idx], self.train_label[idx]
        elif self.train == 'False':
            img, target = self.test_data[idx], self.test_label[idx]
        else:
            img, target = self.data[idx], self.label[idx]
        return img, target

    def __len__(self):
        if self.train == 'True':
            return len(self.train_data)
        elif self.train == 'False':
            return len(self.test_data)
        else:
            return len(self.data)

    def dataLoad(self):
        dataFile = './dataset/Indian_Pines/Indian_pines_corrected.mat'
        dataGt = './dataset/Indian_Pines/Indian_pines_gt.mat'
        rawData = scio.loadmat(dataFile)[
            'indian_pines_corrected']  # 145 145 200
        rawLabel = scio.loadmat(dataGt)['indian_pines_gt']  # 145 145
        
        # 最大值标准化
        # rawData = rawData.transpose(2, 0, 1).astype('float32')
        # for i in range(rawData.shape[0]):
        #     rawData[i] = (rawData[i] - rawData[i].min()) / (rawData[i].max() - rawData[i].min())
        # rawData = rawData.transpose(1, 2, 0)
        
        # 最大值非正规标准化
        rawData = rawData.transpose(2, 0, 1).astype('float32')
        for i in range(rawData.shape[0]):
            rawData[i] = rawData[i] / rawData[i].max()
        rawData = rawData.transpose(1, 2, 0)
        
        # 标准化
        # data = []
        # for i in range(rawLabel.shape[0]):
        #     for j in range(rawLabel.shape[1]):
        #         tmp = list(rawData[i][j])
        #         data.append(tmp)
        # data = preprocessing.StandardScaler().fit_transform(np.array(data))
        # rawData = rawData.astype('float32')
        # idx = 0
        # for i in range(rawLabel.shape[0]):
        #     for j in range(rawLabel.shape[1]):
        #         rawData[i][j] = data[idx]
        #         idx += 1
        
        # 提取有效数据        
        data = []
        label = []
        for i in range(rawLabel.shape[0]):
            for j in range(rawLabel.shape[1]):
                if rawLabel[i][j] != 0:
                    tmp = np.zeros((9, 9, 200), dtype='float32')
                    for k in range(5):
                        for o in range(5):
                            if i - k >= 0 and j - o >= 0:
                                tmp[4 - k][4 - o] = rawData[i - k][j - o]
                            if i + k < rawLabel.shape[0] and j + o < rawLabel.shape[1]:
                                tmp[4 + k][4 + o] = rawData[i + k][j + o]
                            if i - k >= 0 and j + o < rawLabel.shape[1]:
                                tmp[4 - k][4 + o] = rawData[i - k][j + o]
                            if i + k < rawLabel.shape[0] and j - o >= 0:
                                tmp[4 + k][4 - o] = rawData[i + k][j - o]
                    tmp = [tmp]
                    data.append(tmp)
                    label.append(rawLabel[i][j])
        data = np.array(data).transpose(0, 1, 4, 2, 3)  # 10249 1 200 9 9 n c d h w
        label = np.array(label)  # 10249
        data = data.astype('float32')
        label = label.astype('uint8')
        return data, label

    def train_mode(self):
        self.train = True
        
    def valid_mode(self):
        self.train = False

class CaeEncoder(data.Dataset):
    classes = ['1 - Alfalfa', '2 - Corn-notill', '3 - Corn-mintill', '4 - Corn', '5 - Grass-pasture',
               '6 - Grass-trees', '7 - Grass-pasture-mowed', '8 - Hay-windrowed', '9 - Oats', '10 - Soybean-nottill',
               '11 - Soybean-nottill', '12 - Soybean-clean', '13 - Wheat', '14 - Woods', '15 - Buildings-Grass-Trees-Drives',
               '16 - Stone-Steel-Towers']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, rawdata, label, train=True, test_size=0.3):
        self.train = train
        data = []
        for d in rawdata:
            data.append(d.flatten())
        data = np.array(data)
            
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(
            data, label - 1, test_size=test_size)
        if self.train:
            self.train_data, self.train_label = torch.from_numpy(
                self.train_data), torch.from_numpy(self.train_label)
        else:
            self.test_data, self.test_label = torch.from_numpy(
                self.test_data), torch.from_numpy(self.test_label)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    def train_mode(self):
        self.train = True
        
    def valid_mode(self):
        self.train = False
        