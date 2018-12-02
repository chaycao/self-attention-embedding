from preprocess.data import load_data
import numpy as np
from utils import max_len
from keras.preprocessing import sequence


class Dataset():
    """存储准备好的数据集"""

    def __init__(self, settings):
        # -------------------1. 加载数据---------------------------------------------
        # 加载后的数据：data 为2D向量， label 为1D向量
        if settings.isUseSmall:
            data, labels = load_data(settings.small_data_path)
        else:
            data, labels = load_data(settings.data_path)

        self.maxlen = max_len(data)
        data = [[int(n) for n in d] for d in data]
        data = sequence.pad_sequences(data, maxlen=self.maxlen)
        labels = np.array(labels)
        # -------------------2 切分数据---------------------------------------------
        # 将数据切分为训练集、验证集、测试集
        num_data = len(data)
        num_train_data = int(num_data * settings.train_data_factor)
        num_val_data = int(num_data * settings.val_data_factor)
        self.x_train = data[:num_train_data]
        self.y_train = labels[:num_train_data]
        self.x_val = data[num_train_data: num_train_data + num_val_data]
        self.y_val = labels[num_train_data: num_train_data + num_val_data]
        self.x_test = data[num_train_data + num_val_data:]
        self.y_test = labels[num_train_data + num_val_data:]

    def get_train(self):
        return self.x_train, self.y_train

    def get_val(self):
        return self.x_val, self.y_val

    def get_test(self):
        return self.x_test, self.y_test

    def get_train_shape(self):
        return (len(self.x_train), self.maxlen)