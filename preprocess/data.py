import codecs
import numpy as np

'''
数据的预处理
'''

def load_data(path):
    """
    加载数据

    :param
        path: 文件路径
            文件形式为label+text 如：“1 今天 天气 真好”
    :return:
        data：文本数据，如：[[今天，天气，真好],[今天,天气，真差]]
        labels：标签，如：[1,0]
    """
    data = []
    labels = []
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            list = line.split(' ')
            labels.append(list[0])
            data.append(list[1:])
    return data, labels

def shuffle_data(input_path, output_path):
    """
    打乱数据，按行为单位

    :param
        input_path:输入文件路径
    :param
        output_path:打乱后文件路径
    :return:
    """
    data = []
    with codecs.open(input_path, 'r', 'utf-8') as file:
        for line in file:
            data.append(line)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = np.array(data)
    data = data[indices]
    data = data.tolist()
    with codecs.open(output_path, 'w', 'utf-8') as file:
        for line in data:
            file.write(line)


if __name__ == '__main__':
    shuffle_data_input_path = '../data/douban/sequences(1w).txt'
    shuffle_data_output_path = '../data/douban/sequences(1w)_shuffle.txt'
    shuffle_data(shuffle_data_input_path, shuffle_data_output_path)