import codecs
import pickle

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

def shuffle_split():
    '''打乱数据，并分割'''
    data_path = '../data/douban/experiment_data/douban_25w.data'
    data=[]
    with codecs.open(data_path, 'r', 'utf-8') as file:
        for line in file:
            data.append(line)
    size = len(data)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = np.array(data)
    data = data[indices]
    data = data.tolist()
    train_data = data[0:int(size*0.8)]
    eval_data = data[int(size*0.8):int(size*0.9)]
    test_data = data[int(size*0.9):]
    with codecs.open('../data/douban/experiment_data/douban_25w_train.data', 'w', 'utf-8') as file:
        for line in train_data:
            file.write(line)
    with codecs.open('../data/douban/experiment_data/douban_25w_eval.data', 'w', 'utf-8') as file:
        for line in eval_data:
            file.write(line)
    with codecs.open('../data/douban/experiment_data/douban_25w_test.data', 'w', 'utf-8') as file:
        for line in test_data:
            file.write(line)

def remove_lable():
    data_path = '../data/douban/experiment_data/merge_corpus_245w.data'
    output_path = '../data/douban/experiment_data/merge_corpus_245w_text.data'
    with codecs.open(data_path, 'r', 'utf-8') as input_file, \
        codecs.open(output_path, 'w', 'utf-8') as output_file:
        for line in input_file:
            output_file.write(line[2:])

def merge_corpus():
    train_path = '../data/douban/experiment_data/douban_25w_train.data'
    corpus_path = '../data/douban/douban.data'
    output_path = '../data/douban/experiment_data/merge_corpus_245w.data'
    with codecs.open(corpus_path, 'r', 'utf-8') as corpus_file, \
            codecs.open(train_path, 'r', 'utf-8') as train_file, \
            codecs.open(output_path, 'a', 'utf-8') as output_file:
        for i, line in enumerate(corpus_file):
            if (i >= 125000 and i < 2375000):
                output_file.write(line)
        for line in train_file:
            output_file.write(line)

def toSequences(list, word_index):
    list = [word_index[x] for x in list]
    return list

def merge_data_sequence():
    word_index_path = '../data/douban/word_index.pkl'
    train_path = '../data/douban/experiment_data/douban_25w_train.data'
    eval_path = '../data/douban/experiment_data/douban_25w_eval.data'
    test_path = '../data/douban/experiment_data/douban_25w_test.data'
    output_path = '../data/douban/experiment_data/sequence_25w_shuffle.txt'
    with codecs.open(train_path, 'r', 'utf-8') as train_file, \
            codecs.open(eval_path, 'r', 'utf-8') as eval_file, \
            codecs.open(test_path, 'r', 'utf-8') as test_file, \
            codecs.open(word_index_path, 'rb') as word_index_file, \
            codecs.open(output_path, 'w', 'utf-8') as output_file:
        word_index = pickle.load(word_index_file)
        for line in train_file:
            label = line[0]
            list = toSequences(line[2:].strip().split(' '), word_index)
            output_file.write(label+' ')
            for i, item in enumerate(list):
                if i != len(list)-1:
                    output_file.write(str(item)+' ')
                else:
                    output_file.write(str(item) + '\n')
        for line in eval_file:
            label = line[0]
            list = toSequences(line[2:].strip().split(' '), word_index)
            output_file.write(label+' ')
            for i, item in enumerate(list):
                if i != len(list)-1:
                    output_file.write(str(item)+' ')
                else:
                    output_file.write(str(item) + '\n')
        for line in test_file:
            label = line[0]
            list = toSequences(line[2:].strip().split(' '), word_index)
            output_file.write(label+' ')
            for i, item in enumerate(list):
                if i != len(list)-1:
                    output_file.write(str(item)+' ')
                else:
                    output_file.write(str(item) + '\n')

if __name__ == '__main__':
    merge_data_sequence()