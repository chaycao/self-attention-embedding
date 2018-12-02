import os
import codecs
import pickle
from preprocess.data import load_data
from preprocess.embedding import data_to_sequence

'''
首先先融合所有的词语，然后以一个set序列化保存下来
再根据训练测试数据，保留相应的词语和index
'''

def merge(input_path, output_path):
    '''
    融合目录下的所有文件内容，保存为一个set
    '''
    list = []
    files = os.listdir(input_path)
    for filename in files:
        with codecs.open(input_path+'/'+filename, 'r', 'utf-8') as file:
            for line in file:
                list.append(line.strip())
    word_set = set(list)
    with codecs.open(output_path, 'w', 'utf-8') as file:
        for word in word_set:
            file.write(word+'\n')

def sequence(data_path, dict_path, output_path):
    '''
    根据训练测试数据，保留相应的词语和index
    '''
    data, labels = load_data(data_path)
    sequences, word_index = data_to_sequence(data)
    print(len(word_index))
    list = []
    with codecs.open(dict_path, 'r', 'utf-8') as file:
        for line in file:
            word = line.strip()
            if word in word_index:
                list.append(word_index[word])
                print(word)
    print(len(list))
    with codecs.open(output_path, 'wb') as file:
        pickle.dump(list, file)


if __name__ == '__main__':
    # merge('../data/dict/sentiment', '../data/dict/sentiment/sentiment_merge.txt')
    sequence(data_path='../data/ChnSentiCorp.txt',
             # dict_path='../data/dict/sentiment/sentiment_merge.txt',
             # dict_path='../data/dict/intensity-word/程度级别词语（中文）.txt',
             dict_path='../data/dict/negative-word/否定.txt',
             # output_path='../data/dict/sentiment.data'
             # output_path='../data/dict/intensity.data'
             output_path='../data/dict/negative.data'
             )

