import pickle

from preprocess.data import load_data
from keras.preprocessing.text import Tokenizer
import codecs
import numpy as np

def data_to_sequence(data):
    '''
    根据数据，生成单词字典，并将文本转成单词序列
    :param data: 数据
    :return: 单词序列，单词字典
    '''
    texts = []
    for sample in data:
        texts.append(' '.join(sample))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return sequences, word_index

def save_embedding_matrix(wordvec_path, word_index, embedding_matrix_path):
    max_words = len(word_index)
    embedding_dim = 300
    embedding_matrix = np.zeros((max_words+1, embedding_dim))
    count = 0
    with codecs.open(wordvec_path, 'r', 'utf-8') as wordvec:
        for i, line in enumerate(wordvec):
            if (i % 100000 == 0):
                print(str(i))
            # 跳过第一行，腾讯的词向量第一行是数据说明
            if i == 0:
                continue
            values = line.split()
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print("error:"+word)
                continue
            if word in word_index:
                index = word_index[word]
                embedding_matrix[index] = vec
                count += 1
                if count == max_words:
                    break
    print('max_words:' + str(max_words) + '\ncount:' + str(count))
    np.save(embedding_matrix_path, embedding_matrix)


def save_sequences(sequences, labels, sequences_path):
    with codecs.open(sequences_path, 'w', 'utf-8') as file:
        for i, label in enumerate(labels):
            file.write(label + ' ' + ' '.join([str(x) for x in sequences[i]])+'\n')

def generate_embedding_matrix(data_path, wordvec_path, sequences_path,
                              embedding_matrix_path):
    '''根据预训练的词向量，得到Embedding层的权重矩阵'''
    # 读取数据，生成单词字典，并将文本转成单词序列
    data, labels = load_data(data_path)
    sequences, word_index = data_to_sequence(data)

    # 读取预训练词向量，结合数据的单词字典，生成相应的权重矩阵
    save_embedding_matrix(wordvec_path, word_index, embedding_matrix_path)

    # 保存单词序列
    save_sequences(sequences, labels, sequences_path)

def load_word_index(path):
    with codecs.open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    data_path = '../data/douban/douban.data'
    wordvec_path = 'E:\学习资料\预训练词向量\sgns.merge.word\sgns.merge.word'
    sequences_path = '../data/douban/word2vec/sequences.txt'
    embedding_matrix_path = '../data/douban/word2vec/embedding_matrix.npy'
    # generate_embedding_matrix(data_path, wordvec_path, sequences_path, embedding_matrix_path)

    word_index = load_word_index('../data/douban/word_index.pkl')
    save_embedding_matrix(wordvec_path, word_index, embedding_matrix_path)