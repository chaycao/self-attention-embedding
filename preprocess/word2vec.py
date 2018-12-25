import codecs
import multiprocessing
import pickle
import time
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def word2vec():
    start_time = time.time()
    input_file = '../data/douban/experiment_data/merge_corpus_245w_text.data'
    output_model_file = '../data/douban/word2vec/word2vec-245w-100.model'
    output_vector_file = '../data/douban/word2vec/word2vec-245w-100.vector'

    model = Word2Vec(LineSentence(input_file), size=100, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

def generate_embedding_matrix_by_word2vec():
    word_index_path = '../data/douban/word_index.pkl'
    word2vec_path = '../data/douban/word2vec/word2vec-245w-300.vector'
    embedding_matrix_path = '../data/douban/experiment_data/embedding/embedding_matrix_word2vec_300.npy'
    with codecs.open(word_index_path, 'rb') as file:
        word_index = pickle.load(file)
    words_num = len(word_index)
    embedding_dim = 300
    embedding_matrix = np.zeros((words_num+1, embedding_dim))
    with codecs.open(word2vec_path, 'r', 'utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            values = line.split()
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print("error:" + word)
                continue
            if word in word_index:
                index = word_index[word]
                embedding_matrix[index] = vec
    np.save(embedding_matrix_path, embedding_matrix)

if __name__ == '__main__':
    generate_embedding_matrix_by_word2vec()