from models.blstmModel import BLSTM
from models.multiSentimentAttentionModel import MultiSentimentAttentionModle
from models.sentimentAttentionModel import SentimentAttentionModel
from settings import Settings
from dataset import Dataset
from models.bgruModel import BGRU
from models.selfAttentiveEmbeddingModel import SelfAttentiveEmbeddingModel
from models.cnn_blstm_attModel import CnnBlstmAttModel
from draw import draw_history
from keras import backend as K
K.set_image_dim_ordering('tf')

# -------------------0. 配置-------------------------------------------------
settings = Settings()

# -------------------1. 准备数据---------------------------------------------
dataset = Dataset(settings)
x_train, y_train = dataset.get_train()
x_val, y_val = dataset.get_val()
x_test, y_test = dataset.get_test()
train_shape = dataset.get_train_shape()

# -------------------2.开始训练----------------------------------------------

settings.embedding_matrix_path='data/douban/experiment_data/embedding/embedding_matrix_word2vec_100.npy'
blstmModle = BLSTM(train_shape, settings, batch_size=256, epochs=100)
blstmModle.start(x_train, y_train, x_val, y_val, x_test, y_test)


#
# bgruModel = BGRU(train_shape, settings, batch_size=256, epochs=100)
# bgruModel.start(x_train, y_train, x_val, y_val, x_test, y_test)


# selfAttentiveEmbeddingModel = \
#     SelfAttentiveEmbeddingModel(train_shape, settings,
#                                 batch_size=256, epochs=100, use_regularizer=False,
#                                 patience=10, rnn_units=100, useWordvecAtt=False)
# selfAttentiveEmbeddingModel.start(x_train, y_train, x_val, y_val, x_test, y_test)
#
# selfAttentiveEmbeddingModel = \
#     SelfAttentiveEmbeddingModel(train_shape, settings,
#                                 batch_size=256, epochs=100, use_regularizer=False,
#                                 patience=10, rnn_units=100, useWordvecAtt=True)
# selfAttentiveEmbeddingModel.start(x_train, y_train, x_val, y_val, x_test, y_test)
#
# cnnBlstmAttModel = CnnBlstmAttModel(train_shape, settings,
#                                 batch_size=256, epochs=100, use_regularizer=False,
#                                 patience=10, rnn_units=100, useWordvecAtt=False)
# cnnBlstmAttModel.start(x_train, y_train, x_val, y_val, x_test, y_test)
#
# cnnBlstmAttModel = CnnBlstmAttModel(train_shape, settings,
#                                 batch_size=256, epochs=100, use_regularizer=False,
#                                 patience=10, rnn_units=100, useWordvecAtt=True)
# cnnBlstmAttModel.start(x_train, y_train, x_val, y_val, x_test, y_test)

# selfAttentiveEmbeddingModel.evaluate_all(x_test, y_test, 'result/Self-Attentive-Embedding-2018-12-14 20-23-25/')
# selfAttentiveEmbeddingModel.evaluate_single(x_test, y_test,
#                                             'result/Self-Attentive-Embedding-2018-12-14 20-23-25/Self-Attentive-Embedding.18-0.26.h5')

# sentimentAttentionModel = SentimentAttentionModel(train_shape, settings,
#                                 batch_size=256, epochs=100, use_regularizer=False,
#                                 patience=5, rnn_units=100)
# sentimentAttentionModel.start(x_train, y_train, x_val, y_val, x_test, y_test)

# multiSentimentAttentionModle = MultiSentimentAttentionModle(train_shape, settings)
# multiSentimentAttentionModle.start(x_train, y_train, x_val, y_val, x_test, y_test)
# multiSentimentAttentionModle.evaluate_single(x_test, y_test,
#                                              'result/Multi-Sentiment-Attention-2018-12-15 02-05-21/Multi-Sentiment-Attention.98-13.83.h5')