import time
import numpy as np
from keras import Input
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Embedding, Lambda, GRU, Bidirectional, concatenate, \
    Flatten, Dense

from draw import draw_history
from layers.dot import Dot
from layers.batch_dot import Batch_Dot
from layers.merge import Merge
from layers.multiSentimentAttention import MultiSentimentAttention
from metrics import precision, recall, f1
from utils import mkdir, best_model_file, print_metrics, save_to_file
import pickle
import codecs
import keras.backend as K
import tensorflow as tf

'''
融入了情感词、程度词、否定词的注意力模型
'''

class MultiSentimentAttentionModle():

    def __init__(self, input_shape, settings, rnn_unit=100, epochs=100,
                 bathch_size=256):
        # 模型名称
        self.name = 'Multi-Sentiment-Attention'
        # 输入的形状
        self.input_shape = input_shape
        # 创建模型的时间
        self.time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        # 模型结果保存目录
        self.result_path = settings.result_path + self.name + '-' + \
                           self.time_stamp + '/'
        mkdir(self.result_path)
        # 模型的Embedding_matrix位置
        self.embedding_matrix_path = settings.embedding_matrix_path
        self.model = None
        self.setting = settings
        # 加载情感词、程度词、否定词
        self.sentiment = self.load_sentiment_resource(self.setting.sentiment_path)
        self.intensity = self.load_sentiment_resource(self.setting.intensity_path)
        self.negative = self.load_sentiment_resource(self.setting.negative_path)

        # ----------------超参数的设置---------------------
        # 训练迭代次数
        self.epochs = epochs
        # 训练批次
        self.batch_size = bathch_size
        # RNN的单元数
        self.rnn_units = rnn_unit
        # 注意力的维度
        self.da = 350
        # 注意力的行数
        self.r = 1
        # 时间步
        self.t = input_shape[1]

    def model_checkpoint(self):
        return ModelCheckpoint(
            filepath=self.result_path + self.name +
                     '.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True,
            period=1,
            save_weights_only=True
        )

    def early_stoping(self):
        return EarlyStopping(monitor='val_loss', patience=20)

    def csv_logger(self):
        return CSVLogger(filename=self.result_path + 'training.log')

    def load_embedding_matrix(self):
        return np.load(self.embedding_matrix_path)

    def load_sentiment_resource(self, path):
        '''加载情感资源数据，返回ndarray'''
        data = None
        with codecs.open(path, 'rb') as file:
            data = pickle.load(file)
        # data = np.asarray(data)
        data = tf.constant(data)
        return data


    def build(self):
        # 输入层
        main_input = Input(shape=(self.input_shape[1],), name='main_input')
        # 加载编码权重，构建Embedding层
        embedding_matrix = self.load_embedding_matrix()
        embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                    output_dim=embedding_matrix.shape[1],
                                    mask_zero=True,)
        # 将数据经编码层转变
        Wc_T = embedding_layer(main_input)  # (None, t, d)
        Ws_T = embedding_layer(self.sentiment)   # (m, d)
        Wi_T = embedding_layer(self.intensity)   # (k, d)
        Wn_T = embedding_layer(self.negative)    # (p, d)

        # 相关矩阵
        transpose_2D_layer = Lambda(lambda x: K.transpose(x))
        transpose_3D_layer = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))
        Wc = transpose_3D_layer(Wc_T) # (None, d, t)
        Ws = transpose_2D_layer(Ws_T) # (d, m)
        Wi = transpose_2D_layer(Wi_T) # (d, k)
        Wn = transpose_2D_layer(Wn_T) # (d, p)
        Ms = Dot(Ws)(Wc_T)  # (None, t, d)*(d, m) = (None, t, m)
        Mi = Dot(Wi)(Wc_T)  # (None, t, d)*(d, k) = (None, t, k)
        Mn = Dot(Wn)(Wc_T)  # (None, t, d)*(d, p) = (None, t, p)

        #
        Xcs = Batch_Dot(Ms)(Wc) # (None, d, t)*(None, t, m)=(None, d, m)
        Xci = Batch_Dot(Mi)(Wc)  # (None, d, t)*(None, t, k)=(None, d, k)
        Xcn = Batch_Dot(Mn)(Wc)  # (None, d, t)*(None, t, p)=(None, d, p)

        Xs = Dot(Ws_T)(Ms) # (None, t, m)*(m, d)=(None, t, d)
        Xi = Dot(Wi_T)(Mi)  # (None, t, k)*(k, d)=(None, t, d)
        Xn = Dot(Wn_T)(Mn)  # (None, t, n)*(n, d)=(None, t, d)

        # 最终增强上下文为
        Xc = Merge(Xs, Xi)(Xn) # (None, t, d)

        #
        Xcs = transpose_3D_layer(Xcs) # (None, m, d)
        Xci = transpose_3D_layer(Xci) # (None, k, d)
        Xcn = transpose_3D_layer(Xcn) # (None, p, d)

        bgru = Bidirectional(GRU(units=self.rnn_units, return_sequences=True),
                          merge_mode='concat')
        Hc = bgru(Xc) # (None, t, rnn_unit)
        Hs = bgru(Xcs) # (None, m, rnn_unit)
        Hi = bgru(Xci) # (None, k, rnn_unit)
        Hn = bgru(Xcn) # (None, p, rnn_unit)

        a1 = MultiSentimentAttention(Hc, da=self.da, r=self.r, t=self.t)(Hs) # (None, r, t)
        a2 = MultiSentimentAttention(Hc, da=self.da, r=self.r, t=self.t)(Hi) # (None, r, t)
        a3 = MultiSentimentAttention(Hc, da=self.da, r=self.r, t=self.t)(Hn) # (None, r, t)

        o1 = Batch_Dot(Hc)(a1) # (None, r, rnn_unit)
        o1 = Flatten()(o1)
        o2 = Batch_Dot(Hc)(a2)  # (None, r, rnn_unit)
        o2 = Flatten()(o2)
        o3 = Batch_Dot(Hc)(a3)  # (None, r, rnn_unit)
        o3 = Flatten()(o3)

        o = concatenate([o1, o2, o3])
        predictions = Dense(1, activation='sigmoid')(o)
        self.model = Model(inputs=main_input, outputs=predictions)
        self.model.layers[1].set_weights([embedding_matrix])
        self.model.layers[1].trainable = False
        self.model.summary()


    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer='adam',
                      metrics=[precision, recall, f1])

    def fit(self, x_train, y_train, x_val, y_val):
        """
        训练模型
        :param x_train: 训练集的x
        :param y_train: 训练集的y
        :param x_val: 验证集的x
        :param y_val: 验证集的y
        :return:
        """
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[self.model_checkpoint(),
                                            self.early_stoping(),
                                            self.csv_logger()])
        return history

    def evaluate(self, x_test, y_test):
        """
        评估模型
        :param x_test: 测试集的x
        :param y_test: 测试集的y
        :return:
        """
        best_model_name = best_model_file(self.result_path)
        self.model.load_weights(self.result_path+best_model_name)
        test_metrics = self.model.evaluate(x_test, y_test)
        predict_log = print_metrics(test_metrics, self.model.metrics_names)
        save_to_file(self.result_path+'predict.log', predict_log)

    def start(self, x_train, y_train, x_val, y_val, x_test, y_test):
        '''
        模型开始训练
        :param x_train: 训练集的x
        :param y_train: 训练集的y
        :param x_val: 验证集的x
        :param y_val: 验证集的y
        :param x_test: 测试集的x
        :param y_test: 测试集的y
        :return:
        '''
        self.build()
        self.compile()
        history = self.fit(x_train, y_train, x_val, y_val)
        draw_history(history, self.result_path)
        self.evaluate(x_test, y_test)