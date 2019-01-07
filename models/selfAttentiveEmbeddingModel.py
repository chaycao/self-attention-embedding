from keras import Input
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Bidirectional, GRU, Embedding, LSTM, Flatten, \
    Lambda, Dropout, concatenate

from draw import draw_history
from metrics import recall, precision, f1
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import time
from utils import mkdir, best_model_file, print_metrics, save_to_file, list_model
import numpy as np
from layers.selfAttentiveEmbedding import SelfAttentiveEmbedding
from layers.batch_dot import Batch_Dot
from keras import backend as K

class SelfAttentiveEmbeddingModel():
    """双向BGRU网络
    构建模型

    Embedding()
    BLSTM()
    Self-attentive-embedding()
    Flatten()
    Dense()
    """

    def __init__(self, input_shape, settings,
                 epochs=100, batch_size=256, rnn_units=100,
                 da=350, r=30, use_regularizer=True, patience=10,
                 word_da=350, word_r=30, useWordvecAtt=False, useSelfAtt=True):
        # 模型名称
        self.name = "Self-Attentive-Embedding"
        # 输入形状
        self.input_shape = input_shape
        # 创建模型的时间
        self.time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        # 模型结果保存目录
        self.result_path = settings.result_path + self.name + '-' + \
                           self.time_stamp + '/'
        mkdir(self.result_path)
        self.model = None
        # 模型的Embedding_matrix位置
        self.embedding_matrix_path = settings.embedding_matrix_path

        #----------------超参数的设置---------------------
        # 训练迭代次数
        self.epochs = epochs
        # 训练批次
        self.batch_size = batch_size
        # RNN的单元数
        self.rnn_units = rnn_units
        # 注意力的维度
        self.da = da
        # 注意力的行数
        self.r = r
        # 是否使用惩罚项
        self.use_regularizer = use_regularizer
        # early_stopping的等待次数
        self.patience = patience
        # 词向量自注意力的维度
        self.word_da = word_da
        # 词向量自注意力的行数
        self.word_r = word_r
        # 是否使用词向量自注意力
        self.useWordvecAtt = useWordvecAtt
        # 是否使用自注意力
        self.useSelfAtt = useSelfAtt

    def model_infor(self):
        infor = ''
        infor += 'epoch=' + str(self.epochs) + '\n'
        infor += 'batch_size=' + str(self.batch_size) + '\n'
        infor += 'rnn_units=' + str(self.rnn_units) + '\n'
        infor += 'da=' + str(self.da) + '\n'
        infor += 'r=' + str(self.r) + '\n'
        infor += 'use_regularizer=' + str(self.use_regularizer) + '\n'
        infor += 'use_WordvecAtt=' + str(self.useWordvecAtt) + '\n'
        infor += 'use_SelfAtt=' + str(self.useSelfAtt) + '\n'
        return infor

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
        return EarlyStopping(monitor='val_loss', patience=self.patience)

    def csv_logger(self):
        return CSVLogger(filename=self.result_path + 'training.log')

    def load_embedding_matrix(self):
        return np.load(self.embedding_matrix_path)

    def build(self):
        inputs = Input(shape=(self.input_shape[1],))
        embedding_matrix = self.load_embedding_matrix()
        wordvec = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            mask_zero=True,
                            )(inputs)
        if self.useSelfAtt == True:
            H = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True),
                              merge_mode='concat')(wordvec)
            A = SelfAttentiveEmbedding(da=self.da, r=self.r,
                                       use_regularizer=self.use_regularizer)(H)
            M = Batch_Dot(Y=H)(A)
            M = Flatten()(M)
        else:
            M = Bidirectional(LSTM(units=self.rnn_units, return_sequences=False),
                              merge_mode='concat')(wordvec)

        if self.useWordvecAtt == True:
            # 词向量注意力
            A_wordvec = SelfAttentiveEmbedding(da=self.word_da, r=self.word_r,
                                       use_regularizer=self.use_regularizer)(wordvec)
            M_wordvec = Batch_Dot(Y=wordvec)(A_wordvec)
            M_wordvec = Flatten()(M_wordvec)
            # 联合词向量注意力和自注意力
            M = concatenate([M, M_wordvec])

        predictions = Dense(1, activation='sigmoid')(M)
        self.model = Model(inputs=inputs, outputs=predictions)
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
        model_infor = self.model_infor()
        save_to_file(self.result_path+'predict.log', model_infor+predict_log)


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

    def evaluate_all(self, x_test, y_test, result_path):
        self.build()
        self.compile()
        model_names = list_model(result_path)
        # for model_name in model_names:
        #     self.model.load_weights(result_path+model_name)
        #     test_metrics = self.model.evaluate(x_test, y_test)
        #     predict_log = print_metrics(test_metrics, self.model.metrics_names)
        #     print(model_name+'\n'+predict_log+'\n\n')
        self.model.load_weights('result/Self-Attentive-Embedding-2018-12-14 20-23-25/Self-Attentive-Embedding.06-0.29.h5')
        test_metrics = self.model.evaluate(x_test, y_test)
        predict_log = print_metrics(test_metrics, self.model.metrics_names)
        print(predict_log + '\n\n')

    def evaluate_single(self, x_test, y_test, model_path):
        self.build()
        self.compile()
        self.model.load_weights(model_path)
        test_metrics = self.model.evaluate(x_test, y_test)
        predict_log = print_metrics(test_metrics, self.model.metrics_names)
        print(predict_log + '\n\n')