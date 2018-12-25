from keras.models import Sequential, load_model
from keras.layers import Dense, Bidirectional, Masking, GRU, Embedding, LSTM

from draw import draw_history
from metrics import recall, precision, f1
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import time
from utils import mkdir, best_model_file, print_metrics, save_to_file
import numpy as np

class BLSTM():
    """双向BLSTM网络"""

    def __init__(self, input_shape, settings,
                 epochs=100, batch_size=256, rnn_units=100,):
        self.name = "BLSTM"
        self.input_shape = input_shape
        self.time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        self.result_path = settings.result_path + self.name + '-' + \
                           self.time_stamp + '/'
        mkdir(self.result_path)
        self.model_checkpoint = ModelCheckpoint(
            filepath=self.result_path + self.name +
                     '.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True,
            period=1,
            save_weights_only=True
        )
        self.early_stoping = EarlyStopping(monitor='val_loss', patience=3)
        self.csv_logger = CSVLogger(
            filename=self.result_path + 'training.log')
        self.model = None
        self.settings = settings
        self.embedding_matrix = np.load(settings.embedding_matrix_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.rnn_units = rnn_units
        self.train_time = 0

    def model_infor(self):
        infor = ''
        infor += 'embedding_matrix_path=' + str(self.settings.embedding_matrix_path) + '\n'
        infor += 'train_time=' + str(self.train_time) + '\n'
        return infor

    def build(self):
        """
        构建模型
        :return:
        """
        model = Sequential()
        # model.add(Masking(mask_value=0., input_shape=self.input_shape))
        model.add(Embedding(input_dim=self.embedding_matrix.shape[0],
                            output_dim=self.embedding_matrix.shape[1],
                            input_length=self.input_shape[1],
                            mask_zero=True))
        model.add(
            Bidirectional(LSTM(units=self.rnn_units, return_sequences=False),
                          merge_mode='concat'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer='adam',
                      metrics=[precision, recall, f1])
        print(model.summary())
        model.layers[0].set_weights([self.embedding_matrix])
        model.layers[0].trainable = True
        self.model = model
        return self

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
                                 callbacks=[self.model_checkpoint,
                                            self.early_stoping,
                                            self.csv_logger])
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
        return predict_log


    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer='adam',
                      metrics=[precision, recall, f1])

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
        start_time = time.time()
        history = self.fit(x_train, y_train, x_val, y_val)
        end_time = time.time()
        self.train_time = end_time-start_time
        draw_history(history, self.result_path)
        predict_log = self.evaluate(x_test, y_test)
        model_infor = self.model_infor()
        save_to_file(self.result_path + 'predict.log',
                     model_infor + predict_log)