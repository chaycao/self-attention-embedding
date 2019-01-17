from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

class MultiSentimentAttention(Layer):

    def __init__(self, Hc, da, r, d, **kwargs):
        self.Hc = Hc # (None, d, rnn_unit)
        self.da = da
        self.r = r
        self.d = d
        self.supports_masking = True
        # self.activity_regularizer = self_attentive_reg
        super(MultiSentimentAttention, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def build(self, input_shape):
        self.ws1 = self.add_weight(name='Ws1',
                                   shape=(self.da, self.Hc.shape[2].value+1),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.ws2 = self.add_weight(name='Ws2',
                                   shape=(self.r, self.da),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(MultiSentimentAttention, self).build(input_shape)

    def call(self, H):
        q = tf.reduce_mean(H, 2) # (None, d)
        q = tf.expand_dims(q, -1)
        X = tf.concat([self.Hc, q], 2)  # (None, d, rnn_unit) 拼接 (None, d, 1) => (None, d, rnn_unit+1)
        A = K.dot(X, tf.transpose(self.ws1)) # (None, d, rnn_unit+1) * (rnn_unit+1, da) => (None, d, da)
        A = K.tanh(A)
        A = K.dot(A, tf.transpose(self.ws2)) # (None, d, da) * (da, r) => (None, d, r)
        A = tf.transpose(A, [0, 2, 1]) # (None, r, d)
        A = K.softmax(A)
        return A

    def compute_output_shape(self, input_shape):
        return (None, self.r, self.d)
