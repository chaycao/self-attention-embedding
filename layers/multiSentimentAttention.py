from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from regularizers import self_attentive_reg
import numpy as np

class MultiSentimentAttention(Layer):

    def __init__(self, Hc, da, r, t, **kwargs):
        self.Hc = Hc # (None, t, rnn_unit)
        self.da = da
        self.r = r
        self.t = t
        self.supports_masking = True
        self.activity_regularizer = self_attentive_reg
        super(MultiSentimentAttention, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def build(self, input_shape):
        self.ws1 = self.add_weight(name='Ws1',
                                   shape=(self.da, 2*self.Hc.shape[2].value),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.ws2 = self.add_weight(name='Ws2',
                                   shape=(self.r, self.da),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(MultiSentimentAttention, self).build(input_shape)

    def call(self, H):
        H = tf.transpose(H, [0, 2, 1])
        q = tf.reduce_mean(H, 2) # (None, rnn_unit)
        rnn_unit = q.shape[1].value
        q = tf.tile(q, [1, self.t])
        q = tf.transpose(q, [1,0])
        q = tf.reshape(q, [self.t, rnn_unit, -1])
        q = tf.transpose(q, [2,0,1])
        X = tf.concat([self.Hc, q], 2)  # (None, t, rnn_unit) 拼接 (None, t, rnn_unit) => (None, t, 2*rnn_unit)
        A = K.dot(X, tf.transpose(self.ws1)) # (None, t, 2*rnn_unit) * (rnn_unit, da) => (None, t, da)
        A = K.tanh(A)
        A = K.dot(A, tf.transpose(self.ws2)) # (None, t, da) * (da, r) => (None, t, r)
        A = tf.transpose(A, [0, 2, 1]) # (None, r, t)
        A = K.softmax(A)
        return A

    def compute_output_shape(self, input_shape):
        # return (None, self.r, input_shape[2])
        return (None, self.r, self.t)
