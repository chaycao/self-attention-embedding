from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from regularizers import self_attentive_reg_2, self_attentive_reg_3, self_attentive_reg_4, self_attentive_reg_5
import numpy as np

class Attention(Layer):

    def __init__(self, da, **kwargs):
        self.da = da
        super(Attention, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def build(self, input_shape):
        self.ws1 = self.add_weight(name='Ws1',
                                   shape=(self.da, input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.b = self.add_weight(name='b',
                                   shape=(self.da,),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.ws2 = self.add_weight(name='Ws2',
                                   shape=(self.da, 1),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, H):
        # A = softmax(Ws2*tanh(Ws1*H.T))
        A = K.dot(H, tf.transpose(self.ws1)) # (?, maxlen, da)
        A = K.tanh(A+self.b)
        A = K.dot(A, self.ws2) # (?, maxlen, 1)
        # A = tf.transpose(A, [0, 2, 1]) # (?, r, maxlen)
        A = K.softmax(A)
        return A

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], 1)