from keras import backend as K
import tensorflow as tf

def self_attentive_reg(A):
    '''
    自注意力编码的惩罚项
    尽量让A的每一行不一样，获取更多不同的信息
    :param A:
    :return:
    '''
    x = K.batch_dot(A, tf.transpose(A, [0,2,1]))
    x = x-1
    x = tf.norm(x)
    x = 0.0001*x*x
    return x