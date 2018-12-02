from keras import backend as K
from keras.engine.topology import Layer

class Batch_Dot(Layer):
    '''
    用来X乘Y，做batch_dot
    '''

    def __init__(self, Y, **kwargs):
        self.Y = Y
        self.u = Y.shape[2].value
        super(Batch_Dot, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Batch_Dot, self).build(input_shape)

    def call(self, X):
        M = K.batch_dot(X, self.Y)
        return M

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.u)