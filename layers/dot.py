from keras import backend as K
from keras.engine.topology import Layer

class Dot(Layer):
    '''
    用来X乘Y
    '''

    def __init__(self, Y, **kwargs):
        self.Y = Y
        self.u = Y.shape[1].value
        self.supports_masking = True
        super(Dot, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def build(self, input_shape):
        super(Dot, self).build(input_shape)

    def call(self, X):
        M = K.dot(X, self.Y)
        return M

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.u)