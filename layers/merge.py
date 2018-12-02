from keras import backend as K
from keras.engine.topology import Layer

class Merge(Layer):
    '''
    用来融合
    '''

    def __init__(self, A, B, **kwargs):
        self.A = A
        self.B = B
        super(Merge, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Merge, self).build(input_shape)

    def call(self, C):
        return self.A + self.B + C

    def compute_output_shape(self, input_shape):
        return input_shape