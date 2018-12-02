from models.blstmModel import BLSTM
from models.multiSentimentAttentionModel import MultiSentimentAttentionModle
from settings import Settings
from dataset import Dataset
from models.bgruModel import BGRU
from models.selfAttentiveEmbeddingModel import SelfAttentiveEmbeddingModel
from draw import draw_history
from keras import backend as K
K.set_image_dim_ordering('tf')

# -------------------0. 配置-------------------------------------------------
settings = Settings()

# -------------------1. 准备数据---------------------------------------------
dataset = Dataset(settings)
x_train, y_train = dataset.get_train()
x_val, y_val = dataset.get_val()
x_test, y_test = dataset.get_test()
train_shape = dataset.get_train_shape()

# -------------------2.开始训练----------------------------------------------

# selfAttentiveEmbeddingModel = SelfAttentiveEmbeddingModel(train_shape, settings, batch_size=64, epochs=2)
# selfAttentiveEmbeddingModel.start(x_train, y_train, x_val, y_val, x_test, y_test)

# blstmModle = BLSTM(train_shape, settings, batch_size=64, epochs=1)
# blstmModle.start(x_train, y_train, x_val, y_val, x_test, y_test)

bgruModel = BGRU(train_shape, settings, batch_size=64, epochs=1)
bgruModel.start(x_train, y_train, x_val, y_val, x_test, y_test)



# multiSentimentAttentionModle = MultiSentimentAttentionModle(train_shape, settings)
# multiSentimentAttentionModle.start(x_train, y_train, x_val, y_val, x_test, y_test)
