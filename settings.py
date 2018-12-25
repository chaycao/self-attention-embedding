class Settings():
    """存储模型的所有设置"""

    def __init__(self):
        # ----------------1. 数据集设置------------------------------------------
        # 小数据集的设置
        self.isUseSmall = False
        self.small_data_path = 'data/douban/sequences(100)_shuffle.txt'

        # 常规数据集的设置
        self.data_path = 'data/douban/experiment_data/sequences_25w_shuffle.txt'

        # 训练集、验证集、测试集的占比
        self.train_data_factor = 0.8
        self.val_data_factor = 0.1

        # ----------------2.模型训练设置----------------------------------------
        # 模型保存位置
        self.result_path = 'result/'

        # Embedding层的权重矩阵
        self.embedding_matrix_path = 'data/douban/experiment_data/embedding/embedding_matrix_word2vec_100.npy'

        # 情感词的存放文件
        self.sentiment_path = 'data/douban/dict/sentiment.data'

        # 否定词的存放文件
        self.negative_path = 'data/douban/dict/negative.data'

        # 强度词的存放文件
        self.intensity_path = 'data/douban/dict/intensity.data'
