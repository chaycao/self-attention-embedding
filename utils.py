import os
import codecs

def max_len(array):
    '''
    Numpy数组的最大维度
    :param array:
    :return:
    '''
    maxLen = 0
    for i in range(len(array)):
        maxLen = max(maxLen, len(array[i]))
    return maxLen

def mkdir(path):
    """创建文件夹
    如果文件夹不存在，则创建
    :param path:
    :return:
    """
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)

def best_model_file(path):
    '''
    获得最好模型的文件名
    :param path:
    :return:
    '''
    files = os.listdir(path)
    files = [x for x in files if os.path.splitext(x)[1]=='.h5']
    files.sort()
    return files[-1]

def list_model(path):
    return os.listdir(path)


def print_metrics(metrics, metrice_name):
    '''
    打印指标
    :param metrics: 指标的列表
    :param name: 指标的名字
    :return:
    '''
    metrics_log = ''
    for i, metric in enumerate(metrics):
        metrics_log += "%s:%.2f\n" % (metrice_name[i], metric*100)
    print(metrics_log)
    return metrics_log

def save_to_file(filename, content):
    """
    将内容保存在文件
    :param filename: 文件名
    :param content: 内容
    :return:
    """
    with codecs.open(filename, 'w', 'utf-8') as file:
        file.write(content)
