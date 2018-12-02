from matplotlib import pyplot as plt

def draw_history(history, path):
    """绘制F1值和loss
    :param history: 训练历史记录
    :return:
    """
    f1 = history.history['f1']
    val_f1 = history.history['val_f1']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(f1) + 1)

    plt.figure()
    plt.plot(epochs, f1, 'bo', label='Training F1')
    plt.plot(epochs, val_f1, 'b', label='Validation F1')
    plt.title('Traing and validation F1')
    plt.legend()
    plt.savefig(path+'f1.png', dpi=600)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Traing and validation loss')
    plt.legend()
    plt.savefig(path + 'loss.png', dpi=600)
    # plt.show()