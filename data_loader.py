from utils import mnist_reader
import matplotlib.pyplot as plt
import numpy as np

def load_data():

    print("Loading data..")
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # we want to work on a binary classification problem so only class 5 and class 7 will be used
    tr_mask = np.where(np.logical_or(y_train == 5, y_train == 7))[0]
    te_mask = np.where(np.logical_or(y_test == 5, y_test == 7))[0]

    X_train = X_train[tr_mask, :]
    y_train = y_train[tr_mask]
    X_test = X_test[te_mask, :]
    y_test = y_test[te_mask]

    # let class 5 be 0 and let class 7 be 1
    y_train[y_train == 5] = 0
    y_train[y_train == 7] = 1
    y_test[y_test == 5] = 0
    y_test[y_test == 7] = 1

    # Debugging
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # plt.imshow(X_train[1251].reshape(28,28),cmap=plt.cm.gray)
    # plt.show()

    print("Done..")
    return X_train, y_train, X_test, y_test
