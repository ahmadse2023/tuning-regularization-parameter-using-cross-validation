
def rescale(X_train, X_test):

    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, X_test