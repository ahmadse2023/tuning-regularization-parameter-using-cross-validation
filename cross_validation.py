from data_loader import load_data
from preprocess import rescale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def tune_regularization_parameter(model_type, X_train, y_train):
    p = 10
    # add wide range of C values
    C_set = [((p ** j) * 0.000000001) for j in range(4, 9)]

    # that range grows exponentially so stop at C=0.1 and add 5 more values increasing linearly by 0.3
    for i in range(5):
        C_set.append(C_set[len(C_set)-1]+0.3)

    C = pd.DataFrame({
        "C":pd.Series(C_set)
    })
    print(C)
    num_folds = 5
    mean_acc_per_fold = []
    x_trv = np.reshape(X_train, (5, -1, 28 * 28))
    y_trv = np.reshape(y_train, (5,-1))

    for C in C_set:
        print("Model Type: {}".format(model_type))
        idx = C_set.index(C)
        print("Training for C #{}..".format(idx+1))
        fold_res = []
        for fold_idx in range(num_folds):
            print("Training for fold #{}..".format(fold_idx + 1))
            tra_idx_list = list(range(fold_idx)) + list(range(fold_idx + 1, num_folds))
            tra_x = np.concatenate([x_trv[i] for i in tra_idx_list])
            tra_y = np.concatenate([y_trv[i] for i in tra_idx_list])
            v_x = x_trv[fold_idx]
            v_y = y_trv[fold_idx]
            if model_type is "lr":
                cur_fold_acc = LogisticRegression(max_iter=10000000, C=C).fit(tra_x,tra_y).score(v_x,v_y)
            elif model_type is "svm":
                cur_fold_acc = SVC(kernel='linear',C=C).fit(tra_x,tra_y).score(v_x,v_y)
            fold_res.append(cur_fold_acc)
            print("Training for fold #{} Done..".format(fold_idx + 1))
        cur_c_mean_acc = np.mean(fold_res)
        mean_acc_per_fold.append(cur_c_mean_acc)
        print("Training for C #{} Done..".format(idx + 1))

    return C_set[np.argmax(mean_acc_per_fold)]


def main():
    # load training and test data
    X_train, y_train, X_test, y_test = load_data()

    # preprocess
    X_train, X_test = rescale(X_train, X_test)

    svm_C = tune_regularization_parameter("svm", X_train, y_train)
    lr_C = tune_regularization_parameter("lr", X_train, y_train)

    lr_acc = LogisticRegression(max_iter=10000000, C=lr_C).fit(X_train,y_train).score(X_test, y_test)
    svm_acc = SVC(kernel='linear', C=svm_C).fit(X_train,y_train).score(X_test, y_test)
    print("LRACC: ")
    print(lr_acc)
    print("SVMACC")
    print(svm_acc)

if __name__ == "__main__":
    main()


