from data_loader import load_data
from preprocess import rescale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main():

    # load training and test data
    X_train, y_train, X_test, y_test = load_data()

    # preprocess
    X_train, X_test = rescale(X_train, X_test)

    # training and predicting
    p = 10
    test_acc = []
    training_acc = []

    print("Training and Predicting..")
    for i in range(2, 12):
        print("Training Classifier #{}".format(i - 1))
        clf = LogisticRegression(max_iter=10000000, C=(p ** i) * 0.000000001).fit(X_train, y_train)
        test_acc.append(clf.score(X_test, y_test))
        training_acc.append((clf.score(X_train, y_train)))
        print("Classifier #{} acc appended".format(i - 1))
    print("Done..")

    c = [((p ** j) * 0.000000001) for j in range(2, 12)]
    dict = pd.DataFrame({
        "test_acc": pd.Series(test_acc),
        "training_acc": pd.Series(training_acc),
        "C": pd.Series(np.log10(c))
    })
    # visualize
    plt.figure()
    plt.plot("C", "test_acc", data=dict, label="test-data")
    plt.plot("C", "training_acc", data=dict, label="training-data")
    plt.xlabel("Regularization Parameter (in log scale)")
    plt.ylabel("Accuracy")
    plt.legend()

    dict = pd.DataFrame({
        "test_acc": pd.Series(test_acc),
        "training_acc": pd.Series(training_acc),
        "C": pd.Series(c)
    })

    plt.figure()
    plt.plot("C", "test_acc", data=dict, label="test-data")
    plt.plot("C", "training_acc", data=dict, label="training-data")
    plt.xlabel("Regularization Parameter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    print(dict)


if __name__ == "__main__":
    main()