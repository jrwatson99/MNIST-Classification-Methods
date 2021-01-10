import numpy as np
from mnist import MNIST
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def open_menu():
    # load the train and test data
    mnist = MNIST('.\\MNIST dataset\\')
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)

    model = make_pipeline(
        StandardScaler(),
        SGDClassifier()
    )
    model.fit(x_train, y_train)

    print("Train set accuracy: ", model.score(x_train, y_train) * 100)
    print("Test set accuracy: ", model.score(x_test, y_test) * 100)
