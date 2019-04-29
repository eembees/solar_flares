import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from joblib import dump, load
from autosklearn.classification import AutoSklearnClassifier
if __name__ == '__main__':
    INPUT_DIR = '.'
    OUTPUT_DIR = '.'

    INPUT_PATH = Path(INPUT_DIR)
    OUTPUT_PATH = Path(OUTPUT_DIR)

    # c = np.load('../input/npz/fold3Training.npz')
    #
    # X_test = c['data']
    # y_test = c['labels']
    # X_n, X_x, X_y = X_test.shape
    # X_test = X_test.reshape((X_n, X_x * X_y))
    #
    #
    # print('IMPORTED: TEST DATA')
    # print('X : {}, shape: {} '.format(type(X_test), X_test.shape))
    # print('y : {}, shape: {} '.format(type(y_test), y_test.shape))
    #


    cls : AutoSklearnClassifier  = load( INPUT_PATH / 'auto_cls.joblib' )

    # predictions = cls.predict(X_test)
    # print("Accuracy score", accuracy_score(y_test, predictions))


    print(cls.get_models_with_weights())

