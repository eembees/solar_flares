import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
from joblib import dump, load

if __name__ == '__main__':
    INPUT_DIR = '../input/npz/'
    OUTPUT_DIR = '.'

    INPUT_PATH = Path(INPUT_DIR)
    OUTPUT_PATH = Path(OUTPUT_DIR)

    a = np.load('../input/npz/fold1Training.npz')
    b = np.load('../input/npz/fold2Training.npz')
    c = np.load('../input/npz/fold3Training.npz')

    X_train = np.concatenate([a['data'],b['data']])
    y_train = np.concatenate([a['labels'],b['labels']])

    X_test = c['data']
    y_test = c['labels']

    X_n, X_x, X_y = X_train.shape
    X_train = X_train.reshape((X_n, X_x * X_y))


    X_n, X_x, X_y = X_test.shape
    X_test = X_test.reshape((X_n, X_x * X_y))


    print('IMPORTED: TRAINING DATA')
    print('X : {}, shape: {} '.format(type(X_train), X_train.shape))
    print('y : {}, shape: {} '.format(type(y_train), y_train.shape))
    print('IMPORTED: TEST DATA')
    print('X : {}, shape: {} '.format(type(X_test), X_test.shape))
    print('y : {}, shape: {} '.format(type(y_test), y_test.shape))


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cls = AutoSklearnClassifier(time_left_for_this_task=28800, n_jobs=3)

    cls.fit(X_train, y_train)

    dump(cls, OUTPUT_PATH / 'auto_cls.joblib')

    predictions = cls.predict(X_test)
    print("Accuracy score", accuracy_score(y_test, predictions))
