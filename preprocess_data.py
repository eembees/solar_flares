from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import maxabs_scale


def preprocess_data(X, scaler=maxabs_scale):
    shap = X.shape
    # print(shap[1:])
    if shap[1:] != (60, 25):
        raise ValueError('Data shape wrong')
    for i, x_i in enumerate(X):
        x_i_t = np.zeros_like(x_i.transpose())
        for j, series in enumerate(x_i.transpose()):
            series = scaler(series)
            x_i_t[j] = series
        X[i] = x_i_t.transpose()
    return X


if __name__ == '__main__':
    from pathlib import Path
    from reading_data import load_npz_file

    inp = Path('./input/npz/')
    filenames = inp.glob('small.npz')

    for fp in filenames:
        X, y = load_npz_file(fp)
        print(X[0, 0])
        X = preprocess_data(X)
        print(X[0, 0])

        exit()