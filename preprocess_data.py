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
    from reading_data import *

    inp = Path('./input/npz/')
    filenames = inp.glob('*.npz')

    filenames = [f for f in filenames if 'ed' not in str(f)]

    for fp in filenames:
        print('Now treating file: ')
        print(fp.name)
        fpo = Path(str(fp).replace('.npz', '_processed.npz'))
        X, y, ids = load_npz_file(fp, return_ids=True)
        X = preprocess_data(X)
        np.savez(fpo, data=X, labels=y, index=ids)
        X, y = None, None
