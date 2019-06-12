import numpy as np
def makeNewSequenceWithNoise(X_i:np.ndarray, n_frames = 5, noise_frac = 0.3):
    # X_i is shape 60, 25

    X = X_i.transpose()

    X_new = np.zeros_like(X)

    for i, seq in enumerate(X):
        noise = np.std(seq[:n_frames])
        if noise is 0: # from all nan slices being converted to 0
            seq_new = seq
        else:
            seq_new = seq + np.random.normal(loc=0.0, scale=noise * noise_frac, size=len(seq))

        X_new[i] = seq_new

    return X_new.transpose()





if __name__ == '__main__':
    from pathlib import Path
    from reading_data import *
    from sklearn.utils import resample
    import matplotlib.pyplot as plt
    inp = Path('./input/npz/')
    filenames = inp.glob('fold?Training.npz')

    filenames = [f for f in filenames if '3' not in str(f)] # use fold 3 for validation


    fpo = inp.joinpath('fold1-2Training_balanced.npz')

    Xs = []
    ys = []

    for fp in filenames:
        print('Now loading file: ')
        print(fp.name)
        X, y = load_npz_file(fp)
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print('concatenated all files')
    # print(newX.shape)

    ## Split dataset by positive or negative
    X_0 = X[np.argwhere(y == 0)] # type: np.ndarray
    X_1 = X[np.argwhere(y == 1)] # type: np.ndarray


    X_0 = X_0.reshape((-1, 60, 25))
    X_1 = X_1.reshape((-1, 60, 25))

    print(X_0.shape)
    print(X_1.shape)


    # TODO make this again so that we have more data to train on
    # for x in X_1:
    #     plt.plot(x[:,0])
    #     print(x[:,0])
    #     x = makeNewSequenceWithNoise(x)
    #     print(x[:,0])
    #     plt.plot(x[:,0])
    #     plt.show()
    #     exit()


    n_0 = X_0.shape[0]
    n_1 = X_1.shape[0]

    n   = min(n_0, n_1)

    ## Make the dataset 50/50 split

    X_0_new = resample(X_0, n_samples=n, replace=False)
    X_1_new = resample(X_1, n_samples=n, replace=False) # this is to make it bigger

    y_0_new = np.zeros(n, dtype=int)
    y_1_new = np.ones(n, dtype=int)


    X_new = np.concatenate([X_0_new, X_1_new])# type: np.ndarray
    y_new = np.concatenate([y_0_new, y_1_new])# type: np.ndarray



    # mix them up so we don't have any issues with indexing (although it shouldn't happen)
    X, y = resample(X_new, y_new, replace=False) # type: np.ndarray

    print(X.shape)
    print(y.shape)

    ## Save and profit
    ids = np.arange(1, 2*n + 1)

    assert len(ids ) == len(y)

    np.savez(fpo, data=X, labels=y, index=ids)
    print('Saved as: ')
    print(fpo.absolute())
