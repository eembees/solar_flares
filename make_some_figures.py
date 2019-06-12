import numpy as np
from matplotlib import pyplot as plt

from reading_data import *


if __name__ == '__main__':

    data_dir = Path('/Users/mag/PycharmProjects/solar_flares/input/npz')
    out_dir = Path('/Users/mag/PycharmProjects/solar_flares/output/somefigs')
    out_dir.mkdir(exist_ok=True)


    fs = list(data_dir.glob('*Training.npz'))

    f = fs[0]

    fo = f.with_suffix('.png')


    X, y = load_npz_file(f)

    ## Split dataset by positive or negative
    X_0 = X[np.argwhere(y == 0)] # type: np.ndarray
    X_1 = X[np.argwhere(y == 1)] # type: np.ndarray



    N  = 10


    for i in range(N):

        fpo = out_dir / Path(str(fo.name).replace('.png','_{}_0.png'.format(i)))
        fig, axes = plt.subplots(nrows= 5, ncols=5, figsize=(10,6), constrained_layout = True)
        ax = axes.ravel()

        for j, axj in enumerate(ax):
            axj.plot(X_0[i].transpose()[j], c='r')
            axj.set_xticks([])
            axj.set_yticks([])


        fig.savefig(fpo)

        fpo = out_dir / Path(str(fo.name).replace('.png','_{}_1.png'.format(i)))
        fig, axes = plt.subplots(nrows= 5, ncols=5, figsize=(10,6), constrained_layout = True)
        ax = axes.ravel()

        for j, axj in enumerate(ax):
            axj.plot(X_1[i].transpose()[j], c='b')
            axj.set_xticks([])
            axj.set_yticks([])


        fig.savefig(fpo)







