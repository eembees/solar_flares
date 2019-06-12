import os
import sys

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import keras
import tensorflow
from reading_data import load_npz_file

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
GOOGLE_COLAB = "google.colab" in sys.modules
if GOOGLE_COLAB:
    sys.path.append("./gdrive/My Drive/Colab Notebooks/")
    config = tensorflow.ConfigProto(device_count={"GPU": 1})
else:
    config = tensorflow.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
keras.backend.set_session(tensorflow.Session(config=config))


if __name__ == "__main__":
    PLOTDIR = "~/Desktop/Traces"
    DATADIR = "input/npz"
    OUTDIR = "output"
    MODELNAME = "sim_spatialdropout"
    NPZ = "testSet"
    PLOT = True
    CSVNAME = NPZ + MODELNAME + '.csv'


    rootdir = Path(os.getcwd())
    if GOOGLE_COLAB:
        rootdir = "./gdrive/My Drive/Colab Notebooks" + str(rootdir).split("Colab Notebooks")[-1]


    model_name = "{}_best_model.h5".format(MODELNAME)
    datadir = rootdir.joinpath(DATADIR)
    outdir = rootdir.joinpath(OUTDIR)
    plotdir = Path(PLOTDIR).expanduser()
    if not plotdir.exists():
        plotdir.mkdir()

    print("Testing real data examples...")
    model = keras.models.load_model(os.path.join(outdir, model_name))

    X_pred, _ = load_npz_file(datadir.joinpath(NPZ + ".npz"))
    X_pred = sklearn.preprocessing.StandardScaler().fit_transform(X_pred.reshape((X_pred.shape[0], X_pred.shape[1]*X_pred.shape[2]))).reshape(X_pred.shape)

    y_pred = model.predict(X_pred)


    for ax in axes:
        ax.set_xlim(-0.15, 1.15)
    plt.savefig(plotdir.joinpath("hist.pdf"))


    df = pd.DataFrame({'Id':ids,'ClassLabel':np.argmax(y_pred,axis=1)})
    df.to_csv(outdir.join(CSVNAME))