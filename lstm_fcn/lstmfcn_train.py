import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import tensorflow as tf
import sklearn.model_selection
from keras.layers import *
from pathlib import Path

from lstm_fcn.lstmfcn_model import *
from reading_data import load_npz_file

if __name__ == "__main__":
    ROOTDIR = "/Users/mag/Google Drive/Colab Notebooks/solar_flares/"
    DATADIR = "input/npz"
    OUTDIR = "output"
    DATANAME = "lstm_fcn"
    TAG = "spatialdropout"

    TRAIN = True
    NEW_MODEL = True

    EPOCHS = 2
    PERCENTAGE = 100
    BATCH_SIZE = 64
    CALLBACK_TIMEOUT = 15
    N_TIMESTEPS = 60  # Change if Variable length
    INCLUDE_E = True
    INCLUDE_S = True
    SCALER = sklearn.preprocessing.maxabs_scale
    STATIONARY = False

    model_name = "{}_best_model.h5".format(DATANAME)
    if GOOGLE_COLAB:
        ROOTDIR = "./gdrive/My Drive/Colab Notebooks" + str(ROOTDIR).split("Colab Notebooks")[-1]

    rootdir = Path(ROOTDIR)
    datadir = rootdir.joinpath(DATADIR)
    outdir = rootdir.joinpath(OUTDIR)

    X, y = load_npz_file(datadir / 'fold1Training.npz')

    y = keras.utils.to_categorical(y, num_classes=2)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y)

    model = get_model(
        n_features=X_train.shape[-1],
        train=TRAIN,
        new_model=NEW_MODEL,
        model_name=model_name,
        model_path=outdir,
        google_colab=GOOGLE_COLAB,
    )

    if TAG is not None:
        DATANAME += "_" + TAG
        model_name = model_name.replace("best_model", TAG + "_best_model")

    if TRAIN:
        callbacks = generate_callbacks(patience=CALLBACK_TIMEOUT, outdir=outdir, name=DATANAME)
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )
        # lib.plotting.plot_losses(logpath=outdir, outdir=outdir, name=DATANAME)

        if GOOGLE_COLAB:
            print("Converted model from GPU to CPU-compatible")
            cpu_model = create_model(google_colab=False, n_features=X_train.shape[-1])
            gpu_model_to_cpu(
                trained_gpu_model=model, untrained_cpu_model=cpu_model, outdir=outdir, modelname=model_name
            )

    print("Evaluating...")
    y_pred = model.predict(X_test)
    plot_confusion_matrices(
        y_target=y_test, y_pred=y_pred, y_is_binary=True, targets_to_binary=[2, 3], outdir=outdir, name=DATANAME
    )
