import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import tensorflow as tf
import sklearn.model_selection
from keras.layers import *
from pathlib import Path
import pandas as pd
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

    EPOCHS = 10
    PERCENTAGE = 100
    BATCH_SIZE = 128
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

    # X1, y1 = load_npz_file(datadir / 'fold1Training.npz')
    # X2, y2 = load_npz_file(datadir / 'fold2Training.npz')
    # X3, y3 = load_npz_file(datadir / 'fold3Training.npz')
    #
    # X = np.concatenate([X1, X2, X3])
    # y = np.concatenate([y1, y2, y3])


    X, y = load_npz_file(datadir / 'fold1Training.npz')

    X_pred, _ = load_npz_file(datadir / 'testSet.npz')

    y = keras.utils.to_categorical(y, num_classes=2)

    # preprocess x

    X = sklearn.preprocessing.StandardScaler().fit_transform(X.reshape((X.shape[0], X.shape[1]*X.shape[2]))).reshape(X.shape)
    X_pred = sklearn.preprocessing.StandardScaler().fit_transform(X_pred.reshape((X_pred.shape[0], X_pred.shape[1]*X_pred.shape[2]))).reshape(X_pred.shape)

    # print(X.shape)
    # exit()

    # TODO flatten and reshape before scaling
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
    y_pred = model.predict(X_pred)


    #
    # for yp, yt in zip(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)):
    #     print('T:{} P:{}'.format(yt,yp))
    #
    # plot_confusion_matrices(
    #     y_target=np.argmax(y_test,axis=1), y_pred=np.argmax(y_pred,axis=1), y_is_binary=True, outdir=outdir, name=DATANAME
    # )
    #
    ids = np.arange(1, len(y_pred) + 1, dtype=int)

    df = pd.DataFrame({'Id':ids,'ClassLabel':np.argmax(y_pred,axis=1)})
    df.to_csv(outdir/'submission.csv')