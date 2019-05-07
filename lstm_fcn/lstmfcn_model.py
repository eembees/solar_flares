import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import tensorflow as tf
import sklearn.model_selection
from keras.layers import *
from pathlib import Path
from warnings import warn
import mlxtend.plotting
import mlxtend.evaluate
import matplotlib.ticker
from matplotlib import pyplot as plt
import os.path
from sklearn.preprocessing import StandardScaler
from send2trash import send2trash

GOOGLE_COLAB = "google.colab" in sys.modules
if GOOGLE_COLAB:
    sys.path.append("./gdrive/My Drive/Colab Notebooks/solar_flares")
    plt.style.use("default")
    config = tf.ConfigProto(device_count={"GPU": 1})
    keras.backend.set_session(tf.Session(config=config))
else:
    config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    keras.backend.set_session(tf.Session(config=config))


# loss function
# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric


from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: {:.3f} — val_precision: {:.3f} — val_recall {:.3f}".format(_val_f1, _val_precision, _val_recall))
        return



def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def plot_confusion_matrices(y_target, y_pred, name, outdir, targets_to_binary=None, y_is_binary=False):
    """Plots multiclass and binary confusion matrices"""
    if len(y_target.shape) == 3:
        axis = 2
    else:
        axis = 0

    if y_is_binary:
        matrix = mlxtend.evaluate.confusion_matrix(y_target=y_target.argmax(axis=axis).ravel(),
                                                   y_predicted=y_pred.argmax(axis=axis).ravel())
        mlxtend.plotting.plot_confusion_matrix(matrix, show_normed=True, show_absolute=False)
        plt.savefig(os.path.join(outdir, name + "_binary_confusion_matrix.pdf"))
        plt.close()
    else:
        matrix = mlxtend.evaluate.confusion_matrix(y_target=y_target.argmax(axis=axis).ravel(),
                                                   y_predicted=y_pred.argmax(axis=axis).ravel())
        mlxtend.plotting.plot_confusion_matrix(matrix, show_normed=True, show_absolute=False)
        plt.savefig(os.path.join(outdir, name + "_confusion_matrix.pdf"))
        plt.close()

        if targets_to_binary is not None:
            y_target_b, y_pred_b = [lib.ml.labels_to_binary(y, one_hot=True, to_ones=targets_to_binary).ravel() for y in
                                    (y_target, y_pred)]
            matrix = mlxtend.evaluate.confusion_matrix(y_target=y_target_b, y_predicted=y_pred_b)
            mlxtend.plotting.plot_confusion_matrix(matrix, show_normed=True, show_absolute=False)
            plt.savefig(os.path.join(outdir, name + "_binary_confusion_matrix.pdf"))
            plt.close()


## callbacks
def generate_callbacks(outdir, patience, name, monitor="val_loss", mode="min", verbose=1):
    """Generate callbacks for model"""
    log = keras.callbacks.CSVLogger(filename=os.path.join(outdir, name + "_training.log"), append=False)
    early_stopping = keras.callbacks.EarlyStopping(patience=patience, monitor=monitor, verbose=verbose, mode=mode)

    checkpoint_params = dict(save_best_only=True, monitor=monitor, mode=mode, verbose=False)
    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(outdir, name + "_best_model.h5"),
                                                       **checkpoint_params)
    weight_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(outdir, name + "_best_model_weights.h5"),
                                                        save_weights_only=True, **checkpoint_params
                                                        )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, mode="auto", min_delta=0.0001, cooldown=1, min_lr=0
    )
    callbacks = [log, early_stopping, model_checkpoint, weight_checkpoint, reduce_lr]
    return callbacks


def gpu_model_to_cpu(trained_gpu_model, untrained_cpu_model, outdir, modelname):
    """
    Loads a keras GPU model and saves it as a CPU-compatible model.
    The models must be exactly alike.
    """
    weights = os.path.join(outdir, "weights_temp.h5")
    trained_gpu_model.save_weights(weights)
    untrained_cpu_model.load_weights(weights)
    keras.models.save_model(untrained_cpu_model, os.path.join(outdir, modelname))
    try:
        send2trash(weights)
    except OSError:
        warn("Didn't trash file (probably because of Google Drive)", RuntimeWarning)
    except NameError:
        warn("Didn't trash file (probably because of Google Drive)", RuntimeWarning)


def create_model(google_colab, n_features):
    """Creates Keras model"""
    LSTM_ = CuDNNLSTM if google_colab else LSTM

    inputs = Input(shape=(None, n_features))  # Allow for time series that are shorter than 60

    y = Conv1D(filters=128, kernel_size=8, padding="same", kernel_initializer="he_uniform")(inputs)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate=0.3)(y)

    y = Conv1D(filters=256, kernel_size=5, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate=0.3)(y)

    y = Conv1D(filters=128, kernel_size=3, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate=0.3)(y)

    y = AveragePooling1D(1, padding="same")(y)
    # y = GlobalMaxPool1D()(y)

    y = Bidirectional(LSTM_(8, return_sequences=False))(y)
    y = Dropout(0.4)(y)

    x = AveragePooling1D(strides=1, padding="same")(inputs)
    x = Bidirectional(LSTM_(8, return_sequences=False))(x)
    x = Dropout(0.4)(x)

    final = Concatenate()([x, y])

    # final = Lambda(lambda x: x / 0.5)(final)
    outputs = Dense(2, activation="softmax")(final)
    # outputs = Dense(2)(final)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    # optimizer = keras.optimizers.rmsprop(lr=1e-3)
    optimizer = 'adam'
    # model.compile(loss=f1_loss, optimizer=optimizer, metrics=["accuracy"])
    # model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# def create_model(google_colab, n_features):
#     """Creates Keras model"""
#     LSTM_ = CuDNNLSTM if google_colab else LSTM
#
#     inputs = Input(shape=(None, n_features))  # Allow for time series that are shorter than 60
#
#     y = Conv1D(filters=128, kernel_size=8, padding="same", kernel_initializer="he_uniform")(inputs)
#     y = BatchNormalization()(y)
#     y = Activation("relu")(y)
#     y = SpatialDropout1D(rate=0.3)(y)
#
#     y = Conv1D(filters=256, kernel_size=5, padding="same", kernel_initializer="he_uniform")(y)
#     y = BatchNormalization()(y)
#     y = Activation("relu")(y)
#     y = SpatialDropout1D(rate=0.3)(y)
#
#     y = Conv1D(filters=128, kernel_size=3, padding="same", kernel_initializer="he_uniform")(y)
#     y = BatchNormalization()(y)
#     y = Activation("relu")(y)
#     y = SpatialDropout1D(rate=0.3)(y)
#
#     y = AveragePooling1D(1, padding="same")(y)
#     # y = GlobalMaxPool1D()(y)
#
#     y = Bidirectional(LSTM_(8, return_sequences=False))(y)
#     y = Dropout(0.4)(y)
#
#     x = AveragePooling1D(strides=1, padding="same")(inputs)
#     x = Bidirectional(LSTM_(8, return_sequences=False))(x)
#     x = Dropout(0.4)(x)
#
#     final = Concatenate()([x, y])
#
#     final = Lambda(lambda x: x / 0.5)(final)
#     outputs = Dense(2, activation="softmax")(final)
#
#     model = keras.models.Model(inputs=inputs, outputs=outputs)
#     optimizer = keras.optimizers.rmsprop(lr=1e-3)
#     # optimizer = 'adam'
#     # model.compile(loss=f1_loss, optimizer=optimizer, metrics=["accuracy"])
#     model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#     # model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#     return model


def get_model(n_features, train, new_model, model_name, model_path, google_colab, print_summary=True):
    """Loader for model"""
    if train:
        if new_model:
            print("Created new model.")
            model = create_model(google_colab=google_colab, n_features=n_features)
        else:
            try:
                model = keras.models.load_model(str(model_path.joinpath(model_name)))
            except OSError:
                print("No model found. Created new model.")
                model = create_model(google_colab=google_colab, n_features=n_features)
    else:
        print("Loading model from file..")
        model = keras.models.load_model(str(model_path.joinpath(model_name)))

    if print_summary:
        model.summary()
    return model
