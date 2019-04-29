import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import tensorflow as tf
import sklearn.model_selection
from keras.layers import *
from pathlib import Path


GOOGLE_COLAB = "google.colab" in sys.modules
if GOOGLE_COLAB:
    sys.path.append("./gdrive/My Drive/Colab Notebooks/FRETML")
    plt.style.use("default")
    config = tf.ConfigProto(device_count={"GPU": 1})
    keras.backend.set_session(tf.Session(config=config))
else:
    config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    keras.backend.tf_backend.set_session(tf.Session(config=config))


# loss function
# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric


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


def create_model(google_colab, n_features):
    """Creates Keras model"""
    LSTM_ = CuDNNLSTM if google_colab else LSTM

    inputs = Input(shape=(None,n_features)) # Allow for time series that are shorter than 60

    y = Conv1D(filters=128, kernel_size=8, padding="same", kernel_initializer="he_uniform")(inputs)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate = 0.3)(y)

    y = Conv1D(filters=256, kernel_size=5, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate = 0.3)(y)

    y = Conv1D(filters=128, kernel_size=3, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(rate = 0.3)(y)

    y = AveragePooling1D(1, padding="same")(y)
    y = GlobalMaxPool1D(1, data_format='channels_last')(y)

    y = Bidirectional(LSTM_(8, return_sequences=True))(y)
    y = Dropout(0.4)(y)

    x = AveragePooling1D(strides = 1, padding = "same")(inputs)
    x = Bidirectional(LSTM_(8, return_sequences = True))(x)
    x = Dropout(0.4)(x)

    final = Concatenate()([x, y])

    final = Lambda(lambda x: x / 0.5)(final)
    outputs = Dense(6, activation="softmax")(final)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    # optimizer = keras.optimizers.rmsprop(lr=1e-3)
    optimizer = 'adam'
    model.compile(loss=f1_loss, optimizer=optimizer, metrics=["accuracy"])
    return model



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

