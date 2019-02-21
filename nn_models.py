import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import core
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import merge

from config import *
from utils.data_utils import read_ds


def model_1(input_dim=128, output_dim=50):
    input_layer = keras.layers.Input(shape=(input_dim,))
    dense_1 = Dense(256, kernel_initializer='normal', activation='relu')(input_layer)
    dense_2 = Dense(256, kernel_initializer='normal', activation='relu')(dense_1)
    dense_3 = Dense(256, kernel_initializer='normal', activation='relu')(dense_2)
    dense_4 = Dense(256, kernel_initializer='normal', activation='relu')(dense_3)
    dense_5 = Dense(output_dim, kernel_initializer='normal')(dense_4)

    output_layer = dense_5
    model = keras.Model(input_layer, output_layer)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Compile Model
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mean_squared_error'])

    return model


def model_5(input_dim=128, output_dim=50):
    input_layer = keras.layers.Input(shape=(input_dim,))
    bn_1 = BatchNormalization()(input_layer)
    dense_1 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(bn_1)
    dropout_1 = core.Dropout(0.5)(dense_1)
    dense_2 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(dropout_1)
    # model.add(core.Dropout(0.5))
    dense_3 = Dense(128, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_2)
    skip_1 = keras.layers.Add()([dense_1, dense_3])
    # model.add(core.Dropout(0.5))
    dense_4 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(skip_1)
    # model.add(core.Dropout(0.5))
    dense_5 = Dense(128, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_4)
    # dropout_2 = core.Dropout(0.5)(dense_5)
    bn_2 = BatchNormalization()(dense_5)
    dense_6 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(bn_2)
    skip_2 = keras.layers.Add()([dense_4, dense_6])
    # model.add(core.Dropout(0.5))
    dense_7 = Dense(output_dim, kernel_initializer='glorot_normal')(skip_2)

    output_layer = dense_7
    model = keras.Model(input_layer, output_layer)
    adam = keras.optimizers.Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Compile Model
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mean_squared_error'])

    return model


def model_6(input_dim=128, output_dim=50):
    input_layer = keras.layers.Input(shape=(input_dim,))
    bn_1 = BatchNormalization()(input_layer)
    dense_1 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(bn_1)
    dropout_1 = core.Dropout(0.5)(dense_1)
    dense_2 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(dropout_1)
    # model.add(core.Dropout(0.5))
    dense_3 = Dense(128, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_2)
    skip_1 = keras.layers.Add()([dense_1, dense_3])
    # model.add(core.Dropout(0.5))
    dense_4 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(skip_1)
    # model.add(core.Dropout(0.5))
    dense_5 = Dense(128, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_4)
    # dropout_2 = core.Dropout(0.5)(dense_5)
    bn_2 = BatchNormalization()(dense_5)
    dense_6 = Dense(128, kernel_initializer='glorot_normal', activation='relu')(bn_2)
    skip_2 = keras.layers.Add()([dense_4, dense_6])
    # model.add(core.Dropout(0.5))
    dense_7 = Dense(output_dim, kernel_initializer='glorot_normal', activation='sigmoid')(skip_2)

    output_layer = dense_7
    model = keras.Model(input_layer, output_layer)
    adam = keras.optimizers.Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Compile Model
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mean_squared_error'])

    return model


def model_7(input_dim=128, output_dim=50):
    input_layer = keras.layers.Input(shape=(input_dim,))
    bn_1 = BatchNormalization()(input_layer)
    dense_1 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(bn_1)
    dropout_1 = core.Dropout(0.5)(dense_1)
    dense_2 = Dense(512, kernel_initializer='glorot_normal', activation='relu')(dropout_1)
    dense_3 = Dense(1024, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_2)
    bn_2 = BatchNormalization()(dense_3)
    dense_4 = Dense(1024, kernel_initializer='glorot_normal', activation='relu')(bn_2)
    dense_5 = Dense(512, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu')(dense_4)
    # dropout_2 = core.Dropout(0.5)(dense_5)
    bn_2 = BatchNormalization()(dense_5)
    dense_6 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(bn_2)
    dropout_3 = core.Dropout(0.5)(dense_6)
    dense_7 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(dropout_3)
    dense_8 = Dense(output_dim, kernel_initializer='glorot_normal')(dense_7)

    output_layer = dense_8
    model = keras.Model(input_layer, output_layer)
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Compile Model
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mean_squared_error'])

    return model


if __name__ == "__main__":
    x_train, y_train = read_ds(TRAIN_DS_DIR)
    x_val, y_val = read_ds(VAL_DS_DIR)

    # create model
    model = model_7(input_dim=x_train[0].shape[0], output_dim=y_train[0].shape[0])

    # Fit the model
    training_result = model.fit(x_train, y_train, epochs=4000, batch_size=120, validation_data=(x_val, y_val))

    loss, acc = model.evaluate(x_val, y_val)

    print("Loss: ", loss)
    print("Accuracy: ", acc)
