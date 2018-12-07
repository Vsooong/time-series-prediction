import pandas as pd
import numpy as np
from numpy import NaN, nan, isnan, array, split
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


def to_supervised(train, n_input, n_out=7, stride=1):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += stride
    return array(X), array(y)


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def plot_history(history):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('loss', y=0, loc='center')
    pyplot.legend()
    # plot rmse
    pyplot.subplot(2, 1, 2)
    pyplot.plot(history.history['rmse'], label='train')
    pyplot.plot(history.history['val_rmse'], label='test')
    pyplot.title('rmse', y=0, loc='center')
    pyplot.legend()
    pyplot.show()


# train the model
def multichannel_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 2, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=100,
                                   cooldown=0, patience=4, min_lr=1e-5, )
    his = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose
                    , callbacks=[lr_reducer])

    pyplot.figure(figsize=(6, 5))
    pyplot.plot(his.history['loss'][1:], 'red')
    pyplot.legend()
    pyplot.title("multichannel CNN")
    pyplot.show()
    return model


# make a forecast

def multi_channel_forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat.squeeze()
    return yhat


# evaluate a single model
def evaluate_model(model, train, test, n_input, mode=1):
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        if mode == 1:
            yhat_sequence = multi_channel_forecast(model, history, n_input)
        else:
            yhat_sequence = multihead_forecast(model, history, n_input)

        predictions.append(yhat_sequence)
        history.append(test[i, :])
    # evaluate predictions days for each week`
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def multihead_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 2, 60, 32
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # create a channel for each variable
    in_layers, out_layers = list(), list()
    for i in range(n_features):
        inputs = Input(shape=(n_timesteps, 1))
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        flat = Flatten()(pool1)
        # store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # merge heads
    merged = concatenate(out_layers)
    # interpretation
    dense1 = Dense(200, activation='relu')(merged)
    dense2 = Dense(100, activation='relu')(dense1)
    outputs = Dense(n_outputs, )(dense2)
    model = Model(inputs=in_layers, outputs=outputs)
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # fit network
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    model.summary()
    # lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=100,
    #                                cooldown=0, patience=4, min_lr=1e-5, )
    his = model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                    callbacks=[])
    pyplot.figure(figsize=(6, 5))
    pyplot.plot(his.history['loss'][1:], 'red')
    pyplot.legend()
    pyplot.title("multihead CNN")
    pyplot.show()
    return model


def multihead_forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into n input arrays
    input_x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


if __name__ == '__main__':
    dataset = pd.read_csv('load_data/household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                          parse_dates=['datetime'], index_col=['datetime'])

    train, test = split_dataset(dataset.values)
    n_input = 14
    multi_mode = 2

    if multi_mode == 1:

        model = multichannel_model(train, n_input)
        score, scores = evaluate_model(model, train, test, n_input)

        model.save("models/multichannel-cnn-multivariate-%sprior-%d.h5" % (str(n_input), score))
        summarize_scores("CNN", score, scores)
        days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
        pyplot.plot(days, scores, marker='o', label='cnn')
        pyplot.show()
    else:
        model = multihead_model(train, n_input)
        score, scores = evaluate_model(model, train, test, n_input, 2)
        model.save("models/multihead-cnn-multivariate-%sprior-%d.h5" % (str(n_input), score))
        summarize_scores("CNN", score, scores)
        days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
        pyplot.plot(days, scores, marker='o', label='cnn')
        pyplot.show()
