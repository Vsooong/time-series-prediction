from math import sqrt
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import *
import keras.metrics
from CNN_forecast import split_dataset, evaluate_forecasts, summarize_scores
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard


class LSTM_ELE:

    def to_supervised(self, train, n_input, n_out=7):
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
            in_start += 1
        return array(X), array(y)

    def forecast(self, model, history, n_input):
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        input_x = data[-n_input:, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        yhat = model.predict(input_x, verbose=0)
        yhat = yhat[0]
        return yhat

    def build_model(self, train, n_input):
        train_x, train_y = self.to_supervised(train, n_input)
        verbose, epochs, batch_size = 2, 100, 32
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))

        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=10,
                                       cooldown=0, patience=5, min_lr=1e-5, verbose=1)
        tensorboard = TensorBoard(log_dir='tensorboard/ED-lstm-log', write_images=True)

        model.compile(loss='mse', optimizer=Adam(0.01), metrics=['mape'])

        early_stopping = EarlyStopping(monitor='loss', patience=60, min_delta=100)
        model.summary()
        his = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=[early_stopping, ])
        pyplot.figure(figsize=(6, 5))
        pyplot.plot(his.history['loss'], 'red', label='training mse loss')
        pyplot.ylim(200000, 600000)
        pyplot.legend()
        pyplot.show()
        return model

    def evaluate_model(self, model, train, test, n_input):
        history = [x for x in train]
        predictions = list()
        for i in range(len(test)):
            yhat_sequence = self.forecast(model, history, n_input)
            predictions.append(yhat_sequence)
            history.append(test[i, :])
        predictions = array(predictions)
        score, scores = evaluate_forecasts(test[:, :, 0], predictions)
        return score, scores


if __name__ == "__main__":
    dataset = read_csv('load_data/household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                       parse_dates=['datetime'], index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_input = 14
    forecast = LSTM_ELE()
    model = forecast.build_model(train, n_input)

    score, scores = forecast.evaluate_model(model, train, test, n_input)
    model.save("models/ED-lstm-multivariate-%sprior-%d.h5" % (str(n_input), score))
    summarize_scores('lstm', score, scores)
    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    pyplot.plot(days, scores, marker='o', label='lstm')
    pyplot.show()
