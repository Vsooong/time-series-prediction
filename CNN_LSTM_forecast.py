from LSTM_forecast import *


class CNN_LSTM_ELE_uni(LSTM_ELE):
    def forecast(self, model, history, n_input):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, 0]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    # convert history into inputs and outputs
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
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)

    def build_model(self, train, n_input):
        # prepare data
        train_x, train_y = self.to_supervised(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 2, 200, 128
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mse', optimizer=RMSprop(), metrics=['mape'])
        # fit network
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=10,
                                       cooldown=0, patience=5, min_lr=1e-5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=60, min_delta=100)
        model.summary()
        his = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=[])
        pyplot.figure(figsize=(6, 5))
        pyplot.plot(his.history['loss'], 'red', label='training mse loss')
        pyplot.ylim(200000, 600000)
        pyplot.legend()
        pyplot.show()
        return model


class CNN_LSTM_ELE(LSTM_ELE):

    def build_model(self, train, n_input):
        # prepare data
        train_x, train_y = self.to_supervised(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 2, 1000, 128
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mse', optimizer=Nadam(), metrics=['mape'])
        # fit network
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=10,
                                       cooldown=0, patience=5, min_lr=1e-5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=60, min_delta=100)
        model.summary()
        his = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=[])
        pyplot.figure(figsize=(6, 5))
        pyplot.plot(his.history['loss'], 'red', label='training mse loss')
        pyplot.ylim(180000, 600000)
        pyplot.legend()
        pyplot.show()
        return model


class Convlstm_ELE(LSTM_ELE):
    pass


if __name__ == "__main__":
    dataset = read_csv('load_data/household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                       parse_dates=['datetime'], index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_input = 14
    forecast = CNN_LSTM_ELE()
    model = forecast.build_model(train, n_input)

    score, scores = forecast.evaluate_model(model, train, test, n_input)
    model.save("models/ED-cnnlstm-multivariate-%sprior-%d.h5" % (str(n_input), score))
    summarize_scores('lstm', score, scores)
    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    pyplot.plot(days, scores, marker='o', label='lstm')
    pyplot.show()
