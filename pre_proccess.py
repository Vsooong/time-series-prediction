from numpy import *
from matplotlib import pyplot
from pandas import read_csv


# split the dataset by 'chunkID', return a dict of id to rows
def to_chunks(values, chunk_ix=1):
    chunks = dict()
    # get the unique chunk ids
    chunk_ids = unique(values[:, chunk_ix])
    # group rows by chunk id
    for chunk_id in chunk_ids:
        selection = values[:, chunk_ix] == chunk_id
        chunks[chunk_id] = values[selection, :]
    return chunks


def interpolate_hours(hours):
    # find the first hour
    ix = -1
    for i in range(len(hours)):
        if not isnan(hours[i]):
            ix = i
            break
    # fill-forward
    hour = hours[ix]
    for i in range(ix + 1, len(hours)):
        # increment hour
        hour += 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24
    # fill-backward
    hour = hours[ix]
    for i in range(ix - 1, -1, -1):
        # decrement hour
        hour -= 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24


def split_train_test(chunks, row_in_chunk_ix=2):
    train, test = list(), list()
    # first 5 days of hourly observations for train
    cut_point = 5 * 24
    # enumerate chunks
    for k, rows in chunks.items():
        # split chunk rows by 'position_within_chunk'
        train_rows = rows[rows[:, row_in_chunk_ix] <= cut_point, :]
        test_rows = rows[rows[:, row_in_chunk_ix] > cut_point, :]
        if len(train_rows) == 0 or len(test_rows) == 0:
            print('>dropping chunk=%d: train=%s, test=%s' % (k, train_rows.shape, test_rows.shape))
            continue
        # store with chunk id, position in chunk, hour and all targets
        indices = [1, 2, 5] + [x for x in range(56, train_rows.shape[1])]
        train.append(train_rows[:, indices])
        test.append(test_rows[:, indices])
    return train, test


# return a list of relative forecast lead times
def get_lead_times():
    return [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]


def to_forecasts(test_chunks, row_in_chunk_ix=1):
    # get lead times
    lead_times = get_lead_times()
    # first 5 days of hourly observations for train
    cut_point = 5 * 24
    forecasts = list()
    # enumerate each chunk
    for rows in test_chunks:
        chunk_id = rows[0, 0]
        # enumerate each lead time
        for tau in lead_times:
            # determine the row in chunk we want for the lead time
            offset = cut_point + tau
            # retrieve data for the lead time using row number in chunk
            row_for_tau = rows[rows[:, row_in_chunk_ix] == offset, :]
            # check if we have data
            if len(row_for_tau) == 0:
                # create a mock row [chunk, position, hour] + [nan...]
                row = [chunk_id, offset, nan] + [nan for _ in range(39)]
                forecasts.append(row)
            else:
                # store the forecast row
                forecasts.append(row_for_tau[0])
    return array(forecasts)


# convert the test dataset in chunks to [chunk][variable][time] format
def prepare_test_forecasts(test_chunks):
    predictions = list()
    # enumerate chunks to forecast
    for rows in test_chunks:
        # enumerate targets for chunk
        chunk_predictions = list()
        for j in range(3, rows.shape[1]):
            yhat = rows[:, j]
            chunk_predictions.append(yhat)
        chunk_predictions = array(chunk_predictions)
        predictions.append(chunk_predictions)
    return array(predictions)


# calculate the error between an actual and predicted value
def calculate_error(actual, predicted):
    # give the full actual value if predicted is nan
    if isnan(predicted):
        return abs(actual)
    # calculate abs difference
    return abs(actual - predicted)


def summarize_error(name, total_mae, times_mae):
    # print summary
    lead_times = get_lead_times()
    formatted = ['+%d %.3f' % (lead_times[i], times_mae[i]) for i in range(len(lead_times))]
    s_scores = ', '.join(formatted)
    print('%s: [%.3f MAE] %s' % (name, total_mae, s_scores))
    # plot summary
    pyplot.plot([str(x) for x in lead_times], times_mae, marker='.')
    pyplot.show()


# evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
    lead_times = get_lead_times()
    total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
    total_c, times_c = 0, [0 for _ in range(len(lead_times))]
    # enumerate test chunks
    for i in range(len(testset)):
        # convert to forecasts
        actual = testset[i]
        predicted = predictions[i]
        # enumerate target variables
        for j in range(predicted.shape[0]):
            # enumerate lead times
            for k in range(len(lead_times)):
                # skip if actual in nan
                if isnan(actual[j, k]):
                    continue
                # calculate error
                error = calculate_error(actual[j, k], predicted[j, k])
                # update statistics
                total_mae += error
                times_mae[k] += error
                total_c += 1
                times_c[k] += 1
    # normalize summed absolute errors
    total_mae /= total_c
    times_mae = [times_mae[i] / times_c[i] for i in range(len(times_mae))]
    return total_mae, times_mae


# impute missing data
def impute_missing(train_chunks, rows, hours, series, col_ix):
    # impute missing using the median value for hour in all series
    imputed = list()
    for i in range(len(series)):
        if isnan(series[i]):
            # collect all rows across all chunks for the hour
            all_rows = list()
            for rows in train_chunks:
                [all_rows.append(row) for row in rows[rows[:, 2] == hours[i]]]
            # calculate the central tendency for target
            all_rows = array(all_rows)
            # fill with median value
            value = nanmedian(all_rows[:, col_ix])
            if isnan(value):
                value = 0.0
            imputed.append(value)
        else:
            imputed.append(series[i])
    return imputed


# layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5 * 24):
    # lay out whole series
    data = [nan for _ in range(n_steps)]
    # mark all available data
    for i in range(len(chunk_train)):
        # get position in chunk
        position = int(chunk_train[i, 1] - 1)
        # store data
        data[position] = chunk_train[i, col_ix]
    return data


# interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
    # find the first hour
    ix = -1
    for i in range(len(hours)):
        if not isnan(hours[i]):
            ix = i
            break
    # fill-forward
    hour = hours[ix]
    for i in range(ix + 1, len(hours)):
        # increment hour
        hour += 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24
    # fill-backward
    hour = hours[ix]
    for i in range(ix - 1, -1, -1):
        # decrement hour
        hour -= 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24


# created input/output patterns from a sequence
def supervised_for_lead_time(series, n_lag, lead_time):
    samples = list()
    # enumerate observations and create input/output patterns
    for i in range(n_lag, len(series)):
        end_ix = i + (lead_time - 1)
        # check if can create a pattern
        if end_ix >= len(series):
            break
        # retrieve input and output
        start_ix = i - n_lag
        row = series[start_ix:i] + [series[end_ix]]
        samples.append(row)
    return samples


# return true if the array has any non-nan values
def has_data(data):
    return count_nonzero(isnan(data)) < len(data)


# create supervised learning data for each lead time for this target
def target_to_supervised(chunks, rows, hours, col_ix, n_lag):
    train_lead_times = list()
    # get series
    series = variable_to_series(rows, col_ix)
    if not has_data(series):
        return None, [nan for _ in range(n_lag)]
    # impute
    imputed = impute_missing(chunks, rows, hours, series, col_ix)
    # prepare test sample for chunk-variable
    test_sample = array(imputed[-n_lag:])
    # enumerate lead times
    lead_times = get_lead_times()
    for lead_time in lead_times:
        # make input/output data from series
        train_samples = supervised_for_lead_time(imputed, n_lag, lead_time)
        train_lead_times.append(train_samples)
    return train_lead_times, test_sample


# prepare training [var][lead time][sample] and test [chunk][var][sample]
def data_prep(chunks, n_lag, n_vars=39):
    lead_times = get_lead_times()
    train_data = [[list() for _ in range(len(lead_times))] for _ in range(n_vars)]
    test_data = [[list() for _ in range(n_vars)] for _ in range(len(chunks))]
    # enumerate targets for chunk
    for var in range(n_vars):
        # convert target number into column number
        col_ix = 3 + var
        # enumerate chunks to forecast
        for c_id in range(len(chunks)):
            rows = chunks[c_id]
            # prepare sequence of hours for the chunk
            hours = variable_to_series(rows, 2)
            # interpolate hours
            interpolate_hours(hours)
            # check for no data
            if not has_data(rows[:, col_ix]):
                continue
            # convert series into training data for each lead time
            train, test_sample = target_to_supervised(chunks, rows, hours, col_ix, n_lag)
            # store test sample for this var-chunk
            test_data[c_id][var] = test_sample
            if train is not None:
                # store samples per lead time
                for lead_time in range(len(lead_times)):
                    # add all rows to the existing list of rows
                    train_data[var][lead_time].extend(train[lead_time])
        # convert all rows for each var-lead time to a numpy array
        for lead_time in range(len(lead_times)):
            train_data[var][lead_time] = array(train_data[var][lead_time])
    return array(train_data), array(test_data)


def to_naive_train_test():
    dataset = read_csv('D:/data/all/TrainingData.csv', header=0)
    # group data by chunks
    values = dataset.values
    chunks = to_chunks(values)
    print('Total Chunks: %d' % len(chunks))
    train, test = split_train_test(chunks)
    train_rows = array([row for rows in train for row in rows])
    # print(train_rows.shape)
    print('Train Rows: %s' % str(train_rows.shape))
    # reduce train to forecast lead times only
    test_rows = to_forecasts(test)
    print('Test Rows: %s' % str(test_rows.shape))
    savetxt('AirQualityPrediction/naive_train.csv', train_rows, delimiter=',')
    savetxt('AirQualityPrediction/naive_test.csv', test_rows, delimiter=',')


train = loadtxt('AirQualityPrediction/naive_train.csv', delimiter=',')
test = loadtxt('AirQualityPrediction/naive_test.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
test_chunks = to_chunks(test)
# convert training data into supervised learning data
n_lag = 12
train_data, test_data = data_prep(train_chunks, n_lag)
print(train_data.shape, test_data.shape)
# save train and test sets to file
save('AirQualityPrediction/supervised_train.npy', train_data)
save('AirQualityPrediction/supervised_test.npy', test_data)
