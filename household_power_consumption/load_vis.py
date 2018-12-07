import pandas as pd
import numpy as np
from numpy import NaN, nan, isnan
from matplotlib import pyplot

global dataset


def clean_data():
    dataset = pd.read_csv('E:/pattern recognition/household_power_consumption.txt', sep=';', header=0,
                          low_memory=False, infer_datetime_format=True,
                          parse_dates={'datetime': [0, 1]},
                          index_col=['datetime'])
    # summarize
    dataset.replace("?", nan, inplace=True)
    values = dataset.values.astype('float32')

    def fill_missing(values):
        one_day = 60 * 24
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if (values[row, col]):
                    values[row, col] = values[row - one_day, col]

    fill_missing(dataset.values)
    dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 5] + values[:, 5] + values[:, 6])
    print(dataset.shape)
    print(dataset.isnull().sum())
    dataset.to_csv('household_power_consumption.csv')


def plot_lineplot():
    pyplot.figure(figsize=(5, 8))
    for i in range(len(dataset.columns)):
        pyplot.subplot(len(dataset.columns), 1, i + 1)
        name = dataset.columns[i]
        pyplot.plot(dataset[name])
        pyplot.title(name, y=0)
    pyplot.show()

    years = ['2007', '2008', '2009', '2010']
    pyplot.figure()
    for i in range(len(years)):
        # prepare subplot
        ax = pyplot.subplot(len(years), 1, i + 1)
        # determine the year to plot
        year = years[i]
        # get all observations for the year
        result = dataset[str(year)]
        # plot the active power for the year
        pyplot.plot(result['Global_active_power'])
        # add a title to the subplot
        pyplot.title(str(year), y=0, loc='left')
    pyplot.show()


def plot_hist():
    pyplot.figure(figsize=(5, 8))
    for i in range(len(dataset.columns)):
        pyplot.subplot(len(dataset.columns), 1, i + 1)
        name = dataset.columns[i]
        dataset[name].hist(bins=100)
        pyplot.title(name, y=0)
    pyplot.show()
    months = [x for x in range(1, 13)]
    pyplot.figure(figsize=(8, 16))
    for i in range(len(months)):
        # prepare subplot
        ax = pyplot.subplot(len(months), 1, i + 1)
        # determine the month to plot
        month = '2007-' + str(months[i])
        # get all observations for the month
        result = dataset[month]
        # plot the active power for the month
        result['Global_active_power'].hist(bins=100)
        # zoom in on the distribution
        ax.set_xlim(0, 5)
        # add a title to the subplot
        pyplot.title(month, y=0, loc='right')
    pyplot.show()


if __name__ == "__main__":
    # clean_data()
    dataset = pd.read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True,
                          parse_dates=['datetime'], index_col=['datetime'])
    daily_data = dataset.resample('D')
    daily_data = daily_data.sum()
    print(daily_data.shape)
    print(daily_data.head())
    daily_data.to_csv('household_power_consumption_days.csv')

    # plot_hist()
