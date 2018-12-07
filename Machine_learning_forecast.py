import CNN_forecast as cnnf
from numpy import array
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_ML_model(train, n_input):
    pass


def recursive_multi_step_forecast(model, input, n_input, result_steps):
    predict_y = list()
    input_data = list(input)
    for i in range(result_steps):
        X = array(input_data[-n_input:]).reshape(1, n_input)
        y = model.predict(X)[0]
        input_data.append(y)
        predict_y.append(y)
    return predict_y


def univariate_data(input, n_input, n_out=1):
    x, y = cnnf.to_supervised(input, n_input, n_out)
    x = x[:, :, 0]
    y = y[:, 0]
    return x, y


def ML_models(models=dict()):
    models['lr'] = LinearRegression()
    models['sgd'] = SGDRegressor(max_iter=500, tol=1e-3)
    models['rdf'] = RandomForestRegressor(n_estimators=15)
    print('Defined %d models' % len(models))
    steps = list()
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    return pipeline


def evaluate_model(model, train, test, n_input):
    pass


if __name__ == "__main__":
    dataset = pd.read_csv('load_data/household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                          parse_dates=['datetime'], index_col=['datetime'])

    train, test = cnnf.split_dataset(dataset.values)
    n_input = 14

    x, y = univariate_data(train, n_input)

    model = ML_models()
    recursive_multi_step_forecast(model['lr'], x[-1], n_input, 7)
