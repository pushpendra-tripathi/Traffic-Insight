"""
Traffic Flow Prediction with Machine Learning. Driver Code.
"""
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from data.data import pre_process_data

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted result.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print("explained_variance_score:%f" % vs)
    print("mape:%f%%" % mape)
    print("mae:%f" % mae)
    print("mse:%f" % mse)
    print("rmse:%f" % math.sqrt(mse))
    print("r2:%f" % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = "2022-3-4 00:00"
    x = pd.date_range(d, periods=288, freq="5min")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label="True Data")
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel("Traffic Volume")

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def main():
    # models
    stacked_lstm = load_model("model/stacked_lstm.h5")
    lstm = load_model("model/lstm.h5")
    gru = load_model("model/gru.h5")
    saes = load_model("model/saes.h5")

    model_dict = [
        {"name": "Traffic Insight", "model": stacked_lstm},
        {"name": "LSTM", "model": lstm},
        {"name": "GRU", "model": gru},
        {"name": "SAEs", "model": saes},
    ]

    lag = 12
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    _, _, X_test, y_test, scaler = pre_process_data(train_file, test_file, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for obj in model_dict:
        name = obj["name"]
        model = obj["model"]
        if name == "SAEs":
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = "images/" + name + ".png"
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[:288], y_preds, [obj["name"] for obj in model_dict])


if __name__ == "__main__":
    main()
