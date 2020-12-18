#!/usr/bin/env python

"""HAUPTPROJEKT Codebasis für ML Verfahren

In diesem Projekt werden Active Learning Strategien für das Labeln von
unangekündigten Mahlzeiten entwickelt.
Das trainierte Modell soll in ein bereits bestehendes Android Projekt
eingebunden werden.
"""

import csv
import math

import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Models import Profile

__author__ = "Hannes Voß"
__version__ = "0.1"

pd.set_option('display.max_columns', 500)

epochs = 10  # number of epochs per training session
select_per_epoch = 200  # number to select per epoch per label
feature_index = {}  # feature mapping for one-hot encoding

#print("Is CUDA available: ", torch.cuda.is_available())
#device = "cuda" if torch.cuda.is_available() else "cpu"


#class DataLoader(Dataset):
#    def __init__(self, src, tgt):
#        self.input_src = src
#        self.input_tgt = tgt

#    def __len__(self):
#        return len(self.input_src)


def read_profile(filename: str):
    """
    Liest das angegebene Profil aus. Dort ist auch das verwendete Insulin angegeben.

    :return: Gibt das verwendete Profil zurück.
    """
    # TODO: read the actual Profile instead of Dummy-Object

    return Profile("Humalog", "mg/dl")


def get_readable_glucose_data(filename: str):
    """
    Daten müssen vorbereitet werden.
    Timestamps müssen in Millis umgewandelt werden
    SGVs müssen in Ints umgewandelt werden
    """
    glucose_readings = list()
    with open(filename, newline='', encoding="utf-8") as csvfile:
        for row in csv.reader(csvfile, delimiter=',', skipinitialspace=True):
            glucose_readings.append({
                'timestamp': parser.parse(row[0]),
                'sgv': int(row[1])
            })
    return glucose_readings


def get_readable_treatments_data(filename: str):
    """
    Vorbereitung der Treatment Daten.

    :param filename: Dateiname
    :return: Gibt die vorformatierten Treatments als Liste zurück
    """
    treatment_readings = list()
    with open(filename, newline='', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip first line / header line
        for row in csvreader:
            row = [col.strip() for col in row]
            if any(row):
                if row[0] == "":
                    treatment_readings.append({
                        'timestamp': row[1],
                        'duration': row[2],
                        'rate': row[3],
                        '_id': row[4],
                        'eventType': row[5],
                        'carbs': row[6]  # TODO to be continued
                    })
                else:
                    treatment_readings.append({
                        'insulin': float(row[0]),
                        'timestamp': row[1],
                        'duration': row[2],
                        'rate': row[3],
                        '_id': row[4],
                        'eventType': row[5],
                        'carbs': row[6]  # TODO to be continued
                    })
    return treatment_readings


def pre_process_data(profile: Profile, glucose: list, treatments: list):
    """
    Die eingehenden Daten werden hier mit einander verknüpft.
    Dazu wird für jeden CGM Timestamp die aktuelle Insulin-Wirkmenge berechnet.

    :param profile: Das aktuell verwendete Profil.
    :param glucose: Die CGM Glukosedaten.
    :param treatments: Die Insulinabgaben (Meal Bolus und Temp Basal)
    :return: Gibt die für das Training vorbereiteten verknüpften Daten zurück.
    """
    # TODO start with preprocessing the data...
    result = list()
    for item in glucose:
        result.append({
            "sgv": item["sgv"],
            "acting_insulin": 0.0
        })
    return np.asarray(result)


def generate_plot(items: list, x: str, y: str):
    """
    Generiert einen Pandas DataFrame Plot und gibt diesen aus.

    :param items: Die anzuzeigenden Elemente.
    :param x: Der X-Achsenabschnitt.
    :param y: Der Y-Achsenabschnitt.
    """
    df = pd.DataFrame(
        items,
        columns=[x, y]
    )
    df.plot(x=x, y=y)
    plt.show()


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == "__main__":
    profile = read_profile("../../assets/00390014_profile_2017-07-10_to_2017-11-08.json")

    # start with Humalog - other insulin types later (maybe worth to rethink/reevaluate the approach?)
    if profile.used_insulin == "Humalog" and profile.units == "mg/dl":
        # read glucose values (simple)
        glucose = get_readable_glucose_data("../../assets/00390014_entries_2017-07-10_to_2017-11-08.json.csv")
        # print("Available glucose dataset: ", glucose)

        # read treatment values (simple)
        treatments = get_readable_treatments_data("../../assets/00390014_treatments_2017-07-10_to_2017-11-08_aa.csv")
        # print("Available treatments dataset: ", treatments)

        # start with the data preparation
        preprocessed_training_input = pre_process_data(
            profile=profile,
            glucose=glucose,
            treatments=treatments
        )

        # print(preprocessed_training_input)

        # create dataframes for plots
        # generate_plot(glucose, "timestamp", "sgv")
        # generate_plot(treatments, "timestamp", "insulin")

        # create training dataframe
        #train_df = pd.DataFrame({
        #    'inputs': preprocessed_training_input[:5],
        #    'outputs': preprocessed_training_input[5:10]
        #})

        #print(train_df)

        #dataset = tf.data.Dataset.from_tensor_slices(train_df)
        #for item in dataset:
        #    print(item)

        # load dataset
        dataset = read_csv('../../assets/letsgo.csv', header=0, index_col=0)
        values = dataset.values
        print("Raw values: ", values)

        #groups = [0, 1]
        #i = 1
        # plot each column
        #pyplot.figure()
        #for group in groups:
        #    pyplot.subplot(len(groups), 1, i)
        #    pyplot.plot(values[:, group])
        #    pyplot.title(dataset.columns[group], y=0.5, loc='right')
        #    i += 1
        #pyplot.show()

        # TODO use for one-hot values
        # integer encode direction
        # encoder = LabelEncoder()
        # values[:, 4] = encoder.fit_transform(values[:, 4])

        # ensure all data is float
        values = values.astype('float32')

        # values[0][0] -> sgv
        # values[0][1] -> active_insulin
        # print("Encoded values: ", values)
        print("Printing out the values BEFORE scaling with MinMaxScaler...")
        print("Unscaled \'sgv\': ", values[0][0])
        print("Unscaled \'active_insulin\': ", values[0][1])

        print("Unscaled values array: ", values)

        # normalize features
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        print("Scaled values shape: ", scaled.shape)
        print("Scaled values array: ", scaled)

        print("Printing out the values of the first record...")
        print("Scaled \'sgv\': ", scaled[0][0])
        print("Scaled \'active_insulin\': ", scaled[0][1])

        # specify the number of lag entries (TODO recalc due to the cgm specialties)
        n_entries = 12
        n_features = 2

        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_entries, 1)
        print("Reframed values shape: ", reframed.shape)

        # drop columns we don't want to predict (acting_insulin)
        reframed.drop(reframed.columns[[0, 2]], axis=1, inplace=True)
        print(reframed.head())

        # split into train and test sets
        values = reframed.values
        n_train_hours = 24 * 12 * 30  # 30 days (12 entries per hour - times 24 hours)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        print("Split the datasets into train and test...")
        print("train: size -> ", len(train))
        print("test: size -> ", len(test))

        # split into input and outputs
        n_obs = n_entries * n_features
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]
        print(train_X.shape, len(train_X), train_y.shape)

        # train_X shape ist 'mal 6' so groß wegen der temporalen Abhängigkeit zu den nächsten Werten

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((
            train_X.shape[0],
            n_entries,
            n_features
        ))
        test_X = test_X.reshape((
            test_X.shape[0],
            n_entries,
            n_features
        ))

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        print("train_X.shape[1]: ", train_X.shape[1])
        print("train_X.shape[2]: ", train_X.shape[2])

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        print("important: ", train_X, train_y)

        # fit network
        history = model.fit(
            train_X,
            train_y,
            epochs=50,
            batch_size=72,
            validation_data=(test_X, test_y),
            verbose=2,
            shuffle=False
        )

        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        print("test_X -> ", test_X)

        # make a prediction
        yhat = model.predict(test_X)
        print("yhat shape: ", yhat.shape)

        test_X = test_X.reshape(
            (test_X.shape[0], n_entries * n_features)
        )
        print("test_X shape: ", test_X.shape)

        # invert scaling for forecast
        inv_yhat = concatenate(
            (yhat, test_X[:, -1:]),
            axis=1
        )
        print("inv_yhat shape: ", inv_yhat.shape)

        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, -1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        print("Predicted SGV: ", inv_yhat, " - actual SGV: ", inv_y)

        # try to predict on cool data
        cool = np.asarray([[100.0, 10.0], [105.0, 10.0], [109.0, 10.0]])
        cool_scaled = scaler.transform(cool)
        prediction = model.predict(cool_scaled)
        print(prediction)

        # calculate RMSE
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

    # do glucose forecasting
    # train_data = glucose

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # train_window = 12
    # train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    # train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    # print(train_inout_seq[:5])

    # model = GlucoseForecastModel()
    # loss_function = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # print(model)

    # epochs = 150
    # for i in range(epochs):
    #    for seq, labels in train_inout_seq:
    #        optimizer.zero_grad()
    #        model.hidden_cell = (
    #            torch.zeros(1, 1, model.hidden_layer_size),
    #            torch.zeros(1, 1, model.hidden_layer_size)
    #        )

    #        y_pred = model(seq)

    #        single_loss = loss_function(y_pred, labels)
    #        single_loss.backward()
    #        optimizer.step()

    #    if i % 25 == 1:
    #        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
