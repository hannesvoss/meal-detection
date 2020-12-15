#!/usr/bin/env python

"""HAUPTPROJEKT Codebasis für ML Verfahren

In diesem Projekt werden Active Learning Strategien für das Labeln von
unangekündigten Mahlzeiten entwickelt.
Das trainierte Modell soll in ein bereits bestehendes Android Projekt
eingebunden werden.
"""

import csv
import datetime
import math
import re

import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser
import torch
import torch.nn as nn
from numpy.random.mtrand import shuffle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from GlucoseForecastModel import GlucoseForecastModel
from UnannouncedMealClassifier import UnannouncedMealClassifier

__author__ = "Hannes Voß"
__version__ = "0.1"

epochs = 10  # number of epochs per training session
select_per_epoch = 200  # number to select per epoch per label
feature_index = {}  # feature mapping for one-hot encoding

print("Is CUDA available: ", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


class DataLoader(Dataset):
    def __init__(self, src, tgt):
        self.input_src = src
        self.input_tgt = tgt

    def __len__(self):
        return len(self.input_src)


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


def make_feature_vector(features, feature_index):
    vec = torch.zeros(len(feature_index))
    for feature in features:
        if feature in feature_index:
            vec[feature_index[feature]] += 1
    return vec.view(1, -1)


def train_model(training_data, validation_data="", evaluation_data="", num_labels=2, vocab_size=0):
    """Train model on the given training_data
        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """

    model = UnannouncedMealClassifier(num_labels, vocab_size)
    # let's hard-code our labels for this example code
    # and map to the same meaningful booleans in our data,
    # so we don't mix anything up when inspecting our data
    label_to_ix = {"not_disaster_related": 0, "disaster_related": 1}

    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label

        shuffle(training_data)  # randomize the order of the training data
        related = [row for row in training_data if '1' in row[2]]
        not_related = [row for row in training_data if '0' in row[2]]

        epoch_data = related[:select_per_epoch]
        epoch_data += not_related[:select_per_epoch]
        shuffle(epoch_data)

        # train our model
        for item in epoch_data:
            features = item[1].split()
            label = int(item[2])

            model.zero_grad()

            feature_vec = make_feature_vector(features, feature_index)
            target = torch.LongTensor([int(label)])

            log_probs = model(feature_vec)

            # compute loss function, do backward pass, and update the gradient
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

    fscore, auc = evaluate_model(model, evaluation_data)
    fscore = round(fscore, 3)
    auc = round(auc, 3)

    # save model to path that is alphanumeric and includes number of items and accuracies in filename
    timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":",
                                                                                                                 "")
    training_size = "_" + str(len(training_data))
    accuracies = str(fscore) + "_" + str(auc)

    model_path = "models/" + timestamp + accuracies + training_size + ".params"

    torch.save(model.state_dict(), model_path)
    return model_path


def evaluate_model(model, evaluation_data):
    """Evaluate the model on the held-out evaluation data
    Return the f-value for disaster-related and the AUC
    """

    related_confs = []  # related items and their confidence of being related
    not_related_confs = []  # not related items and their confidence of being _related_

    true_pos = 0.0  # true positives, etc
    false_pos = 0.0
    false_neg = 0.0

    with torch.no_grad():
        for item in evaluation_data:
            _, text, label, _, _, = item

            feature_vector = make_feature_vector(text.split(), feature_index)
            log_probs = model(feature_vector)

            # get confidence that item is disaster-related
            prob_related = math.exp(log_probs.data.tolist()[0][1])

            if label == "1":
                # true label is disaster related
                related_confs.append(prob_related)
                if prob_related > 0.5:
                    true_pos += 1.0
                else:
                    false_neg += 1.0
            else:
                # not disaster-related
                not_related_confs.append(prob_related)
                if prob_related > 0.5:
                    false_pos += 1.0

    # Get FScore
    if true_pos == 0.0:
        fscore = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = (2 * precision * recall) / (precision + recall)

    # GET AUC
    not_related_confs.sort()
    total_greater = 0  # count of how many total have higher confidence
    for conf in related_confs:
        for conf2 in not_related_confs:
            if conf < conf2:
                break
            else:
                total_greater += 1

    denom = len(not_related_confs) * len(related_confs)
    auc = total_greater / denom

    return [fscore, auc]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


if __name__ == "__main__":
    glucose = get_readable_glucose_data("assets/00390014_entries_2017-07-10_to_2017-11-08.json.csv")
    print("Available glucose dataset: ", glucose)
    treatments = get_readable_treatments_data("assets/00390014_treatments_2017-07-10_to_2017-11-08_aa.csv")
    print("Available treatments dataset: ", treatments)

    df = pd.DataFrame(glucose, columns=['timestamp', 'sgv'])
    df.plot(x='timestamp', y='sgv')
    plt.show()

    df = pd.DataFrame(treatments, columns=['timestamp', 'insulin'])
    df.plot(x='timestamp', y='insulin')
    plt.show()

    # do glucose forecasting
    train_data = glucose

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    train_window = 12
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    print(train_inout_seq[:5])

    model = GlucoseForecastModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)

    epochs = 150
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size)
            )

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
