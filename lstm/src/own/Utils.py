from datetime import datetime
import pandas as pd

import numpy as np

from own.Dataset import Dataset


class Utils:
    @staticmethod
    def prepare_arrays(n_past, n_future, dataset_train, training_set_scaled):
        x_train = []
        y_train = []
        # Shape the arrays
        for i in range(n_past, len(training_set_scaled) - n_future + 1):
            x_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
            y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        print("X_train shape == {}".format(x_train.shape))
        print("y_train shape == {}".format(y_train.shape))
        return x_train, y_train

    @staticmethod
    def datetime_to_timestamp(x):
        """
        x : a given datetime value (datetime.date)
        """
        return x  # datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

    @staticmethod
    def load_dataset(url: str):
        dataset_train = pd.read_csv(url, header=0, index_col=0)

        # Select features (columns) to be involved into training and prediction
        cols = list(dataset_train)[0:1]

        print("Features (for prediction & training): ", cols)

        # Extract dates (will be used in visualization)
        datelist_train = list(dataset_train['sgv'].index)
        print("Dates (for visualization): ", datelist_train)

        ds = Dataset(
            dataset=dataset_train,
            cols=cols,
            datelist_train=datelist_train
        )

        # Using multiple predictors (features)
        training_set = dataset_train.to_numpy()
        print("Shape of training set == {}".format(training_set.shape))

        return ds, training_set
