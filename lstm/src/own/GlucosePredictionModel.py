from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
import numpy as np


class GlucosePredictionModel:
    """
    This class holds the keras model for glucose prediction (prediction of sgv)
    """

    def __init__(self, n_past, dataset_train):
        """
        :param n_past: Number of past entries we want to use to predict the future
        :param dataset_train: Dataset for training
        """

        print("Going to train with: ", dataset_train)

        # Build the model
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=(n_past, dataset_train.shape[1] - 1), return_sequences=True))  # 64 // (n_past, dataset_train.shape[1] - 1)
        self.model.add(LSTM(units=10, return_sequences=False))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=1, activation="linear"))  # activation="linear"
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_squared_error")

        #self.es = EarlyStopping(monitor="val_loss", min_delta=1e-10, patience=10, verbose=1)
        #self.rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        self.mcp = ModelCheckpoint(
            filepath="weights.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        self.tb = TensorBoard("logs")

    def train(self, x_train: np.array, y_train: np.array):
        """
        Call to train the model (fit the model)
        :return:
        """
        print("important: ", y_train)

        # Start the training
        return self.model.fit(
            x_train, y_train,
            shuffle=False,  # really?
            epochs=30,
            callbacks=[],  # self.es, self.rlr, self.mcp, self.tb
            validation_split=0.2,
            verbose=1,
            batch_size=64  # 256
        )

    def predict(self, values: list):
        """
        Call to predict values from the keras model
        :return: Returns the prediction of the keras model
        """
        # Start predicting
        return self.model.predict(
            values
        )
