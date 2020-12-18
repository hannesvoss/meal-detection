import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from own.GlucosePredictionModel import GlucosePredictionModel
from own.Utils import Utils

# Load the data via Utils
dataset, training_set = Utils.load_dataset("F:/PycharmProjects/meal-detection/assets/letsgo.csv")

# Feature scaling
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

print("Predicting scaling: ", training_set[:, 0:1])

sc_predict = MinMaxScaler()
scaled_values = sc_predict.fit_transform(training_set[:, 0:1])

print("Scaled values: ", scaled_values)

n_future = 6  # number of entries we want to predict into the future
n_past = 12  # number of past entries we want to use to predict the future

# Shape the arrays
X_train, y_train = Utils.prepare_arrays(
    n_past=n_past,
    n_future=n_future,
    dataset_train=dataset.dataset,
    training_set_scaled=training_set_scaled
)

# Build & train the model
model = GlucosePredictionModel(
    n_past=n_past,
    dataset_train=dataset.dataset
)
print("Los gehts: ", X_train, y_train)
history = model.train(X_train, y_train)

# ---------------------------------

# Make predictions
# print("datelist_train[0] -> ", dataset.datelist_train[0])
datelist_future = pd.date_range(dataset.datelist_train[0], periods=n_future, freq="5T").tolist()  # freq T for minutes
print("datelist_future -> ", datelist_future)
datelist_future_ = []
for current in datelist_future:
    datelist_future_.append(current.date())

predictions_future = model.predict(X_train[-n_future:])
predictions_train = model.predict(X_train[n_past:])

print("pred_future scale: ", predictions_future)

# Inverse the predictions to original measurements
y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(
    y_pred_future,
    columns=['sgv']
).set_index(
    pd.Series(datelist_future)
)

PREDICTION_TRAIN = pd.DataFrame(
    y_pred_train,
    columns=['sgv']
).set_index(
    pd.Series(dataset.datelist_train[2 * n_past + n_future - 1:])
)

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series()  # .apply(Utils.datetime_to_timestamp)
PREDICTION_TRAIN.head(3)

print(PREDICTIONS_FUTURE)
print(PREDICTION_TRAIN)

# Parse training set timestamp for better visualization
dataset_train = pd.DataFrame(dataset.dataset, columns=dataset.cols)
dataset_train.index = dataset.datelist_train
dataset_train.index = pd.to_datetime(dataset.dataset.index)

# Set plot size
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2017-11-06'  # 2017-11-06

# Plot of predicted time series
plt.plot(
    PREDICTIONS_FUTURE.index,
    PREDICTIONS_FUTURE['sgv'],
    color='r',
    label='Predicted SGV'
)

# actual_range = dataset_train.iloc[::-1]
actual_range = dataset_train.iloc[::-1].loc[START_DATE_FOR_PLOTTING:]
print("Actual SGV timestamps: ", actual_range.index)
plt.plot(
    actual_range.index,
    actual_range['sgv'],
    color='b',
    label='Actual SGV'
)

#prediction_range = PREDICTION_TRAIN.iloc[::-1].loc[START_DATE_FOR_PLOTTING:]
#print("Predicted timestamps: ", prediction_range.index)
#plt.plot(
#    prediction_range.index,
#    prediction_range['sgv'],
#    color='orange',
#    label='Training predictions'
#)

plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions and Actual SGV', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('SGV', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
