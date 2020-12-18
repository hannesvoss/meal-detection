import random

from pandas import read_csv, DataFrame

from IOBWizard import IOBWizard

dataset = read_csv('../../assets/00750271_entries.json.csv', parse_dates=True, index_col=0)
treatments = read_csv('../../assets/treatments.csv', parse_dates=True, index_col=0)

# manually specify column names
dataset.columns = ['sgv']
dataset.index.name = 'timestamp'

# handle the treatments
treatments.columns = ['insulin', 'created_at', 'duration', 'rate', '_id', 'eventType', 'carbs', 'temp', 'enteredBy',
                      'absolute', 'absorptionTime', 'unabsorbed', 'type', 'programmed', 'foodType']
treatments.index.name = 'timestamp'

# prepare treatments
treatments['insulin'].fillna(0, inplace=True)
treatments['absolute'].fillna(0, inplace=True)

# mark all NA values with 0
dataset['sgv'].fillna(0, inplace=True)

# TODO build up the new resulting dataframe
iob_calculator = IOBWizard(
    temp_basal=list(),
    bolus=list(),
    insulin="Humalog"
)

'''for timestamp, item in dataset.iterrows():
    df = DataFrame(
        {
            'sgv': dataset['sgv'],
            'acting_insulin': iob_calculator.get_iob(timestamp)
        },
        columns=['sgv', 'acting_insulin']
    )'''

# FIXME precisely calculate the acting insulin for the CGM entries
# for 'Meal Bolus' the value is in 'insulin'
# for 'Temp Basal' the value is in 'absolute'
# FIXME try to put 'eventType' as one-hot-encoded value with 'absolute' & 'insulin' values
dataset.insert(1, 'acting_insulin', 5.0)
dataset['acting_insulin'] = dataset['acting_insulin'].apply(lambda x: float(random.randrange(0, 15)))

#for item in dataset:
#    item.update(DataFrame({
#        'acting_insulin': float(random.randrange(0, 15))
#    }))

# convert all values to floating point
dataset = dataset.astype(float)

# drop the first 24 hours (24 * 12)
dataset = dataset[(24 * 12):]

# summarize first 5 rows
print(dataset.head(5))

# save to file
dataset.to_csv('../../assets/doagain.csv')
