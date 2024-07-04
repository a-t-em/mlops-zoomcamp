import pandas as pd
from datetime import datetime
import batch
from batch import prepare_data
import pickle

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

ans = [
    (-1, -1, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
]

df_ans = pd.DataFrame(ans, columns=columns)
df_ans['duration'] = [9.0, 8.0]

df_prep = prepare_data(df, columns)
df_prep['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
df_prep['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
df_prep['PULocationID'] = df_prep['PULocationID'].astype(int)
df_prep['DOLocationID'] = df_prep['DOLocationID'].astype(int)

# with open('model.bin', 'rb') as f_in:
#     dv, lr = pickle.load(f_in)

# categorical = ['PULocationID', 'DOLocationID']

# dicts = df[categorical].to_dict(orient='records')
# X_val = dv.transform(dicts)
# y_pred = lr.predict(X_val)

# print('predicted total duration:', sum(y_pred))

assert df_ans.equals(df_prep)

