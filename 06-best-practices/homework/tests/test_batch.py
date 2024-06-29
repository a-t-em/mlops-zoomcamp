import pandas as pd
from datetime import datetime
from batch import prepare_data


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
    (-1, -1, dt(1, 1), dt(1, 10), 9),
    (1, 1, dt(1, 2), dt(1, 10), 8),
]

df_ans = pd.DataFrame(ans, columns=columns.append('duration'))
df_prep = prepare_data(df, columns)

assert df_ans == df_prep

