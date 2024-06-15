#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


# In[ ]:


import pickle
import pandas as pd
import numpy as np

# In[ ]:

categorical = ['PULocationID', 'DOLocationID']

# In[ ]:
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# In[ ]:
def apply_model(df, output_file):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(np.mean(y_pred))

    df_result = df[['ride_id']]
    df_result['predictions'] = y_pred
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )
# In[ ]:

import sys 
def run():
    taxi_color = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])

    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_color}_tripdata_{year:04d}-{month:02d}.parquet'
    output_filename = f'{taxi_color}-taxi-outputs'
    df = read_data(filename)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    apply_model(df, output_filename)

if __name__ == '__main__':
    run()