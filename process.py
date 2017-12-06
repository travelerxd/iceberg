import numpy as np
import pandas as pd

from tqdm import tqdm
import cv2


def one_hot(label,len):
    Y=np.zeros([len,2])
    for i in range(0,len):
       Y[i][label[i]]=1

    return Y




def read_data(file):
    df = pd.read_json(file)
    # df = df[:100]
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    # print(df['inc_angle'].value_counts())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2, 0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands


def process(df, bands):
    w, h = 224, 224

    bands = 0.5 + bands / 100.
    X = []

    for i in tqdm(bands, miniters=100):
        x = cv2.resize(i, (w, h)).astype(np.float32)
        X.append(x)

    X = np.array(X)



    return X,df['is_iceberg'].values



