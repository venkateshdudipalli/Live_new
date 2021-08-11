

# Create your models here.
from django.db import models

import numpy as np
import pickle
#import pyreadstat
import json
import pandas as pd 



Data = pd.read_csv('test.csv')

lr =  pickle.load(open("Earth_lr.pkl", 'rb'))
knn =  pickle.load(open("Earth_knn.pkl", 'rb'))
dt =  pickle.load(open("Earth_dt.pkl", 'rb'))

def predict(row,algo):
    print(row)
    print(algo)
    test_data = Data.iloc[row]
    print(test_data.shape)
    test_data = np.expand_dims(test_data, axis = 0)
    print(test_data.shape)
    if algo == "lr":
        y_pred = lr.predict(test_data)
        print(y_pred)
        #y_final = scaler.inverse_transform(y_pred).ravel()
        return y_pred[0]
    else:
        y_pred = dt.predict(test_data)
        return y_pred[0]



    
