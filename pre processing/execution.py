import preprocess
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import os 
import numpy as np 
os.chdir(r'D:\NITT\interns\iiit research summer intern\datasets\bp\blood pressure dl\csv dataset')
data = pd.read_csv('green_ppg_values_newaug.csv')
data = data.drop('index', axis=1)
data = np.array(data)
scaler = MinMaxScaler()
store_address = r'D:\NITT\interns\iiit research summer intern\datasets\bp\blood pressure dl\csv dataset'
# print(preprocess.process_signal(data[10]))
print(preprocess.prepare_dataset(data, store_address))
