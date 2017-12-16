#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:24:46 2017

@author: See
"""

import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

df = quandl.get("WIKI/GOOGL")
print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume']]
print(df.head())

#Percentage between High and Low - to understand vol
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#forecasting Adjusted Close price
forecast_col = 'Adj. Close'
##Machine Learning cannot handle N/A
df.fillna(-99999, inplace=True)

#math.ceil rounds up decimal numbers, and returns a float, hence needed a int
#try to predict price in 10 days
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

#X denotes Features
#y denotes Labels

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

#scaling (skip for high frequency)
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#X = X[:-forecast_out+1]
df.dropna(inplace=True)
y=np.array(df['label'])

print(len(X), len(y))

#use 20% of the data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#classifier

clf = LinearRegression(n_jobs=-1)
#can switch between algorithms
##support vector machine
#clf = svm.SVR()
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#train data needs to be seperated from test data otherwise, 
#the answer would be known already

#predicted 30 days stock value
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast']=np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 60*60*24
next_unix = last_unix + one_day 

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Adj. Price')
plt.show()