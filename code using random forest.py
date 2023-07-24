# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:12:11 2021

@author: Swaroop Honrao
"""

#Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import dataset
data = pd.read_csv('skoda.csv')
x = data.iloc[:, [0,1,3,4,5,6,7,8]].values
y = data.iloc[:, 2].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(x[:, [1,3,5,6,7]])
x[:, [1,3,5,6,7]] = imputer.transform(x[:, [1,3,5,6,7]])

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0,2,4])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#Spliting dataset into training set and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 20:] = sc.fit_transform(x_train[:, 20:])
x_test[:, 20:] = sc.transform(x_test[:, 20:])

#Linear regression method
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
regressor = RandomForestRegressor(n_estimators=150)
regressor.fit(x_train,y_train)

#Predicting test results
y_pred = regressor.predict(x_test)

#Rsquared error and MSE
m12 = metrics.r2_score(y_test, y_pred)*100
print('R squared error :', m12)
m22 = metrics.mean_squared_error(y_test, y_pred)
print('MSE :', m22)