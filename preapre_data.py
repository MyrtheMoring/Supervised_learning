# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:48:02 2019

@author: Zubair Irshad
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

import itertools
import timeit
import time

# Load data
datapath=os.path.join('E:\Education - Grad\Machine Learning\Supervised learning project','german_credit.csv')
def load_data(file_path):
    return pd.read_csv(file_path, sep=' ', header = None)

data = load_data(datapath)

print data.shape


new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header #set the header row as the df header


# One hot encoding data and data engineering

col_1hot = ['checking_status','credit_history','purpose','savings_status','employment','personal_status','other_parties','property_magnitude','other_payment_plans','housing','job','own_telephone','foreign_worker']

data_1hot = data[col_1hot]

data_1hot = pd.get_dummies(data_1hot).astype('category')

data_other_cols = data.drop(col_1hot,axis=1)

data = pd.concat([data_other_cols,data_1hot],axis=1)

column_order = list(data)

column_order.insert(0, column_order.pop(column_order.index('class')))

data = data.loc[:, column_order]

print data.shape

data['class'].replace('2',0,inplace=True)

print data['class']
#data['y'].replace("yes",1,inplace=True)
#


#data['y'] = data['y'].astype('category')

numer_ccols = ['duration','credit_amount','installment_commitment','residence_since','age','existing_credits','num_dependents']
data_num = data[numer_ccols]


data_categorical = data.drop(numer_ccols,axis=1)
#
data = pd.concat([data_categorical,data_num],axis=1)


##Assign inputs and outputs
o = data.shape[0]
p = data.shape[1]


#print list(data) 
#
#print data.values[:,-7:]
#print data.values[:,1:55]


X = np.array(data.values[:,1:p], dtype='float')
y = np.array(data.values[:,0],dtype='int64')

X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)


mm_scalar = MinMaxScaler()

X_train[:,-7:]= mm_scalar.fit_transform(X_train[:,-7:])
X_val[:,-7:] = mm_scalar.transform(X_val[:,-7:])
X_test[:,-7:] = mm_scalar.transform(X_test[:,-7:])

print X_train.shape
print X_val.shape
print X_test.shape




