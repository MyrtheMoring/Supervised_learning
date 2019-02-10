# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:18:23 2019

@author: Zubair Irshad
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc,f1_score

from sklearn.model_selection import learning_curve
import time

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import RandomizedSearchCV

def hyper_boosting(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title):
    
    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1,250,40).astype('int')
    for i in n_estimators:         
            clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth/2), 
                                             min_samples_leaf=int(min_samples_leaf/2), random_state=100,)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
      
    plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(n_estimators, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

                
#LOAD DATA
datapath="E:\Education - Grad\Machine Learning\Supervised learning project\wdbc.data"
def load_data(file_path):
    return pd.read_csv(file_path, header = None)

data = load_data(datapath)
data.columns = ['ID','Diagnosis','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ab','ac','ad','ae']

s = data['Diagnosis']
mymap = {'M':1, 'B':-1}
list_numbers= [mymap[item] for item in s]
list_array = np.array(list_numbers)

data['diag_num'] = list_array

del data['Diagnosis']

corr_matrix = data.corr()

k = corr_matrix["diag_num"].sort_values(ascending=False)


#Assign inputs and outputs
o = data.shape[0]
p = data.shape[1]

data = data.values
X= data[:,1:p-1]

#y=y

y = data[:,p-1]


X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)

