# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:35:14 2019

@author: Zubair Irshad
"""

# Load data
datapath="E:\Education - Grad\Machine Learning\Supervised learning project\wdbc.data"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.ensemble import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

import itertools
import timeit
import time




datapath=os.path.join('E:\Education - Grad\Machine Learning\Supervised learning project','bank-additional-full.csv')
def load_data(file_path):
    return pd.read_csv(file_path, sep=';', header = None)

data = load_data(datapath)


new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header #set the header row as the df header


# One hot encoding data and data engineering

col_1hot = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

data_1hot = data[col_1hot]

data_1hot = pd.get_dummies(data_1hot).astype('category')

data_other_cols = data.drop(col_1hot,axis=1)

data = pd.concat([data_other_cols,data_1hot],axis=1)

column_order = list(data)

column_order.insert(0, column_order.pop(column_order.index('y')))

data = data.loc[:, column_order]

data['y'].replace("no",0,inplace=True)
data['y'].replace("yes",1,inplace=True)

data['y'] = data['y'].astype('category')

numer_ccols = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
data_num = data[numer_ccols]


data_categorical = data.drop(numer_ccols,axis=1)
#
data = pd.concat([data_categorical,data_num],axis=1)


##Assign inputs and outputs
o=data.shape[0]
p= data.shape[1]


data = data.values

X= data[:,1:p]
y = data[:,0]

X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)


mm_scalar = MinMaxScaler()

X_train[:,-10:]= mm_scalar.fit_transform(X_train[:,-10:])
X_val[:,-10:] = mm_scalar.transform(X_val[:,-10:])
X_test[:,-10:] = mm_scalar.transform(X_test[:,-10:])


def hyperKNN(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    klist = np.linspace(1,250,25).astype('int')
    for i in klist:
        clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
        
    plt.plot(klist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
def final_classifier_results(clf,X_train, X_test, y_train, y_test):
    
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["1","-1"], title='Confusion Matrix')
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



hyperKNN(X_train, y_train, X_val, y_val,title="Model Complexity Curve for kNN (Phishing Data)\nHyperparameter : No. Neighbors")


#Check unique number of M and B in output
unique_elements, counts_elements = np.unique(y_test, return_counts=True)

near_neighbour = np.arange(1, 31, 2)

train_results = []
val_results = []

score_train =[]
score_val=[]


for n_neighbors in near_neighbour:
   dt = KNeighborsClassifier(n_neighbors=n_neighbors)
   dt.fit(X_train, y_train)
   
   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score= dt.score(X_train, y_train)
   
   score_train.append(score)
   train_results.append(roc_auc)
   
   val_pred = dt.predict(X_val)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, val_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score= dt.score(X_val, y_val)
   
   score_val.append(score)
   
   val_results.append(roc_auc)
   
line1 = plt.plot(near_neighbour, train_results, 'b', label="Train AUC")
line2 = plt.plot(near_neighbour, val_results, 'r', label="Validation AUC")
   
plt.ylabel('AUC score_n_neighbours')
plt.xlabel('near_neighbour')
plt.show()

line1 = plt.plot(near_neighbour, score_train, 'b', label="Train score")
line2 = plt.plot(near_neighbour,score_val, 'r', label="Validation score")
   
plt.ylabel('Score_n_neighbours')
plt.xlabel('near_neighbour')
plt.show()

#Plot Learning curve

estimator = KNeighborsClassifier()

title = "Learning Curves (K_Neighbor)"


plt.show()

# Random search

params = {"n_neighbors": np.arange(1, 31, 2),"metric": ["euclidean", "cityblock"]}
print("[INFO] tuning hyperparameters via random search")
model = KNeighborsClassifier()



rand = RandomizedSearchCV(model, params,random_state=100, cv=5)
start = time.time()
rand.fit(X_train, y_train)

acc = rand.score(X_val, y_val)

print ("Randomized search took:", time.time() - start)
print ("Randomized search accuracy:", acc*100)
print ("Randomized search best parameters:", rand.best_params_)

n_neighbour_best = rand.best_params_['n_neighbors']
metric_best = rand.best_params_['metric']

# Learning curve and final classfication results

estimator_final = KNeighborsClassifier(n_neighbors = n_neighbour_best, metric=metric_best)

plot_learning_curve(estimator_final, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=4)

final_classifier_results(estimator_final, X_train, X_test, y_train, y_test)