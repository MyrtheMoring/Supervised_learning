# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:08:10 2019

@author: Zubair Irshad
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import itertools


from sklearn.model_selection import learning_curve

import timeit


#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import time

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

def hyperNN(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    hlist = np.linspace(1,150,30).astype('int')
    for i in hlist:         
            clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                                learning_rate_init=0.05, random_state=100)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
      
    plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(hlist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Hidden Units')
    
    print f1_test[17:23]
    print hlist[17:23]
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def hyperNN_num_layers(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    hlist = np.arange(1,20,1).astype('int')
    for i in hlist:         
            clf = MLPClassifier(hidden_layer_sizes=(110,i), solver='adam', activation='relu', 
                                learning_rate_init=0.001, random_state=100)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
            
    print f1_test[1:10]
    print hlist[1:10]
            
      
    plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(hlist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Hidden Layers')
    
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
    
    auc_sc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    
    cm = cm.astype('float')
    
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]
    
    
    # percentual confusion matrix values
    total = X_test.shape[0]
    
    GB= float((TP+TN)/total)
    BA = float((FN+FP)/total)
    print("Good Accepted + Bad Rejected: %0.2f "% (GB))
    print("Bad Accepted + Good Rejected: %0.2f "% (BA))

    # Associated cost - from cost matrix
    cost = (FP*1) + (FN*1)
    print "Associated cost: %0.2f" % cost

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc_sc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    cm = cm.astype('int')
    plot_confusion_matrix(cm, classes=["Yes","No"], title='Confusion Matrix')
    plt.show()


#LOAD DATA
datapath="E:\Education - Grad\Machine Learning\Supervised learning project\wdbc.data"
def load_data(file_path):
    return pd.read_csv(file_path, header = None)

data = load_data(datapath)
data.columns = ['ID','Diagnosis','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ab','ac','ad','ae']

#GET OUTPUT IN NUMBER FORM M=1 and B=-1

s = data['Diagnosis']
mymap = {'M':1, 'B':-1}
list_numbers= [mymap[item] for item in s]
list_array = np.array(list_numbers)

data['diag_num'] = list_array

del data['Diagnosis']

corr_matrix = data.corr()

k = corr_matrix["diag_num"].sort_values(ascending=False)

#data.hist(bins=50, figsize=(20,15))
#plt.show()

#PLOT HISTOGRAM OF DAA+TA

#data.hist(bins=50, figsize=(20,15))
#plt.show()

#Assign inputs and outputs
o=data.shape[0]
p= data.shape[1]


data = data.values
X= data[:,1:p-1]

y = data[:,p-1]

#y = y
#
#print ("shape of y:",y.shape)





X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)


#Check unique number of M and B in output
unique_elements, counts_elements = np.unique(y_test, return_counts=True)

#Feature_Scaling

from sklearn.preprocessing import MinMaxScaler
mm_scalar = MinMaxScaler()

X_train= mm_scalar.fit_transform(X_train)
X_val = mm_scalar.transform(X_val)
X_test = mm_scalar.transform(X_test)


#
#scaler.fit(X_train)
#
#StandardScaler(copy=True, with_mean=True, with_std=True)
#
#
#X_train = scaler.transform(X_train)
#X_val = scaler.transform(X_val)
#X_test = scaler.transform(X_test)

#Implement Neural Networks

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
#print score

# Random search

def NNRandomSearchCV(X_train,y_train):
    
    params = {'learning_rate': ["constant","invscaling","adaptive"],'activation': ["relu", "tanh","logistic"],'alpha':np.arange(0.0001,0.01,10),'hidden_layer_sizes':np.arange(10,150,20)}
    print("[INFO] tuning hyperparameters via random search")
    model = MLPClassifier()
    
    
    rand = RandomizedSearchCV(model, param_distributions=params,random_state=100, cv=5)
    start = time.time()
    rand.fit(X_train, y_train)
    
    acc = rand.score(X_val, y_val)
    
    print ("Randomized search took:", time.time() - start)
    print ("Randomized search accuracy:", acc*100)
    print ("Randomized search best parameters:", rand.best_params_)
    return rand.best_params_['learning_rate'], rand.best_params_['activation'],rand.best_params_['alpha'],rand.best_params_['hidden_layer_sizes']


hyperNN(X_train, y_train, X_val, y_val,title="Model Complexity Curve for Neural Networks (Breast Cancer Data)\nHyperparameter : No. of Hidden units")

hyperNN_num_layers(X_train, y_train, X_val, y_val,title="Model Complexity Curve for Neural Networks (Breast Cancer Data)\nHyperparameter : No. of Hidden layers")

opt_LR, opt_act, opt_alpha, opt_hls= NNRandomSearchCV(X_train,y_train)

estimator_final = MLPClassifier(hidden_layer_sizes=(110,10), solver='adam', activation=opt_act, alpha=opt_alpha,learning_rate = opt_LR, random_state=100)

title = "Learning Curves (Neural Networks - Breast Cancer Data)"
plot_learning_curve(estimator_final, title, X, y, ylim=(0.3, 1.01), cv=5, n_jobs=4)

final_classifier_results(estimator_final, X_train, X_test, y_train, y_test)


