# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:15:13 2019

@author: Zubair Irshad
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve,auc,f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit

from sklearn.model_selection import learning_curve
import time

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV


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

def hyperSVM(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    kernel_func = ['linear','poly','rbf','sigmoid']
    for i in kernel_func:         
            if i == 'poly':
                for j in [2,3]:
                    clf = SVC(kernel=i, degree=j, gamma ='auto', random_state=100)
                    clf.fit(X_train, y_train)
                    y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)
                    f1_test.append(f1_score(y_test, y_pred_test))
                    f1_train.append(f1_score(y_train, y_pred_train))
            else:    
                clf = SVC(kernel=i, gamma ='auto', random_state=100)
                clf.fit(X_train, y_train)
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                f1_test.append(f1_score(y_test, y_pred_test))
                f1_train.append(f1_score(y_train, y_pred_train))
                
    xvals = ['linear','poly2','poly3','rbf','sigmoid']
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')
    
    print f1_test
    print xvals
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    
def SVMRandomSearchCV(X_train,y_train):
    
    params = {'kernel': ["linear","poly","rbf","sigmoid"],'C': [1e-4, 1e-3, 1e-2, 1e01, 1],'gamma':["auto","scale"]}
    print("[INFO] tuning hyperparameters via random search")
    model = SVC()
    
    
    rand = RandomizedSearchCV(model, param_distributions=params,random_state=100, cv=5)
    start = time.time()
    rand.fit(X_train, y_train)
    
    acc = rand.score(X_val, y_val)
    
    print ("Randomized search took:", time.time() - start)
    print ("Randomized search accuracy:", acc*100)
    print ("Randomized search best parameters:", rand.best_params_)
    return rand.best_params_['kernel'], rand.best_params_['C'],rand.best_params_['gamma']

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

hyperSVM(X_train, y_train, X_val, y_val,title="Model Complexity Curve for SVM (Breast Cancer Data)\nHyperparameter : Kernel function")

opt_kernel, opt_C, opt_gamma= SVMRandomSearchCV(X_train,y_train)

estimator_final = SVC(kernel=opt_kernel, gamma = opt_gamma,C= opt_C, random_state=100)

title = "Learning Curves (SVM)"
plot_learning_curve(estimator_final, title, X, y, ylim=(0.3, 1.01), cv=5, n_jobs=4)

final_classifier_results(estimator_final, X_train, X_test, y_train, y_test)

