# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:24:56 2019

@author: Zubair Irshad
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve,auc,f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import time

import itertools

import timeit
#from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV


def hyperKNN(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    klist = np.linspace(1,100,25).astype('int')
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
    
    print f1_test[0:10]
    print klist[0:10]
    
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
    plot_confusion_matrix(cm, classes=["Yes","No"], title='Confusion Matrix - Breast Cancer')
    plt.show()

#LOAD DATA
datapath="E:\Education - Grad\Machine Learning\Supervised learning project\wdbc.data"
def load_data(file_path):
    return pd.read_csv(file_path, header = None)

data = load_data(datapath)
data.columns = ['ID','Diagnosis','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ab','ac','ad','ae']


ab= data[data['Diagnosis'] == 'M'].shape[0]
bc =  data[data['Diagnosis'] == 'B'].shape[0]

print ab
print bc

print ab*1.0/(ab+bc)

print bc*1.0/(ab+bc)
 


s = data['Diagnosis']
mymap = {'M':1, 'B':-1}
list_numbers= [mymap[item] for item in s]
list_array = np.array(list_numbers)

data['diag_num'] = list_array

del data['Diagnosis']

corr_matrix = data.corr()

k = corr_matrix["diag_num"].sort_values(ascending=False)





#PLOT HISTOGRAM OF DAA+TA

#data.hist(bins=50, figsize=(20,15))
#plt.show()


#Assign inputs and outputs
o=data.shape[0]
p= data.shape[1]

data = data.values
X= data[:,1:p-1]
y = data[:,p-1]

X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)

hyperKNN(X_train, y_train, X_val, y_val,title="Model Complexity Curve for kNN (Breast Cancer)\nHyperparameter : No. Neighbors")



from sklearn.preprocessing import MinMaxScaler
mm_scalar = MinMaxScaler()

print X_train

X_train= mm_scalar.fit_transform(X_train)
X_val = mm_scalar.transform(X_val)
X_test = mm_scalar.transform(X_test)

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

estimator_final = KNeighborsClassifier(n_neighbors = n_neighbour_best, metric=metric_best)

plot_learning_curve(estimator_final, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=4)

final_classifier_results(estimator_final, X_train, X_test, y_train, y_test)






