# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:41:17 2019

@author: Zubair Irshad
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve,auc,f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


import itertools

from sklearn.model_selection import learning_curve

from sklearn.ensemble import GradientBoostingClassifier

import timeit

import time

#LOAD DATA
datapath="E:\Education - Grad\Machine Learning\Supervised learning project\wdbc.data"
def load_data(file_path):
    return pd.read_csv(file_path, header = None)

data = load_data(datapath)
data.columns = ['ID','Diagnosis','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ab','ac','ad','ae']


#df_1hot = data['Diagnosis']
#df_1hot = pd.get_dummies(df_1hot)
#
#print df_1hot






#GET OUTPUT IN NUMBER FORM M=1 and B=-1

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

print o, p

data = data.values
X= data[:,1:p-1]

y = data[:,p-1]



X_inter, X_test, y_inter, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_inter, y_inter, test_size=0.25, random_state=42)


#Check unique number of M and B in output
unique_elements, counts_elements = np.unique(y_test, return_counts=True)

#Implement decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()    
    
def hyperTree(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    max_depth = list(range(1,31))
    for i in max_depth:         
            clf = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
      
    plt.plot(max_depth, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(max_depth, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Max Tree Depth')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

hyperTree(X_train, y_train, X_test, y_test,title="Score vs Maximum paraemter depth curve (Breast Cancer)\nHyperparameter : Maximum Tree Depth")

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

# Tune max_depth 
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []
val_results = []

score_train =[]
score_val=[]
max_depth1 =[]

for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
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
   max_depth1.append(max_depth)
   
#print np.transpose(np.array(score_val))
#
#print max_depth1

#print np.transpose(score_val.shape)
   
#print np.transpose(score_train)
#
#print np.transpose(max_depth1)
#
#print np.transpose(score_val)
   
line1 = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2 = plt.plot(max_depths, val_results, 'r', label="Validation AUC")
   
plt.ylabel('AUC score_max_depth')
plt.xlabel('Tree depth')
plt.show()

line1 = plt.plot(max_depths, score_train, 'b', label="Train score")
line2 = plt.plot(max_depths,score_val, 'r', label="Validation score")
   
plt.ylabel('Score_max_depth')
plt.xlabel('Tree depth')
plt.show()

#Tune min_samples_split

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
val_results = []

score_train =[]
score_val=[]

for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(X_train, y_train)
   
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score = dt.score(X_train, y_train)
   
   score_train.append(score)
   train_results.append(roc_auc)
   
   val_pred = dt.predict(X_val)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, val_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score= dt.score(X_val, y_val)
   
   score_val.append(score)
   
   val_results.append(roc_auc)
   
line1 = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2 = plt.plot(min_samples_splits, val_results, 'r', label="Validation AUC")
   
plt.ylabel('AUC score_min_samples_split')
plt.xlabel('Tree depth')
plt.show()

line1 = plt.plot(min_samples_splits, score_train, 'b', label="Train score")
line2 = plt.plot(min_samples_splits,score_val, 'r', label="Validation score")
   
plt.ylabel('Score_min_samples_split')
plt.xlabel('Tree depth')
plt.show()

# Tune min_samples_leaf

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []
val_results = []

score_train =[]
score_val=[]

for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   dt.fit(X_train, y_train)
   
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score = dt.score(X_train, y_train)
   
   score_train.append(score)
   train_results.append(roc_auc)
   
   val_pred = dt.predict(X_val)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, val_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score= dt.score(X_val, y_val)
   
   score_val.append(score)
   
   val_results.append(roc_auc)
   
line1 = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2 = plt.plot(min_samples_leafs, val_results, 'r', label="Validation AUC")
   
plt.ylabel('AUC score_min_samples_leaf')
plt.xlabel('Tree depth')
plt.show()

line1 = plt.plot(min_samples_leafs, score_train, 'b', label="Train score")
line2 = plt.plot(min_samples_leafs,score_val, 'r', label="Validation score")
   
plt.ylabel('Score_min_samples_leaf')
plt.xlabel('Tree depth')
plt.show()


# Tune max_features

max_features = list(range(1,X_train.shape[1]))

train_results = []
val_results = []

score_train =[]
score_val=[]

for max_feature in max_features:
   dt = DecisionTreeClassifier(max_features=max_feature)
   dt.fit(X_train, y_train)
   
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score = dt.score(X_train, y_train)
   
   score_train.append(score)
   train_results.append(roc_auc)
   
   val_pred = dt.predict(X_val)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, val_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   score= dt.score(X_val, y_val)
   
   score_val.append(score)
   
   val_results.append(roc_auc)
   
line1 = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2 = plt.plot(max_features, val_results, 'r', label="Validation AUC")
   
plt.ylabel('AUC score_max_feature')
plt.xlabel('Tree depth')
plt.show()

line1 = plt.plot(max_features, score_train, 'b', label="Train score")
line2 = plt.plot(max_features,score_val, 'r', label="Validation score")
   
plt.ylabel('Score_max_feature')
plt.xlabel('Tree depth')
plt.show()

# Random search



from sklearn.model_selection import RandomizedSearchCV

params = {"min_samples_split": np.linspace(0.1, 1.0, 10,endpoint=True),"max_depth": np.linspace(1, 32, 32, endpoint=True), "max_features":list(range(1,X_train.shape[1])),"min_samples_leaf" : np.linspace(0.1, 0.5, 5, endpoint=True)}
print("[INFO] tuning hyperparameters via grid search")
model = DecisionTreeClassifier()



rand = RandomizedSearchCV(model, params)
start = time.time()
rand.fit(X_train, y_train)

acc = rand.score(X_val, y_val)

min_sample_opt= rand.best_params_['min_samples_split']
max_depth_opt= rand.best_params_['max_depth']
max_features_opt= rand.best_params_['max_features']
min_sample_leaf_opt= rand.best_params_['min_samples_leaf']



print ("Randomized search took:", time.time() - start)
print ("Randomized search accuracy:", acc*100)
print ("Randomized search best parameters:", rand.best_params_)


#Learning Curves

estimator = DecisionTreeClassifier(max_depth=max_depth_opt, min_samples_leaf = min_sample_leaf_opt, max_features=max_features_opt, min_samples_split=min_sample_opt)

title = "Learning Curves (Decision tree)"

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



plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=4)

plt.show()

final_classifier_results(estimator, X_train, X_test, y_train, y_test)



#Implement final classifier evaluation

#IMPLEMENT PRUNING


#Implementing boosting for Decision tree

def hyper_boosting(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title):
    
    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1,100,30).astype('int')
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
    
    print f1_test
    print n_estimators
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def BoostRandomSearchCV(X_train,y_train):
    
    start_leaf_n = round(0.005*len(X_train))
    end_leaf_n = round(0.05*len(X_train)) 
    
    params = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
                  'max_depth': np.arange(1,4),
                  'n_estimators': np.linspace(10,100,3).round().astype('int'),
                  'learning_rate': np.linspace(.001,.1,3)}
    print("[INFO] tuning hyperparameters via random search")
    model = GradientBoostingClassifier()
    
    
    rand = RandomizedSearchCV(model, param_distributions=params,random_state=100, cv=5)
    start = time.time()
    rand.fit(X_train, y_train)
    
    acc = rand.score(X_val, y_val)
    
    print ("Randomized search took:", time.time() - start)
    print ("Randomized search accuracy:", acc*100)
    print ("Randomized search best parameters:", rand.best_params_)
    return rand.best_params_['min_samples_leaf'], rand.best_params_['max_depth'],rand.best_params_['n_estimators'],rand.best_params_['learning_rate']


hyper_boosting(X_train, y_train, X_val, y_val, 3, 50, title="Model Complexity Curve for Boosted Tree (Breast_Canver)\nHyperparameter : No. of Estimators")

opt_msl, opt_md, opt_ne, opt_lr= BoostRandomSearchCV(X_train,y_train)

estimator_final = GradientBoostingClassifier(max_depth=opt_md, min_samples_leaf=opt_msl, 
                                              n_estimators=opt_ne, learning_rate=opt_lr, random_state=100)
title = "Learning Curves (Boosting)"

plot_learning_curve(estimator_final, title, X, y, ylim=(0.3, 1.01), cv=5, n_jobs=4)

final_classifier_results(estimator_final, X_train, X_test, y_train, y_test)


    

       
   
   
   