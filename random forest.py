# -*- coding: utf-8 -*-


"""
Created on Tue Aug 20 11:39:32 2019

@author: Robert Schuldtd

SKLEARN Practice for Random Forest using my outcome and the unmet 
caregiver needs. Test for using in dissertation and feature selection
to properly weight the dimensions of unmet caregiver needs in the study
"""

import pandas as pd
import sklearn as sk

#load my data set
data = pd.read_csv(*******\randomforest.csv", header = 0)

#check variable names

var_list = data.head()

data = data[['outcomes', 'un_adl', 'un_iadl', 'un_mdctn', 'un_equip', 'un_prcdr', 'un_sprvsn', 'un_advcy', 'post_acute_pat' ]]

from sklearn.model_selection import train_test_split

X = data[[ 'un_adl', 'un_iadl', 'un_mdctn', 'un_equip', 'un_prcdr', 'un_sprvsn', 'un_advcy']] 
#name my features and name my classes
feature_names = ['Unmet ADL', 'Unmet IADL', 'Unmet Medication', 'Unmet Equipment', 'Unmet Procedure', 'Unmet Supervision', 'Unmet Advocacy']
target_names = ['No_bad','hosp_stay', 'snf_stay','long_stay', 'death' ]
#Outcome variable 
y= data[['outcomes']]

#Split data set into test and train
#Set 70% to train and 30% to test
X_Train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

#Import my classifier

from sklearn.ensemble import RandomForestClassifier

#create my Gaussian Classifier
clf=RandomForestClassifier(bootstrap=True, class_weight=None , criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

clf.fit(X_Train, y_train.values.ravel())

y_pred=clf.predict(X_test)


feature_imp = pd.Series(clf.feature_importances_ ,index= feature_names).sort_values(ascending=False)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:" , cf)

from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns 


#create a bar plot of features

sns.barplot(x=feature_imp, y=feature_imp.index)
#add labels
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend
plt.show()

#visualize confusion Matrix
def plot_confusion_matrix(cm, 
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize = True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    accuracy = np.trace(cm) /float(np.sum(cm))
    misclass= 1- accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
        plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
plot_confusion_matrix(cm = cf,
                      normalize = False,
                      target_names = target_names,
                      title = 'Confusion Matrix')    
