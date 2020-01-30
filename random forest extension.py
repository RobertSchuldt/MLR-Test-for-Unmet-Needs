# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:34:39 2020

@author: 3043340
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:28:01 2020

@author: Robert Schuldt

MLR Model Random Forest
"""



import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import seaborn as sns

csv_file = "*****\Python\\MLR Testing2.csv"
c_size = 50000

data_set = pd.DataFrame()

for gm_chunk in pd.read_csv(csv_file, chunksize=c_size, header= 0):
    data_set = data_set.append(gm_chunk)
    print(gm_chunk.shape)
   

    
view = data_set.describe()    



data_set = data_set

target_names = ['No_bad', 'Hospital','LTC']

X = data_set.drop(columns =['outcome', 'STATE_CODE', 'COUNTY_CD', 'fips_state',
                             'any_hosp_stay', 'M0010_MEDICARE_ID', 'M0030_SOC_DT', 
                             'M1020_PRI_DGN_ICD', 'nutrition', 'age_mark', 'ffs',
                             'inpat_event_pre', 'Employment_Rate_16_', 'Education_Level',
                             'death_30', 'death_90', 'index' ])

varlist = X.describe()
    
feature_names= list(X.columns.values)

y = data_set[['outcome']]


y.describe()
id_y = y['outcome'].value_counts()
# Split the data set into test and training'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)


#Adjust sampling b/c we have imbalanced Data set
from imblearn.over_sampling import SMOTE

smote= SMOTE('not majority', random_state = 42)
X_sm, y_sm =  smote.fit_sample(X_train, y_train)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


#Want to place my acc and feature weights into an empy

model_acc = []

feature_meaures = []

#Now to apply my randomized grid to the random forest classifer
clf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 4)
# Fit the random search model
rf_random.fit(X_sm, y_sm)

train_pred= clf.predict(X_sm)
y_pred= clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
model_acc.append(acc)
print("Accuracy of %s is %s"%(clf, acc))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of %s is %s"%(clf, cm))

#Create my feature importance matrix 
        
        
feature_imp = pd.Series(clf.feature_importances_ , index= feature_names).sort_values(ascending=False)
feature_meaures.append(feature_imp)
import matplotlib.pyplot as plt
            
        
#create a bar plot of features
        
feature_plot = sns.barplot(x=feature_imp, y=feature_imp.index)
#add labels
            
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.tick_params(axis= 'y', which= 'major', pad = 10)
plt.title("Visualizing Important Features")
plt.legend
plt.tight_layout()
plt.show(feature_plot)
plt.figure(figsize = (12, 12))
plt.rcParams["ytick.labelsize"] = 3
feature_plot.figure.savefig('Z:\\DATA\\Dissertation\\Python\\feature_importance.png', dpi = 300)

best_fit = rf_random.best_params_




#visualize confusion Matrix
def plot_confusion_matrix(cm, 
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize = True):
    
    import matplotlib.pyplot as plt
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



plot_confusion_matrix(cm = cm,
                      normalize = False,
                      target_names = None,
                      title = 'Confusion Matrix')



