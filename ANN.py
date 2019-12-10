# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:25:12 2019

@author: Robert Schuldt

Attempt at Neural Network for predicting home health inpatient event
or death among beneficiaries. Using the NN to classify into the three categories
using Keras Deep learning library.

email: rschuldt@uams.edu
"""

import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import layers


csv_file = "Z:\\DATA\\Dissertation\\Python\\MLR Testing.csv"
c_size = 50000

data_set = pd.DataFrame()

for gm_chunk in pd.read_csv(csv_file, chunksize=c_size):
    data_set = data_set.append(gm_chunk)
    print(gm_chunk.shape)
    
#select variables I am interested in using for the analysis. 

data_set = data_set[['outcome_code', 'un_adl' ,'un_iadl', 'un_mdctn' ,'un_equip',
 'un_prcdr' ,'un_sprvsn', 'un_advcy', 'post_acute_pat', 'psc_other' , 'community_pat', 
 'female', 
 'white','black', 'hispanic', 'other_race',  'dual', 'AGE_AT_END_REF_YR', 'urban' , 'micro_metro' , 'adj_rural', 
 'remote_rural',  'percap_pcp_15', 'percap_hosp_bed15', 'income' ,'poverty' ]]


#Naming targets for future analysis, may do the same with the other variables
target_names = ['No_bad','Inpatient', 'Death' ]

#Generate my data sets 

X = data_set[['un_adl' ,'un_iadl', 'un_mdctn' ,'un_equip',
 'un_prcdr' ,'un_sprvsn', 'un_advcy', 'post_acute_pat', 'psc_other' , 'community_pat', 
 'female', 
 'white','black', 'hispanic', 'other_race',  'dual',  'AGE_AT_END_REF_YR', 'urban' , 'micro_metro' , 'adj_rural', 
 'remote_rural',  'percap_pcp_15', 'percap_hosp_bed15', 'income' ,'poverty' ]]

y = data_set[['outcome_code']]

#Properly encode my outcome variable
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
#Convert to dummy
dummy_y = np_utils.to_categorical(encoded_y)


#create test and training data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,dummy_y,test_size=0.3, shuffle=True)


#Do feature scaling to transform the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Work on adjusting the data set to beter balance the data

from imblearn.over_sampling import SMOTE

smote= SMOTE('minority')
X_sm, y_sm =  smote.fit_sample(X_train, y_train)


#Define our baseline model
from keras.optimizers import Adam

opt = Adam(lr = 0.0001)
def baseline_model():
    #create the model
    model = Sequential()
    
    model.add(Dense(51, input_dim = 25, activation = 'relu'))
    #model.add(Dropout(0.5, input_shape = (25,)))
    model.add(Dense(25, input_dim = 25, activation = 'relu'))
     #model.add(Dropout(0.5, input_shape = (25,)))
    model.add(Dense(10, input_dim = 25, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    #compile Model
    model.compile(loss='categorical_crossentropy' , optimizer = opt  , metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn = baseline_model, epochs =100, batch_size = 36)

hstory = estimator.fit(X_sm, y_sm,
                       validation_split =0.2,
                       verbose = 1)



predictions = estimator.predict(X_test)

kfold = KFold(n_splits=2, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

