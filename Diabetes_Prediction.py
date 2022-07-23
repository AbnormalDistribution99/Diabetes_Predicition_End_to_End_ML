## importing all the required libraries.
import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn import ensemble

## Reading cvs file
dataframe = pd.read_csv("diabetes.csv")

## Seperate predictors and outcome(labels)
X = dataframe.drop(columns='Outcome')
y = dataframe['Outcome']

## Splitting the dataset into train and test to check the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

## Random forest model, found the best parameter using Randomized Search CV
## Cross-Validation == 5
model_rfr = ensemble.RandomForestClassifier(
 n_estimators =  1100,
 max_leaf_nodes = 50,
 max_features = 'log2',
 max_depth =  100
 )

## Fit method
model_rfr.fit(X_train, y_train)

## Dumping the model in pickle file (Pickling)

filename = 'Diabetes_Prediction.pkl'
pickle.dump(model_rfr, open(filename, 'wb'))