"""
@author : ruthvicp
date : 2/18/2019
"""

#Import GaussianNB from scikit-Learn libraries
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#loading irisdataset
#irisdataset = datasets.load_iris()
irisdataset = pd.read_csv('iris.csv')
# x is predictor and y is the target data
x = irisdataset.iloc[:, :-1]  # remove last column
y = irisdataset.iloc[:, -1]   # consider only the last column

#training set
x_train , x_test, y_train , y_test = train_test_split(x,y)
# creating Gaussain Naive Bayes object for classification
model = GaussianNB()
#Training the Model
model.fit(x_train, y_train)
#Predictig the Output
y_predict = model.predict(x_test)
print(y_predict)
#Printing the  accuracy
print("Accuracy", metrics.accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))


