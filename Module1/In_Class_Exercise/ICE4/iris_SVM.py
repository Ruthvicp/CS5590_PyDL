"""
@author : ruthvicp
date : 2/18/2019
"""

#import SVC From Scikit- Learn library
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
#Loading iris data
#irisdataset = datasets.load_iris()
irisdataset = pd.read_csv('iris.csv')
# x is predictor and y is the target data
x = irisdataset.iloc[:, :-1] # remove last column
y = irisdataset.iloc[:, -1] # consider only the last column
#training set
x_train , x_test, y_train , y_test = train_test_split(x,y, test_size= 0.2)

#using linear SVC Classification Object
svclassifier = SVC(kernel='linear')
#training the data
svclassifier.fit(x_train, y_train)
#pridicting the output
y_pred = svclassifier.predict(x_test)
print(y_pred)
list = y_pred

print("Accuracy", metrics.accuracy_score(y_test, y_pred))
# for i in list:
#     print(y.get(i))
#classification report to display the accuracy
print(classification_report(y_test, y_pred))