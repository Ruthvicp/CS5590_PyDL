import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn import svm
from matplotlib import pyplot as plt
# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv")
print(dataset.shape)

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(dataset.iloc[1:,1].values)
X_train, X_test, Y_train, Y_test = train_test_split(dataset.iloc[1:,2:32].values, y,
                                                    test_size=0.25, random_state=87)

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(40, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
scores = my_first_nn.evaluate(X_test, Y_test, verbose=0)
Y_pred = my_first_nn.predict(X_test)


average_precision = average_precision_score(Y_test,Y_pred)
print(average_precision)

t = np.arange(0,142,step=1)
plt.scatter(t,Y_test, color='r')
plt.scatter(t,Y_pred, alpha=0.45)
plt.show()