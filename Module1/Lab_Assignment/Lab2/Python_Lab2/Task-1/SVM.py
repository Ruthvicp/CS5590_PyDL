#import packages for support vector , accuracy, split train data and datasets
import inline as inline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def Svm():
    #load the iris datasets
    data=datasets.load_iris()
    #load x and y data
    x=data.data
    y=data.target
    #split training and test data for both x and y for linear kernel
    train_x, test_x, train_y, test_y=train_test_split(x, y, test_size=0.2, random_state=21)
    #split training and test data for both x and y for rbf kernel
    train_x1, test_x1, train_y1, test_y1=train_test_split(x, y, test_size=0.2, random_state=23)
    #define the model for poly kernel
    p_model = SVC(kernel="poly", degree=4, gamma=0.1, C=1)
    p_model1 = SVC(kernel="poly", degree=4, gamma =100, C = 1000)
    #define the model for rbf kernel
    rmodel=SVC(kernel='rbf', gamma=0.1, C=1)
    rmodel1 = SVC(kernel='rbf', gamma=100, C=1000)
    #fit training data into linear kernel
    p_model.fit(train_x, train_y)
    p_model1.fit(train_x, train_y)

    #predict the test data using linear kernel
    prediction=p_model.predict(test_x)
    prediction1 = p_model1.predict(test_x)
    #calc accuracy score for linear kernel
    print("poly kernel Accuracy score  : ", accuracy_score(prediction, test_y))
    print(prediction)
    print("poly kernel Accuracy score is", accuracy_score(prediction1, test_y))
    print(prediction1)
    #fit training data into rbc kernel
    rmodel.fit(train_x1, train_y1)
    rmodel1.fit(train_x1, train_y1)
    #predict the test data for rbc kernel
    pred=rmodel.predict(test_x1)
    pred1 = rmodel1.predict(test_x1)
    #calc accuracy for rbc kernel
    print("RBF kernel accuracy score is", accuracy_score(pred, test_y1))
    print("RBF kernel accuracy score is", accuracy_score(pred1, test_y1))
    print(pred)
    print(pred1)

def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min()-1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min()-1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  # Meshgrid : return coordinate matrices from coordinate vectors.
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))

  plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.show()

if __name__ == '__main__':
    Svm()
    # import some data to play with
    iris1 = datasets.load_iris()
    X = iris1.data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = iris1.target
    gammas = [0.1, 100]
    # gamma is a parameter for non linear hyperplanes.
    # The higher the gamma value it tries to exactly fit the training data set
    for gamma in gammas:
        p_model = SVC(kernel="poly", degree= 4)
        svc = SVC(kernel='rbf', gamma=gamma).fit(X, y)
# increasing gamma leads to overfitting as the classifier tries to perfectly fit the training data
        plotSVC('gamma=' + str(gamma))

    cs = [0.1, 1000]
    # C is the penalty parameter of the error term. It controls the trade off between
    # smooth decision boundary and classifying the training points correctly.
    for c in cs:
        svc = SVC(kernel='rbf', C=c).fit(X, y)
    # Increasing C values may lead to overfitting the training data.
        plotSVC('C=' + str(c))