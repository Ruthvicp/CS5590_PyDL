import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


iris = datasets.load_iris()


x = iris.data
y = iris.target
target_names = iris.target_names

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


modelNB=GaussianNB()
modelNB.fit(x_train,y_train)
print("The Score is",modelNB.score(x_train,y_train))
y_pred1 = modelNB.predict(x_test)
print("The Accuracy classification score of testing : ",metrics.accuracy_score(y_test, y_pred1))

modelKN = KNeighborsClassifier(n_neighbors=11 )

modelKN.fit(x_train, y_train)

y_pred = modelKN.predict(x_test)

modelLDA = LinearDiscriminantAnalysis(n_components=3)
A_R = modelLDA.fit(x_test, y_pred).transform(x)
colors = ['red', 'green', 'blue']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    # Scattering the data
    plt.scatter(A_R[y == i, 0], A_R[y == i, 1], alpha=1, color=color, label=target_name)
#places a legend in the axes
plt.legend(loc='best', shadow=False, scatterpoints=1)
# Show the scattered points on the graph
plt.show()