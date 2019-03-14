from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split


def k_Nearest():
    #loading the iris dataset from sklearn
    irisdataset = datasets.load_iris()
    # Assigning data for both x and y
    x = irisdataset.data
    y = irisdataset.target
    # Dividing the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    k_range = range(1,50) # setting the K range
    scores = []
    for k in k_range:
        # Assigining the K value to the neighbours for the model
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k)
        # fitting the data into model
        knn.fit(x_train, y_train)
        # Predict the value y
        y_pred = knn.predict(x_test)
        # Accuracy for the predicted value
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        # Appending the accuracy values to the scores to plot the graph
        scores.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores)
    plt.xlabel("K value")
    plt.ylabel("Accuracy/Testing")
    plt.show()

if __name__ == '__main__':
    k_Nearest()