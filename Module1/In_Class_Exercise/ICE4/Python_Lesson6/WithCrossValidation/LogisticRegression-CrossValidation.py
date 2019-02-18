from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split

irisdatasets = datasets.load_iris()
x = irisdatasets.data
y = irisdatasets.target

# split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x, y)
print(model.score(x, y))
