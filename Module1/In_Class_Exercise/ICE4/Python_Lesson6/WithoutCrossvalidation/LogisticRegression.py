from sklearn.linear_model import LogisticRegression
from sklearn import datasets

irisdatasets = datasets.load_iris()
x = irisdatasets.data
y = irisdatasets.target

model = LogisticRegression()
model.fit(x, y)
print(model.score(x, y))
