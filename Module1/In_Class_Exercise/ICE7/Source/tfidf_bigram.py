from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer(ngram_range= (2,3))
X_train_range = tfidf_Vect.fit_transform(twenty_train.data)
set(stopwords.words('english'))
# print(tfidf_Vect.vocabulary_)
clf = KNeighborsClassifier(n_neighbors= 3)
clf.fit(X_train_range, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_range = tfidf_Vect.transform(twenty_test.data)
pred = clf.predict(X_test_range)
score = metrics.accuracy_score(twenty_test.target, pred)
print("bigram",score)