"""
@author : ruthvicp
date : 2/22/2019
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# method to rate the graduate rating column
def assignRating(dataset):
    if dataset['Grad.Rate'] <= 25:
        return 0
    elif dataset['Grad.Rate'] > 25 and dataset['Grad.Rate'] <= 50:
        return 1
    elif dataset['Grad.Rate'] > 50 and dataset['Grad.Rate'] <= 75:
        return 2
    elif dataset['Grad.Rate'] > 75:
        return 3

sns.set(style="white", color_codes=True)
warnings.filterwarnings("ignore")
dataset = pd.read_csv('College.csv')
# drop non numerical columns instead of vectorizing them
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset.drop(['Private'], axis=1)
# add an additional column with the calculated rating
dataset['Rating'] = dataset.apply(assignRating,axis=1)
# take columns 2 to 13
x = dataset.iloc[:, 2:13]
# take only last column
y = dataset.iloc[:, :-1]
print(x)

# Here I am calculating the no. of clusters based on max & min values
# which is a trial and error type. I will evalaute this value at the end
# using elbow method
y_max = pd.DataFrame(y).max()
print(y_max)
y_min = pd.DataFrame(y).min()
print(y_min)
n = (y_max - y_min) / 20
n = n["Grad.Rate"]

# see how many samples we have of each species
#print(dataset[0].value_counts())

# Plot the initial data set
sns.FacetGrid(dataset, hue="Grad.Rate", size=4).map(plt.scatter, "Enroll", "S.F.Ratio")
sns.FacetGrid(dataset, hue="Rating", size=4).map(plt.scatter, "Enroll", "Grad.Rate").add_legend()
plt.show()
# transform the data here
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

nclusters = int(n) # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
centroids = km.cluster_centers_
# print(y_cluster_kmeans)

# Legend Colors for clusters
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'g',
                   3 : 'y',
                   4: 'm'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in y_cluster_kmeans]

# import PCA for reducing dimensions of the data from 13 columns to 2 columns
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xx = pca.fit_transform(X_scaled_array)
#print(xx)
# Clustering Plot
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)
plt.scatter(xx[:,0], xx[:,1], c=label_color)
plt.title('Identified Clusters')
plt.show()

from sklearn import metrics
wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(xx)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

print("Silhouette Score:", silhouette_score(x, y_cluster_kmeans,metric='euclidean'))
