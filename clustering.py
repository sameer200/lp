from sklearn.cluster import KMeans
import numpy as np
X = np.array([[0.1, 0.6], [0.15, 0.71], [0.08, 0.9], [0.16, 0.85],
			  [0.2, 0.3], [0.25, 0.5], [0.24,0.1], [0.3,0.2]])

centres = np.array([[0.01,0.06],[0.3,0.2]])

print("initial centroids :\n",centres)

#creating model
k_model = KMeans(n_clusters=2, init=centres, n_init=1).fit(X)
k_model.fit(X)
print("Labels:",k_model.labels_)

#find P6 location
print("P6 balongs to cluster",k_model.labels_[5])

#using labels to find population around centroid
print("no of population around cluster 2:",np.count_nonzero(k_model.labels_==1))

#find new centroids
print("new centroids:\n",k_model.cluster_centers_)