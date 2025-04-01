#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset: Annual Income and Spending Score of Customers
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76], [21, 6], [22, 94], [23, 3], [24, 71],
    [25, 77], [26, 75], [27, 5], [28, 95], [29, 40], [30, 75], [31, 8], [32, 90], [33, 3], [34, 73]
])

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.xlabel("Annual Income (in $1000s)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means Clustering")
plt.legend()
plt.savefig("k_means_plot.png")
