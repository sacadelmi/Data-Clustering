# Sacad Elmi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Load dataset into pandas DataFrame
df = pd.read_csv('dataset', delimiter=' ', header=None)

# Extract numerical features from DataFrame
X = df.iloc[:, 1:].values.astype(float)

def distance(X, Y):
    # Calculate the Euclidean distance between points X and Y
    return np.linalg.norm(X - Y)

def kMeans(data, k):
    # Perform k-Means clustering on the given data
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]

    while True:
        # Assign each data point to the closest centroid
        distances = np.zeros((data.shape[0], k))
        for i in range(k):
            distances[:, i] = np.apply_along_axis(distance, 1, data, centroids[i])
        clusters = np.argmin(distances, axis=1)

        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids

    return clusters, centroids

def kMeansPP(data, k):
    # Perform k-Means++ clustering on the given data
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0], 1), :]

    for i in range(1, k):
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            distances[:, j] = np.apply_along_axis(distance, 1, data, centroids[j])
        min_dists = np.min(distances, axis=1)
        probs = min_dists / np.sum(min_dists)
        centroids[i] = data[np.random.choice(data.shape[0], 1, p=probs), :]

    cluster_assignments, centroids = kMeans(data, k)

    return centroids, cluster_assignments

def bisectingkMeans(data, s):
    # Perform bisecting k-Means clustering on the data
    clusters = np.zeros(data.shape[0], dtype=int)
    centroids = [data.mean(axis=0)]
    hierarchy = [(clusters, centroids)]

    while len(hierarchy) < s:
        sse = [np.sum([distance(x, c) ** 2 for x in data[clusters == i]]) for i, c in enumerate(centroids)]
        largestClusterSSE = np.argmax(sse)
        clusterData = data[clusters == largestClusterSSE]
        new_assignments, new_centroids = kMeans(clusterData, 2)
        new_assignments[new_assignments == 0] = largestClusterSSE
        new_assignments[new_assignments == 1] = len(centroids)
        clusters[clusters == largestClusterSSE] = new_assignments
        centroids.pop(largestClusterSSE)
        centroids += list(new_centroids)
        hierarchy.append((clusters.copy(), centroids.copy()))

    return hierarchy

# Function to compute Silhouette coefficient for given clusters
def silhouetteCoefficient(data, clusters):
    n = len(data)
    a = np.zeros(n)
    b = np.zeros(n)

    for i in range(n):
        a[i] = np.mean(np.apply_along_axis(distance, 1, data[clusters == clusters[i]], data[i]))
        other_clusters = np.unique(clusters[clusters != clusters[i]])
        b[i] = np.min([np.mean(np.apply_along_axis(distance, 1, data[clusters == j], data[i])) for j in other_clusters])

    s = (b - a) / np.maximum(a, b)
    return np.mean(s)

# Function to plot Silhouette coefficients
def silhouetteCoefficientPlot(s_values, silhouette_coefficients, title):
    plt.plot(s_values, silhouette_coefficients)
    plt.xlabel('k' if title != 'Bisection' else 's')
    plt.ylabel('Silhouette coefficient')
    plt.title(title)
    plt.show()

k_values = range(2, 10)
s_values = range(2, 10)

# Compute Silhouette coefficients for k-Means
kMeans_silhouette_coefficients = []
for k in k_values:
    clusters_kMeans, centroids_kMeans = kMeans(X, k)
    kMeans_silhouette_coefficients.append(silhouetteCoefficient(X, clusters_kMeans))

print("K-Means Silhouette Coefficients:")
for k, s in zip(k_values, kMeans_silhouette_coefficients):
    print(f"k value: {k}, Silhouette coefficient: {s}")

silhouetteCoefficientPlot(k_values, kMeans_silhouette_coefficients, 'kMeans')

# Compute Silhouette coefficients for k-Means++
kMeansPP_silhouette_coefficients = []
for k in k_values:
    centroids_kMeansPP, clusters_kMeansPP = kMeansPP(X, k)
    kMeansPP_silhouette_coefficients.append(silhouetteCoefficient(X, clusters_kMeansPP))

print('\n')
print("K-Means++ Silhouette Coefficients:")
for k, s in zip(k_values, kMeansPP_silhouette_coefficients):
    print(f"k value: {k}, Silhouette coefficient: {s}")

silhouetteCoefficientPlot(k_values, kMeansPP_silhouette_coefficients, 'kMeans++')

# Compute Silhouette coefficients for bisecting k-Means
hierarchy = bisectingkMeans(X, s=9)
bisecting_silhouette_coefficients = []
for s in s_values:
    clusters, centroids = hierarchy[s - 1][0], hierarchy[s - 1][1]
    bisecting_silhouette_coefficients.append(silhouetteCoefficient(X, clusters))

print('\n')
print('Bisecting K-Means Silhouette Coefficients:')
for s, SC in zip(s_values, bisecting_silhouette_coefficients):
    print(f"s value: {s}, Silhouette coefficient: {SC}")

silhouetteCoefficientPlot(s_values, bisecting_silhouette_coefficients, 'bisectingkMeans')
