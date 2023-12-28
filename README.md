# Data-Clustering

## Overview

This repository contains the implementation and evaluation of three different clustering algorithms in Python. The objective is to explore and evaluate the performance of these algorithms in clustering unlabelled data.

## Clustering Algorithms

Word Embeddings Dataset:

This dataset contains word embeddings represented as 300-dimensional feature vectors corresponding to various words in a semantic space. Each line begins with a word followed by its 300 features, capturing the semantic context and meaning of that word. Word embeddings are commonly utilized in natural language processing tasks to represent words in a numerical format, encapsulating their semantic relationships within a vector space.

This dataset serves as a valuable resource for tasks involving language understanding, text analysis, and machine learning models that rely on capturing semantic similarities between words.

### K-Means

K-means is a popular clustering algorithm that aims to partition a dataset into K clusters based on feature similarity. It starts by randomly choosing K data points as initial centroids and assigns each point to the closest centroid. The algorithm iteratively updates the centroids by computing the mean of points in each cluster until convergence.

#### Pseudocode:
k-Means Clustering (Number of clusters: k, Dataset: data)
1. Initialize k cluster representatives Y1,...Yk from the dataset randomly
2. Repeat until convergence: 
    a. Assign all objects in the dataset to the closest representative 
    b. Compute the new representatives Y1,...Yk as the means of the current clusters
3. Return the final cluster representatives Y1,...Yk

### K-Means++

K-means++ improves the initial selection of centroids in K-means by ensuring that the centroids are well-separated. It selects the first centroid randomly and the subsequent centroids with associated probabilities to maximize the distance between them.

#### Pseudocode:
k-Means++ (Number of clusters: k, Dataset: data)
1. Initialize k cluster representatives using K-means++ initialization method
2. Proceed with standard k-means using these initial cluster representatives
3. Return the final cluster representatives

### Bisecting K-Means

Bisecting K-means is a hybrid approach that combines partitional and hierarchical clustering. It starts with a single cluster and iteratively splits the cluster with the highest sum of squared errors using the K-means algorithm until the desired number of clusters is obtained.

#### Pseudocode:
Bisecting k-Means (Number of clusters: k, Dataset: data)
1. Initialize a single cluster containing all data points
2. While the number of clusters in the hierarchy is less than s:
    a. Find the cluster with the largest sum of squared errors
    b. Split the chosen cluster into two using the k-means clustering algorithm
3. Return the final cluster representatives Y1,...Yk, which are the centroids of the leaf clusters

The parameter 's' represents the desired number of clusters or the stopping criterion for the algorithm. It controls the hierarchical splitting of clusters, guiding the algorithm to continue splitting clusters until it reaches the specified number of clusters ('s'). Adjusting 's' affects the depth of the hierarchy and the final number of clusters generated by the Bisecting K-Means algorithm.

## Evaluation and Discussion

### Evaluating Clustering Algorithms

To evaluate the performance of the implemented clustering algorithms (K-Means, K-Means++, and Bisecting K-Means), we used the Silhouette Coefficient. This metric measures the quality of clusters by evaluating the similarity of data points within clusters compared to neighboring clusters.

The Silhouette Coefficient ranges between -1 and 1, where:
- Values close to +1 indicate that data points are well-clustered.
- Values close to 0 indicate overlapping clusters.
- Values close to -1 suggest data points might be assigned to the wrong clusters.

### Computing the Silhouette Coefficient

We calculated the Silhouette Coefficient for varying cluster sizes (K/S) across each algorithm. This involved clustering the given dataset for different K or S values and computing the average Silhouette Coefficient.
The figures below represent the Silhouette coefficient vs the number of clusters for each of the algorithms.


#### Figure 1: Silhouette Coefficient vs. Number of Clusters (K/S) for K-Means

![K-Means](https://github.com/sacadelmi/Data-Clustering/blob/main/kmeans.png)

#### Figure 2: Silhouette Coefficient vs. Number of Clusters (K/S) for K-Means++
![K-Means++](https://github.com/sacadelmi/Data-Clustering/blob/main/kmeans%2B%2B.png)

#### Figure 3: Silhouette Coefficient vs. Number of Clusters (K/S) for Bisecting K-Means
![Bisecting K-Means](https://github.com/sacadelmi/Data-Clustering/blob/main/bisectingkmeans.png)

### Further Results: Optimum Silhouette Coefficient

| Algorithm          | Optimal Setting | Silhouette Coefficient       |
| ------------------ | --------------- | ---------------------------- |
| K-Means            | K=2             | 0.1525277616533619           |
| K-Means++          | K=2             | 0.1525277616533619           |
| Bisecting K-Means  | S=2             | 0.1525277616533619           |

This table details the optimal settings and their respective silhouette coefficients for the three clustering algorithms: K-Means, K-Means++, and Bisecting K-Means. Each algorithm achieves a silhouette coefficient of 0.1525277616533619 at the specified optimal settings.


### Analysis of Results
The silhouette score at K/S = 2 demonstrates the optimum clustering for our dataset across all three algorithms: K-Means, K-Means++, and Bisecting K-Means. Surprisingly, each algorithm obtained identical silhouette scores at this value.

#### Implications of Silhouette Coefficient Results

The consistency of silhouette scores across algorithms at K/S = 2 indicates that this setting provides the best clustering quality for our dataset. 

### Conclusion

Although K-Means++, and Bisecting K-Means offer theoretical improvements over K-Means, our evaluation suggests an equivalent performance among these algorithms for our dataset. Therefore, K/S = 2 emerges as the optimal number of clusters, as supported by the Silhouette Coefficient.

