# kmeans
Implementing the K-means clustering algorithm from scratch in Python is a great way to understand how the algorithm works. Here's a step-by-step guide on how to do it:

### Step 1: Import Necessary Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Initialize Centroids

To start, you need to randomly initialize `k` centroids. 

```python
def initialize_centroids(X, k):
    # Randomly choose k unique data points as initial centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    return centroids
```

### Step 3: Assign Clusters

For each data point, assign it to the nearest centroid.

```python
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        # Compute the distance between x and each centroid
        distances = np.linalg.norm(x - centroids, axis=1)
        # Assign the data point to the closest centroid
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)
```

### Step 4: Update Centroids

Update the centroids by calculating the mean of all data points assigned to each cluster.

```python
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        # Get all data points belonging to cluster i
        cluster_points = X[clusters == i]
        # Compute the mean of the points in cluster i
        centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(centroid)
    return np.array(new_centroids)
```

### Step 5: Implement the K-Means Algorithm

Now, combine the above steps into the K-means algorithm.

```python
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    
    for i in range(max_iters):
        # Assign clusters based on current centroids
        clusters = assign_clusters(X, centroids)
        
        # Save the current centroids for convergence check
        old_centroids = centroids
        
        # Update centroids
        centroids = update_centroids(X, clusters, k)
        
        # Check for convergence
        if np.all(np.abs(centroids - old_centroids) < tol):
            break
    
    return centroids, clusters
```

### Step 6: Visualize the Results (Optional)

If you have a 2D dataset, you can visualize the clusters.

```python
def plot_kmeans(X, centroids, clusters):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
    plt.show()
```

### Step 7: Run the K-means Algorithm

Let's test the implementation with some sample data.

```python
# Generate synthetic data
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Run K-means
k = 4
centroids, clusters = kmeans(X, k)

# Visualize the clusters
plot_kmeans(X, centroids, clusters)
```

### Explanation:

1. **initialize_centroids**: This function initializes the centroids by randomly selecting `k` unique data points from the dataset.
2. **assign_clusters**: This function assigns each data point to the closest centroid.
3. **update_centroids**: This function updates the centroids by computing the mean of all data points assigned to each cluster.
4. **kmeans**: This is the main function that iteratively assigns clusters and updates centroids until the algorithm converges (i.e., the centroids do not change significantly between iterations) or the maximum number of iterations is reached.
5. **plot_kmeans**: This function visualizes the final clusters and centroids (optional for 2D data).

### Output

When you run the script, it will generate a plot showing the clusters and the final centroids marked with an "x".

This is a basic implementation of the K-means algorithm from scratch in Python. You can extend this by adding more features, such as different distance metrics, or optimizing the algorithm for larger datasets.
