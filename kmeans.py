import numpy as np

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        new_centroids = np.array([X[np.where(labels == i)].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return labels, centroids

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
k = 2

labels, centroids = kmeans(X, k)

print("Labels:", labels)
print("Centroids:", centroids)