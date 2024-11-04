import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
data = np.random.rand(1000, 2) * 50
tuple_array = np.array(data)

cluster1 = tuple_array[np.random.choice(tuple_array.shape[0])]
cluster2 = tuple_array[np.random.choice(tuple_array.shape[0])]
cluster3 = tuple_array[np.random.choice(tuple_array.shape[0])]
cluster4 = tuple_array[np.random.choice(tuple_array.shape[0])]

threshold = 0.0001
max_iterations = 100
labels = []
c1, c2, c3, c4 = [], [], [], []

print("Initial centroids:")
print(cluster1, cluster2, cluster3, cluster4)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

for iteration in range(max_iterations):
    old_cluster1, old_cluster2, old_cluster3, old_cluster4 = cluster1, cluster2, cluster3, cluster4
    c1, c2, c3, c4 = [], [], [], []
    labels = []

    for i in data:
        d1 = euclidean_distance(cluster1, i)
        d2 = euclidean_distance(cluster2, i)
        d3 = euclidean_distance(cluster3, i)
        d4 = euclidean_distance(cluster4, i)
        d = np.argmin([d1, d2, d3, d4])

        if d == 0:
            c1.append(i)
        elif d == 1:
            c2.append(i)
        elif d == 2:
            c3.append(i)
        else:
            c4.append(i)

        labels.append(d)

    if c1:
        cluster1 = np.mean(c1, axis=0)
    if c2:
        cluster2 = np.mean(c2, axis=0)
    if c3:
        cluster3 = np.mean(c3, axis=0)
    if c4:
        cluster4 = np.mean(c4, axis=0)

    centroid_shifts = [
        euclidean_distance(cluster1, old_cluster1),
        euclidean_distance(cluster2, old_cluster2),
        euclidean_distance(cluster3, old_cluster3),
        euclidean_distance(cluster4, old_cluster4),
    ]

    if max(centroid_shifts) < threshold:
        print(f"Converged after {iteration + 1} iterations")
        break

print("Final centroids:")
print(cluster1, cluster2, cluster3, cluster4)

plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c=labels, label='Data Points', alpha=0.5)
plt.scatter(cluster1[0], cluster1[1], color='violet', marker='o', s=100, label='Centroid 1')
plt.scatter(cluster2[0], cluster2[1], color='cyan', marker='o', s=100, label='Centroid 2')
plt.scatter(cluster3[0], cluster3[1], color='blue', marker='o', s=100, label='Centroid 3')
plt.scatter(cluster4[0], cluster4[1], color='yellow', marker='o', s=100, label='Centroid 4')

plt.legend()
plt.title('Clustering Visualization with Convergence')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
