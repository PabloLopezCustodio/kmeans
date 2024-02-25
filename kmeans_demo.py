import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function for printing clusters
def print_clusters(membership, data):
    for i in range(int(np.max(membership))+1):
        names = []
        for j in range(membership.size):
            if membership[j] == i:
                names.append(data.iloc[j]["name"])
        print('cluster', i+1, ':', names)


# colorblind-friendly palette
cb_palette = [(108/255, 108/255, 108/255),
              (213/255, 94/255, 0/255),
              (230/255, 159/255, 0/255),
              (240/255, 228/255, 66/255),
              (0/255, 158/255, 115/255),
              (86/255, 180/255, 233/255),
              (0/255, 114/255, 178/255),
              (204/255, 121/255, 167/255)]

# number of clusters
K = 4

# load data and convert to xy coordinates
data = pd.read_csv('beer.csv')
x = data["price"]
y = data["alcohol"]
points = list(zip(x,y))

# centres of clusters are chosen randomly for the first iteration
ids = np.random.choice(len(points), size=K, replace=False)
centres = [points[i] for i in ids]

# initialise variables for the algorithm
clusters = [[] for _ in range(K)]
membership = np.zeros(len(points))
new_membership = np.zeros(len(points))
colour_ids = np.random.choice(len(cb_palette), size=K, replace=False)

# k-means loop
iteration = 1
while True:
    # compute centres
    if iteration != 1:
        for i, cluster in enumerate(clusters):
            centres[i] = np.average(cluster, axis=0)
    # calculate distances and assign clusters
    clusters = [[] for _ in range(K)]
    for i, point in enumerate(points):
        distances = [(centre[0] - point[0])**2 + (centre[1] - point[1])**2 for centre in centres]
        clusters[np.argmin(distances)].append(point)
        new_membership[i] = np.argmin(distances)
    print('Iteration', iteration)
    print_clusters(new_membership, data)
    # plot clusters
    plt.figure()
    ax = plt.gca()
    for i in range(K):
        plt.scatter([point[0] for point in clusters[i]], [point[1] for point in clusters[i]], color=cb_palette[colour_ids[i]], label=f"cluster {i+1}")
        plt.scatter(centres[i][0], centres[i][1], color=cb_palette[colour_ids[i]], edgecolors='black', marker="*", s=180)
    plt.xlabel('price (GBP)')
    plt.ylabel('alcohol percentage')
    ax.set_aspect('equal')
    ax.legend()
    # exit condition
    if not any(membership - new_membership):
        plt.title(f"FINAL RESULT in {iteration} iterations")
        plt.show()
        break
    plt.title(f"iteration {iteration}")
    plt.show()
    membership = np.copy(new_membership)
    iteration = iteration + 1
