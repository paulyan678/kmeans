import numpy as np
import matplotlib.pyplot as plt
def kmeans(x, y, k, converge_change):
    '''
    x: data points (number of points, number of features)
    y: labels (number of points, )
    k: number of clusters
    converge_change: the threshold of change of centroids for convergence. If the biggeset change in centroids is smaller than converge_change, then the algorithm converges.
    '''
    num_points, num_dim = x.shape

    # (number of points, number of features + 1)
    data = np.hstack((x, y.reshape(-1, 1)))
    # randomly choose k points as initial centroids
    random_indics = np.random.choice(num_points, k, replace=False)

    # (k, number of features + 1)
    old_centroids = data[random_indics, :]
    data_reshaped = np.reshape(data, (1, num_points, num_dim + 1))



    cur_change = np.inf
    while cur_change > converge_change:
        old_centroids_reshape = np.reshape(old_centroids, (k, 1, num_dim + 1))
        diff = data_reshaped - old_centroids_reshape
        dist = np.linalg.norm(diff, axis=2)
        assignment = np.argmin(dist, axis=0)
        # update centroids 
        new_centroids = np.zeros((k, num_dim + 1))
        for i in range(k):
            new_centroids[i, :] = np.mean(data[assignment == i, :], axis=0)
        # compute the change of centroids
        cur_change = np.max(np.linalg.norm(new_centroids - old_centroids, axis=1))
        old_centroids = new_centroids


    return assignment, new_centroids
        
def plot_clusters(x, y, assignment, centroids):
    plt.scatter(x, y, c=assignment)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='k')
    plt.show()

if __name__ == '__main__':
    # random draw n 2d datapoint from a normal distrubtuion
    n = 100
    x = np.random.normal(size=n, loc=0, scale=1)
    x = np.vstack((x, np.random.normal(size=n, loc=5, scale=1)))
    y = np.random.normal(size=n, loc=0, scale=1)
    y = np.vstack((y, np.random.normal(size=n, loc=5, scale=1)))
    x = np.reshape(x, (2*n, 1))

    k = 1
    converge_change = 0.1
    assignment, centroids = kmeans(x, y, k, converge_change)
    plot_clusters(x, y, assignment, centroids)
    











