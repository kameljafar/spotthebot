from collections import defaultdict
import scipy
import numpy as np
import sys
import time

from itertools import  product
from scipy.special import gamma
from scipy.spatial.distance import  euclidean
from math import sqrt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import  silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from s_dbw import S_Dbw, SD

import gc

from scipy.spatial import cKDTree

import math

import numpy as np
from scipy.special import gamma
from scipy.spatial import cKDTree
from collections import defaultdict

import matplotlib.pyplot as plt
from constants import data_path,file_for_labels_path



def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)

def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    return max_diff >= h


def find_centroids(data, labels):
    centroids = []
    for label in np.unique(labels):
        centroids.append(data[labels == label].mean(axis=0))
    return centroids

def get_validation_scores_u(data, labels):

    scores = {}


    #scores['7 silhouette'] = silhouette_score(data, labels)
    #print('7 silhouette', scores['7 silhouette'], flush=True)
    #gc.collect()

    scores['4 CH'] = calinski_harabasz_score(data, labels)
    print('4 CH', scores['4 CH'], flush=True)
    gc.collect()

    

    #scores['10 SD'] = SD(data, labels)
    #print('10 SD', scores['10 SD'], flush=True)
    #gc.collect()

    return scores

class WishartClusterization:
    clusters_to_objects: defaultdict
    object_labels: np.ndarray
    clusters: np.ndarray
    kd_tree: cKDTree

    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors
        self.significance_level = significance_level

    def fit(self, X, workers=-1, batch_weight_in_gb=30):
        self.kd_tree = cKDTree(data=X)

        distances = np.empty(0).ravel()

        batch_size = batch_weight_in_gb * (1024 ** 3) // 8
        batches_count = X.shape[0] // (batch_size // (self.wishart_neighbors + 1))
        if batches_count == 0:
            batches_count = 1

        batches = np.array_split(X, batches_count)
        itr=0
        for batch in batches:
            itr += 1
            batch_dists, _ = self.kd_tree.query(x=batch, k=self.wishart_neighbors + 1, workers=workers)
            batch_dists = batch_dists[:, -1].ravel()
            distances = np.hstack((distances, batch_dists))
            print(f"{itr} / {len(batches)} iterations")

        indexes = np.argsort(distances)
        X = X[indexes]
        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype=int) - 1

        # index in tuple
        # min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)

        batches = np.array_split(X, batches_count)
        idx_batches = np.array_split(indexes, batches_count)
        del X, indexes
        itr2 = 0
        for batch, idx_batch in zip(batches, idx_batches):
            itr2 += 1
            print(f"{itr2} / {len(batches)} iterations 2")
            _, neighbors = self.kd_tree.query(x=batch, k=self.wishart_neighbors + 1, workers=workers)
            neighbors = neighbors[:, 1:]
            # print("len(idx_batch)", len(idx_batch))
            # print("len(neighbors)", len(neighbors))
            # print("enumerate(idx_batch)", enumerate(idx_batch))
            for real_index, idx in enumerate(idx_batch):
                # print("real_index, idx", real_index, idx)
                # print("neighbors[real_index]", neighbors[real_index])
                # print("self.object_labels", self.object_labels)
                # print("(self.object_labels)", (self.object_labels))
                # print("self.object_labels[neighbors[real_index]]", self.object_labels[neighbors[real_index]])

                neighbors_clusters = np.concatenate(
                    [self.object_labels[neighbors[real_index]], self.object_labels[neighbors[real_index]]])
                unique_clusters = np.unique(neighbors_clusters).astype(int)
                unique_clusters = unique_clusters[unique_clusters != -1]

                if len(unique_clusters) == 0:
                    self._create_new_cluster(idx, distances[idx])
                else:
                    max_cluster = unique_clusters[-1]
                    min_cluster = unique_clusters[0]
                    if max_cluster == min_cluster:
                        if self.clusters[max_cluster][-1] < 0.5:
                            self._add_elem_to_exist_cluster(idx, distances[idx], max_cluster)
                        else:
                            self._add_elem_to_noise(idx)
                    else:
                        my_clusters = self.clusters[unique_clusters]
                        flags = my_clusters[:, -1]
                        if np.min(flags) > 0.5:
                            self._add_elem_to_noise(idx)
                        else:
                            significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                            significan *= self.wishart_neighbors
                            significan /= size
                            significan /= np.power(np.pi, dim / 2)
                            significan *= gamma(dim / 2 + 1)
                            significan_index = significan >= self.significance_level

                            significan_clusters = unique_clusters[significan_index]
                            not_significan_clusters = unique_clusters[~significan_index]
                            significan_clusters_count = len(significan_clusters)
                            if significan_clusters_count > 1 or min_cluster == 0:
                                self._add_elem_to_noise(idx)
                                self.clusters[significan_clusters, -1] = 1
                                for not_sig_cluster in not_significan_clusters:
                                    if not_sig_cluster == 0:
                                        continue

                                    for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                        self._add_elem_to_noise(bad_index)
                                    self.clusters_to_objects[not_sig_cluster].clear()
                            else:
                                for cur_cluster in unique_clusters:
                                    if cur_cluster == min_cluster:
                                        continue

                                    for bad_index in self.clusters_to_objects[cur_cluster]:
                                        self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                    self.clusters_to_objects[cur_cluster].clear()

                                self._add_elem_to_exist_cluster(idx, distances[idx], min_cluster)

        self.labels_ = self.clean_data()
        return self.labels_

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq: index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype=int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis=0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)




alg='wishartbatch'
def main(R, n, language, type, embedding, k, h):
    from constants import data_path
    if language == 'npy':
        data_path = data_path
        data =  np.load(data_path)

        
    else:
        data_path = data_path
        file = open(data_path, 'r')
        i = 0
        lines = file.readlines()
        # print(lines[0])
        data = []
        for line in lines:
            data.append(list(map(float, line.split())))
        file.close()
    print('all n-gramms:', len(data))

    data = np.unique(data, axis=0)
    print('new n-gramms:', len(data))
    gc.collect()

    #Clustering
    print('creating cluster algo', flush=True)
    wish = WishartClusterization(wishart_neighbors=k, significance_level=h)
    gc.collect()
    
    st_ = time.time() 
    print('start clustering', flush=True)
    wish.fit(data)
    gc.collect()

    print('Clustered', time.time() - st_, flush = True)
    #Finished clustering
    #Checking amount of clusters

    
    print('n_clusters =', len(np.unique(wish.labels_)), flush=True)
    gc.collect()

    #if 1 cluster, no need to calc metrics
    if len(np.unique(wish.labels_)) == 1:
        print('one cluster', flush=True)
        return data, wish

    #if 2 clusters, but one is noise
    if len(np.unique(wish.labels_)) == 2 and (0 in np.unique(wish.labels_)):
        print('one cluster and noise', flush=True)
        return  data, wish

    #if every point is a cluster
    if len(np.unique(wish.labels_)) == len(data):
        print('every point is a cluster', flush=True)
        return data, wish
    
    clean_data = data[wish.labels_ != 0]
    clean_labels = wish.labels_[wish.labels_ != 0]
    print('Deleting noise', len(clean_data), np.min(clean_labels), flush=True)
    #calc metrics
    st_ = time.time() 
    print('start calculating metrics', flush=True)

    scores_dict = get_validation_scores_u(clean_data, 
                                          clean_labels)
    print(scores_dict, time.time()- st_, flush = True)



    #Writing labels
    file_for_labels = open( file_for_labels_path, 'w+')
    file_for_labels.write(' '.join(list(map(str, wish.labels_))))
    gc.collect()
    return data, wish


import pickle


R = 10
n = 3
language = "npy" ##  "txt" or "npy"
type = "random5newlit"
embedding = "SVD"

ks = np.array([10,15])
hs = np.array([0.005,0.05])
fig, axs = plt.subplots(ncols=len(hs), nrows = len(ks), figsize=(20,20))

for i in range(len(ks)):
    #index_row=[]
    for j in range(len(hs)):
        data, wish = main(R, n, language, type, embedding, ks[i], hs[j])
        # Save data and wish using pickle
        with open(f"./data_{ks[i]}_{hs[j]}.pkl", "wb") as f:
            pickle.dump(data, f)
        with open(f"./wish{ks[i]}_{hs[j]}.pkl", "wb") as f:
            pickle.dump(wish, f)
        
        for col in np.unique(wish.labels_):
            if col == 0:
                axs[i][j].scatter(data[wish.labels_ == col][:,0], data[wish.labels_ == col][:,1], color='black')
            else:
                axs[i][j].scatter(data[wish.labels_ == col][:,0], data[wish.labels_ == col][:,1])
        print('k = ' +str(ks[i]) + ' h = ' + str(hs[j]) + ' n_clusters = ' + str(len(np.unique(wish.labels_))))
        axs[i][j].set_title('k =' +str(ks[i]) + ' h = ' + str(hs[j]) + ' n_clusters = ' + str(len(np.unique(wish.labels_))))
        fig.savefig(f"plot_k={ks[i]}_h={hs[j]}.png")