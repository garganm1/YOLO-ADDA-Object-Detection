import numpy as np
import glob
from utils.utils import bbox_iou 
import torch
import random


# print(cluster_centroid.shape)
# input()
def k_means(dataset, k):
    '''
    Function which takes dataset and number of clusters as
    arguments and returns centroids for different clusters and cluster
    assignment for all the samples of dataset
    Arguments:
    dataset: input dataset
    k: number of clusters
    Returns:
    cluster_centroid: centroid of different clusters
    custer_assignment: cluster assignment for each sample in the dataset
    '''
    n_samples = dataset.shape[0]
    d = dataset.shape[1]

    # Assigning cluster centroids as first k points in the dataset
    # cluster_centroid = dataset[:k,:]
    # print(list(range(dataset.shape[0])))
    # input()
    l = random.choices(list(range(dataset.shape[0])), k=k)
    cluster_centroid = dataset[l,:]
    # print(cluster_centroid.shape)
    
    cluster_centroid = torch.FloatTensor(np.concatenate((np.zeros((len(cluster_centroid), 2)), np.array(cluster_centroid)), 1)) 
    
    # Creating array of shape n_sample*1 to store cluster assignment for each samples
    cluster_assignment = np.zeros((n_samples, 1))
    epochs = 1000

    for e in range(epochs):
    # Assigning cluster to each sample of dataset
        for i in range(n_samples):
            
            gw = dataset[i,:][0]
            gh = dataset[i,:][1]
            point = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            iou = bbox_iou(point, cluster_centroid)
            cluster_assignment[i] = 1 - torch.argmin(iou)# Updating cluster centroids
        for j in range(k):
            points = dataset[list((cluster_assignment==j).squeeze(1)),:]
            print(points)
            input()
            cluster_cen = np.mean(points, axis = 0).reshape(1,-1)
            print(cluster_cen)
            input()
            cluster_cen = torch.FloatTensor(np.concatenate((np.zeros((len(cluster_cen), 2)), np.array(cluster_cen)), 1)) 
            cluster_centroid[[j],:] = cluster_cen
        print(cluster_centroid)
    return(cluster_centroid, cluster_assignment)




label_files = glob.glob('./data/labels_for_clustering/*.txt')
X = None
for file in label_files:
    with open(file, 'r') as f:
        l = f.readline().strip().split(' ')
        h = float(l[3])
        w = float(l[4])
        
        if X is None:
            X = np.array([[h,w]])
        else:
            X = np.vstack((X, np.array([h,w])))


# X_13 = X*13
# X_26 = X*26
# X_52 = X*52
X_416 = X*416
c, d = k_means(X_416, 9)
print(c)