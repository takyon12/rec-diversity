# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:00:51 2019

@author: PC
"""

from math import sqrt, floor
import scipy.spatial.distance as metric
import numpy as np
import pandas as pd
from ast import literal_eval

DEBUG = True
diversified_list = []

def add_to_divlist(item):
    
    if diversified_list:
        last = float(diversified_list[-1])
        print(f"LAST ITEM IS : {last}")
        print(f"ITEM TO BE ADDES IS {item}")
        x = float(float(item)/last)
        print(f"DIVISION IS : {x}")
        if x <= 0.10:
            print("OMITTED...")
            pass
        else:
            print("ADDED....")
            diversified_list.append(item)
    else:
        print("FIRST ITEM IN THE LIST...")
        diversified_list.append(item)
        
        
def get_nparray(series):
    np_array = np.array(literal_eval(series))
    return np_array


def cluster(ds, k):

    # Number of rows in dataset
    m = np.shape(ds)[0]
    print("m:")
    print(m)

    # Hold the instance cluster assignments
    cluster_assignments = np.mat(np.zeros((m, 1)))
    print("cluster_assignments:")
    print(cluster_assignments)

    # Initialize centroids
    cents = initialize(ds, k)
    print("cents:")
    print(cents)
    
    # Preserve original centroids
    cents_orig = cents.copy()
    print("cents_orig:")
    print(cents_orig)
    
    changed = True
    num_iter = 0

    #cluster_assignments[0, :] = 0.0
    
    # Loop until no changes to cluster assignments
    while changed:

        changed = False

        # For every instance (row in dataset)
        for i in range(m):

            # Track minimum distance, and vector index of associated cluster
            min_dist = np.inf
            min_index = -1

            # Calculate distances
            for j in range(k):

#                dist_ji = euclidean_dist(cents[j,:], ds[i,:])
                dist_ji = euclidean_dist(cents[j], ds[i])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j

            # Check if cluster assignment of instance has changed
            if cluster_assignments[i, 0] != min_index: 
                changed = True

            # Assign instance to appropriate cluster
            cluster_assignments[i, :] = min_index#, min_dist**2

        # Update centroid location
        for cent in range(k):
            points = ds[np.nonzero(cluster_assignments[:,0].A==cent)[0]]
            cents[cent,:] = np.mean(points, axis=0)

        # Count iterations
        num_iter += 1

    # Return important stuff when done
    return cents, cluster_assignments, num_iter, cents_orig

def euclidean_dist(A, B):
   
#    return metric.euclidean(A, B)
    return np.linalg.norm(A - B, axis = 1)

def initialize(ds, k):

    # Number of attributes in dataset
    n = np.shape(ds)[1]
    
    # The centroids
    centroids = np.mat(np.zeros((k,n)))

    # Create random centroids (get min, max attribute values, randomize in that range)
    for j in range(n):
        min_j = min(ds[:,j])
        range_j = float(max(ds[:,j]) - min_j)
        centroids[:,j] = min_j + range_j * np.random.rand(k, 1)

    centroids[0,:] = ds[0,:]
    # Return centroids as numpy array
    return centroids

def main():
    
    df = pd.read_csv('query-hive-697613.csv')
    
    if DEBUG:
        print("full dataframe loaded")
        print(list(df.columns.values))
        
    #convert string representation of vector to numpy array
    #df['lab.oc_vector'] = df['lab.oc_vector'].apply(get_nparray)
            
    ordercodes = df['lab.ordercode'].values
    embeddings = df['lab.oc_vector'].values
    rankings = df['lab.revenue'].values
    brandname = df['run.brandname'].values
    familyname = df['run.familyname'].values
    
   
    ranked_indices = np.argsort(rankings)
    ranked_indices = ranked_indices[::-1] #reverse because argsort is ascending
    
    rankings = rankings[ranked_indices]
    ordercodes = ordercodes[ranked_indices]
    embeddings = embeddings[ranked_indices] 
    familyname = familyname[ranked_indices]
    brandname = brandname[ranked_indices] 
    
    #from now all arrays are sorted according to revenue
    
        
    df = pd.DataFrame(data = embeddings)
    df[0] = df[0].str.replace('[','').str.replace(']','')
    df = df[0].str.split(',', expand = True).add_prefix('coord')   
    df = df.convert_objects(convert_numeric = True)    
    embeddings = df.values

    print("embeddings shape")
    print(embeddings.shape)
    
    centroids, cluster_assignments, iters, orig_centroids = cluster(embeddings, 3)
    
    A = np.squeeze(np.asarray(cluster_assignments))
    
    print("cluster assignments:")
    print(A)
    
    rev_01 = rankings[np.where(A == 0.0)]
    rev_02 = rankings[np.where(A == 1.0)]
    rev_03 = rankings[np.where(A == 2.0)]
    
    
    with open('results', 'w') as f:
        for i in range(np.shape(A)[0]):
            f.write(f"{A[i]}, {rankings[i]}\n")
#    
    # Output results
    print('Number of iterations:', iters)
    print('\nFinal centroids:\n', centroids)
    print('\nCluster membership and error of first 10 instances:\n', cluster_assignments[:10])
    print('\nOriginal centroids:\n', orig_centroids)
    
    f = open('diversified_list.csv','w')

    
    for i in range(np.shape(A)[0]):
        try:   
            add_to_divlist(rev_01[i])
            add_to_divlist(rev_02[i])
            add_to_divlist(rev_03[i])
            print(f"{rev_01[i]}")
            f.write(f"{rev_01[i]},CLUSTER: 1\n")
            print(f"{rev_02[i]}")
            f.write(f"{rev_02[i]},CLUSTER: 2\n")
            print(f"{rev_03[i]}")
            f.write(f"{rev_03[i]},CLUSTER: 3\n")
        except:
            pass   
    
    f.close()
    
if __name__ == "__main__":
    main()
