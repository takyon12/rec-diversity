# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:05:07 2019

@author: Katarina_Bedejova
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from copy import deepcopy

DEBUG = True

def get_nparray(series):
    np_array = np.array(literal_eval(series))
    return np_array

def diversify(array_ordercodes, array_embeddings, array_rankings):
    
    #order arrays according to rankings
    ranked_indices = np.argsort(array_rankings)
    ranked_indices = ranked_indices[::-1] #reverse because argsort is ascending
    print(ranked_indices)
    
    array_rankings = array_rankings[ranked_indices]
    array_ordercodes = array_ordercodes[ranked_indices]
    array_embeddings = array_embeddings[ranked_indices]  
    
    df = pd.DataFrame(data = array_embeddings)
    df[0] = df[0].str.replace('[','').str.replace(']','')
    df = df[0].str.split(',', expand = True).add_prefix('coord')   
    df = df.convert_objects(convert_numeric = True)    
    array_embeddings = df.to_numpy()
    
           
    k = 4
    n = array_embeddings.shape[0]
    c = array_embeddings.shape[1]
    
    if DEBUG:
        print(f"clusters: {k}")
        print(f"rows: {n}")
        print(f"features: {c}")
 
    mean = np.mean(array_embeddings, axis = 0)
    std = np.std(array_embeddings, axis = 0)
    centroids = np.random.randn(k,c)*std + mean
    
    print(centroids)
    
    centroids_old = np.zeros(centroids.shape)
    centroids_updated = deepcopy(centroids)
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    
    err = np.linalg.norm(centroids_updated - centroids_old)
    
    while err!=0 :
        for i in range(k):
            distances[:,i] = np.linalg.norm(array_embeddings - centroids[i], axis = 1)
        clusters = np.argmin(distances, axis = 1)
    
        centroids_old = deepcopy(centroids_updated)
        
        for i in range(k):
            centroids_updated[i] = np.mean(array_embeddings[clusters == i], axis = 0)
        err = np.linalg.norm(centroids_updated - centroids_old)
        
    print(clusters)
    print(clusters.shape)
    
    cluster1 = np.where(clusters==0)
    cluster2 = np.where(clusters==1)
    cluster3 = np.where(clusters==2)
    cluster4 = np.where(clusters==3)
    
    n_items=0
    row=0
    diversified_list = []
    
    while n_items!=n:
        try:
            diversified_list.append(cluster1[row])
            diversified_list.append(cluster2[row])
            diversified_list.append(cluster3[row])
            diversified_list.append(cluster4[row])
            n_items= n_items+4
            row=row+1
        except:
            pass
    
    return diversified_list
        

def main():  
    df = pd.read_csv('query-hive.csv')
    
    if DEBUG:
        print("full dataframe loaded")
        print(list(df.columns.values))
        
    #convert string representation of vector to numpy array
    #df['lab.oc_vector'] = df['lab.oc_vector'].apply(get_nparray)
            
    ordercodes = df['lab.ordercode'].to_numpy()
    embeddings = df['lab.oc_vector'].to_numpy()
    rankings = df['lab.revenue'].to_numpy() 
    
    result = diversify(ordercodes, embeddings, rankings)
         
    
if __name__ == "__main__":
    main()

