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
diversified_list = []
bulharska_konstanta = 0.10

def add_to_list(diversified_list, revenue, clust_nr, file = None):
        
    print(f"{float(revenue)},{clust_nr}")    
    
    if not diversified_list:
        pass
    else:
        last = diversified_list[-1]
    
    if (float(last)/float(revenue)) > 15:
        print("gap too big")
    else:
        file.write(f"{revenue},{clust_nr}\n")
        print("gap ok")
        diversified_list.append(float(revenue))
        
    

def get_nparray(series):
    np_array = np.array(literal_eval(series))
    return np_array



def diversify(array_ordercodes, array_embeddings, array_rankings, array_fname = None, array_bname = None):
    
    #order arrays according to rankings
    ranked_indices = np.argsort(array_rankings)
    ranked_indices = ranked_indices[::-1] #reverse because argsort is ascending
    print(ranked_indices)
    
    array_rankings = array_rankings[ranked_indices]
    array_ordercodes = array_ordercodes[ranked_indices]
    array_embeddings = array_embeddings[ranked_indices] 
    array_fname = array_fname[ranked_indices]
    array_bname = array_bname[ranked_indices]  
    
    #check
    with open('RANKINGS.csv', 'w') as f:
        for item in array_rankings:
            f.write("%s\n" % item)
    
    
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
    centroids[0] = array_embeddings[0]
    centroids[1] = array_embeddings[1]
    centroids[2] = array_embeddings[2]
    centroids[3] = array_embeddings[3]
       
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
       
    
    cluster1 = np.where(clusters==0) #indexy 
    cluster2 = np.where(clusters==1)
    cluster3 = np.where(clusters==2)
    cluster4 = np.where(clusters==3)
    
    clust1_ordercodes = array_ordercodes[cluster1]
    clust1_bnames = array_bname[cluster1]
    clust1_fnames = array_fname[cluster1]
    clust1_rev = array_rankings[cluster1]
    
    clust2_ordercodes = array_ordercodes[cluster2]
    clust2_bnames = array_bname[cluster2]
    clust2_fnames = array_fname[cluster2]
    clust2_rev = array_rankings[cluster2]
    
    clust3_ordercodes = array_ordercodes[cluster3]
    clust3_bnames = array_bname[cluster3]
    clust3_fnames = array_fname[cluster3]
    clust3_rev = array_rankings[cluster3]
    
    clust4_ordercodes = array_ordercodes[cluster4]
    clust4_bnames = array_bname[cluster4]
    clust4_fnames = array_fname[cluster4]
    clust4_rev = array_rankings[cluster4]
    
       
    f = open('diversified_list.csv','w')
    
    diversified_list.append(clust1_rev[0])
    
    for i in range(1,n):
        try:
            pass
            add_to_list(diversified_list, clust1_rev[i], 1, f)
            add_to_list(diversified_list, clust2_rev[i], 2, f)   
            add_to_list(diversified_list, clust3_rev[i], 3, f)   
            add_to_list(diversified_list, clust4_rev[i], 4, f)               
#            print(f"{clust1_ordercodes[i]} {clust1_bnames[i]} {clust1_fnames[i]}")
#            f.write(f"{clust1_ordercodes[i]},{clust1_bnames[i]},{clust1_fnames[i]},{clust1_rev[i]},CLUSTER: 1\n")
#            print(f"{clust2_ordercodes[i]} {clust2_bnames[i]} {clust2_fnames[i]}")
#            f.write(f"{clust2_ordercodes[i]},{clust2_bnames[i]},{clust2_fnames[i]},{clust2_rev[i]},CLUSTER: 2\n")
#            print(f"{clust3_ordercodes[i]} {clust3_bnames[i]} {clust3_fnames[i]}")
#            f.write(f"{clust3_ordercodes[i]},{clust3_bnames[i]},{clust3_fnames[i]},{clust3_rev[i]},CLUSTER: 3\n")
#            print(f"{clust4_ordercodes[i]} {clust4_bnames[i]} {clust4_fnames[i]}")
#            f.write(f"{clust4_ordercodes[i]},{clust4_bnames[i]},{clust4_fnames[i]},{clust4_rev[i]},CLUSTER: 4\n")
        except:
            pass   
    
    f.close()
    
#    #return diversified_list
        

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
    brandname = df['run.brandname'].to_numpy() 
    familyname = df['run.familyname'].to_numpy() 
    
    #result = 
    diversify(ordercodes, embeddings, rankings, familyname, brandname)
         
    
if __name__ == "__main__":
    main()

