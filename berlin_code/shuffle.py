import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import random
from scipy.spatial.distance import pdist

met_dict = {864691132772384381: 'met7', 864691132656139260: 'met7', 864691132679532125: 'met7',
            864691132857128518: 'met7', 864691132752141543: 'met7', 864691132635887345: 'met7',
            864691132679146333: 'met8', 864691132636753137: 'met8', 864691132636263975: 'met8',
            864691132710743877: 'met8', 864691132724827721: 'met6', 864691132607980867: 'met6',
            864691132921234401: 'met6', 864691132737950875: 'met6', 864691132681743407: 'met6',
            864691132685582532: 'met6'}

cluster_dict = {('met7', 'met8'):'m6_only', ('met6', 'met8'):'m7_only',('met6', 'met7'):'m8_only',
                  ('met8',):'m67_only', ('met7',):'m68_only', ('met6',):'m78_only', ():'all'}


def pull_cluster_sizes(df, metric = 'sokalsneath', threshold = 0, cluster_dict = cluster_dict):
    distance_matrix = pdist(df.T, metric='sokalsneath')
    # hierarchical clustering
    Z = hierarchy.linkage(distance_matrix, method='average')
    
    cluster_labels = hierarchy.fcluster(Z, threshold, criterion='distance')
    #print(cluster_labels)
    # I wnat this to return the number of each cluster
    
    val, count = np.unique(cluster_labels, return_counts=True)
    clust_count = {}
    
    for clust_num, clust_num_count in list(zip(val, count)):
        #print(clust_num, clust_num_count)
        # get the first index of that array 
        
        clust_idx = np.where(cluster_labels==clust_num)[0][0]
        #print(f'clust_num:{clust_num}, clust_idx:{clust_idx}')
        # now get the met types with 0 connections here - should give keys to cluster_dict
        clust_labels_key = tuple(df.iloc[:,clust_idx][df.iloc[:,clust_idx] == 0].index)
        #print(f'clust_labels_key:{clust_labels_key}')
        met_clust = cluster_dict[clust_labels_key]
        #print(clust_labels_key, met_clust)
        #print(met_clust, clust_num_count)
        clust_count[met_clust] = clust_num_count

    # check for met clusters that are not present in clust_count and add them with 0 as value
    
    missing_clusts = set(clust_count.keys()) ^ set(cluster_dict.values())
    if len(missing_clusts) > 0:
        for(met_clust) in missing_clusts:
            clust_count[met_clust] = 0
    
    return(clust_count)


def shuffle_clust_hist(df, pivot_df, n = 1000):
    
    # get real dist 
    real_clust_n = pull_cluster_sizes(pivot_df)
    # now shuffle x times! 
    hist_df = pd.DataFrame(columns = real_clust_n.keys())
    
    for i in range(n):
        shuffle_test_df = df.copy()
        # make new shuffle of met dict 
        shuffled_dict_values = list(met_dict.values())
        random.shuffle(shuffled_dict_values)
        shuffled_met_dict = dict(zip(met_dict.keys(), shuffled_dict_values))
        df['shuffled_met'] = df['pre_pt_root_id'].map(shuffled_met_dict)
        # make this into a pivot table 
        df_pivot = df.pivot_table(columns = 'post_pt_root_id', index = 'shuffled_met', values = 'size', fill_value = 0)
        #recluster
        new_cluster_sizes = pull_cluster_sizes(df_pivot)
        hist_df = pd.concat([hist_df,pd.DataFrame([new_cluster_sizes])])
    
    return hist_df