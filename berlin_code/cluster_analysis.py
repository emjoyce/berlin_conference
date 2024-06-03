import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree




def pull_cluster_sizes(df, metric = 'sokalsneath', threshold = 0,
         cluster_dict = {('met7', 'met8'):'m6_only', ('met6', 'met8'):'m7_only',('met6', 'met7'):'m8_only',
           ('met8',):'m67_only', ('met7',):'m68_only', ('met6',):'m78_only', ():'all'}):
    distance_matrix = pdist(df.T, metric=metric)
    # hierarchical clustering
    Z = hierarchy.linkage(distance_matrix, method='average')
    
    cluster_labels = hierarchy.fcluster(Z, threshold, criterion='distance')
    # I want this to return the number of each cluster
    
    val, count = np.unique(cluster_labels, return_counts=True)
    
    clust_count = {}
    for clust_num, clust_num_count in list(zip(val, count)):
        
        # get the first index of that array 
        clust_idx = np.where(cluster_labels==clust_num)[0][0]
        
        # now get the met types with 0 connections here
        clust_labels_key = tuple(df.iloc[:,clust_idx][df.iloc[:,clust_idx] == 0].index)
        met_clust = cluster_dict[clust_labels_key]
        
        clust_count[met_clust] = clust_num_count

    # check for met clusters that are not present in clust_count and add them with 0 as value
    
    missing_clusts = set(clust_count.keys()) ^ set(cluster_dict.values())
    if len(missing_clusts) > 0:
        for(met_clust) in missing_clusts:
            clust_count[met_clust] = 0
    
    return(clust_count)


def shuffle_clust_hist(df, n = 1000):
    
    # get real dist 
    pivot_table_real = pd.pivot_table(df, values='size', index=['l6_group'], columns=['post_pt_root_id'], aggfunc=np.sum, fill_value=0)
    log_pivot_table_real = np.log1p(pivot_table_real)
    real_clust_n = pull_cluster_sizes(log_pivot_table_real)
    
    # now shuffle x times! 
    hist_df = pd.DataFrame(columns = real_clust_n.keys())
    
    for i in range(n):
        shuffle_test_df = df.copy()
        shuffle_test_df['shuffled_l6_group'] = np.random.permutation(shuffle_test_df['l6_group'].values)

        pivot_table_shuffled = pd.pivot_table(shuffle_test_df, values='size', index=['shuffled_l6_group'], columns=['post_pt_root_id'], aggfunc=np.sum, fill_value=0)

        log_pivot_table_shuffled = np.log1p(pivot_table_shuffled)

        new_line = pull_cluster_sizes(log_pivot_table_shuffled)
        hist_df = pd.concat([hist_df,pd.DataFrame([new_line])])
    
    return hist_df



def calculate_between_clusters(pre_points, post_points, max_dist = 2500):
    '''finds clusters of presyn points that are max_dist away from any given postsyn points'''
    
    # pre and post array clusters where index matches them 
    pre_cluster = []
    post_cluster = []
    
    # Create a KDTree for efficient nearest neighbor searches
    tree = KDTree(post_points)
    
    for pre_ind, pre_pt in enumerate(pre_points):
        # Check for points that are less than max_dist apart
        # Get the index of the post_pt that this pre_pt is less than max_dist apart from 
        post_inds = tree.query_radius([pre_pt], r=max_dist)[0]
        
        # If there are no close post points, continue to the next iteration
        if len(post_inds) == 0:
            continue
        
        # Check if any of the post_inds show up in post_cluster
        # This finds indices in post_cluster that have overlapping post_inds
        inds_to_remove = [sub_ind for sub_ind, sublist in enumerate(post_cluster) if not set(sublist).isdisjoint(post_inds)]

        # If there are no overlaps, create a new cluster
        if len(inds_to_remove) == 0:
            pre_cluster.append([pre_ind])
            post_cluster.append(list(post_inds))
            continue
        
        # Merge the pre inds and post inds into a single list that will be a cluster
        # First, create a new_pre_cluster that will have all the pre_inds for the current pre_ind 
        # but also from all the inds_to_remove indices
        new_pre_cluster = [pre_ind] + [item for i in inds_to_remove for item in pre_cluster[i]]
        
        # Same for new_post_cluster
        new_post_cluster = list(post_inds) + [item for i in inds_to_remove for item in post_cluster[i]]
        
        # Now remove the clusters that were merged
        # Sorting inds_to_remove in reverse order so popping doesn't mess up the indices
        inds_to_remove.sort(reverse=True)
        for ind in inds_to_remove:
            pre_cluster.pop(ind)
            post_cluster.pop(ind)
        
        # Finally, add the new merged clusters
        pre_cluster.append(new_pre_cluster)
        post_cluster.append(new_post_cluster)
        
    pre_cluster_points = [[pre_points[i] for i in cluster] for cluster in pre_cluster]
    post_cluster_points = [[post_points[i] for i in cluster] for cluster in post_cluster]

    return pre_cluster_points, post_cluster_points



def pull_real_syns(pre_rids, post_rids, syn_table, res = [9.7, 9.7, 45], syn_table_column = 'pre_pt_position'):
    
    '''
    syn table needs to have pre and post partners with post partners only having a soma and self contacts removed
    '''
    syn_table = syn_table[(syn_table['pre_pt_root_id'].isin(pre_rids))&(syn_table['post_pt_root_id'].isin(post_rids))]
    
    if len(syn_table)>0:
        return np.vstack(syn_table[syn_table_column])*res
    else:
        return []
        
# func that will check what percentage of clusters have at least one actual synapse between the two clusters
def cluster_real_syn_percentage(cluster_syns, real_syns):
    matches = 0 
    
    for cluster_syn in cluster_syns:
        found_match = False
        for cluster_syn_loc in cluster_syn:
            
            for real_syn_loc in real_syns:
                if np.array_equal(cluster_syn_loc, real_syn_loc):
                    found_match = True
                    break
            if found_match:
                break
        
        if found_match:
            matches+=1
    
    return (matches / len(cluster_syns))*100