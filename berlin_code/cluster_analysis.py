import numpy as np
import sklearn as skl

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree


cluster_dict = {('met7', 'met8'):'m6_only', ('met6', 'met8'):'m7_only',('met6', 'met7'):'m8_only',
                  ('met8',):'m67_only', ('met7',):'m68_only', ('met6',):'m78_only', ():'all'}


def pull_shuffle_cluster_met_sizes(pivot_df, return_as_percentage = True, metric = 'sokalsneath', threshold = 0, 
                       cluster_dict = cluster_dict):
    distance_matrix = pdist(pivot_df.T, metric='sokalsneath')
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
        clust_labels_key = tuple(pivot_df.iloc[:,clust_idx][pivot_df.iloc[:,clust_idx] == 0].index)
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
    
    if return_as_percentage:
        total_n_post = sum(clust_count.values())
        clust_count = {k:v/total_n_post for k, v in clust_count.items()}
    return(clust_count)





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
    
    syn_table = syn_table[(syn_table['pre_pt_root_id'].isin(pre_rids))&(syn_table['post_pt_root_id'].isin(post_rids))]
    
    if len(syn_table)>0:
        return np.vstack(syn_table[syn_table_column])*res
    else:
        return []
        
# func that will check what percentage of clusters have at least one actual synapse between the two clusters
def cluster_real_or_close_syn_percentage(pre_clusters, post_clusters, max_close_or_real_syn_dist = 0):
    
    '''
    if max_close_or_real_syn_dist == 0, will return real synapses only 
    if max_close_or_real_syn_dist > 0, then will return non partners that are that distance apart
    
    '''
    
    num_clust_with_syn = 0
    
    for pre_cluster, post_cluster in zip(pre_clusters, post_clusters):
        num_close = sum(skl.metrics.pairwise_distances_argmin_min(np.vstack(pre_cluster), np.vstack(post_cluster))[1] <= max_close_or_real_syn_dist)
        if num_close > 0:
            num_clust_with_syn += 1
            continue
            
    return(num_clust_with_syn/ len(pre_clusters)*100)

def calculate_synapse_odds(df, synapse_id_col, post_pt_root_id_col):
    # Group by the synapse_id column and count occurrences of post_pt_root_id within each group
    count_per_group = df.groupby([synapse_id_col, post_pt_root_id_col]).size().reset_index(name='post_count_in_cube')
    
    # Merge the counts back to the original DataFrame
    df = df.merge(count_per_group, on=[synapse_id_col, post_pt_root_id_col], how='left')
    
    # Calculate the total count for each synapse_id group
    total_per_group = df.groupby(synapse_id_col)[synapse_id_col].transform('size')
    df['total_syn_per_cube'] = total_per_group
    
    # Calculate the odds and add as a new column
    df['odds'] = df['post_count_in_cube'] / total_per_group
    
    return df