import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import random
from scipy.spatial.distance import pdist
import time

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


def shuffle_clust_hist(df, pivot_df, nuc_df_solorids = None, n = 1000):

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
        multi_post_df = df[df['post_pt_root_id'].isin(nuc_df_solorids)]
        # make this into a pivot table 
        df_pivot = df.pivot_table(columns = 'post_pt_root_id', index = 'shuffled_met', values = 'size', fill_value = 0)
        #recluster
        new_cluster_sizes = pull_cluster_sizes(df_pivot)
        hist_df = pd.concat([hist_df,pd.DataFrame([new_cluster_sizes])])
    
    return hist_df

def single_spatial_shuffle(pre_id, client, nuc_df_solorids = None, max_dist = 2500, 
                           compare_num_soma_partners = True, max_retries = 5, 
                           record_all = True, 
                   ):
    '''
    takes the pre_ids and does a spatial shuffle on all postsyn partners
    postysyn partners will be dropped if they do not belong to a body with a soma
    
    '''
    soma_df = client.materialize.synapse_query(pre_ids = [pre_id]).reset_index(drop = True)
    synapse_res = soma_df.attrs['dataframe_resolution']
    
    print(f'number of synapses to shuffle for body {pre_id}: {len(soma_df)}')

    if nuc_df_solorids is None:
        nuc_df = client.materialize.query_table('nucleus_detection_v0')
        nuc_df_solorids = pd.DataFrame(nuc_df['pt_root_id'].value_counts())
        nuc_df_solorids= list(nuc_df_solorids[nuc_df_solorids['count'] == 1].index)
    
    if compare_num_soma_partners:
        len_true_somad_post = len(soma_df[soma_df['post_pt_root_id'].isin(nuc_df_solorids)])
        print(f'number of true synapses onto a neuron with a soma: {len_true_somad_post}')
    
    # now go through each presyn and get all postsyn partners within max dist
    if not record_all:
        spatial_shuffled_rids = []
        sizes = []
    if record_all:
        all_shuffled_data = []
        
    for i in range(len(soma_df)):
        syn_loc = soma_df.loc[i, 'ctr_pt_position']

        
        x, y, z = syn_loc[0], syn_loc[1], syn_loc[2]
        bounding_box = [[(x-max_dist/synapse_res[0]), (y-max_dist/synapse_res[1]), (z-max_dist//synapse_res[2])],[(x+max_dist/synapse_res[0]), (y+max_dist//synapse_res[0]), (z+max_dist//synapse_res[2])]]
        # now get all postsyn sites within max_dist
        # this has been failing from the server side so trying to add a fix so that it will run overnight
        retry_count = 0
        success = False
        while retry_count < max_retries and not success:
            try:
                post_syns_in_dist = client.materialize.synapse_query(bounding_box = bounding_box).reset_index(drop = True)
                success = True
            except:
                print(f"Attempt {retry_count + 1} for location {x, y, z} on neuron {pre_id} failed")
                #client = caveclient.CAVEclient('v1dd')
                retry_count += 1
                time.sleep(10)
        
        if record_all:
            
            real_synapse_id = soma_df.loc[i, 'id']
            pre_id = soma_df.loc[i, 'pre_pt_root_id']
            real_post_id = soma_df.loc[i, 'post_pt_root_id']
            real_size = soma_df.loc[i, 'size']
            
            shuffled_synapse_ids = post_syns_in_dist['id'].values
            shuffled_post_ids = post_syns_in_dist['post_pt_root_id'].values
            shuffled_sizes = post_syns_in_dist['size'].values
            
            data_length = len(post_syns_in_dist)
            
            all_shuffled_data.append(pd.DataFrame({
                'real_synapse_id': np.full(data_length, real_synapse_id),
                'pre_root_id': np.full(data_length, pre_id),
                'real_post_pt_root_id': np.full(data_length, real_post_id),
                'ctr_pt_position': [syn_loc] * data_length,
                'real_size': np.full(data_length, real_size),
                'shuffled_synapse_id': shuffled_synapse_ids,
                'shuffled_post_id': shuffled_post_ids,
                'shuffled_size': shuffled_sizes
                    }))
            
        
        if not record_all:
            # now pick random post partner from post_syns_in_dist
            ind = random.choice(range(len(post_syns_in_dist)))
            new_post = post_syns_in_dist.loc[ind, 'post_pt_root_id']
            new_size = post_syns_in_dist.loc[ind, 'size']

            # now update the soma_df with the new post_id
            spatial_shuffled_rids+=[new_post]
            sizes += [new_size]
        
        if i % 25 == 0:
            print(f'finished {i/len(soma_df)*100}% of synapses')
#     if compare_num_soma_partners:
#         len_shuffled_soma_post = len(soma_df[soma_df['shuffled_post_id'].isin(nuc_df['pt_root_id'])])
#         print(f'number of shuffled synapses onto a neuron with a soma: {len_shuffled_soma_post}')
        
        
    if not record_all:
        soma_df['shuffled_post_id'] = spatial_shuffled_rids
        soma_df['shuffled_syn_size'] = sizes
        
    if record_all:
        spatial_shuffled_df = pd.concat(all_shuffled_data, ignore_index=True)
        spatial_shuffled_df['pre_root_soma'] = spatial_shuffled_df['pre_root_id'].isin(nuc_df_solorids)
        spatial_shuffled_df['real_post_root_soma'] = spatial_shuffled_df['real_post_pt_root_id'].isin(nuc_df_solorids)
        spatial_shuffled_df['shuffled_post_id_soma'] = spatial_shuffled_df['shuffled_post_id'].isin(nuc_df_solorids)
        return spatial_shuffled_df
        

    return soma_df



def spatial_shuffle_multiple(shuffle_all_df, n_shuffles = 1000):

    # this is supposed to take the shuffled df, then group bby 
    # pre_pt_root_id and then by real synapse id

    grouped_df = shuffle_all_df.groupby(['real_synapse_id'])
    # now for each group, go through and select a single row from that group
    # then append that to a list
    all_shuffled_data = []
    for name, group in grouped_df:
        
        group.reset_index(drop = True, inplace = True)
        i = random.choices(range(len(group)), k = n_shuffles)
        all_shuffled_data.append(group.iloc[i].values)

    return pd.DataFrame(np.vstack(all_shuffled_data), columns = shuffle_all_df.columns)

