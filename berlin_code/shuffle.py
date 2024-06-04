import pandas as pd
import numpy as np
from . import cluster_analysis
import random
import time


def shuffle_clust_hist(df, n = 1000):
    
    # get real dist 
    pivot_table_real = pd.pivot_table(df, values='size', index=['l6_group'], columns=['post_pt_root_id'], aggfunc=np.sum, fill_value=0)
    log_pivot_table_real = np.log1p(pivot_table_real)
    real_clust_n = cluster_analysis.pull_cluster_sizes(log_pivot_table_real)
    
    # now shuffle x times! 
    hist_df = pd.DataFrame(columns = real_clust_n.keys())
    
    for i in range(n):
        shuffle_test_df = df.copy()
        shuffle_test_df['shuffled_l6_group'] = np.random.permutation(shuffle_test_df['l6_group'].values)

        pivot_table_shuffled = pd.pivot_table(shuffle_test_df, values='size', index=['shuffled_l6_group'], columns=['post_pt_root_id'], aggfunc=np.sum, fill_value=0)

        log_pivot_table_shuffled = np.log1p(pivot_table_shuffled)

        new_line = cluster_analysis.pull_cluster_sizes(log_pivot_table_shuffled)
        hist_df = pd.concat([hist_df,pd.DataFrame([new_line])])
    
    return hist_df

def spatial_shuffle(pre_id, client, max_dist = 5000, compare_num_soma_partners = True, 
                    nuc_df = None, synapse_res = [9.7,9.7,45], max_retries = 5):
    '''
    takes the pre_ids and does a spatial shuffle on all postsyn partners
    postysyn partners will be dropped if they do not belong to a body with a soma
    
    '''
    soma_df = client.materialize.synapse_query(pre_ids = [pre_id]).reset_index(drop = True)
    print(f'number of synapses to shuffle for body {pre_id}: {len(soma_df)}')

    if nuc_df is None:
        nuc_df = client.materialize.query_table('nucleus_detection_v0')
    
    if compare_num_soma_partners:
        len_true_somad_post = len(soma_df[soma_df['post_pt_root_id'].isin(nuc_df['pt_root_id'])])
        print(f'number of true synapses onto a neuron with a soma: {len_true_somad_post}')
    
    # now go through each presyn and get all postsyn partners within max dist
        
    for i in range(len(soma_df)):
        soma_loc = np.vstack(soma_df.loc[i, 'ctr_pt_position'])
        x, y, z = soma_loc[0][0], soma_loc[1][0], soma_loc[2][0]
        bounding_box = [[(x-max_dist/synapse_res[0]), (y-max_dist/synapse_res[1]), (z-max_dist//synapse_res[2])],[(x+max_dist/synapse_res[0]), (y+max_dist//synapse_res[0]), (z+max_dist//synapse_res[2])]]
        # now get all postsyn sites within max_dist
        # this has been failing from the server side so trying to add a fix so that it will run overnight
        retry_count = 0
        success = False
        while retry_count < max_retries and not success:
            try:
                post_syns_in_dist = client.materialize.synapse_query(bounding_box = bounding_box)
                success = True
            except:
                print(f"Attempt {retry_count + 1} for neuron failed")
                retry_count += 1
                time.sleep(30)
        # now pick random post partner from post_syns_in_dist
        new_post = int(random.choice(list(post_syns_in_dist['post_pt_root_id'])))

        # now update the soma_df with the new post_id
        soma_df.loc[i, 'shuffled_post_id'] = new_post
        
        if i % 25 == 0:
            print(f'finished {i/len(soma_df)*100}% of synapses')
    
    if compare_num_soma_partners:
        len_shuffled_soma_post = len(soma_df[soma_df['shuffled_post_id'].isin(nuc_df['pt_root_id'])])
        print(f'number of shuffled synapses onto a neuron with a soma: {len_shuffled_soma_post}')
    
    return soma_df