import pandas as pd
import numpy as np
import random
import time
from . import cluster_analysis

met_dict = {864691132772384381: 'met7', 864691132656139260: 'met7', 864691132679532125: 'met7',
            864691132857128518: 'met7', 864691132752141543: 'met7', 864691132635887345: 'met7',
            864691132679146333: 'met8', 864691132636753137: 'met8', 864691132636263975: 'met8',
            864691132710743877: 'met8', 864691132724827721: 'met6', 864691132607980867: 'met6',
            864691132921234401: 'met6', 864691132737950875: 'met6', 864691132681743407: 'met6',
            864691132685582532: 'met6'}






def shuffle_clust_hist(df, pivot_df, nuc_df_solorids, n = 1000):
    
    # get real dist 
    real_clust_n = cluster_analysis.pull_shuffle_cluster_met_sizes(pivot_df)
    # now shuffle x times! 
    hist_df = pd.DataFrame(columns = real_clust_n.keys())
    
    for i in range(n):
        # make new shuffle of met dict 
        shuffled_dict_values = list(met_dict.values())
        random.shuffle(shuffled_dict_values)
        shuffled_met_dict = dict(zip(met_dict.keys(), shuffled_dict_values))
        df['shuffled_met'] = df['pre_pt_root_id'].map(shuffled_met_dict)
        multi_post_df = df[df['post_pt_root_id'].isin(nuc_df_solorids)]
        # make this into a pivot table 
        df_pivot = multi_post_df.pivot_table(columns = 'post_pt_root_id', index = 'shuffled_met', values = 'size', fill_value = 0)
        #recluster
        new_cluster_sizes = cluster_analysis.pull_shuffle_cluster_met_sizes(df_pivot)
        
        hist_df = pd.concat([hist_df,pd.DataFrame([new_cluster_sizes])])
    hist_df = hist_df.reset_index(drop = True)
    
        
    return hist_df


def spatial_shuffle_query_rid(pre_id, client, nuc_df_solorids = None, max_dist = 2500, 
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


def spatial_shuffle_n_times(shuffle_all_possibilities_df, n_shuffles = 1000):

    # this is supposed to take the shuffled df, then group bby 
    # pre_pt_root_id and then by real synapse id

    grouped_df = shuffle_all_possibilities_df.groupby(['real_synapse_id'])
    # now for each group, go through and select a single row from that group
    # then append that to a list
    all_shuffled_data = []
    for name, group in grouped_df:
        
        group.reset_index(drop = True, inplace = True)
        i = random.choices(range(len(group)), k = n_shuffles)
        all_shuffled_data.append(group.iloc[i].values)

    return pd.DataFrame(np.vstack(all_shuffled_data), columns = shuffle_all_possibilities_df.columns)


def spatial_shuffle_to_single_pivot(spatial_shuffle_n_times_df, nuc_df_solorids, multiple_post_count = 3):
    '''
    takes spatial_shuffle_n_times_df, which has every possible post synaptic partner for each presynaptic partner
    within a spatial cube, and picks one random post synaptic partner for each presynaptic partner. 
    returns the pivot table of those shuffled synapses. 
    '''
    grouped_df = spatial_shuffle_n_times_df.groupby('real_synapse_id')
    all_shuffled_data = []
    
    for name, group in grouped_df:
        # pick one random row in the real_syn group and append 
        all_shuffled_data.append(group.sample(1).values)
    final_df = pd.DataFrame(np.vstack(all_shuffled_data), columns = spatial_shuffle_n_times_df.columns)
    multiple_post = pd.DataFrame(final_df['shuffled_post_id'].value_counts()).reset_index()
    multiple_post = multiple_post[multiple_post['count'] > multiple_post_count]['shuffled_post_id'].values
    
    final_df = final_df[(final_df['shuffled_post_id'].isin(multiple_post)) & (final_df['shuffled_post_id'].isin(nuc_df_solorids))].reset_index(drop = True)

    return final_df.pivot_table(columns = 'shuffled_post_id', index = 'pre_met', values = 'shuffled_size', fill_value = 0)

 
def spatial_shuffle_multi_met_dist(m678_shuffle_all_df, real_pivot, n_shuffles, nuc_df_solorids, multiple_post_count = 3):
    
    spatial_shuffle_n_times_df = spatial_shuffle_n_times(m678_shuffle_all_df)
    real_clust_n = cluster_analysis.pull_shuffle_cluster_met_sizes(real_pivot)
    hist_df = pd.DataFrame(columns = real_clust_n.keys())

    ordered_keys = list(real_clust_n.keys())
    cluster_distributions = []

    for i in range(n_shuffles):
        if i%50==0:
            print(f'finished {i/n_shuffles*100}% of shuffles')
        current_pivot = spatial_shuffle_to_single_pivot(spatial_shuffle_n_times_df, nuc_df_solorids, multiple_post_count = multiple_post_count)
        new_cluster_met_dict = cluster_analysis.pull_shuffle_cluster_met_sizes(current_pivot)
        ordered_values = [new_cluster_met_dict[key] for key in ordered_keys]
        cluster_distributions.append(ordered_values)
    hist_df = pd.DataFrame(cluster_distributions, columns=ordered_keys)
    return hist_df


