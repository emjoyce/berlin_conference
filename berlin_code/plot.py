import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import sklearn as skl
from . import cluster_analysis





def adjust_color_lightness(color, amount=.5):
    """ Adjust the brightness of the given color """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_clusters(pre_clusters, post_clusters, ax = None, min_points=2, post_alpha = .3, pre_alpha = 1, 
                 title = 'Pre and Post Synaptic Clusters', x = 'x', y = 'y',plot_close_or_real = True,
                  max_close_or_real_syn_dist = 0):
    '''Plots all pre and post clusters on a single graph, with matching colors for each pair
    when max_close_or_real_syn_dist == 0, they are real synapses 
    '''
    
    # Define a cycle of colors
    if ax is None:
        ax = plt.gca()
    
    ax_dict = {'x':0, 'y':1, 'z':2}
    colors = cycle(plt.cm.Dark2.colors)  # Color cycle from a colormap

    for pre_cluster, post_cluster in zip(pre_clusters, post_clusters):
        if len(pre_cluster) >= min_points or len(post_cluster) >= min_points:
            # Get the next color from the cycle
            base_color = next(colors)

            # Adjust the color for post-synaptic points
            adjusted_color = adjust_color_lightness(base_color, amount=1.5)

            # Unpacking the points
            pre_x, pre_y = np.vstack(pre_cluster)[:,ax_dict[x]], np.vstack(pre_cluster)[:,ax_dict[y]]
            post_x, post_y, = np.vstack(post_cluster)[:,ax_dict[x]], np.vstack(post_cluster)[:,ax_dict[y]]

            # Plotting the pre_cluster points
            ax.scatter(pre_x, pre_y, color=base_color, label='Pre-synaptic', alpha = pre_alpha)

            # Plotting the post_cluster points
            ax.scatter(post_x, post_y, color=adjusted_color, label='Post-synaptic', alpha = post_alpha)
            
    # plot the real or very close synapses
    if plot_close_or_real:
        # pull out the real_synapses that are less than max_dist away
        close_real_syn_mask = skl.metrics.pairwise_distances_argmin_min(np.vstack(pre_clusters), np.vstack(post_clusters))[1] <= max_close_or_real_syn_dist
        if sum(close_real_syn_mask) > 0:
            close_real_synapses = np.vstack(pre_clusters)[close_real_syn_mask]
            close_real_x, close_real_y = np.vstack(close_real_synapses)[:,ax_dict[x]], np.vstack(close_real_synapses)[:,ax_dict[y]]
            if max_close_or_real_syn_dist == 0:
                label = 'real syanpses'
            else:
                label = f'synapses under {max_close_or_real_syn_dist/100} microns away'
            ax.scatter(close_real_x, close_real_y, color='black', label=label, alpha = 1, s = 5)


    # Configure and show the plot
    ax.set_xlabel(f'{x} coordinate')
    ax.set_ylabel(f'{y} coordinate')
    ax.set_title(title)
    ax.invert_yaxis()
    #ax.legend()


def calculate_and_plot_clusters(pre_points, post_points, ax, pre_met_type, post_met_type,
                                max_cluster_dist = 2500, min_points=2, post_alpha=0.3, pre_alpha=1, 
                                x='x',y='y',plot_close_or_real=True, max_close_or_real_syn_dist=0
                               ):
    
    
    pre_clusters, post_clusters = cluster_analysis.calculate_between_clusters(pre_points, post_points, max_dist = max_cluster_dist)
    
    real_perc = cluster_analysis.cluster_real_or_close_syn_percentage(pre_clusters, post_clusters)
    
    title = f'met {pre_met_type} and met {post_met_type} top targeters:\n {round(real_perc, 2)}% of clusters with real synapses out of {len(pre_clusters)} clusters'
    
    plot_clusters(pre_clusters, post_clusters, min_points=min_points, title = title, ax = ax,
                 post_alpha=post_alpha, pre_alpha=pre_alpha, x=x,y=y,plot_close_or_real=plot_close_or_real,
                  max_close_or_real_syn_dist=max_close_or_real_syn_dist,)