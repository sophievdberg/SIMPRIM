"""

Functions related to clustering the journeys.

"""

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
#from sklearn import metrics

from clustering import evaluation
from clustering.center_init import kmeans_plusplus_initializer, class_based_initializer
from pyclustering.cluster.kmedoids import kmedoids


def pairwise_distance_matrix(X, metric='cosine'):
    """
    Finds the pairwise similarity matrix of journeys stored in input matrix X. 
    It is expected that X only contains the required features and weights are 
    yet applied if necessary.
    
    Parameters:
    -----------
        X:        [numpy matrix] A matrix with journey profiles with only features 
                  based on which to calculate similarity, i.e. no std columns like 
                  journey_id. Shape: nr_journeys, nr_features
        metric:   [str] either 'cosine' or 'jaccard'
        
    Returns:
        Pairwise similarity matrix with the shape nr_journeys,nr_journeys 
    """
    if metric=='cosine':
        pairwise_dists = cosine_distances(X)
    elif metric=='jaccard':
        pairwise_dists = squareform(pdist(X, metric='jaccard')) # slower than cosine
    return pairwise_dists


def kMedoids_custom(X, trace_df, k, metric, random_state=4, dist_matrix=None, 
                    medoids_initializer='kmeans++', return_model=True):
    """
    The kMedoids Clustering algorithm used to cluster the journeys. The implementation 
    of pyclustering forms the basis of this function. See 
    https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/kmedoids.py
    for the documentation.
    
http://localhost:8888/edit/Google%20Drive/aEducation/Master/Thesis/Paper/BPIC%202012/clustering/clusterer.py#    Parameters:
    --------------
        X:         [numpy matrix] matrix with journey vectors containing only the features 
                   on which simlarity should be calculated and the weights are yet applied.
        trace_df:  [DataFrame] with journey profile vectors.
        medoids_initializer: 'kmeans++' or 'class_based' that samples one journey from one 
                   class as a starting point (see module center_init for these functions)
    
    """
    # Calculate distance matrix (if not provided)
    if dist_matrix is None:
        pairwise_distances = pairwise_distance_matrix(X, metric=metric)
    else:
        pairwise_distances = dist_matrix 
        
    # Clustering
    if medoids_initializer == 'class_based':
        initial_medoids_idx = class_based_initializer(trace_df, k, random_state, 
                                                      return_index=True)
    else:
        initial_medoids_idx = kmeans_plusplus_initializer(X, k, 
                              **{'random_state':random_state}).initialize(return_index=True)
        
    # model
    model = kmedoids(pairwise_distances, initial_medoids_idx, tolerance=0.00000001, 
                     **{'data_type':'distance_matrix'})
    model.process()

    # results
    medoids_idx = model.get_medoids()
    cluster_idx = model.get_clusters()
    cluster_labels = evaluation.cluster_idx_to_label_list(cluster_idx, len(X))  
    
    if return_model:
        return medoids_idx, cluster_labels, model
    else:
        return medoids_idx, cluster_labels # model is expensive to store




def clustering_quality(X, trace_df, k, true_labels, metric, random_states,
                       medoids_initializer='kmeans++', verbose=True, return_cluster_info=True):
    """"
    The cluster and cluster class labels are returned from the last clustering. 
    This is not necessarily the best one.
    """
    if verbose:
        print('\tCalculating pairwise distance matrix...')
    pairwise_distances = pairwise_distance_matrix(X, metric=metric)
    
    if (true_labels is None) & (medoids_initializer=='class_based'):
        print('Note: for this configuration, there are no labels available.')
        print('The kmeans++ initializer is used instead')
        medoids_initializer = 'kmeans++'
            
    scores = list()       
    for idx, RS in enumerate(random_states):
        if verbose:
            print('\tClustering with Random State:', RS)

        medoids_idx, cluster_labels, trained_model = kMedoids_custom(X, trace_df, k,
                                                        metric, random_state=RS, 
                                                        dist_matrix=pairwise_distances, 
                                                        medoids_initializer=medoids_initializer)
      
        scores = evaluation.cluster_scorer(trained_model, X, 
                                           title = 'PAM (RS={})'.format(RS), 
                                           pred_cluster_labels = cluster_labels, 
                                           true_labels = true_labels, 
                                           centers_ids = medoids_idx, 
                                           metric = metric, # calculate SDBw correclty
                                           score_table = scores)
        if true_labels is None:
            cluster_class_labels = []
        else:  # Calculate Rand Index to have some feeling for cluster accuracy
            # Hungarian assignment of cluster to class
            cluster_class_labels = evaluation.hungarian_cluster_labels(cluster_labels,
                                                                       true_labels, k)
            RI = evaluation.rand_index_score(cluster_class_labels, true_labels) 
            scores[idx].append(RI)
    
    scores.append(evaluation.summarize_statistics(scores, summary_type='avg'))
    scores.append(evaluation.summarize_statistics(scores, summary_type='std'))
    #evaluation.print_scores(scores)

    if return_cluster_info:
        return scores, cluster_class_labels, cluster_labels, medoids_idx
    else:
        return scores