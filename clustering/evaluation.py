"""

All general cluster evaluation functions.

"""

import pandas as pd
import numpy as np
from collections import Counter
from tabulate import tabulate

from scipy.special import comb
from scipy.optimize import linear_sum_assignment

from sklearn import metrics
from s_dbw import S_Dbw # check the documentation for the settings: https://pypi.org/project/s-dbw/
# Other S_Dbw implementation: https://github.com/fanfanda/S_Dbw/blob/master/S_Dbw.py

from clustering.center_init import kmeans_plusplus_initializer
from pyclustering.cluster.kmedoids import kmedoids
 

### SCORING OF CLUSTER OUTPUT USING INTERNAL AND EXTERNAL VALIDITY MEASURES
    
def cluster_scorer(model, X, title, pred_cluster_labels, true_labels=None, 
                   pred_class_labels=None, centers_ids=None, metric='cosine', 
                   score_table=[]):
    """
    Calculates unsupervised and supervised cluster evaluation indices (SDbw & AMI) 
    and stores adds these scores to a score table that can be printed with the 
    function print_scores().
    
    Parameters:
    -----------
        model:    trained kMedoids model as a result from the kMedoids_custom() 
                  function in the clustering.clusterer module
        X:        [numpy array] journey profiles 
        title:    [str] name of the model that you trained
        pred_cluster_labels: [list] predicted cluster labels for journeys in X 
        true_labels: [list] original labels for journeys in X. 
                     If None, no supervised metric is calculated.
        pred_labels: [list of len(X)] if pred_cluster_labels not specified, 
                     labels are retrieved from the model object. 
                     Since this is only allowed for scikit-learn model objects, 
                     the predicted labels can also be specified seperately.
        centers_ids: [list] medoid indices that resulted from the clustering algorithm
        metric:      [str] using 'cosine' or 'jaccard' to calculate similarity
        score_table: [list of lists] if provided, scores are added to this table. 
                     If not provided, a new table is created

    Returns:
        A list of lists with scores that can be printed with the print_scores() function
    
    """
    
    if len(title) > 18: # formatting
        title = title[:18]
        
    if not pred_class_labels:
        pred_class_labels = pred_cluster_labels
    
    score_list = [title]
    if not true_labels is None:
        score_list.append(metrics.homogeneity_score(true_labels, pred_class_labels))
        score_list.append(metrics.adjusted_mutual_info_score(true_labels, pred_class_labels, 
                                                             average_method='arithmetic'))
    score_list.append(S_Dbw(X, pred_cluster_labels, centers_id=centers_ids, method='Halkidi', 
                            alg_noise='bind', metric=metric, centr='median')) 
    
    score_table.append(score_list) 
    return score_table


def print_scores(score_table, extended=False):
    """
    Prints the score table in a nice, readable format.
    """
    metric_list = ['Homogen.', 'AMI', 'S_Dbw']  
    metric_list_extended = ['Homogen.', 'AMI', 'V-Measure', 'NMI', 'ARI', 
                            'S_Dbw', 'Silhouette', 'CH']  

    if extended:
        headers = metric_list_extended
    elif len(score_table[0]) == 5:
        headers = metric_list + ['RandIndex']     
    elif len(score_table[0]) == 2: 
        headers = metric_list[2:]
    else:
        headers = metric_list  
        
    print(tabulate(score_table, headers, tablefmt="fancy_grid"))
    
    
    
def summarize_statistics(score_table, summary_type='avg', model_types=None):
    """
    Summarizes the scores of the different clustering outcomes that are based 
    on different random_states, i.e. different sets of initial medoids. 
    Either the average of a the metrics is calculated or the std. dev.
    
    Parameters:
    ------------
        score_table:   [list of lists] score table 
        summary_types: [str] 'avg' for average or 'std' for std. dev.
        model_types:   [list of strings] titles of the score table (first 
                       column) to group the scores on before calculating 
                       the summary statistics.
    Returns:
        Input score table complemented with a row that describes the indicated 
        summary statistic.
    """
    if model_types: # if multiple models combine in one score table
        summary_table = []
        for model_type in model_types:
            scores = [i for i in score_table if model_type+' ' in i[0]]
            if summary_type == 'avg':
                summary_table.append( ['Avg {}'.format(model_type, len(scores))] + \
                     [np.array([score[i] for score in [sublist for sublist in scores]]).mean() \
                      for i in range(1,len(score_table[0]))] )
            elif summary_type == 'std':
                summary_table.append( ['Std {}'.format(model_type, len(scores))] + \
                     [np.array([(score[i]) for score in [sublist for sublist in scores]]).std() \
                      for i in range(1,len(score_table[0]))] )

    else: # all scores are from the same model
        if summary_type == 'avg':
            summary_table = ['Average'.format(len(score_table))] + \
                  [np.array([score[i] for score in [sublist for sublist in score_table]]).mean() \
                   for i in range(1,len(score_table[0])) ]
        elif summary_type == 'std':
            summary_table = ['Std Dev'.format(len(score_table))] + \
                  [np.array([(score[i]) for score in [sublist for sublist in score_table]]).std() \
                   for i in range(1,len(score_table[0]))] 
            
    return summary_table


def rand_index_score(clusters, classes):
    """Rand Index Score (not adjusted)"""
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def cluster_idx_to_label_list(clusters, nr_journeys):
    """
    Transforms the cluster type output of PyClustering to the required list of labels needed 
    for scikit-learn operations.
    
    Parameters:
    -------------
        clusters: [list of lists] with object indices per cluster
    
    Returns:
        A list of predicted cluster labels 
    
    """
    labels = [None] * nr_journeys
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    return labels

def variance_score_list(score_table, feature_type, score_type):
    """
    Parameters:
        feature_type : choose between 'L1', 'FCBF', 'Variance', 'Laplacian'
        score_type : choose between 'AMI' or 'S_DBw'   
    """
    if score_type == 'AMI':
        idx=2
    elif score_type == 'S_DBw':
        idx=3
    else:
        print('Wrong input for score_type. Pick either AMI or S_DBw')
    
    scores = [score[idx] for score in [sublist for sublist in score_table if feature_type in sublist[0]]]
    return np.var(scores)


### COMPARING THE CLUSTERING OUTPUT WITH THE ORIGINAL LABELS

def hungarian_cluster_labels(cluster_labels, true_labels, k):
    """
    Transforming the cluster labels to the optimal corresponding
    class labels based on the hungarian optimization algorithm
    (assignment problem). Used to create a confusion matrix.
    """
    # Cost function:
    C = np.empty((k,k))
    for i in range(k): # assigned cluster label
        for j in range(k): # true label 
            C[i][j] = len(set([idx for idx, n in enumerate(cluster_labels) if\
                               n==i])-set([idx for idx,n in enumerate(true_labels) \
                                           if n==j]))

    # Find optimal assignment        
    row_ind, col_ind = linear_sum_assignment(C)

    # Map cluster labels to class labels
    cluster_map_dict = {c:l for c,l in zip(row_ind, col_ind)}
    cluster_class_labels = np.array(pd.Series(cluster_labels).map(cluster_map_dict))
    return cluster_class_labels
    