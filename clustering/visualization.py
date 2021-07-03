# Standard
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

from sklearn.metrics import confusion_matrix
import skopt.plots


def plot_confusion_matrix(true_labels, cluster_class_labels, size=(13,13)):   
    """
    Confusion Matrix that shows how the cluster class labels are
    overlapping with the original journey class label. The hungarian
    algorithm is used to map a cluster label to a class label with minimal
    errors.
    """
    k = max(len(set(true_labels)), len(set(cluster_class_labels)))
    
    plt.figure(figsize=size)
    
    if k > 10:
        plt.subplot(3, 1, 1)
    else:
        plt.subplot(1, 3, 1)
    og_mat = confusion_matrix(true_labels, true_labels)
    sns.heatmap(og_mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=[i for i in range(k)],
                yticklabels=[i for i in range(k)])
    plt.title('Original Distribution', fontsize=14)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    if k > 10:
        plt.subplot(3, 1, 2)
    else:
        plt.subplot(1, 3, 2)
    mat = confusion_matrix(true_labels, cluster_class_labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=[i for i in range(k)],
                yticklabels=[i for i in range(k)])
    plt.title('Optimal Cluster-Label Assignment', fontsize=14)
    plt.xlabel('true label')

    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    
    return fig1

def plot_weight_optimization(optimization_results, figsize=(10,6)):   
    plt.figure(figsize=(10,6))
    skopt.plots.plot_convergence(optimization_results)
    plt.title('Weight Convergence Plot', fontsize=16)
    plt.show()
    

def plot_average_results(avg_scores_list_ami, nr_features_list, labels, avg_scores_list_sdbw=None):
        
    xi = list(range(len(nr_features_list)))
    plt.figure(figsize=(14,4))

    plt.subplot(1, 2, 1)
    plt.plot(xi, avg_scores_list_ami, marker='o', linestyle='--', color='r', label=labels[0]) 
    plt.ylim((0.25, 0.5))
    plt.xlabel('nr of features')
    plt.ylabel('Average AMI') 
    plt.xticks(xi, nr_features_list)
    plt.legend() 
    
    if avg_scores_list_sdbw is not None:
        plt.subplot(1, 2, 2)
        plt.plot(xi, avg_scores_list_sdbw, marker='o', linestyle='--', color='b', label=labels[1]) 
        plt.ylim((0.5, 1.5))
        plt.xlabel('nr of features')
        plt.ylabel('Average S_Dbw') 
        plt.xticks(xi, nr_features_list)
        
    plt.legend()     
    plt.show()
