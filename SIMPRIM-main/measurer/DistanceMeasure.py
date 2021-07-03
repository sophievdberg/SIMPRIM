# Standard libraries
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

# Own libraries
from utilities import utils
from preprocessing import vectorizing as vec
from preprocessing import general as pp
from clustering import clusterer  

# Function specific libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances



### Input trace data
def get_train_test_trace_data(trace_df, trainset, testset):
    "Select the training and test data sets based on the journey separation made as part of the methodology."
    
    trainingIDs = utils.read_list_from_txt('journeyIDs_training_set_{}.txt'.format(trainset))
    training_trace_df = trace_df[trace_df.journey_id.isin(trainingIDs)] 
    print('\nShape training data: {}'.format(training_trace_df.shape))

    testIDs = utils.read_list_from_txt('journeyIDs_test_set_{}.txt'.format(testset))       
    test_trace_df = trace_df[trace_df.journey_id.isin(testIDs)]
    print('Shape test data: {}'.format(test_trace_df.shape))

    return training_trace_df, test_trace_df



# Similarity measure format
def trace_df_to_similarity_meausure_format(trace_df, features=None, weights=None, return_scaler_object=False):
    """
    The order of the weights must match the order of the features
    """
    if return_scaler_object:
        X_scaled, scaler = pp.scaler(trace_df.loc[:, features], stype='standardization', return_scaler_object=True)  
        return X_scaled * weights, scaler
    
    else:
        X_scaled = pp.scaler(trace_df.loc[:, features], stype='standardization')  
        return X_scaled * weights


    
def distance_to_medoids(full_journey_df, jID, 
                        medoids, state_events, scaler, trace_type, features, weights, 
                        step=None, return_closest_medoid=False):
    """
    Parameters:
        original data: dataframe that contains the full journeys (cleaned & without Business Ruling)
        medoid   : numpy array consisting of the indicated features
        jID      : journey ID of journey of interest
        scaler   : scaler object that is fitted during the clustering that led to the medoids
        features : list of feature names to include in the journey vector
        feature_idx: ordered list of feature indices that corresponds to the weight vector
    """      
    # Full single journey
    single_journey_df = full_journey_df[full_journey_df['journey_id']==jID].reset_index()
    single_journey_df = single_journey_df.sort_values(by='timestamp')
    
    if step:
        # only a part of the journey is selected
        single_journey_df['is_state_event'] = single_journey_df['event_type'].apply(lambda x: 1 if x in state_events else 0)
        # find current point in full journey based on the provided 'step'
        for x in range(len(single_journey_df)):
            if sum(single_journey_df['is_state_event'].iloc[:x+1]) == step:
                step_idx = x
                break
        single_journey_df = single_journey_df.iloc[:step_idx+1, :] 
        del single_journey_df['is_state_event']
    
    journey_vector = vec.single_journey_to_vector(single_journey_df, features, trace_type) 
    journey_vector = scaler.transform(journey_vector.loc[:, features]) # scaler is fitted on the selected features
    journey_vector = journey_vector * weights
    
    distance = cosine_distances(journey_vector.reshape(1, -1), medoids)[0] / 2 
    
    if return_closest_medoid:
        return np.argmin(distance)
    else:
        return distance
    
    
    
#### FUNCTIONS THAT CAN SELECT A GROUP OF SIMILAR JOURNEY BASED ON AN EXAMPLE JOURNEY


def find_journey_with_pattern(journey_df, pattern, pattern_type='contact_type', random_state=1):
    journey_df = journey_df.sample(frac=1, random_state=random_state) # shuffle the df such that every time another journey is found that matches the pattern 
    journey_df = journey_df.rename({'type_contact': 'contact_type'}, axis=1)
    for jID in journey_df.journey_id:
        sequence = journey_df[journey_df.journey_id==jID].sort_values(by='starttijd')[pattern_type].tolist()
        if any(pattern == sequence[i:i+len(pattern)] for i in range(len(sequence) - 1)): # if pattern in sequence (in order)
            print('Journey ID:', jID)
            return journey_df[journey_df.journey_id==jID].sort_values(by='starttijd')
    print('No journey with such a pattern found')
    

def find_similar_traces(sample_jID, trace_df, distance, distance_metric, ordered_features, 
                          weights, pattern=None, pattern_type=None, journey_df=None, zero_to_one_range=True):
    # Get traces
    X = trace_df_to_similarity_meausure_format(trace_df, ordered_features, weights)
    # Get the selected trace
    idx_sample = trace_df.reset_index(drop=True).index[trace_df['journey_id']==sample_jID].tolist()[0]
    X_sample = X[idx_sample]
    # Find pairwise distances
    pairwise_distances = cosine_distances(X_sample.reshape(1, -1), X)
    #pairwise_distances = clusterer.pairwise_distance_matrix(X, metric=distance_metric)  
    if zero_to_one_range:
        pairwise_distances = pairwise_distances / 2
    close_journeyIDs = np.where(pairwise_distances<=distance)[1]
    
    similar_traces = trace_df.reset_index(drop=True).iloc[close_journeyIDs]
    
    if pattern is None:
        return similar_traces
    else:
        jIDs_with_pattern = []
        journey_df = journey_df.rename({'type_contact': 'contact_type'}, axis=1)
        for jID in similar_traces.journey_id:
            sequence = journey_df[journey_df.journey_id==jID].sort_values(by='starttijd')[pattern_type].tolist()
            if any(pattern == sequence[i:i+len(pattern)] for i in range(len(sequence) - 1)): # if pattern in sequence (in order)
                jIDs_with_pattern.append(jID)
        similar_traces_with_pattern = similar_traces[similar_traces.journey_id.isin(jIDs_with_pattern)]
        return similar_traces_with_pattern
    
def distance_to_sampled_journey(jID, jIDsample, trace_df, X=None, ordered_features=None, weights=None, zero_to_one_range=True):
    """
    
    Parameters:
        jID: journey id of a surrounding journey
        jIDsample: journey id of the journey of interest
        X: matrix in similarity measure format (ftrs,weights,scaled). 
            Does not have to be provided but is computationally more efficient.
            If not provided, ordered features and weights should be given.
    """
    if X is None:
        X = trace_df_to_similarity_meausure_format(trace_df, ordered_features, weights)
        
    idx_jID = trace_df.index[trace_df['journey_id']==jID].tolist()[0]
    idx_jIDsample = trace_df.index[trace_df['journey_id']==sample_jID].tolist()[0]
    distance = cosine_distances(X[idx_jID].reshape(1, -1), X[idx_jIDsample].reshape(1, -1))[0][0]
    if zero_to_one_range:
        return distance / 2
    else:
        return distance
    
    


