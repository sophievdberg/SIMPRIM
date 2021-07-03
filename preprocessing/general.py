import numpy as np
"""

All general pre-processing functions used.

"""

import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

from utilities import utils


##### CLEANING BCIP 2012 DATASET

def format_data(data):
    """
    Formatting BCIP 2012 data such that columns are easier to interpret.
    Necessary in this formatting is that the unique case identifier
    is referred to as 'journey_id'. This will be used in the 
    remainder of the code.
    """
    data.rename(columns={'case:concept:name' : 'journey_id',
                         'case:AMOUNT_REQ' : 'loan_amount',
                         'case:REG_DATE' : 'registration_date',
                         'org:resource' : 'resource',
                         'concept:name' : 'event_type',
                         'time:timestamp' : 'timestamp'
                       }, inplace=True)

    # Drop transition information
    data.drop(columns=['lifecycle:transition'], inplace=True)

    # Change format
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    data['registration_date'] = pd.to_datetime(data['registration_date'], 
                                               utc=True)
    data['loan_amount'] = data['loan_amount'].astype(int)

    # Replace missing resource with 'Missing'
    data['resource'].fillna('Missing', inplace=True)
    
    return data

def remove_running_journeys(data):
    """
    BCIP 2012 specific function. Filters out all journeys without 
    a clear application result (i.e. running journeys).
    """
    result_events = ['A_CANCELLED','A_REGISTERED','A_ACTIVATED','A_APPROVED', 
                     'A_DECLINED','O_CANCELLED','O_DECLINED','O_ACCEPTED'] 
    
    complete_IDs = data[data['event_type'].isin(result_events)]['journey_id'].unique()    
    print('Removed running journeys ({:.2f}%)'.format(\
                      (1-len(complete_IDs)/data['journey_id'].nunique())*100))
    data = data[data['journey_id'].isin(complete_IDs)]
    return data
    
    
    
##### FEATURE REDUNDANCY

def get_uniform_features(df):
    uniform = list()
    for col in df.columns:
        if df[col].nunique() == 1:
            uniform.append(col)
    return uniform

def remove_uniform_features(trace_df):
    uniform_ftrs = get_uniform_features(trace_df)
    print('Removed: {} features'.format(len(uniform_ftrs)))
    return trace_df.drop(uniform_ftrs, axis=1)

def get_correlated_features(corr_df, threshold=1):
    corr_df = corr_df.abs()
    upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape),k=1).astype(np.bool))
    correlated_features = [column for column in upper_tri.columns if \
                           any(upper_tri[column] >= threshold)]
    return correlated_features
      
def remove_correlated_features(trace_df, threshold=1):
    """
    Dealing with collinearity. Features are removed if their correlation 
    coefficient > threshold. Only one of the features from a correlating 
    pair is removed. 
    """
    corr_df = trace_df.drop(['journey_id'], axis=1).corr()
    correlated_ftrs = get_correlated_features(corr_df, threshold=threshold)
   
    print('Removed: {} features'.format(len(correlated_ftrs)))
    trace_df = trace_df.drop(correlated_ftrs, axis=1)
    return trace_df

def remove_redundant_features(df):
    print('\nRemoving non-variance features...')
    df = remove_uniform_features(df)
    print('\nRemoving perfectly correlating features...')
    df = remove_correlated_features(df, threshold=1)
    return df 


### JOURNEY CLASSES

def add_journey_class(data):
    """
    BCIP 2012 specific function that assigns a journey class to each journey
    based on the application outcome since no class labels exist in the 
    benchmark dataset. We distinguish 6 classes: applications that are 
        (1) accepted directly, meaning after only one offer; 
        (2) accepted after some optimization of the offer; 
        (3) rejected straight away, 
        (4) rejected after an offer was drafted, 
        (5) cancelled before an offer was drafted and 
        (6) cancelled after an offer was drafted.
    
    The function returns the DataFrame with an additional column: journey_class
    """
    # Results events per types
    accepted = ['A_REGISTERED', 'A_APPROVED', 'A_ACTIVATED', 'O_ACCEPTED']
    declined = ['A_DECLINED', 'O_DECLINED']
    cancelled = ['A_CANCELLED', 'O_CANCELLED']
    
    classified_IDs = [] # init
    
    # Accepted applications
    accepted_IDs = list(data[data['event_type'].isin(accepted)]['journey_id'].unique())
    classified_IDs += accepted_IDs
    
    accepted_straight = [] # class type
    accepted_after_optimization = [] # class type
    #print('Finding Accepted Applications')
    for jID in accepted_IDs:
        journey_events = list(data[data['journey_id']==jID]['event_type'])
        if journey_events.count('O_SENT')==1:
            accepted_straight.append(jID)
        else:
            accepted_after_optimization.append(jID)
    
    # Declined applications
    remaining_data = data[~data['journey_id'].isin(classified_IDs)]
    declined_IDs = list(remaining_data[remaining_data['event_type'].isin(declined)]\
                                                           ['journey_id'].unique())
    classified_IDs += declined_IDs
    
    declined_straight = [] # class type
    declined_after_offer = [] # class type
    #print('Finding Declined Applications')
    for jID in declined_IDs:
        journey_events = list(remaining_data[remaining_data['journey_id']==jID]\
                                                                 ['event_type'])
        if 'O_DECLINED' in journey_events:
            declined_after_offer.append(jID)
        else:
            declined_straight.append(jID)
    
    # Cancelled applications
    remaining_data = data[~data['journey_id'].isin(classified_IDs)]
    cancelled_IDs = list(remaining_data[remaining_data['event_type'].isin(cancelled)]\
                                                             ['journey_id'].unique())
    classified_IDs += cancelled_IDs
    
    cancelled_straight = [] # class type
    cancelled_after_offer = [] # class type
    # If O_cancelled in journey but still an accepted offer, this one is yet in accepted journeys
    #print('Finding Cancelled Applications')
    for jID in cancelled_IDs:
        journey_events = list(remaining_data[remaining_data['journey_id']==jID]['event_type'])
        if 'O_CANCELLED' in journey_events:
            cancelled_after_offer.append(jID)
        else:
            cancelled_straight.append(jID)
            
    if not len(classified_IDs)==data['journey_id'].nunique():
        print('Warning: Not all journeys are classified.')
        
    # Store journey class per journey ID in a dict
    class_dict = {jID: 'C_straight' for jID in cancelled_straight}
    class_dict.update({jID: 'C_offer' for jID in cancelled_after_offer})
    class_dict.update({jID: 'R_straight' for jID in declined_straight})
    class_dict.update({jID: 'R_offer' for jID in declined_after_offer})
    class_dict.update({jID: 'A_optimized' for jID in accepted_after_optimization})
    class_dict.update({jID: 'A_straight' for jID in accepted_straight})
    
    # Add results to dataframe
    data['journey_class'] = data['journey_id'].map(class_dict)
    
    return data
    


#### SCALING

def scaler(X, stype, return_scaler_object=False):
    """
    Scales the columns of a numpy array. 
        - X: [numpy matrix] journey profiles
        - stype: [str] Either 'standardization' or 'range' (i.e. MinMax) 
        indicating the scaler type that is used. 
    """
    if stype=='range':
        scaler = MinMaxScaler()
    else: 
        if not stype=='standardization':
            print('No correct scaler type is selected. Standardization is used.')
        scaler = StandardScaler()
    
    scaler.fit(X)
    if return_scaler_object:
        return scaler.transform(X), scaler
    else:
        return scaler.transform(X)


    
##### TRAINING AND TEST DATASETS
    
def make_train_test_datasets(data, nr_trainsets, nr_testsets, test_size, 
                             stratified=True):
    
    sampled_journeyIDs = []
    
    print('\n## Creating {} journey ID sets for testing'.format(nr_testsets))
    test_dict = {i+1:None for i in range(nr_testsets)}
    journeys_per_testset = math.floor((data.shape[0]*test_size) / nr_testsets)
    for i in range(nr_testsets):
        df = data[~data.journey_id.isin(sampled_journeyIDs)]
        # Sample 
        if stratified:
            sample_df = stratified_sample(df, n=journeys_per_testset)
        else:
            sample_df = df.sample(n=journeys_per_testset, random_state=2)
        # Store
        sampled_journeyIDs += list(sample_df.journey_id.unique())
        test_dict[i+1] =  list(sample_df.journey_id.unique())
        
    print('{}% of the total set is tested with'.format(\
                  round((nr_testsets*journeys_per_testset)/len(data), 4)*100))    
    for nr, IDs in test_dict.items():
        utils.write_list_to_txt(IDs, 'journeyIDs_test_set_{}.txt'.format(nr))
    
    print('\n## Creating {} journey ID sets for training'.format(nr_trainsets))
    training_full_dict = {i+1:None for i in range(nr_trainsets)}
    journeys_per_trainset = math.floor((data.shape[0]-len(sampled_journeyIDs))\
                                                                 /nr_trainsets)
    for i in range(nr_trainsets):
        df = data[~data.journey_id.isin(sampled_journeyIDs)] #already sampled
        if i+1==nr_trainsets:
            sample_df = df.copy() # use all if this is the final training set
        else:
            if stratified:
                sample_df = stratified_sample(df, n=journeys_per_trainset)
            else:
                sample_df = df.sample(n=journeys_per_trainset, random_state=2)      
        # Store
        sampled_journeyIDs += list(sample_df.journey_id.unique())
        training_full_dict[i+1] =  list(sample_df.journey_id.unique())
    for nr, IDs in training_full_dict.items():
        utils.write_list_to_txt(IDs, 'journeyIDs_training_set_{}.txt'.format(nr))
    print('{}% of the total set is trained with'.format(\
                   round((nr_trainsets*journeys_per_trainset)/len(data), 3)*100)) 
    
    print('\nTotal number of sampled journeys:', len(sampled_journeyIDs))
    
    
def stratified_sample(trace_df, n):
    """
    Selects a sample that preserves the original class distribution 
    of the full trace dataframe. 
    
    Parameters:
    ------------
        trace_df: [DataFrame] trace dataframe that you want to sample 
                  from that at least contains a column journey_id and 
                  a column journey_class 
        n: [int] number of journeys to sample, where n<len(trace_df)
    
    Returns
        Smaller dataframe (size n) with the same distribution of classes 
        as the original trace dataframe.
    
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    np.random.seed(2020) 
    
    if n>len(trace_df):
        print('n is bigger than the dataframe')
        return trace_df
    
    data = trace_df.loc[:, ['journey_id', 'journey_class']]
    data.drop_duplicates(subset=['journey_id'], inplace=True)
    X = data.drop(['journey_class'], axis=1)

    le = LabelEncoder()
    le.fit(data['journey_class'])
    y = le.transform(data['journey_class'])

    # The train set X are the sampled journeys
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, 
                                                stratify=y, random_state=42)
    
    return trace_df[trace_df.journey_id.isin(list(X_train['journey_id']))]
    