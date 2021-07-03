import pandas as pd 
import numpy as np

from collections import Counter
import itertools
import re

# Own modules
#from preprocessing import cleaning
#from utilities import utils


#### CONTACT TYPES AND ACTIVITY COUNT JOURNEY PROFILES 

def one_hot_encoding(eventlog_df):
    """
    Function that creates the Activity and Channel journey perspectives
    for each journey in the event log. 
    
    Input:
        A event log dataframe where one row is one touchpoint
    
    Returns:
        A journey dataframe where one row is a journey vector according
        to the perspectives used.
    """  
    # Initialize df with ddloc & journey IDs
    hot_encoded_df = pd.DataFrame({'journey_id':eventlog_df.journey_id})

    # Get one hot encoding of the event columns
    # They are called event_type and resource for the BCIP 2012 dataset
    # Adjust for your own dataset if needed
    e = pd.get_dummies(eventlog_df['event_type'])
    r = pd.get_dummies(eventlog_df['resource'], prefix='R')

    # Join the dfs 
    hot_encoded_df = hot_encoded_df.join(e)
    hot_encoded_df = hot_encoded_df.join(r)
    
    return hot_encoded_df.groupby('journey_id').sum()# hot_encoded_df


def transform_into_feature_vec(eventlog_df, trace_type, verbose=True):
    """
    Transforms the initial eventlog dataset into a dataframe with trace vectors. 
    Perspectives Activity and Channel are included in the journey profiles.
    Based on these profiles, similarity can be calculated.
    
    Params:
        eventlog_df: [DataFrame] clean eventlog (one touchpoint per row)
        trace_type:  [str] either 'BOA' or 'SOA' feature representation
        
    Returns:
        A dataframe with the Activities and Channel journey profile vectors. 
        One row corresponds to one journey profile.
    """
    if verbose:
        print('Numerically encoding the activities...')
    trace_df = one_hot_encoding(eventlog_df)
    
    if trace_type == 'SOA':
        if verbose:
            print('Binarizing the count features (trace type = SOA)...')
        trace_df = trace_df.apply(lambda x: (x>0).astype(np.int8), axis=1) # make binary instead of count         

    return trace_df



#### ORDER JOURNEY PROFILE - BIGRAM (2GRAM) FUNCTIONS FOR CONTACT TYPE TRANSITIONS 

def get_ngrams(input_list, N=2):
    "k-grams"
    grams = [input_list[i:i+N] for i in range(len(input_list)-N+1)]
    return dict(Counter(tuple(i) for i in grams))

def get_bigram_data(bigram_dict, combinations_list):
    """
    Transforms a count dictionary in an ordered feature vector.
    
    Parameters:
        bigram_dict  : dict with a combinations count
        combinations : list of tuples, where each tuple is a combination of two 
                       contact types (strings)
    Returns:
        ordered count list (feature vector)
    """
    return [bigram_dict.get(combi, 0) for combi in combinations_list]

def bigram_feature_vectors(eventlog_df, trace_type, verbose=True):
    """
    Transition Perspective in journey profile. These features are indicated 
    with the prefix 'T' from Transition.
    
    Input:
        eventlog_df: [DataFrame] clean eventlog (one touchpoint per row)
        trace_type:  [str] either 'BOA' or 'SOA' feature representation
    
    Output:
        A dataframe with the Transition perspective vectors. 
        One row corresponds to one journey.
    """   
    # All possible bigrams of Contact Type information
    bigrams_same = [(i,i) for i in eventlog_df['event_type'].unique()] 
    bigrams_diff = list(itertools.permutations(eventlog_df['event_type'].unique(), 2)) 
    bigram_names = ['T_{}&{}'.format(i,j) for (i,j) in bigrams_same+bigrams_diff] 

    # Get bigram dictionary
    kgram_df = eventlog_df.groupby('journey_id').apply(lambda x: \
                                                 get_ngrams(x['event_type'].tolist()))
    kgram_df = pd.DataFrame(kgram_df, columns=['bigram_dict']).reset_index()

    # Get bigram feature vector
    kgram_df['bigram_vector'] = kgram_df['bigram_dict'].apply(get_bigram_data, **{'combinations_list':bigrams_same+bigrams_diff})
    
    # Spread the features over columns in the dataframe
    kgram_df[bigram_names] = pd.DataFrame(kgram_df['bigram_vector'].tolist(), index=kgram_df.index)
    kgram_df.drop(['bigram_vector', 'bigram_dict'], axis=1, inplace=True)
    
    # Binarize features if trace type is SOA
    if trace_type == 'SOA':
        if verbose:
            print('\nBinarizing the bigram features (trace type = SOA)...')
        for name in bigram_names:
            kgram_df[name] = kgram_df[name].apply(lambda x: int(x>0))
            
    return kgram_df



#### PERFORMANCE JOURNEY PROFILE 

def calculate_duration(eventlog_df):
    """
    Calculates the duration of journeys in days as part of the Performance Perspective.
    
    Returns:
        Dataframe with two columns: journey_id and duration per journey
    """
    eventlog_df['timestamp'] = pd.to_datetime(pd.Series(eventlog_df['timestamp']), format='%Y-%m-%d %H:%M')
    eventlog_df['registration_date'] = pd.to_datetime(pd.Series(eventlog_df['registration_date']), format='%Y-%m-%d %H:%M')

    start_df = pd.DataFrame(eventlog_df.groupby('journey_id')['registration_date'].min()).reset_index()
    end_df = pd.DataFrame(eventlog_df.groupby('journey_id')['timestamp'].max()).reset_index()
    duration_df = pd.merge(start_df, end_df, on='journey_id')

    # Calculate the total duration in days
    duration_df['duration'] = duration_df.apply(lambda x: (x['timestamp'] - x['registration_date']).days, axis=1)
    duration_df.drop(['timestamp', 'registration_date'], axis=1, inplace=True)
    return duration_df


def get_additional_features(eventlog_df):
    """
    Transforms the initial eventlog dataset into a dataframe with trace vectors that describe
    the Peformance Perspective of joruneys.
    
    Input:
        eventlog_df: [DataFrame] clean eventlog (one touchpoint per row)
    
    Output:
        A dataframe with the Transition perspective vectors. 
        One row corresponds to one journey.
    """   
    # Additional features
    nr_resources_df = pd.DataFrame(eventlog_df.groupby('journey_id')['resource'].nunique()).reset_index().rename(columns={'resource':'nr_resources'})
    nr_events_df = pd.DataFrame(eventlog_df.groupby('journey_id').size(), columns=['nr_events']).reset_index()
    duration_df = calculate_duration(eventlog_df)
    
    # Add:
    # number of W_ events?
        
    # Merge 
    add_ftrs_df = nr_resources_df.merge(nr_events_df, on='journey_id')
    add_ftrs_df = add_ftrs_df.merge(duration_df, on='journey_id')
    return add_ftrs_df


### LOAN AMOUNT

def bin_loan_amount(data):
    data['loan<5k'] = data['loan_amount'].apply(lambda x: 1 if x<5000 else 0)
    data['loan_5k_10k'] = data['loan_amount'].apply(lambda x: 1 if (x>=5000 and x<10000) else 0)
    data['loan_10k_20k'] = data['loan_amount'].apply(lambda x: 1 if (x>=10000 and x<20000) else 0)
    data['loan>20k'] = data['loan_amount'].apply(lambda x: 1 if (x>=20000) else 0)
    del data['loan_amount']
    return data


#### CALL ALL FUNCTIONS - MAIN FUNCTIONS OF THIS MODULE
      
def eventlog_to_vectors(data, trace_type='BOA', bigram_features=True, additional_features=True, 
                        remove_result_events=True, verbose=True): 
    """
    Transforms a raw event log into a dataframe containing journey profiles of customer journeys. 
    It is featurizing the event log.
    
    Parameters:
    ------------
        data:                  [DataFrame] cleaned event log (one touchpoint per row)
        trace_type:            [str] 'BOA' or 'SOA' indicating featue representation type
        bigram_features:       [boolean] whether or not to add the Transition Perspective
        additional_features:   [boolean] whether or not to add the Performance Perspective
        remove_result_events:  [boolean] whether or not to remove events that indicate the journey class 
        
    Returns:
        A feature vector dataframe with one trace per row
    """
    
    data = data.sort_values(by='timestamp')
    trace_df = transform_into_feature_vec(data, trace_type, verbose)

    if bigram_features:
        if verbose:
            print('Extracting bigram features..') 
        # Sort based on time such that after a groupby, the events are in the correct order
        data = data.sort_values(by='timestamp')
        # Get bigram dataframe and merge with trace_df
        bigram_df = bigram_feature_vectors(data, trace_type, verbose)
        trace_df = trace_df.merge(bigram_df, on='journey_id')
 
                
    if additional_features:
        if verbose:
            print('Extracting additional features..')
        add_ftrs_df = get_additional_features(data)
        trace_df = trace_df.merge(add_ftrs_df, on='journey_id')
    
    # Add loan amount attributes
    case_data = data[['journey_id', 'loan_amount', 'journey_class']].drop_duplicates(subset=['journey_id']) #'registration_date'
    if verbose:
        print('Adding class labels..')
    # Add loan amount and journey class attributes
    trace_df = trace_df.merge(case_data, on='journey_id', how='inner')

    if verbose:
        print('Bin loan amount..')
    trace_df = bin_loan_amount(trace_df)

    if remove_result_events:
        if verbose:
            print('Removing events that include a result (class information)..')
        result_events = ['A_CANCELLED', 'A_REGISTERED', 'A_ACTIVATED', 'A_APPROVED', 'A_DECLINED',
                         'O_CANCELLED', 'O_DECLINED', 'O_ACCEPTED'] 
        trace_df.drop(columns=result_events, errors='ignore', inplace=True, axis=1)  

    # Put standard cols at the beginning (visually nice)
    std_cols = ['journey_id', 'journey_class']#, 'loan_amount', 'registration_date']
    feature_cols = [col for col in trace_df.columns if col not in std_cols]
    trace_df = trace_df[std_cols+feature_cols]
        
    if verbose:
        print('Done!')   
    return trace_df     

