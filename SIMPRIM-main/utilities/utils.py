import random
import scipy.sparse as sparse
import scipy.io
import numpy as np
import pandas as pd

def write_csv(df, filename, index=True):
    print("Saved results in a .csv file called '{}'".format(filename))
    df.to_csv(filename, index=index)
    
    
def write_pickle(df, filename):
    print("Saved results in a .pickle file called '{}'".format(filename))
    df.to_pickle(filename)
    
    
def write_list_to_csv(a_list, filename):
    import csv
    print("Saved feature names in a .csv file called '{}'".format(filename))
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(a_list)
        
        
def write_list_to_txt(a_list, filename):
    with open(filename, 'w') as file:
        for item in a_list:
            file.write("%s," % item) # comma separated
    print("Saved feature names in a .txt file called '{}'".format(filename))
    
    
def read_list_from_txt(filename):
    file = open(filename,"r+") 
    return file.read().split(',')
        
    
def stratified_sample(df, n):
    """
    Sample that adheres to the original class distribution
    Distribution of both classes
    Based on journey_id split since labels are involved
    
    Parameters:
    ------------
    df : Df that you want to sample from that at least contains
                  a journey IDs and journey class labels
    n  : [int] number of journeys that are sampled
    
    Returns:
        Smaller dataframe (size n) with the same distribution of 
        classes as the original dataframe.
    
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    if n>len(df):
        print('n is bigger than the dataframe')
        return df
    
    data = df.loc[:, ['journey_id', 'journey_class']]
    data.drop_duplicates(subset=['journey_id'], inplace=True)
    X = data.drop(['journey_class'], axis=1)
   
    le = LabelEncoder()
    le.fit(data['journey_class'])
    y = le.transform(data['journey_class'])

    # The train set X are the sampled journeys
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y, random_state=42)
    
    return df[df.journey_id.isin(list(X_train.journey_id))]
    
