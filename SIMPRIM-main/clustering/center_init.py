"""
Code to initialize the centers of the kMedoids (medoids) algorithm. Three techniques are included: (1) random, (2) class-based, i.e. semi-supervised, and (3) kmeans++. 

Note: the code of kmeans++ is from the pyclustering library but this did not work when importing directly from online: https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/center_initializer.py
Note: when you indicate the random_state (int) in the kwargs the selected centroids are reproducible. 

Example:
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from cluster.center_init import kmeans_plusplus_initializer, get_center_indices

k=15
initial_medoids = kmeans_plusplus_initializer(X, amount_centers=k, **{'random_state':4}).initialize()
initial_medoids = kmeans_plusplus_initializer(X, amount_centers=k, **{'random_state':4}).initialize(return_index=True)
#initial_medoids_idx = get_center_indices(X, initial_medoids)

Example:
from cluster.center_init import random_initializer
initial_medoids = random_initializer(X, amount_centers=15, output_arrays=True, **{'random_state':5})
initial_medoids_idx = random_initializer(X, amount_centers=15, output_arrays=False, **{'random_state':5})

"""

import numpy as np
import random
import warnings


### Random initializer
def random_initializer(data, amount_centers, return_index=False, **kwargs):
    """
    
    Params:
        data (array_like): input matrix, list of arrays
        amount_centers (int): Amount of centers that should be initialized
        output_arrays (bool): Whether or not to output the actual centers or not in array
                              format. The default is False, which will return the indices.
    
        **kwargs: Arbitrary keyword arguments. Available arguments: 'random_state' (int).
                  to make results reproducible.
    returns:
        Indices or actual arrays of randomly selected centers
    """
    np.random.seed(kwargs.get('random_state', None))
    medoids_idx = np.random.choice(list(range(len(data))), amount_centers) 
    if return_index:
        medoids_idx
    else:
        return X[medoids_idx, :] 


### Semi Supervised Initializer
def class_based_initializer(trace_df, k, random_state, return_index=True):
    # Sample a medoid from each class
    initial_medoids_df = trace_df.groupby('journey_class').apply(lambda x: x.sample(n=1, random_state=random_state)).reset_index(drop=True)
    
    # Find indices of these medoids
    trace_df_reset_idx = trace_df.reset_index(drop=True)
    medoids_idx = []
    for jID in list(initial_medoids_df['journey_id']):
        idx = trace_df_reset_idx.index[trace_df_reset_idx['journey_id']==jID].tolist()[0]
        medoids_idx.append(idx)
    if return_index:
        return medoids_idx
    else:
        return trace_df_reset_idx.iloc[medoids_idx,:]

    
### Kmeans ++ initializer
def get_center_indices(data, centers):
    """
    Outputs the indices of the given centroids in the original dataset.
    In case multiple similar arrays are present in the dataset, the first
    match is returned. Required since K++ initializer outputs the 
    actual centroid arrays while the kmedoids implementation needs indices.
    
    data: input matrix, list of arrays
    centroids: list of centroid arrays
    
    """
    indices = list()
    for medoid in centers:
        indices.append([i for i, m in enumerate(data) if list(m) == list(medoid)][0])
    return indices 


class kmeans_plusplus_initializer:
    """!
    @brief K-Means++ is an algorithm for choosing the initial centers for algorithms like K-Means or X-Means.
    @details K-Means++ algorithm guarantees an approximation ratio O(log k). Clustering results are depends on
              initial centers in case of K-Means algorithm and even in case of X-Means. This method is used to find
              out optimal initial centers.
    Algorithm can be divided into three steps. The first center is chosen from input data randomly with
    uniform distribution at the first step. At the second, probability to being center is calculated for each point:
    \f[p_{i}=\frac{D(x_{i})}{\sum_{j=0}^{N}D(x_{j})}\f]
    where \f$D(x_{i})\f$ is a distance from point \f$i\f$ to the closest center. Using this probabilities next center
    is chosen. The last step is repeated until required amount of centers is initialized.   
    """

    ## Constant denotes that only points with highest probabilities should be considered as centers.
    FARTHEST_CENTER_CANDIDATE = "farthest"

    
    def __init__(self, data, amount_centers, amount_candidates=None, **kwargs):
        """!
        @brief Creates K-Means++ center initializer instance.
        
        @param[in] data (array_like): List of points where each point is represented by list of coordinates.
        @param[in] amount_centers (uint): Amount of centers that should be initialized.
        @param[in] amount_candidates (uint): Amount of candidates that is considered as a center, if the farthest points
                    (with the highest probability) should be considered as centers then special constant should be used
                    'FARTHEST_CENTER_CANDIDATE'. By default the amount of candidates is 3.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'random_state').
        <b>Keyword Args:</b><br>
            - random_state (int): Seed for random state (by default is `None`, current system time is used).
        @see FARTHEST_CENTER_CANDIDATE
        """
        
        self.__data = np.array(data)
        self.__amount = amount_centers
        self.__free_indexes = set(range(len(self.__data)))

        if amount_candidates is None:
            self.__candidates = 3
            if self.__candidates > len(self.__data):
                self.__candidates = len(self.__data)
        else:
            self.__candidates = amount_candidates

        self.__check_parameters()

        random.seed(kwargs.get('random_state', None))


    def __check_parameters(self):
        """!
        @brief Checks input parameters of the algorithm and if something wrong then corresponding exception is thrown.
        """
        if (self.__amount <= 0) or (self.__amount > len(self.__data)):
            raise ValueError("Amount of cluster centers '" + str(self.__amount) + "' should be at least 1 and "
                             "should be less or equal to amount of points in data.")

        if self.__candidates != kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE:
            if (self.__candidates <= 0) or (self.__candidates > len(self.__data)):
                raise ValueError("Amount of center candidates '" + str(self.__candidates) + "' should be at least 1 "
                                 "and should be less or equal to amount of points in data.")

        if len(self.__data) == 0:
            raise ValueError("Data is empty.")


    def __calculate_shortest_distances(self, data, centers):
        """!
        @brief Calculates distance from each data point to nearest center.
        
        @param[in] data (numpy.array): Array of points for that initialization is performed.
        @param[in] centers (numpy.array): Array of indexes that represents centers.
        
        @return (numpy.array) List of distances to closest center for each data point.
        
        """

        dataset_differences = np.zeros((len(centers), len(data)))
        for index_center in range(len(centers)):
            center = data[centers[index_center]]

            dataset_differences[index_center] = np.sum(np.square(data - center), axis=1).T

        with warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            shortest_distances = np.nanmin(dataset_differences, axis=0)

        return shortest_distances


    def __get_next_center(self, centers):
        """!
        @brief Calculates the next center for the data.
        @param[in] centers (array_like): Current initialized centers represented by indexes.
        @return (array_like) Next initialized center.<br>
                (uint) Index of next initialized center if return_index is True.
        """

        distances = self.__calculate_shortest_distances(self.__data, centers)

        if self.__candidates == kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE:
            for index_point in centers:
                distances[index_point] = np.nan
            center_index = np.nanargmax(distances)
        else:
            probabilities = self.__calculate_probabilities(distances)
            center_index = self.__get_probable_center(distances, probabilities)

        return center_index


    def __get_initial_center(self, return_index):
        """!
        @brief Choose randomly first center.
        @param[in] return_index (bool): If True then return center's index instead of point.
        @return (array_like) First center.<br>
                (uint) Index of first center.
        """

        index_center = random.randint(0, len(self.__data) - 1)
        if return_index:
            return index_center

        return self.__data[index_center]


    def __calculate_probabilities(self, distances):
        """!
        @brief Calculates cumulative probabilities of being center of each point.
        @param[in] distances (array_like): Distances from each point to closest center.
        @return (array_like) Cumulative probabilities of being center of each point.
        """

        total_distance = np.sum(distances)
        if total_distance != 0.0:
            probabilities = distances / total_distance
            return np.cumsum(probabilities)
        else:
            return np.zeros(len(distances))


    def __get_probable_center(self, distances, probabilities):
        """!
        @brief Calculates the next probable center considering amount candidates.
        @param[in] distances (array_like): Distances from each point to closest center.
        @param[in] probabilities (array_like): Cumulative probabilities of being center of each point.
        @return (uint) Index point that is next initialized center.
        """

        index_best_candidate = 0
        for i in range(self.__candidates):
            candidate_probability = random.random()
            index_candidate = -1

            for index_object in range(len(probabilities)):
                if candidate_probability < probabilities[index_object]:
                    index_candidate = index_object
                    break

            if index_candidate == -1:
                index_best_candidate = next(iter(self.__free_indexes))
            elif distances[index_best_candidate] < distances[index_candidate]:
                index_best_candidate = index_candidate

        return index_best_candidate


    def initialize(self, **kwargs):
        """!
        @brief Calculates initial centers using K-Means++ method.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'return_index').
        <b>Keyword Args:</b><br>
            - return_index (bool): If True then returns indexes of points from input data instead of points itself.
        @return (list) List of initialized initial centers.
                  If argument 'return_index' is False then returns list of points.
                  If argument 'return_index' is True then returns list of indexes.
        
        """

        return_index = kwargs.get('return_index', False)

        index_point = self.__get_initial_center(True)
        centers = [index_point]
        self.__free_indexes.remove(index_point)

        # For each next center
        for _ in range(1, self.__amount):
            index_point = self.__get_next_center(centers)
            centers.append(index_point)
            self.__free_indexes.remove(index_point)

        if not return_index:
            centers = [self.__data[index] for index in centers]

        return centers