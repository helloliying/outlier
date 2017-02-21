 #!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import warnings
from numpy import *
import numpy as np



def distance_euclidean_test(A,B):
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA()/2)**0.5
    return np.matrix(ED)


class LOF:
    """Helper class for performing LOF computations and instances normalization."""
    def __init__(self, instances, normalize=True, distance_function=distance_euclidean_test):
        self.instances = instances
        self.normalize = normalize
        self.distance_function = distance_function
        if normalize:
            self.normalize_instances()

    def compute_instance_attribute_bounds(self):
        min_values = [float("inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * n.inf
        max_values = [float("-inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * -1 * n.inf
        for instance in self.instances:
            min_values = tuple(map(lambda x,y: min(x,y), min_values,instance)) #n.minimum(min_values, instance)
            max_values = tuple(map(lambda x,y: max(x,y), max_values,instance)) #n.maximum(max_values, instance)

        diff = [dim_max - dim_min for dim_max, dim_min in zip(max_values, min_values)]
        if not all(diff):
            problematic_dimensions = ", ".join(str(i+1) for i, v in enumerate(diff) if v == 0)
            warnings.warn("No data variation in dimensions: %s. You should check your data or disable normalization." % problematic_dimensions)

        self.max_attribute_values = max_values
        self.min_attribute_values = min_values

    def normalize_instances(self):
    
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()
        new_instances = []
        for instance in self.instances:

            new_instances.append(self.normalize_instance(instance)) # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances

    def normalize_instance(self, instance):
        return tuple(map(lambda value,max,min: (value-min)/(max-min) if max-min > 0 else 0,
                 instance, self.max_attribute_values, self.min_attribute_values))


    def local_outlier_factor(self, min_pts, instance):
        if self.normalize:
            instance = self.normalize_instance(instance)
        return local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)


def k_distance(k, instance, instances, distance_function=distance_euclidean_test):
    instances = np.matrix(list(instances))
    distances_test = distance_euclidean_test(instances,instance)
    sort_matrix = distances_test.argsort(axis = 0)
    distance_sort = distances_test[sort_matrix]
    distance_diff = np.diff(distance_sort[:,:,0],axis=0)
    n = 0
    unzero_sum = 0 
    while unzero_sum < k and n <= distance_diff.shape[0]+1:
        unzero_sum = np.sum(distance_diff[:n,0] != 0)
        n = n+1
   
    neighbours_test = instances[sort_matrix[0:(n-1)]]
    k_distance_value_test = distance_sort[n-2] if distance_sort.shape[0] >=k else distance_sort[-1]
    return k_distance_value_test, neighbours_test


# def k_distance(k, instance, instances, distance_function=distance_euclidean_test):

#     distances_test = distance_euclidean_test(instances,instance)
#     sort_matrix = distances_test.argsort(axis = 0)
#     distance_sort = zeros((sort_matrix.shape[1],sort_matrix.shape[0],1))

#     for i in range(distances_test.shape[1]):
#        distance_sort[i,:,:] = distances_test[:,i][sort_matrix[:,i]][:,0,0]
#     distance_diff = np.diff(distance_sort[:,:,0],axis=1)
#     n_list = [0]*distance_diff.shape[0]
   
#     for j in range(distance_diff.shape[0]): 
#         unzero_sum = 0
#         n = 0
#         while unzero_sum < k and n <= distance_diff.shape[1]+1:
#             unzero_sum = np.sum(distance_diff[j,:n] != 0,axis=0)
#             n = n+1
#             n_list[j]=n

#     neighbours_test = zeros((sort_matrix.shape[1],,2))
#     for m in range(len(n_list)):
#         neighbours_test = instances[sort_matrix[:,m][0:(n_list[m]-1)]][:,0,:]
    # k_distance_value_test = distance_sort[n-2] if distance_sort.shape[0] >=k else distance_sort[-1]
    # return k_distance_value_test, neighbours_test

def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean_test):
    (k_distance_value, neighbours_test) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])


def local_reachability_density(min_pts, instance, instances, **kwargs):
    (k_distance_value, neighbours_test) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array_test = [0]*neighbours_test.shape[0]

    for i in range(neighbours_test.shape[0]):
        neighbour = neighbours_test[i,:,:]
        reachability_distances_array_test[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs)

    if not any(reachability_distances_array_test):
        warnings.warn("Instance %s (could be normalized) is identical to all the neighbors. Setting local reachability density to inf." % repr(instance))
        return float("inf")
    else:
        return neighbours_test.shape[0] / sum(reachability_distances_array_test)


def local_outlier_factor(min_pts, instance, instances, **kwargs):

    instance = np.matrix(instance)
    (k_distance_value, neighbours_test) = k_distance(min_pts, instance, instances, **kwargs)
    instance_lrd = local_reachability_density(min_pts, instance, instances, **kwargs)

    lrd_ratios_array = [0]* neighbours_test.shape[0]
   
    for i in range(neighbours_test.shape[0]):
        instances_without_instance = set(instances)
        neighbour = neighbours_test[i,:,:]
        neighbour2touple =  (neighbour[0,0],neighbour[0,1])
        instances_without_instance.discard(neighbour2touple)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd

    return sum(lrd_ratios_array) / neighbours_test.shape[0]


def outliers(k,instances, candidate=None,**kwargs):
    """Simple procedure to identify outliers in the dataset."""
    instances_value_backup = instances
    outliers = []
    if not candidate:
        candidate = instances
    """ import k-means calucate outlier candidate dataset ClOF """

    for i,instance in enumerate(candidate):
        instance = tuple(instance)
        instances = list(instances_value_backup)
        instances.remove(instance)
        l = LOF(instances, **kwargs)
        value = l.local_outlier_factor(k, instance)

        if value[0,0] > 1:
            outliers.append({"lof": value[0,0], "instance": instance, "index": i})

    outliers.sort(key=lambda o: o["lof"], reverse=True)
    return outliers



