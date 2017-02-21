#!/usr/bin/python
# -*- coding: utf8 -*-
from numpy import *  
import time  
import matplotlib.pyplot as plt  
from Euclidean import Euclidean 
import numpy as np


class KMeans(object):
    
    def __init__(self):
        self.euclideanVector = Euclidean()   

    def euclDistance(self,vector1, vector2):  
        return sqrt(sum(power(vector2 - vector1, 2)))  
  
    def initCentroids(self,dataSet, k):  
        numSamples, dim = dataSet.shape 

        centroids = zeros((k, dim))  
        for i in range(k):  
            index = int(random.uniform(0, numSamples))
            centroids[i, :] = dataSet[index, :] 
        return centroids  
  


    def kmeans(self,dataSet, k,ifCandidate = False):  
        numSamples = dataSet.shape[0]  
        clusterAssment = mat(zeros((numSamples, 2)))  
        clusterChanged = True  
  
        centroids = self.initCentroids(dataSet, k)  
  
        while clusterChanged:  
            clusterChanged = False  

            for i in xrange(numSamples):  
                minDist  = 100000.0  
                minIndex = 0  
 
                for j in range(k): 

                    distance = self.euclDistance(centroids[j, :], dataSet[i, :])  
                    if distance < minDist:  
                        minDist  = distance  
                        minIndex = j 
     
                if clusterAssment[i, 0] != minIndex:  
                    clusterChanged = True  
                    clusterAssment[i, :] = minIndex, minDist**2

            candidate = []
            for j in range(k):
                pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  
                centroids[j, :] = mean(pointsInCluster, axis = 0)
                if not clusterChanged and ifCandidate:
                    distanceMatrix = self.euclideanVector.caculate(np.matrix(pointsInCluster),np.matrix(centroids[j,:]))
                    distanceMean = mean(distanceMatrix,axis = 0)
                    candidateSet = pointsInCluster[nonzero(distanceMatrix[:, 0].A >= distanceMean)[0]].tolist()
            
                    if candidate:
                        candidate = candidate +candidateSet
                    else:
                        candidate = candidateSet    

        return centroids, clusterAssment ,candidate

  

    def showCluster(self,dataSet, k, centroids, clusterAssment):  
        numSamples, dim = dataSet.shape  
        if dim != 2:  
            print "Sorry! I can not draw because the dimension of your data is not 2!"  
            return 1   
        
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] 

        for i in xrange(numSamples):  
            markIndex = int(clusterAssment[i, 0])  
            plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
      
        for i in range(k):  
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
      
        plt.show()  
    



