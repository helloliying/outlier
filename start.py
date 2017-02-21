#!/usr/bin/python
# -*- coding: utf8 -*-
import lof
from mysqlclient import MysqlClient
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import random
import vincent
import time
from kmeans import KMeans
import pandas as pd
import numpy as np
import json
import time
import re
from Euclidean import Euclidean
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def saveFile(user_outer):
    line = json.dumps(dict(user_outer),ensure_ascii=False)
    outer_file.write(line+'\n')
    outer_file.flush()

def saveCentroids(centroids):
    line = json.dumps(dict(centroids),ensure_ascii=False)
    centroids_file.write(line+'\n')
    centroids_file.flush()

def test_normalization_problems(k,user,instances,user_review,candidate=None,count=None):
    temp_list = []
    outers = lof.outliers(k,instances, candidate)
    user_outer["user"] = user
    for i, outer in enumerate(outers):
        index = outer["index"]
        temp_list.append(dict(outer,**user_review[index]))
    user_outer["outer"] = temp_list
    user_outer["count"] = count
    saveFile(user_outer)
    return outers


def list2tuple(list_1,list_2):
    m = []
    for i in range(len(list_1)):
        m.append((list_1[i],list_2[i]))
    return m

def plot(x_lab,y_lab,outer_l,outer_t,member_id):
    f1 = plt.figure(1)
    plt.scatter(x_lab,y_lab,c="blue")
    plt.scatter(outer_l,outer_t,c="red")
    #plt.show()
    plt.savefig("/Users/homelink/dianping/outer_pic/"+member_id+'.png')

def k_mean(k,user,dataSet,ifCandidate = False,count = None):
    data = mat(dataSet)
    centroids, clusterAssment,candidateList = k_means.kmeans(data, k,ifCandidate)
    user_centroids["user"] = user
    user_centroids["centroids"] = centroids.tolist()
    user_centroids["count"] = count
    if not ifCandidate:
        saveCentroids(user_centroids)
    #k_means.showCluster(data, k, centroids, clusterAssment)
    return candidateList

if __name__ == '__main__':
     review = open("/Users/homelink/dianping/access_1.txt","r")
     centroids_file = open("/Users/homelink/dianping/review_etl/centroids.txt","a")
     outer_file = open("/Users/homelink/dianping/review_etl/outer.txt","a")
     k_means = KMeans()
     for line in review.readlines():
         location = []
         user_centroids = {}
         user_outer = {}
         review_dic = json.loads(line)
         user = review_dic.keys()[0]

         user_review = review_dic.values()[0]
         count = len(user_review)
         if count >15:
             for each in user_review :
                 location.append((float(each["longitude"]),float(each["latitude"])))

             candidate = None
             candidate = k_mean(2,user,location,ifCandidate=True)

             outer = test_normalization_problems(count/2,user,location,user_review,candidate,count)

             for i in range(len(outer)):
                 lof = outer[i]["lof"]
                 if lof >1.2:
                     # outer_l.append(outer[i]["instance"][0])
                     # outer_t.append(outer[i]["instance"][1])
                     temp = outer[i]["instance"]
                     location.remove(temp)

         # plot(x_lab,y_lab,outer_l,outer_t,member_id)
             k_mean(2,user,location,False,count)



