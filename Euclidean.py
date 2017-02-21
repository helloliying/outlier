#!/usr/bin/python
#-*- coding: utf8 -*-

import numpy as np

class Euclidean(object):

	def __init__(self):
		pass

	def __checkDim__(self):
		pass

	def caculate(self,A,B):
		BT = B.transpose()
		vecProd = A * BT
		SqA =  A.getA()**2
		sumSqA = np.matrix(np.sum(SqA, axis=1))
		sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
		SqB = B.getA()**2
		sumSqB = np.sum(SqB, axis=1)
		sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
		SqED = sumSqBEx + sumSqAEx - 2*vecProd   
		ED = (SqED.getA())**0.5
		return np.matrix(ED)




