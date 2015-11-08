# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:07:57 2015

@author: Wasit
"""
from matplotlib import pyplot as plt
import pickle

'''
clmax: a maximum number of classes
theta_dim: How many elements in the theta for each proposal generated in dataset.getParam()?
theta_range: the range for individual elements in the theta_dim
size: a number of samples
samples: class of samples. Originally this is a data locator to identify where is the raw data, e.g. paths/files or (r,c) of the images.
I: the input data
'''

print 'Loading the input file'
pickleFile = open('dataset00.pic', 'rb')
clmax,theta_dim,theta_range,size,samples,I,unused_pos = pickle.load(pickleFile)
pickleFile.close()

print 'Drawing the graph'
mark_sym=['*','+','.','|','x','^','s','o']
mark_colr=['r','g','b','k']
for i in xrange(size):
    cl=samples[i]
    mark=mark_colr[cl%len(mark_colr)]+mark_sym[(cl//len(mark_colr))%len(mark_sym)]
    plt.plot(I[i,0],I[i,1],mark)
#    plt.hold(True)