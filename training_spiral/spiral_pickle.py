# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:22:57 2015

@author: Wasit
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
clmax=10
spc=1e3
theta_range=2
samples=np.zeros(spc*clmax,dtype=np.uint32)
I=np.zeros((spc*clmax,theta_range),dtype=np.float32)
mark_sym=['*','+','.','|','x','^','s','o']
mark_colr=['r','g','b','k']
N=8 #number of datasets being generated
path="spc%.1g/"%spc
if not os.path.exists(path):
    os.makedirs(path)
for n in xrange(N):
    for cl in xrange(clmax):
        xo=cl*spc
        #define label
        samples[xo:xo+spc]=cl
        phi = np.linspace(0, 2*np.pi, spc) + \
        np.random.randn(spc)*0.4*np.pi/clmax + \
        2*np.pi*cl/clmax
        r = np.linspace(0.1, 1, spc)
        I[xo:xo+spc,:]=np.transpose(np.array([r*np.cos(phi), r*np.sin(phi)]))
        #mark=mark_colr[cl%len(mark_colr)]+mark_sym[(cl//len(mark_colr))%len(mark_sym)]
        #plt.plot(I[xo:xo+spc,0],I[xo:xo+spc,1],mark)
        #plt.hold(True)
    
    with open(path+'dataset%02d.pic'%(n), 'wb') as pickleFile:
    #write label and feature vector
        theta_dim=1
        pickle.dump((clmax,theta_dim,theta_range,len(samples),samples,I,None), pickleFile, pickle.HIGHEST_PROTOCOL)
    #pickleFile.close()