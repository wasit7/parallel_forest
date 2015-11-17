# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:22:57 2015

@author: Wasit
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

def gen_data():
    clmax=5
    spc=1e3
    theta_range=2
    samples=np.zeros(spc*clmax,dtype=np.uint32)
    I=np.zeros((spc*clmax,theta_range),dtype=np.float32)
    mark_sym=['*','+','.','|','x','^','s','o']
    mark_colr=['r','g','b','k']
    N=8 #number of datasets being generated
    #path="%.1g/"%spc
    path="train/"
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

def train():
    from pforest import master
    m=master.master()
    m.reset()
    m.train()
    
    pickleFile = open('out_tree.pic', 'wb')
    pickle.dump(m.root, pickleFile, pickle.HIGHEST_PROTOCOL)
    pickleFile.close()



def show_result():
    import pickle
    from matplotlib import pyplot as plt      
    from pforest import dataset
      
    pickleFile = open('out_tree.pic', 'rb')
    root = pickle.load(pickleFile)
    pickleFile.close()
    
    #init the test tree
    t=tree()
    t.settree(root)
    t.show()
    #compute recall rate
    dset=dataset()
    correct=0;
    for x in xrange(dset.size):
        L=t.getL(np.array([x]),dset)
        if dset.getL(x) == L:
            correct=correct+1
        dset.setL(x,L)
    print("recall rate: {}%".format(correct/float(dset.size)*100))


        
    #setup the new test-set
    d=0.01
    y, x = np.mgrid[slice(-1, 1+d, d), slice(-1, 1+d, d)]
    #create dataset       
    dset2=dataset()
    
    #start labeling   
    
    L=np.zeros(x.shape,dtype=int)
    for r in xrange(x.shape[0]):
        for c in xrange(x.shape[1]):
            Prob=t.classify(( x[r,c],y[r,c] ))
            L[r,c]=np.argmax(Prob)
    
    #plot the lalbel out put
    plt.close('all')
    plt.axis([-1,1,-1,1])
    plt.pcolor(x,y,L)
    plt.show()
    
    #overlaying new input data
    plt.hold(True)
    plt.set_cmap('jet')
    marker=['bo','co','go','ro','mo','yo','ko',
            'bs','cs','gs','rs','ms','ys','ks']
    z=np.random.randint(0,dset.size,1000)
    for i in z:
        plt.plot(dset2.I[i,0],dset2.I[i,1],marker[dset2.samples[i]])

if __name__ == '__main__':
    gen_data()
    train()
    show_result()