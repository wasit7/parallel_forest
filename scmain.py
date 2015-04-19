"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Mon Oct 13 18:50:34 2014

@author: Wasit
"""
import sys
import pickle
from scmaster import master
from sctree import tree
import imp
import numpy as np
import time
import datetime
def timestamp(ti=time.time()):
    tf=time.time()    
    print("    took: %.2f sec"%(tf-ti))
    return tf
def train(dsetname='dataset_pickle'):
    ts=time.time()
    strtime=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print("Starting time: "+strtime)
    print("----main::train>> init dset, dsetname: {}".format(dsetname))
    #training
    m=master(dsetname)
    ts=timestamp(ts)
        
    print("----main::train>> m.reset()")   
    m.reset()
    ts=timestamp(ts)
    
    #print("main>>H,Q:".format(m.reset()))
    print("----main::train>>m.trian()")
    strtime=datetime.datetime.fromtimestamp(ts).strftime('%m%d_%H%M_%S')
    m.train(strtime)     
    ts=timestamp(ts)
    #recording the tree pickle file
    print("----main::train>>recording")
    rfile=dsetname+'/'+strtime+'.pic'
    pickleFile = open(rfile, 'wb')
    pickle.dump(m.root, pickleFile, pickle.HIGHEST_PROTOCOL)
    pickleFile.close()
    return rfile

def recall(dsetname='dataset_pickle', rfile=''):
    ts=time.time()
    strtime=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print("Starting time: "+strtime)
    
    #reading the tree pickle file
    pickleFile = open(rfile, 'rb')
    root = pickle.load(pickleFile)
    pickleFile.close()
    #init the test tree
    t=tree()
    t.settree(root)
    print("----main::recall::loadtree") 
    ts=timestamp(ts)
    
    #compute recall rate
    loader= imp.load_source('dataset', dsetname+'.py')
    dset=loader.dataset()
    correct=0;
    for x in xrange(dset.size):
        cL=dset.getL(x)#cL:correct label
#print prob
        p=t.getP(np.array([x]),dset)
        ids = p.argsort()[::-1][:3]
        L=ids[0]        
        
#print max likelihood
        #L=t.getL(np.array([x]),dset)
        
        if  cL== L:
            correct=correct+1
        print("\n%03d: correct L"%cL)
        for i in xrange(len(ids)):
            print("%03d_%03d"%(ids[i],100*p[ids[i]])),

        dset.setL(x,L)
    print("recall rate: {}%".format(correct/float(dset.size)*100))
    print("----main::recall::evaluate") 
    ts=timestamp(ts)
    return t, dset

    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:main.py dsetname [optional: mode]')
        print(">>ipython main.py dsetname")
        rfile=train('dataset_pickle')
        t,dset=recall('dataset_pickle',rfile)
        dset.show()
    elif len(sys.argv) == 2:
        rfile=train(sys.argv[1])
        t,dset=recall(sys.argv[1],rfile)
    elif len(sys.argv) == 3:
        if sys.argv[2]=='show':
            t,dset=recall(sys.argv[1],rfile)
            dset.show()
        if sys.argv[2]=='profile':
            import cProfile, pstats, StringIO
            pr = cProfile.Profile()
            pr.enable()
            # ... do something ...
            rfile=train(sys.argv[1])
            pr.disable()
            s = StringIO.StringIO()
            #sortby = 'cumulative'
            sortby = 'tot'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(50)
            print s.getvalue()