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
import os

import argparse



class log:
    def __init__(self,path=""):
        self.initTime=time.time()
        self.lastTime=self.initTime
        if os.path.isfile(path):
            self.log_file=open(path, 'w')
        else:
            self.log_file=open(path, 'a')
    def __del__(self):
        self.finished("__del__")
        if self.log_file:
            self.log_file.close()
    def finished(self,processName="processName"):
        t=time.time()
        msg="[%7.2fs] [%7.2fs] {%s}"%(t-self.lastTime,t-self.initTime,processName)
        self.log_file.write(msg+"\n")
        self.lastTime=t
        return msg
    def current(self,msg=""):
        self.log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+"\t"+msg+"\n")

def timestamp(ti=time.time()):
    tf=time.time()
    print("    took: %.2f sec"%(tf-ti))
    return tf

def train(dsetname='dataset_pickle'):
    global mylog
    mylog.current("train")
    #training
    m=master(dsetname)
    mylog.finished("main::train>> m=master(dsetname)\nmaster: %s"%m)
    m.reset()
    print(mylog.finished("main::train>> m.reset()"))

    #print("main>>H,Q:".format(m.reset()))
    m.train(uniquename)
    print(mylog.finished("main::train>> m.train(uniquename)"))
    #recording the tree pickle file

    tree_file=os.path.join(path,uniquename+'.pic')
    pickleFile = open(tree_file, 'wb')
    pickle.dump(m.root, pickleFile, pickle.HIGHEST_PROTOCOL)
    pickleFile.close()
    print(mylog.finished("main::train>> recording tree"))
    return tree_file

def recall(dsetname='dataset_pickle', rfile=''):
    global mylog
    mylog.current("recall")

    #reading the tree pickle file
    pickleFile = open(rfile, 'rb')
    root = pickle.load(pickleFile)
    pickleFile.close()
    #init the test tree
    t=tree()
    t.settree(root)
    print(mylog.finished("main::train>> loading tree"))

    #compute recall rate
    loader= imp.load_source('dataset', dsetname+'.py')
    dset=loader.dataset()
    print(mylog.finished("main::train>> loading dset\ndataset: %s"%dset))

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
 #       print("\n%03d: correct L"%cL)
 #       for i in xrange(len(ids)):
 #           print("%03d_%03d"%(ids[i],100*p[ids[i]])),

        dset.setL(x,L)
    print(mylog.finished("main::train>> classifying\nrecall rate: {}%".format(correct/float(dset.size)*100)))

    return t, dset

mylog=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the random forest")
    parser.add_argument("dsetname", help="dataset name")
    parser.add_argument("-m","--mode", help="running mode", choices=["show", "profile"])
    args = parser.parse_args()

    global mylog
    if args.mode == None:
        ##init log file
        dsetname=args.dsetname
        uniquename=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        path=dsetname
        if not os.path.exists(path):
            os.makedirs(path)
        mylog=log(os.path.join(path,uniquename+'.log'))

        tree_file=train(dsetname)
        #print rfile #"0426_1711_51.pic"

        t,dset=recall('dataset_pickle',tree_file)
        #dset.show()
    elif args.mode == "show":
        t,dset=recall(sys.argv[1],rfile)
        dset.show()
    elif args.parse == "profile":
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
