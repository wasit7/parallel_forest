"""
The engine of the pforest

This module contain definition of the engine class that is
use by the pforest.

GNU GENERAL PUBLIC LICENSE Version 2
Created on Mon Oct 13 18:56:19 2014

@author: Wasit
"""

import numpy as np
#from scdataset_spiral import dataset
#import imp

class engine:
    """
    The engine class that will be load to each instant of IPython's engine.

    This class provide ways for the master node to manipulate the bags
    that are distributed to each IPython's engins.

    Normally, the parallel forest is represent as a tree, but that could
    introduce a large amount of network traffic to keep trees in sync
    when the trees are distributed across IPython's parallel engines.
    Therefore, the engine class utilize the queue concept to reduce the
    command need to invoke to maintain the trees in sync with the others
    across IPython's engines.
    """

    def __init__(self,dset):
        """
        Initialize the engine.

        The dset argument is the dataset that will be used by this engine.
        """

        self.bag=None
        self.queue=None
        self.dset=dset
#        self.dsetname=dsetname
#        loader= imp.load_source('dataset', self.dsetname+'.py')
#        self.dset==loader.dataset()
    def __del__(self):
        del self.bag
        del self.queue
        del self.dset
    def __repr__(self):
        return "        dset:{}[{}]\n".format(self.dset.__module__,self.dset.size)

    def pop(self):
        """Pop the next bag to be manipulate out of the queue."""
        self.bag=self.queue.pop()
        
    def reset(self):
        """
        Reset the variable in the engine.

        The reset is usually for the new calculation.

        intput: 
            void
        output:
            H [float] entropy the root bag
            Q [float] entropy the root bag
        """

        del self.bag
        del self.queue        
        self.bag=self.dset.getX()
        self.queue=[self.bag]
        H,Q=self.getH(self.bag)
        return H,Q
    
    def getH(self,bag):
        """
        Calculate and return the entropy and size of the current bag.

        input:
            void
        output:
            H [float] entropy of the current bag
            Q [integer] size of the current
        Description:
            to get entropy of the current bag
        """
        Q=len(bag)
#        if Q>0:
#            p=np.bincount(self.dset.getL(bag),minlength=self.dset.clmax)
#        else:
#            p=np.zeros(self.dset.clmax,dtype=np.float32)
        p=np.bincount(self.dset.getL(bag),minlength=self.dset.clmax)
        H=entropy(p)
#        print('engine>>getH H:{}'.format(H))
#        print('engine>>getH p:{}'.format(p))
#        print('engine>>getH Q:{}'.format(Q))
        return H, Q

    def getHist(self):
        """
        Calculate and return the histogram of the current bag.

        This is being called from the master node.

        The returned histrogram is in numpy array format.
        """
        if  0 < len(self.bag):
            hist = np.bincount(self.dset.getL(self.bag), minlength=self.dset.clmax)
            hist = np.array(hist, dtype=np.float)
        else:
            hist = np.zeros(self.dset.clmax, dtype=np.float)
        return hist

    def getParam(self):
        """Shortcut to get the parameter from dataset."""
        return self.dset.getParam(self.bag)

    def getQH(self, all_thetas, all_taus):
        """
        master passes all_thetas and all_taus to this function
        in order to find ensemble entropy
        """
        #number of attamp
        att=len(all_taus)        
        QHs=np.zeros(att)
        for a in xrange(att):
            bagL=self.bag[self.dset.getI(all_thetas[a,:],self.bag)<all_taus[a]]
            bagR=self.bag[self.dset.getI(all_thetas[a,:],self.bag)>=all_taus[a]]
            HL,QL=self.getH(bagL)            
            HR,QR=self.getH(bagR)
#            print("engine>>bagL:{}".format(bagL))
#            print("engine>>bagR:{}\n".format(bagR))
            QHs[a]=QL*HL+QR*HR
        
        return QHs,len(self.bag)
        
    def split(self,theta,tau):
        '''
        Splitting the current bag by the final theta and tau.

        After theta and tau are decided by the master node, the master
        node will pass them to each engine object to be used to split
        the current bag.

        The two bags that are the result of the operation will be added
        to the queue for processing later.

        This method also return the the result of the spliting back to
        the master node.

        input:
            theta [1D ndarray] the numner of element equals to
                dataset.dim_theta
            tau [float] threshold
        output:
            (HL,QL,HR,QR) [all  scalar]
        master passes theta and tau to this function 
        in order to split the current bag
        '''
        bagL=self.bag[self.dset.getI(theta,self.bag)<tau]
        bagR=self.bag[self.dset.getI(theta,self.bag)>=tau]
        self.queue.append(bagL)
        self.queue.append(bagR)
#        print("engine>>split()")
        HL,QL=self.getH(bagL)            
        HR,QR=self.getH(bagR)
        
        return HL,QL,HR,QR
        
def entropy(p):
    """Helper function for the engine.getH"""
    p=p + np.finfo(np.float32).tiny
    p=p/np.sum(p)
    return np.inner(-p,np.log2(p))

if __name__ == '__main__':
    from dataset_pickle import dataset
    dset=dataset()
    eng=engine(dset)
    eng.reset()

    all_thetas,all_taus=eng.dset.getParam(eng.bag)
#    print all_thetas
#    print all_taus
#    print eng.bag
#    bagL=eng.bag[eng.dset.getI(all_thetas[:,0],eng.bag)<all_taus[0]]
#    print bagL
    QHs,Q=eng.getQH(all_thetas,all_taus)
    print("QHs:\n{}".format( QHs)) 
    print("Q:{}".format( Q))