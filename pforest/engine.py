"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Mon Oct 13 18:56:19 2014

@author: Wasit
"""

import numpy as np
#from scdataset_spiral import dataset
#import imp
class engine:
    def __init__(self,dset):
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
        self.bag=self.queue.pop()
        
    def reset(self):
        '''
        intput: 
            void
        output:
            H [float] entropy the root bag
            Q [float] entropy the root bag
        '''
        del self.bag
        del self.queue        
        self.bag=self.dset.getX()
        self.queue=[self.bag]
        H,Q=self.getH(self.bag)
        return H,Q
    
    def getH(self,bag):
        '''
        input:
            void
        output:
            H [float] entropy of the current bag
            Q [integer] size of the current
        Description:
            to get entropy of the current bag
        '''
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
        '''
        histogram of the current bag
        '''
        if  0 < len(self.bag):
            hist = np.bincount(self.dset.getL(self.bag), minlength=self.dset.clmax)
            hist = np.array(hist, dtype=np.float)
        else:
            hist = np.zeros(self.dset.clmax, dtype=np.float)
        return hist
    def getParam(self):
        return self.dset.getParam(self.bag)
    def getQH(self, all_thetas, all_taus):
        '''
        master passes all_thetas and all_taus to this function
        in order to find ensemble entropy
        '''
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
        input:
            theta [1D ndarray] the numner of element equals to dataset.dim_theta\n
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