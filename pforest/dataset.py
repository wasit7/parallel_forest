"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Tue Oct 14 18:52:01 2014

@author: Wasit
"""
import numpy as np
import os
#from PIL import Image
#from scipy.ndimage import filters
try:
    import json
except ImportError:
    import simplejson as json
#1800
#num_img=100
#spi=5
#
#rootdir="dataset"
#mrec=64
#mtran=64
#margin=mrec+mtran
class dataset:
    def __init__(self,index=0,n_proposal=100):
        '''
        To create and initialise        
        self.dimtheta--(m)dimension of theta. theta is a column vector
        self.size------(n)number of samples in the root bag
        self.I---------prepocessed data
        self.samples---the marix which has size of [(p+1)xn],
                       where p is size of vector that identify location 
                       of a sample in self.I. 
                       Note that the fist row of self.sample is label
        '''
        
        #1 self.cmax: maximum number of classes
        #2 self.spi: number of samples per image [removed]
        #3 self.theta_dim: the number of elements in a theta (a number of parameter in theta)    
        #4 self.size: number of all samples in the root bag
        #5 self.I: the data
        #6 self.samples: samples[x]=[class]
        #7 self.theta_range: range of theta for generating value in getParam()
        '''
        Example: In order to extract LBP feature, the possible setup is theta_dim=5 
        when 4 dimensions is used to indicate the 2 corners of rectangular window. 
        The last dimension represent the bin of the LBP histogram.  
        Then we can set theta=[r1, c1, r2, c2, bin]^T
        In this particular case (|theta| = 5 ). The theta dimension is called "theta_dim"
        In the getParam() the random proposals are generated by random funtion within a curtain range, which is called "theta_range".
        #3 self.theta_dim: 
        # r1,r2 {margin~rmax-margin},
        # c1,c2 {margin~cmax-margin}, 
        # bin {0~3}
        # L1(r1c1)----L2(r1c2)
        #  |            |
        # L3(r2c1)----L4(r2c2)
        '''
        import pickle
        self.n_proposal=n_proposal
        self.index=index
        self.path='train/dataset%02d.pic'%(self.index)
        pickleFile = open(self.path, 'rb')
        self.clmax,self.theta_dim,self.theta_range,self.size,self.samples,self.I,pos = pickle.load(pickleFile)
        if self.samples is None:
            self.samples=np.zeros(self.I.shape[0])
        self.samples.astype(np.uint8)
        pickleFile.close()                   
    def __str__(self):
        return '\tdatset_pickle: path=./"%s" cmax=%d, theta_dim=%d, theta_range=%d \n\
        \tsize=%d, label.shape=%s, I.shape=%s'\
        %(self.path,self.clmax,self.theta_dim,self.theta_range,self.size,self.samples.shape,self.I.shape)
    def __del__(self):
        del self.clmax
        del self.theta_dim
        del self.theta_range
        del self.size
        del self.samples#samples contains only label
        del self.I
    def getX(self):
        '''
        input: 
            void
        output: 
            [1D ndarray dtype=np.uint32]
        '''
        return np.random.permutation(self.size)
    def getL(self,x):
        '''
        input: 
            [1D ndarray dtype=np.uint32]
        output: 
            [1D ndarray dtype=np.uint32]
        '''
        return self.samples[x]
    def setL(self,x,L):
        '''
        input:
            x: [1D ndarray dtype=np.uint32]
            L: [1D ndarray dtype=np.uint32]
        '''
        self.samples[x]=L
###here
    def getIs(self,thetas,x):
        '''
        input:
            x: [1D ndarray dtype=np.uint32]\n
            thetas: [2D ndarray float]
        output: 
            [1D ndarray dtype=float]
        Description:
            In spiral case, it uses only first row of the thetas
        '''
        #dataset.getParam() calls this
        #theta and x have same number of column
        #3 self.theta_dim: [0_r1, 1_c1, 2_r2, 3_c2, 4_bin]^T
        # r1,r2 {margin~rmax-margin},
        # c1,c2 {margin~cmax-margin}, 
        # bin {0~3}
        # L1(r1c1)----L2(r1c2)
        #  |            |
        # L3(r2c1)----L4(r2c2)
    ##########
        #6 self.samples: samples[x]=[0_class, 1_img, 2_row, 3_column]^T        
#        r1=self.samples[2,x]+thetas[0,:]
#        c1=self.samples[3,x]+thetas[1,:]
#        r2=self.samples[2,x]+thetas[2,:]
#        c2=self.samples[3,x]+thetas[3,:]
#        bins=thetas[self.theta_dim-1,:]
#        f=np.zeros(len(x))
#        for i,ix in enumerate(x):
#            img=self.samples[1,ix]
#            L1=self.I[img][r1[i],c1[i],bins[i]]
#            L2=self.I[img][r1[i],c2[i],bins[i]]
#            L3=self.I[img][r2[i],c1[i],bins[i]]
#            L4=self.I[img][r2[i],c2[i],bins[i]]
#            f[i]=float(L4+L1-L2-L3)

##need to check
        f=np.zeros(len(x))
        for i in xrange(len(x)):
            f[i]=self.I[x[i],thetas[i,0]]
        return f
        
    def getI(self,theta,x):
        '''
        input:
            x: [1D ndarray dtype=np.uint32]\n
            theta: [1D ndarray float]
        output: 
            [1D ndarray dtype=float]
        Description:
            In spiral case, it uses only first row of the thetas
        '''
        #engine.getQH() call this
##original        
#        r1=self.samples[2,x]+theta[0]
#        c1=self.samples[3,x]+theta[1]
#        r2=self.samples[2,x]+theta[2]
#        c2=self.samples[3,x]+theta[3]
#        bins=theta[self.theta_dim-1]
#        f=np.zeros(len(x))
#        for i,ix in enumerate(x):
#            img=self.samples[1,ix]
#            L1=self.I[img][r1[i],c1[i],bins]
#            L2=self.I[img][r1[i],c2[i],bins]
#            L3=self.I[img][r2[i],c1[i],bins]
#            L4=self.I[img][r2[i],c2[i],bins]
#            f[i]=float(L4+L1-L2-L3)
#        return f
        f=np.zeros(len(x))
        f=self.I[x[:],theta[0]]
        return f
        
    def getParam(self,x):
        '''
        input:
            x: [1D ndarray dtype=np.uint32]
        output:
            thetas: [2D ndarray float] rmax=len(x), cmax=theta_dim
            taus: [1D ndarray dtype=np.uint32]
        Description:
            In spiral case, it uses only first row of the thetas
        '''
        #3 self.theta_dim: [0_r1, 1_c1, 2_r2, 3_c2, 4_bin]
        #6 self.samples: samples[x]=[0_class, 1_img, 2_row, 3_column]^T
        
#        N=len(x)//1 #divided by minbagsize  
        N=len(x)
        if N>self.n_proposal:
            x=np.random.permutation(x)[:self.n_proposal]
#        else:
#            x=np.random.permutation(x)[:N]
#        print x
        #ux=np.random.randint(-mtran,mtran,size=len(x))
        #uy=np.random.randint(-mtran,mtran,size=len(x))
        #hx=np.random.randint(8,mrec,size=len(x))
        #hy=np.random.randint(8,mrec,size=len(x))
        #bins=np.random.randint(0,self.dim_bin,size=len(x))
        
        thetas=np.zeros((len(x),self.theta_dim))
        thetas[:,0]=np.random.randint(0,self.theta_range,size=len(x))
        thetas.astype(int)
        taus = self.getIs(thetas, x)
        return thetas,taus
    
    def show(self):
        #show dataset
        print self.samples
if __name__ == '__main__':
#    import matplotlib.pyplot as plt
    dset=dataset()
    print dset
    x=dset.getX()
    
    
    
    
#    print("number of images: {}".format(len(dset.I)))    
#    markers=['ko','ro','go','bo','po']
#    for i in xrange(len(dset.jsonfiles)):
#        f=open(dset.jsonfiles[i],"r")
#        js=json.loads(f.read())
#        f.close()
#        img_path= rootdir + js['path'][1:]
#        print(img_path)
#        im=np.array(Image.open(img_path).convert('L'))
#        plt.hold(False)        
#        plt.imshow(im)
#        plt.hold(True)
#        for j in range(dset.size):
#            #samples[x]=[0_class,1_img, 2_row, 3_column]^T
#            if dset.samples[1,j]==i:
#                plt.plot(dset.samples[3,j],dset.samples[2,j],markers[dset.samples[0,j]])
#        plt.set_cmap('gray')
#        plt.show()
#        plt.ginput()
#    plt.close('all')
#--