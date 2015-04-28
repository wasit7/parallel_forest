# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:48:08 2015

@author: Sarunya
"""
import os
import sys
import pickle
import numpy as np


##create clients
from IPython import parallel
c = parallel.Client(packer='pickle')
c.block = True
#print(c.ids)

##create direct view
dview = c.direct_view()
dview.block = True
#print len(dview)


#if __name__ == '__main__':
#    global 
dsetname = './dataset'
ddesname = 'fft_dataset'
clmax = 11 #clmax is amount of class
theta_dim = 1
images_files = []
class_files = []

isTrain = True #train (test: False)
for root, dirs, files in os.walk(dsetname):
    for f in files:
        if f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('PNG'):
            # read image to array (PIL) 
            images_files.append(os.path.join(root,f))
            
            img_name = os.path.basename(os.path.join(root,f))
            file_name = img_name.split(".")
            if isTrain is True:
                # check image don't have file type 'bmp'
                if os.path.isfile(os.path.join(root , 'bmp/' + file_name[0] + '.bmp')) == False:
                    print "plese label" , root , img_name
                    sys.exit()#break
                else:
                    class_files.append(os.path.join(root , 'bmp/' + file_name[0] + '.bmp'))

dview.execute("from Engine_fft import *")
dview['images_files']=images_files
dview['class_files']=class_files
dview['isTrain']=isTrain
dview.execute("vs ,cs = getVector(images_files,class_files,20,isTrain)")
vs=np.array(dview.gather('vs'))
cs=np.array(dview.gather('cs'))
k = 0
if cs[0] is None:
    cs = None
    
if not os.path.exists(ddesname):
    os.makedirs(ddesname)
rfile = ddesname +'/'+ 'dataset%02d.pic'%(k)
pickleFile = open(rfile, 'wb')
theta_range = vs.shape[1]
size = vs.shape[0]
samples = cs
I = vs
pickle.dump((clmax,theta_dim,theta_range,size,samples,I), pickleFile, pickle.HIGHEST_PROTOCOL)
pickleFile.close()
k = k+1



