# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:31:34 2015

@author: Sarunya
"""
import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy.ndimage
from scipy.ndimage import filters


sys.setrecursionlimit(10000)
bs = 200
wd = 5# theta_range=wd*wd*2
clmax = 11 #clmax is amount of class

def normFFT(images_file):
    # apply to array
    img = np.array(images_file)
    #converte image to frequency domain
    #f=np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    f = np.log(np.abs(np.fft.fft2(img)))
    #scaling
    s=(100./f.shape[0],100./f.shape[1])
    
    #normalized frequency domian
    return scipy.ndimage.zoom(f,s,order = 2)

def G(x,mu,s):
    return 1.0/ np.sqrt(2.0*np.pi)*np.exp(((x-mu)**2)/(-2.0*s**2))
    
def getValue(images):
 
    f = normFFT(images) #f=[100,100]
    rmax,cmax = f.shape 

    sg = np.zeros((2*wd,wd)) #sg[60,30]
    sg[0:wd,:]=np.log(np.abs(f[rmax-wd:rmax,0:wd])) #sg[0:30,:] = f[70:100,0:30]
    sg[wd:2*wd,:]=np.log(np.abs(f[0:wd,0:wd])) #sg[30:60,:] = f[0:30,0:30]
    
    #filters.gaussian_filter(sg, (3,3), (0,0), sg)
   
#    fsg=np.zeros(wd)
#    for b in xrange(wd):
#        for r in xrange(wd):
#            for c in xrange(wd):
#                rad=np.sqrt(r**2+c**2)            
#                fsg[b]=fsg[b]+sg[wd+r,c]*G(rad,float(b),0.2)+sg[wd-r,c]*G(rad,float(b),0.2)
#        fsg[b]=fsg[b]/(np.pi*float(b+1.0))
#        fsg=fsg/np.linalg.norm(fsg)
#        fsg.astype(np.float32)
    return sg.reshape(-1)
    

def getVector(images_files,class_files,samples, isTrain):
    sub_img = []
    sub_cs = []
    bb = bs//2
    
    for f in xrange(len(images_files)):
        img = Image.open(images_files[f]).convert('L')
        w , h = img.size
        pixels=[]
        for i in xrange(samples):
            r = np.random.randint(bb, h-bb)
            c = np.random.randint(bb, w-bb)
            pixels.append((c,r))
            box = (c-bb, r-bb, c + bb, r + bb)
            output_img = img.crop(box)
            sub_img.append(getValue(output_img))
    
        if isTrain:
            cimg = Image.open(class_files[f]).convert('L')
            for p in pixels:   
                sub_cs.append(cimg.getpixel(p))
    if isTrain:
        sub_img=np.array(sub_img,dtype=np.float32)
        sub_cs=np.array(sub_cs,dtype=np.uint32)
        sub_cs[sub_cs==255]= clmax - 1
    else:
        sub_cs=None       
    return (sub_img ,sub_cs)

"""
if __name__ == '__main__':
    dsetname = './random'
    images_files = []
    class_files = []
    for root, dirs, files in os.walk(dsetname):
        for f in files:
            if f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('PNG'):
                # read image to array (PIL) 
                images_files.append(os.path.join(root,f))
                
                img_name = os.path.basename(os.path.join(root,f))
                file_name = img_name.split(".")
                # check image don't have file type 'bmp'
                if os.path.isfile(os.path.join(root , 'bmp/' + file_name[0] + '.bmp')) == False:
                    print "plese label" , root , img_name
                    cross = 1
                else:
                    class_files.append(os.path.join(root , 'bmp/' + file_name[0] + '.bmp'))

    
    vs ,cs = getVector(images_files,class_files,5,isTrain=True) """
    