#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.measurements import label, center_of_mass


def get_blobs(fname):

    print("==-- extracting blobs from {0} --==".format(fname))
    fac   = 8
    nbuff = 101
    cap   = cv2.VideoCapture(fname)
    skip  = 0
    print("skipping {0} frames...".format(skip))
    try:
        nfrm  = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
        dum   = cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip)
        nrow  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        ncol  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    except:
        nfrm  = int(np.round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        dum   = cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
        nrow  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ncol  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img   = np.zeros([nrow, ncol, 3], dtype=np.uint8)
    frm   = np.zeros_like(img)
    bkg   = np.zeros(img.shape, dtype=float)
    buff  = np.zeros([nbuff] + list(img.shape), dtype=np.uint8)
    diff  = np.zeros(img.shape, dtype=float)
    diffb = np.zeros([nrow//fac, ncol//fac], dtype=float)
    blbs  = np.zeros_like(diffb)
    
    print("filling buffer...")
    for ii in range(nbuff):
        st, buff[ii] = cap.read()
    
    print("setting background...")
    bkg = buff.mean(0)
    
    cnt = 0
    
    mask = np.ones_like(diffb)
    mask[:,:40] = 0.0
    mask[:13,:] = 0.0
    mask[88:,:] = 0.0
    
    print("extracting blobs...")
    all_blobs = {}
    
    t0 = time.time()
    for ii in range(nfrm-nbuff):
        if (ii+1)%50==0:
            print("  {0} of {1}".format(ii+1, nfrm-nbuff))
    
        bind       = cnt % nbuff
        img[...]   = buff[(bind + nbuff//2) % nbuff]
        diff[...]  = 1.0*img - bkg
        diffb[...] = np.abs(diff) \
            .mean(2) \
            .reshape(nrow//fac, fac, ncol//fac, fac) \
            .mean(1).mean(-1)
    
        st, frm[...] = cap.read()
        bkg[...]    -= buff[bind] / float(nbuff)
        bkg[...]    += frm / float(nbuff)
        buff[bind]   = frm
    
        blbs[...]                       = (gf(diffb*mask,(3,1))>20)
        labs                            = label(blbs)
        all_blobs[ii + skip + nbuff//2] = center_of_mass(blbs, labs[0], 
                                                         range(1,labs[1]+1))
        cnt += 1
    
    print("dt = {0}".format(time.time()-t0))
    
    print("writing to json...")
    fopen = open("../output/blobs_{0}.json" \
                 .format(fname.split("/")[-1].split(".")[0]), "w")
    json.dump(all_blobs, fopen, sort_keys=True, indent=4)
    fopen.close()
    
    return


# -- get file list
dpath = os.path.join(os.path.expanduser("~"),
                     "data/healthy_neighborhoods/Camera Observation/" + \
                     "Syracuse - Near Westside/")
flist = [i for i in sorted(glob.glob(os.path.join(dpath,"*.AVI")))]

for fname in flist:
    get_blobs(fname)
