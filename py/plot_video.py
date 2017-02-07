#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

ii = 2

# -- load the blobs
bfile = "../output/blobs_TLC{0:05}.json".format(ii)
blobs = json.load(open(bfile, "r"))

# -- load the video
vfile = os.path.join(os.path.expanduser("~"), "data/healthy_neighborhoods/" 
                     "Camera Observation/Syracuse - Near Westside", 
                     "TLC{0:05}.AVI".format(ii))
cap = cv2.VideoCapture(vfile)

# -- initialize the image
skip = 1500
cnt  = skip
print("skipping {0} frames...".format(skip))
st, fr = cap.read()
try:
    dum   = cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip)
except:
    dum   = cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
print("done")

fsx = 8.0
fsy = fsx * float(fr.shape[0]) / float(fr.shape[1])
fig, ax = plt.subplots(figsize=(fsx, fsy))
fig.subplots_adjust(0, 0, 1, 1)
tblobs  = blobs[str(cnt)]
npnts   = len(tblobs)
xs, ys  = [], [] if npnts==0 else np.array(tblobs).T
pnts,   = ax.plot(ys*8, xs*8, "ro", ms=15)
im      = ax.imshow(fr[...,::-1])
txt     = ax.text(fr.shape[1], fr.shape[0] - 0.05 * fr.shape[0], 
                  "frame:{0}, counts:{1}".format(cnt, npnts), color="w", 
                  ha="right", fontsize=15)
plt.ion()
plt.show()

# -- loop through frames
for ii in range(0, 3000):
    cnt += 1
    st, fr = cap.read()
    tblobs  = blobs[str(cnt-1)]
    npnts   = len(tblobs)
    if npnts > 0:
        xs, ys  = np.array(tblobs).T
        pnts.set_data(ys*8, xs*8)
    else:
        pnts.set_data([], [])
    im.set_data(fr[...,::-1])
    txt.set_text("frame:{0}, counts:{1}".format(cnt, npnts))
    fig.canvas.draw()
    plt.pause(0.2)
