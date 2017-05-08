#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import copy

# -- utilities
rad2  = 9.0
gdict = {}


# -- read in blobs
bname = os.path.join("..", "output","blobs_TLC00001.json")
bdict = json.load(open(bname, "r"))
tss   = sorted([int(i) for i in bdict.keys()])
hdict = copy.copy(bdict)


# -- for each frame, 
#        for each blob, 
#            if there is a blob in the next frame or previous frame within 
#            the radius remove it and add this index to the pop dict
for ts0, ts1, ts2 in zip(tss[:-2], tss[1:-1], tss[2:]):
    sts = str(ts1)
    bc1 = bdict[sts]
    nbc = len(bc1)
    if nbc == 0:
        continue
    else:
        gdict[sts] = np.ones(nbc, dtype=bool)
    bcc = bdict[str(ts0)] + bdict[str(ts2)]

    for ii in range(nbc):
        for jj in range(len(bcc)):
            dr2 = (bc1[ii][0] - bcc[jj][0])**2 + (bc1[ii][1] - bcc[jj][1])**2
            if dr2 <= rad2:
                gdict[sts][ii] = False
                break


# -- for each entry in the pop dict, pop those blobs from the blob dict
for key in gdict:
    bdict[key] = [list(i) for i in np.array(bdict[key])[gdict[key]]]
