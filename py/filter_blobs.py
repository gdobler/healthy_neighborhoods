#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import copy
import numpy as np

# -- utilities
rad2  = 9.0


for ifile in (1, 2, 3, 4, 5):

    # -- initialize "good" blobs dictionary
    gdict = {}


    # -- read in blobs
    print("read file {0}".format(ifile))
    bname = os.path.join("..", "output","blobs_TLC{0:05}.json".format(ifile))
    bdict = json.load(open(bname, "r"))
    tss   = [str(i) for i in sorted([int(i) for i in bdict.keys()])]
    hdict = copy.copy(bdict)


    # -- for each frame, 
    #        for each blob, 
    #            if there is a blob in the next frame or previous frame within 
    #            the radius remove it from the "good" blobs dictionary
    print("  filtering stationary blobs...")
    for ts0, ts1, ts2 in zip(tss[:-2], tss[1:-1], tss[2:]):
        bc1 = bdict[ts1]
        nbc = len(bc1)
        if nbc == 0:
            continue
        else:
            gdict[ts1] = np.ones(nbc, dtype=bool)
        bcc = bdict[ts0] + bdict[ts2]

        for ii in range(nbc):
            for jj in range(len(bcc)):
                dr2 = (bc1[ii][0] - bcc[jj][0])**2 + \
                    (bc1[ii][1] - bcc[jj][1])**2
                if dr2 <= rad2:
                    gdict[ts1][ii] = False
                    break


    # -- for each entry in the pop dict, pop those blobs from the blob dict
    for key in gdict:
        bdict[key] = [list(i) for i in np.array(bdict[key])[gdict[key]]]

    # -- write to file
    print("  writing to json...")
    fopen = open(os.path.join("..", "output","blobs_filtered_TLC{0:05}.json" \
                            .format(ifile)), "w")
    json.dump(bdict, fopen, sort_keys=True, indent=4)
    fopen.close()

