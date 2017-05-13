#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2


def write_frame(fr, fname):

# -- get the path to videos
bpath  = os.path.join(os.path.expanduser("~"), "data", 
                      "healthy_neighborhoods", "Camera Observation")

# -- loop through directories
for rr, dd, ff in os.walk(bpath):
