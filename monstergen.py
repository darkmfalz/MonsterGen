# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:11:58 2018

@author: Durandal
"""

import cv2
import numpy as np
import tensorflow as tf
import os

os.chdir('C:\\Users\\Durandal\\Pokemon')
img = cv2.imread('(1).png')
print(img[40][38][:])
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()