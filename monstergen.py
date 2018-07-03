# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:11:58 2018

@author: Durandal
"""
import cv2 as cv
import numpy as np
import tensorflow as tf
import os

def test():
    os.chdir('C:\\Users\\Durandal\\Pokemon')
    
    img1 = read_image('(900).png')
    img2 = read_image('(899).png')
    
    mse = ((img1.astype(float) - img2.astype(float)) ** 2.0).mean(axis=None)
    print(mse)
    
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    ret, img1_bin = cv.threshold(img1_gray, 127, 255, 0)
    img1_cont, contours, hierarchy = cv.findContours(img1_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img1, contours, -1, (0,255,0), 3)
    cv.namedWindow( "Components", 1 );
    cv.imshow( "Components", img1);
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    name = np.full([10], 2**31-1, dtype = int)
    dist = np.full([10], 2.0**31.0-1.0, dtype = float)
    for x in range(1, 2437):
        temp_img = read_image("(" + str(x) + ").png")
        mse = ((img1.astype(float) - temp_img.astype(float)) ** 2.0).mean(axis=None)
        if mse < np.amax(dist):
            dist[np.argmax(dist)] = mse
            name[np.argmax(dist)] = x
    
    for x in name:
        temp_img = read_image("(" + str(x) + ").png")
        cv.imshow( "Components", temp_img);
        cv.waitKey(0)
        cv.destroyAllWindows()
        
def read_image(name):
    img_raw = cv.imread(name, cv.IMREAD_UNCHANGED)
    #Extract alpha channel and color channels seperately
    img_alpha = img_raw[:,:,3]
    img = img_raw[:,:,:3]
    #Turn transparent pixels white
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img_alpha[x][y] == 0:
                img[x][y] = [255, 255, 255]
    #Copy to avoid typing error -- necessary, but unsure why
    img = img.copy()
    return img

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

#estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
#x = np.array([1., 2., 3., 4.])
#y = np.array([0., -1., -2., -3.])
#input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
#estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
#print(estimator.evaluate(input_fn=input_fn, steps=10))

def sim(img1, img2):
    fimg1 = img1.astype(float)
    fimg2 = img2.astype(float)
    mean = 0.0
    for x in range(len(fimg1)):
        for y in range(len(fimg1[0])):
            for c in range(3):
                mean += (fimg1[x,y,c] - fimg2[x,y,c]) ** 2 / (len(fimg1) * len(fimg1[0]) * 3)
    return mean

def filter_img(img):
    img_gauss = cv.GaussianBlur(img, (3, 3), 0.1)
    img_gauss = cv.cvtColor(img_gauss, cv.COLOR_BGR2GRAY)
    img_laplace = cv.Laplacian(img_gauss, cv.CV_16S, 3)
    abs_img_laplace = cv.convertScaleAbs(img_laplace)
    (thresh, img_bin) = cv.threshold(abs_img_laplace, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    return img_bin

#print(sim(img, img2))

#Incomplete image similarity metric
#blur = cv.GaussianBlur(img, (3, 3), 0.1)
#blur_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
#blur_laplace = cv.Laplacian(blur_gray, cv.CV_16S, 3)
#abs_blur_laplace = cv.convertScaleAbs(blur_laplace)
#(thresh, bin_img) = cv.threshold(abs_blur_laplace, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#kernel = np.array([[2, 4, 8], [256, 1, 16], [128, 64, 32]], np.int)
#filter_img = np.array(np.zeros((96,96), dtype=np.int))
#for x in range(96):
#    for y in range(96):
#        trans = 0
#        for i in range(3):
#            for j in range(3):
#                if 96 > (x+i-1) >= 0 and 96 > (y+j-1) >= 0:
#                    trans += kernel[i][j] * bin_img[x+i-1][y+j-1]
#        filter_img[x][y] = trans

#for s in np.arange(0.0, 1.0, 0.05):
#    blur = cv.GaussianBlur(img, (3, 3), s)
#    blur_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
#    blur_laplace = cv.Laplacian(blur_gray, cv.CV_16S, 3)
#    abs_blur_laplace = cv.convertScaleAbs(blur_laplace)
#    (thresh, bin_img) = cv.threshold(abs_blur_laplace, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#    cv.imshow('image',bin_img)
#    cv.waitKey(0)
#    cv.destroyAllWindows()

#for i in 1:2436:
#    img2 = cv.imread('(899).png')
#    mse = ((img - B) ** 2).mean(axis=None)

test()