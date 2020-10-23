#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:24:44 2020

@author: ericlyons
"""
import numpy as np
from numpy import sin,cos,pi,sqrt,zeros,size,arctan2,exp,cross
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim,ylim,xlabel,ylabel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize
from scipy import stats
from scipy.optimize import curve_fit
from numpy.random import randn
import pandas as pd

data = np.loadtxt('30.txt',skiprows = 1)
#x = data[:,0]
#y = data[:,1]

data = np.loadtxt('60.txt',skiprows = 1)
#x1 = data[:,0]
#y1 = data[:,1]

data = np.loadtxt('90.txt',skiprows = 1)
#x2 = data[:,0]
#y2 = data[:,1]

data = np.loadtxt('120.txt', skiprows = 1)
#x3 = data[:,0]
#y3 = data[:,1]

data = np.loadtxt('300.txt', skiprows = 1)
#x4 = data[:,0]
#y4 = data[:,1]




#data = np.loadtxt('pic296.txt',skiprows = 1)
#x2 = data[:,0]
#y2 = data[:,1]
#
#data = np.loadtxt('pic300.txt',skiprows = 1)
#x3 = data[:,0]
#y3 = data[:,1]
#
#
#data = np.loadtxt('pic312.txt',skiprows = 1)
#x4 = data[:,0]
#y4 = data[:,1]



import imageio

image1 = (imageio.imread('(0).tif'))
x1 = np.array([i for i in range(6000)])
y1 = 4000 - np.argmax(image1, axis=0)

#image2 = (imageio.imread('(30).tif'))
#x2 = np.array([i for i in range(6000)])
#y2 = 4000 - np.argmax(image2, axis=0)

image3 = (imageio.imread('(60).tif'))
x3 = np.array([i for i in range(6000)])
y3 = 4000 - np.argmax(image3, axis=0)

image4 = (imageio.imread('(90).tif'))
x4 = np.array([i for i in range(6000)])
y4 = 4000 - np.argmax(image4, axis=0)

image5 = (imageio.imread('(120).tif'))
x5 = np.array([i for i in range(6000)])
y5 = 4000 - np.argmax(image5, axis=0)


plt.figure(figsize = (12,9))
plt.plot(x1,y1,'b.', label = '0 seconds')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.ylim(1700,1850)
plt.xlim(2625,4600)


#plt.plot(x2,y2, 'r.', label = '30 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')


plt.plot(x3,y3,'g.', label = '30 seconds')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')


plt.plot(x4,y4,'m.', label = '60 seconds')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')


plt.plot(x5,y5,'k.', label = '120 seconds')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc = "upper left")

plt.figure(figsize = (12,9))
plt.plot(x1,y1,'b.', label = '300 seconds')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

z = scipy.stats.pearsonr(x1,y1)
k = np.corrcoef(x1,y1)
c = pd.DataFrame([(1,2,3),(4,5,6),(7,8,9)])

print(z)
print(k)
print(c)

    





#plt.figure(figsize=(12,9))
#plt.ylim(0,3999.5)
#plt.xlim(0,6000)
#plt.xlabel('x(pixels)')
#plt.ylabel('y(pixels)')
#plt.plot(x,y, 'b.')
#print(y)


























#plt.figure(figsize = (9,6))
#plt.plot(x,y,'b.', label = '0 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.ylim(1100,1400)
#plt.xlim(1000,4900)
#plt.legend()

#plt.plot(x1,y1, 'r.', label = '30 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.ylim(1100,1400)
#plt.xlim(1000,4900)
#plt.legend()

#plt.plot(x2,y2,'g.', label = '60 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.ylim(1100,1400)
#plt.xlim(1000,4900)
#plt.legend()

#plt.plot(x3,y3,'m.', label = '90 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.ylim(1100,1400)
#plt.xlim(1000,4900)
#plt.legend()

#plt.plot(x4,y4,'k.', label = '300 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.ylim(1100,1400)
#plt.xlim(1000,4900)
#plt.legend()
#plt.plot(x3,y3, 'y.', label = '140 seconds')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')

#plt.ylim(1600,1900)
#plt.xlim(2200,4900)

