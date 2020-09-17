from __future__ import print_function
# coding: utf-8

import math
import numpy as np
# import random


# In[16]:

PI = math.pi


# In[17]:

DBL_MAX = 10**37
MAX_1D_ENTRY = 262145
MAX_ENTRY = MAX_1D_ENTRY
MAX_2D_ENTRY = 2050


# In[18]:

get = {}


# In[19]:

def read_data(filename):
    try:
        fpin = open(filename,"r")
        raw = f.read()
        x = [int(j) for j in raw.split()]
        num = len(get[x])
        fpin.close()
        
        return (x, num)
    except:
        print("Cannot open input file(s)")


# In[155]:

def read_2ddata(filename):
    try:
        fpin = open(filename,"r")
        raw = fpin.readlines()
        nrow, ncol = [int(j) for j in raw[0].split()]
        if (nrow > MAX_2D_ENTRY) or (ncol > MAX_2D_ENTRY):
            print("Image too large.  Try increase MAX_2D_ENTRY.")
        more_data = reduce(lambda a,b:a+" "+b, raw[1:]).split()
        x = np.array([[0.0]*ncol]*nrow)
        
        k = 0
        for i in range(nrow):
            for j in range(ncol):
                x[i,j] = float(more_data[k])
                k += 1
        if k <len(more_data):
            print("wrong image dimension")
        fpin.close()
        return(x, nrow, ncol)
    except:
        print("Cannot open input file(s)")


# In[158]:

# In[157]:


# In[83]:

def save_data(x, filename, num):
    try:
        fpout = open(filename,"w")
        for i in xrange(num):
            fpout.write(" ".join([str(j) for j in x[i]]))
            fpout.write("\n")
        fpout.close()
    except:
        print("Cannot open output file(s)")


# In[84]:

#save_data(x,r"C:\Users\kevin\study\image_segmentation\rwork.revision\try",3)


# In[146]:

def save_image(x, filename, nrow, ncol):
    try:
        fpout = open(filename, "w")
        fpout.write("{} {}\n" .format(str(nrow), str(ncol)))
        
        for i in range(nrow):
            for j in range(ncol):
                fpout.write("{} " .format(str(x[i,j])))
            fpout.write("\n")
        fpout.close()
    except:
        print("Cannot open output file(s)")


# In[106]:

#filename = r"C:\Users\kevin\study\image_segmentation\rwork.revision\try"


# In[147]:

#save_image(x,r"C:\Users\kevin\study\image_segmentation\rwork.revision\try",5,2)


# In[148]:

def save_char_image(x, filename, nrow, ncol):
    try:
        fpout = open(filename, "w")
        fpout.write("{} {}\n" .format(str(nrow), str(ncol)))
        
        for i in range(nrow):
            for j in range(ncol):
                fpout.write("{} " .format(str(int(x[i,j]))))
            fpout.write("\n")
        fpout.close()
    except:
        print("Cannot open output file(s)")


# In[154]:

#random.gauss(0,1)

