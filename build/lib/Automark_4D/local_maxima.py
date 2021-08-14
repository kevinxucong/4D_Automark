import numpy as np
import scipy.ndimage.filters as filters

def median_smooth(data,size=3):
    '''
    data: m*n np.array
    '''
    return(filters.median_filter(data, size=size))

def local_maximum(data_smooth, size=3,threshold=1):
    '''
    data_smooth: m*n np.array
    threshold: return as local_max only if local_max-local_min > threshold
    return: a Boolean np.array
    '''
    data_max = filters.maximum_filter(data_smooth, size)
    maxima = (data_smooth == data_max)
    data_min = filters.minimum_filter(data_smooth, size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    return(maxima)