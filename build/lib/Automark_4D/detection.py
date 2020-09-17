import numpy as np
import math
import os
from scipy.special import factorial
# import time
# import json
# import itertools

# from multiprocessing import Pool, Lock

from Automark_4D import srgrow_poisson_t_multi_band_exptime
from Automark_4D import mdlsegi_poisson_exptime
from Automark_4D import local_maxima

def get_initial_seed(data, n_grid=8, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1):
    
    """
    Get the initial seeds based on local maximum and regular grids
    
    Input:
    data: (num_band, T1, m, n) np.array
    n_grid: nonnegative int, number of regular grids per row/column
    par_median_smooth: positive int, window size for median smoothing
    par_local_maximum_size: positive int, window size for local maximum
    par_local_maximum_threshold: nonnegetive int, threshold for local maximum
    
    Output:
    label_im: (m*n) list, -1 denote unassigned pixels, and consecutive nonnegative integers start from 0 denotes initial seeds
    """
    
    num_band, T, m, n = data.shape
    local_max = np.zeros((m,n))
    for nb in range(num_band):
        local_max += local_maxima.local_maximum(local_maxima.median_smooth(
            np.sum(data[nb], axis=0), par_median_smooth), par_local_maximum_size, par_local_maximum_threshold)
    local_max = local_max > 0

    label_im = [-1] * m * n
    n_init = -1
    for i in range(m):
        for j in range(n):
            if local_max[i, j]:
                n_init += 1
                label_im[i * n + j] = n_init
            elif n_grid > 0:
                if i%math.ceil(m/float(n_grid)) == (m//(2*n_grid)) and j%math.ceil(n/float(n_grid)) == (n//(2*n_grid)):
                    n_init += 1
                    label_im[i * n + j] = n_init
    return label_im

def get_fitted(data, bg, exptimes, init_seed_im=None, n_grid=8, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1):
     
    """
    Get the fitted labels and the corresponding MDL
    
    Input:
    data: (num_band, T1, m, n) np.array
    bg: num_band list
    exptimes: T1 list
    init_seed_im: (m*n) list
    n_grid: nonnegative int, number of regular grids per row/column
    par_median_smooth: positive int, window size for median smoothing
    par_local_maximum_size: positive int, window size for local maximum
    par_local_maximum_threshold: nonnegetive int, threshold for local maximum
    
    Output:
    label_im: (m*n) list, pixel(s) with the same nonnegative 
    """
    num_band, T1, m, n = data.shape
    observe_im = np.zeros((num_band, T1, m*n))
    for i in range(m):
        for j in range(n):
            observe_im[:, :, i * n + j] = data[:, :, i, j]

    exptimes1 = exptimes
    if not init_seed_im:
        label_im = get_initial_seed(data, n_grid, par_median_smooth, par_local_maximum_size, par_local_maximum_threshold)
    else:
        label_im = init_seed_im
        
    srgrow_poisson_t_multi_band_exptime.rsgrow(observe_im, label_im, m, n, bg, exptimes1)

    a = min(label_im)
    label_im = [i - a + 1 for i in label_im]
    num_merge = 1
    mdl_out_grey = [[0.0] * m * n] * num_band
    mdl_out_region = [0] * m * n
    num_over = max(label_im)
    mdl_curve = [0.0] * 2 * (num_over - num_merge)
    mdlsegi_poisson_exptime.segment_mdl_ind(observe_im, label_im, m, n, num_merge, num_band, bg, exptimes1,
                                            mdl_out_region, mdl_out_grey, mdl_curve)
    num_region = max(mdl_out_region)
    permtx = mdlsegi_poisson_exptime.init_perimeter_matrix(mdl_out_region, m, n, num_region)
    graphs = [0] * num_band
    for nb in range(num_band):
        graphs[nb] = mdlsegi_poisson_exptime.image_to_graph(observe_im[nb], mdl_out_region, permtx, m, n, num_region, exptimes1)
    MDL = mdlsegi_poisson_exptime.compute_mdl(graphs, permtx, m, n, num_band, bg)
    return MDL, mdl_out_region

def backward_elimination(data, bg, exptimes, init_breaks, init_seed_im=None, n_grid=8, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1):
    """
    Using backward elimination to remove redundant change points.
    
    Input:
    data: (num_band, T1, m, n) np.array
    bg: num_band list
    exptimes: T1 list
    init_breaks: list, 1 <= init_breaks[i] < init_breaks[j] < T for all i < j
    init_seed_im: (m*n) list or None
    n_grid: nonnegative int, number of regular grids per row/column
    par_median_smooth: positive int, window size for median smoothing
    par_local_maximum_size: positive int, window size for local maximum
    par_local_maximum_threshold: nonnegetive int, threshold for local maximum
    
    Output:
    label_im: (m*n) list, pixel(s) with the same nonnegative 
    """
    num_band, T, m, n = data.shape
    breaks = init_breaks
    
    M = len(breaks) + 1
    data_list = [0] * M
    data_list[0] = (0, breaks[0])
    for i in range(1, M - 1):
        data_list[i] = (breaks[i - 1], breaks[i])
    data_list[M - 1] = (breaks[M - 2], T)
    
    break_list = [0] * M
    break_list[0] = data_list
    label_list = [[0 for i in range(M - j)] for j in range(M)]
    mdl_list = [[0 for i in range(M - j)] for j in range(M)]
    dict_mdl = {}
    dict_label = {}
    
    # fit models for each time interval
    for l in range(M):
        data1 = data[:, break_list[0][l][0]:break_list[0][l][1]]
        exptimes1 = exptimes[break_list[0][l][0]:break_list[0][l][1]]
        MDL, mdl_out_region = get_fitted(data1, bg, exptimes1, init_seed_im, n_grid, par_median_smooth, par_local_maximum_size, par_local_maximum_threshold)
        
        label_list[0][l] = mdl_out_region
        mdl_list[0][l] = MDL
        
    # merge neighboring time intervals
    for q in range(1, M):
        temp_label_list = [0] * (M - q)
        temp_mdl_list = [0] * (M - q)
        for r in range(M - q):
            if (break_list[q - 1][r][0], break_list[q - 1][r + 1][1]) in dict_mdl:
                temp_label_list[r] = dict_label[(break_list[q - 1][r][0], break_list[q - 1][r + 1][1])]
                temp_mdl_list[r] = dict_mdl[(break_list[q - 1][r][0], break_list[q - 1][r + 1][1])]
            else:
                data1 = data[:, break_list[q - 1][r][0]:break_list[q - 1][r + 1][1]]
                exptimes1 = exptimes[break_list[q - 1][r][0]:break_list[q - 1][r + 1][1]]
                MDL, mdl_out_region = get_fitted(data1, bg, exptimes1, None, n_grid=8, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1)
                temp_label_list[r] = mdl_out_region
                temp_mdl_list[r] = MDL
                
                dict_label[(break_list[q - 1][r][0], break_list[q - 1][r + 1][1])] = mdl_out_region
                dict_mdl[(break_list[q - 1][r][0], break_list[q - 1][r + 1][1])] = MDL
                
        comparing_list = [0] * (M - q)
        for r in range(M - q):
            comparing_list[r] = temp_mdl_list[r] - mdl_list[q - 1][r] - mdl_list[q - 1][r + 1]

        min_increase = comparing_list[0]
        K = 0
        for r in range(1, M - q):
            if comparing_list[r] < min_increase:
                min_increase = comparing_list[r]
                K = r

        break_list[q] = break_list[q - 1][0:K] + [(break_list[q - 1][K][0], break_list[q - 1][K + 1][1])] + \
                        break_list[q - 1][K + 2:M]
        label_list[q] = label_list[q - 1][0:K] + [temp_label_list[K]] + label_list[q - 1][K + 2:M]
        mdl_list[q] = mdl_list[q - 1][0:K] + [temp_mdl_list[K]] + mdl_list[q - 1][K + 2:M]

    merge_mdl_curve = [0] * M
    for q in range(M):
        merge_mdl_curve[q] = (M - q - 1) * math.log(T) + sum(mdl_list[q])

    min_mdl = merge_mdl_curve[0]
    K = 0
    for r in range(1, M):
        if merge_mdl_curve[r] <= min_mdl:
            min_mdl = merge_mdl_curve[r]
            K = r
    final_break_ls = break_list[K]
    final_label_ls = label_list[K]
    return final_break_ls, final_label_ls, min_mdl, merge_mdl_curve, K, break_list, label_list, mdl_list

