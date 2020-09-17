# modify the factorial
# use Stirling's approximation: log(n!) approx= n*log(n) - n + 0.5*log(2*pi*n) + 1/(12*n)
# when n is large

# also use approximation when sorting the neighboring pixels

# Suppose observations from different bands are independent

import math
import sys
import os
import numpy as np
from .basic import *
from scipy.special import factorial


PI = math.pi
MAX_NBR = 100000  # max no. of nbr pixels of a region


class PIXEL:  # neighboring pixel
    def __init__(self, pos, value):
        self.pos = pos  # int: position
        self.value = value  # num_band*T np.matrix


class GREGION:
    def __init__(self, size, tmean, nnbr, pxl_lst, bg, exptimes):
        self.size = size # int
        self.tmean = tmean # num_band-list
        self.nnbr = nnbr # int
        self.pxl_lst = pxl_lst # list of PIXEL
        self.bg = bg #num_band-list
        self.exptimes = exptimes # list of floats

    def num_band(self):
        if len(self.tmean) == len(self.bg):
            return len(self.tmean)
        else:
            print("Error")

    def sort_n(self):
        nband = self.num_band()
        par = np.array([0.0 for i in range(nband)])
        for i in range(nband):
            par[i] = max(self.tmean[i], self.bg[i])
        exptimes = np.array(self.exptimes)

#         self.pxl_lst.sort(key=lambda x: -np.sum(np.log(np.dot(par[:,None], exptimes[None,:]))*x.value) + np.sum(np.log(factorial(x.value, exact=False))))
        # negative log likelihood
        self.pxl_lst.sort(key=lambda x: -np.sum(np.log(np.dot(par[:,None], exptimes[None,:]))*x.value) + np.sum(np.array(list(map(fac_robust, x.value.flatten())))))
        
    def remove_head(self):
        otmean = self.tmean
        self.tmean = (otmean * self.size + np.sum(self.pxl_lst[0].value, axis=1)/float(sum(self.exptimes))) / float(
            self.size + 1)
        self.size += 1
        self.nnbr -= 1
        self.pxl_lst = self.pxl_lst[1:]

    def remove_neck(self, neck):
        matched = False
        idx = -1
        for i in range(self.nnbr):
            if self.pxl_lst[i].pos == neck:
                idx = i
                matched = True
                break
        if matched:
            self.nnbr -= 1
            self.pxl_lst = self.pxl_lst[:idx] + self.pxl_lst[(idx + 1):]


def log2(x):
    return (math.log(x) / math.log(2))


def logstar(x):
    ans = 1.518535
    if x > 1:
        temp = log2(x)
        while temp > 0:
            ans = ans + temp
            temp = log2(temp)
    return (ans)

def fac_robust(k):
    """
    Stirling's approximation for log(n!)
    """
    if k <= 170:
        return np.log(factorial(k))
    else:
        return k*np.log(k) - k + 0.5*np.log(2*np.pi*k) + 1/(12*k)

def check_and_fix_nbr(allregions, obsim, outim, i, direction):
    """
    allregions: nseed-list of GREGION (list of all regions)
    i: current position
    direction: up(-n), down(n), left(-1) or right(1)
    outim: (m*n)-list (region IDs for all the pixels) (IDs from 0 to (nseed-1), -1 for ungrouped)
    obsim: (num_band,T,m*n)-np.array
    """
    if outim[i + direction] >= 0:
        tlab = outim[i + direction]
        tnbr = allregions[tlab].nnbr
        for j in range(tnbr):
            if allregions[tlab].pxl_lst[j].pos == i:
                break
        else:
            # allregions[tlab].pos[tnbr] = i
            # allregions[tlab].diff[tnbr] = (allregions[tlab].tmean - obsim[:, :, i].T).T  # tmean/diff[i]: num_band*T
            allregions[tlab].pxl_lst.append(PIXEL(i,obsim[:, :, i]))
            allregions[tlab].nnbr += 1


def init_allregions(obsim, outim, m, n, bg, exptimes):
    up = -n
    down = n
    left = -1
    right = 1
    nseed = max(outim) + 1 # ID: from 0 to (nseed-1)

    nlab = 0

    allregions = [0]*nseed
    for i in range(nseed):
        allregions[i] = GREGION(0,0,0,[],bg, exptimes)

    for i in range(m * n):
        if outim[i] >= 0:
            allregions[outim[i]].size += 1
            allregions[outim[i]].tmean += obsim[:, :, i]
            nlab += 1
            # print(outim[i],allregions[outim[i]].size,allregions[1].nnbr)

    for i in range(nseed):
        allregions[i].tmean = np.sum(allregions[i].tmean, axis=1) / float(allregions[i].size * sum(exptimes))  # num_band vector

    # print(allregions[1].size,allregions[1].nnbr)

    # four corners
    i = 0
    if outim[i] == -1:
        check_and_fix_nbr(allregions, obsim, outim, i, down)
        check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)

    i = n - 1
    if outim[i] == -1:
        check_and_fix_nbr(allregions, obsim, outim, i, down)
        check_and_fix_nbr(allregions, obsim, outim, i, left)
    # print(allregions[1].size,allregions[1].nnbr)

    i = (m - 1) * n
    if outim[i] == -1:
        check_and_fix_nbr(allregions, obsim, outim, i, up)
        check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)

    i = m * n - 1
    if outim[i] == -1:
        check_and_fix_nbr(allregions, obsim, outim, i, up)
        check_and_fix_nbr(allregions, obsim, outim, i, left)
    # print(allregions[1].size,allregions[1].nnbr)

    # top row
    for c in range(1, n - 1):
        i = c
        if outim[i] == -1:
            check_and_fix_nbr(allregions, obsim, outim, i, down)
            check_and_fix_nbr(allregions, obsim, outim, i, left)
            check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)

    # bottom row
    for c in range(1, n - 1):
        i = (m - 1) * n + c
        if outim[i] == -1:
            check_and_fix_nbr(allregions, obsim, outim, i, up)
            check_and_fix_nbr(allregions, obsim, outim, i, left)
            check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)

    # left column
    for r in range(1, m - 1):
        i = r * n
        if outim[i] == -1:
            check_and_fix_nbr(allregions, obsim, outim, i, up)
            check_and_fix_nbr(allregions, obsim, outim, i, down)
            check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)

    # right column
    for r in range(1, m - 1):
        i = r * n + (n-1)
        if outim[i] == -1:
            check_and_fix_nbr(allregions, obsim, outim, i, up)
            check_and_fix_nbr(allregions, obsim, outim, i, down)
            check_and_fix_nbr(allregions, obsim, outim, i, left)
    # print(allregions[1].size,allregions[1].nnbr)

    # remaining
    for r in range(1, m - 1):
        for c in range(1, n - 1):
            i = r * n + c
            if outim[i] == -1:
                check_and_fix_nbr(allregions, obsim, outim, i, up)
                check_and_fix_nbr(allregions, obsim, outim, i, down)
                check_and_fix_nbr(allregions, obsim, outim, i, left)
                check_and_fix_nbr(allregions, obsim, outim, i, right)
    # print(allregions[1].size,allregions[1].nnbr)
    # print(allregions[1].pos)

    for i in range(nseed):
        allregions[i].sort_n()
    initmdl = 0

    return allregions, nlab, initmdl


def get_best_region(allregions, exptimes):
    nseed = len(allregions)

    bregion = 0 # best region
    tregion = 0
    while allregions[tregion].nnbr == 0:
        tregion += 1
        if tregion >= nseed: break
    bregion = tregion

    nband = allregions[bregion].num_band()

    par = np.array([0.0 for j in range(nband)])
    for j in range(nband):
        par[j] = max(allregions[bregion].tmean[j], allregions[bregion].bg[j])

    x = allregions[bregion].pxl_lst[0].value # num_band*T np.matrix
    exptimes = np.array(exptimes)

    bestscore = -np.sum(np.log(np.dot(par[:,None], exptimes[None,:])) * x) + np.sum(par)*np.sum(exptimes) + \
                np.sum(np.array(list(map(fac_robust, x.flatten()))))

    for i in range(tregion + 1, nseed):
        if allregions[i].nnbr > 0:
            par = np.array([0.0 for nb in range(nband)])
            for j in range(nband):
                par[j] = max(allregions[i].tmean[j], allregions[i].bg[j])

            x = allregions[i].pxl_lst[0].value  # num_band*T np.matrix

            tempscore = -np.sum(np.log(np.dot(par[:,None], exptimes[None,:])) * x) + np.sum(par)*np.sum(exptimes) + \
                        np.sum(np.array(list(map(fac_robust, x.flatten()))))

            if tempscore < bestscore:
                bestscore = tempscore
                bregion = i

    if allregions[bregion].nnbr == 0:
        print("t: {}, b: {}, bscore: {}".format(tregion, bregion, bestscore))
        return

    singlemdl = bestscore
    return bregion, singlemdl


def add_nbr(region1, obsim, newpos):
    for i in range(region1.nnbr):
        if region1.pxl_lst[i].pos == newpos:
            break
    else:
        region1.pxl_lst.append(PIXEL(newpos,obsim[:, :, newpos]))
        region1.nnbr += 1


def update_allregions(allregions, bregion, obsim, outim, m, n):
    bpos = allregions[bregion].pxl_lst[0].pos
    allregions[bregion].remove_head()

    if bpos >= n:
        if outim[bpos - n] == -1:
            add_nbr(allregions[bregion], obsim, bpos - n)
        elif outim[bpos - n] >= 0 and outim[bpos - n] != bregion:
            allregions[outim[bpos - n]].remove_neck(bpos)

    if bpos % n != 0:
        if outim[bpos - 1] == -1:
            add_nbr(allregions[bregion], obsim, bpos - 1)
        elif outim[bpos - 1] >= 0 and outim[bpos - 1] != bregion:
            allregions[outim[bpos - 1]].remove_neck(bpos)

    if bpos % n != n - 1:
        if outim[bpos + 1] == -1:
            add_nbr(allregions[bregion], obsim, bpos + 1)
        elif outim[bpos + 1] >= 0 and outim[bpos + 1] != bregion:
            allregions[outim[bpos + 1]].remove_neck(bpos)

    if bpos < m * n - n:
        if outim[bpos + n] == -1:
            add_nbr(allregions[bregion], obsim, bpos + n)
        elif outim[bpos + n] >= 0 and outim[bpos + n] != bregion:
            allregions[outim[bpos + n]].remove_neck(bpos)

    allregions[bregion].sort_n()


def rsgrow(obsim, outim, m, n, bg, exptimes):
    runningmdl = 0

    allregions, nlabelled, initmdl = init_allregions(obsim, outim, m, n, bg, exptimes)

    while nlabelled < m * n:
        bregion, singlemdl = get_best_region(allregions, exptimes)
        runningmdl += singlemdl

        bpos = allregions[bregion].pxl_lst[0].pos
        outim[bpos] = bregion

        update_allregions(allregions, bregion, obsim, outim, m, n)
        nlabelled += 1
