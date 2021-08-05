# modify the factorial
# use Stirling's approximation: log(n!) approx= n*log(n) - n + 0.5*log(2*pi*n) + 1/(12*n)
# when n is large

# use priority queue (heapq) 

# update .sort_n() in case when vals are the same, wrong element is put in the tail

import numpy as np
import heapq
from scipy.special import factorial

class PIXEL:  # neighboring pixel
    def __init__(self, pos, value):
        self.pos = pos  # int: position
        self.value = value  # num_band*T np.matrix

class GREGION:
    def __init__(self, size, tmean, nnbr, pxl_lst, exptimes, bg):
        self.size = size # int
        self.tmean = tmean # num_band-list
        self.nnbr = nnbr # int
        self.pxl_lst = pxl_lst # list of neighboring PIXEL, where the last element has the smallest neg logilikelihood
        self.exptimes = exptimes # list of floats
        self.bg = bg #num_band-list
        
    def num_band(self):
        return self.bg.shape[0]

    def sort_n(self):
        if self.nnbr > 0:
            par = np.array([max(a,b) for a,b in zip(self.tmean, self.bg)])
            exptimes = np.array(self.exptimes)
            # negative log likelihood
            min_val = float('inf')
            min_val_pos = -1
            idx = 0
            for i,x in enumerate(self.pxl_lst):
                val = -np.sum(np.log(np.dot(par[:,None], exptimes[None,:]))*x.value) + np.sum(np.array(list(map(fac_robust, x.value.flatten()))))
                if val < min_val or (val == min_val and x.pos > min_val_pos):
                    idx = i
                    min_val = val
                    min_val_pos = x.pos
            self.pxl_lst[-1], self.pxl_lst[idx] = self.pxl_lst[idx], self.pxl_lst[-1]
        
    def remove_tail(self):
        # absorb the PIXEL in the last location of self.pxl_lst
        otmean = self.tmean
        self.tmean = (otmean * self.size + np.sum(self.pxl_lst[-1].value, axis=1)/float(sum(self.exptimes))) / float(
            self.size + 1)
        self.size += 1
        self.nnbr -= 1
        self.pxl_lst.pop()

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
            self.pxl_lst.pop(idx)
        
                 
def fac_robust(k):
    """
    Stirling's approximation for log(n!)
    """
    if k <= 170:
        return np.log(factorial(k))
    else:
        return k*np.log(k) - k + 0.5*np.log(2*np.pi*k) + 1.0/(12*k)
    

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
            
        
def init_allregions(obsim, outim, m, n, exptimes, bg):
    up = -n
    down = n
    left = -1
    right = 1
    nseed = max(outim) + 1 # ID: from 0 to (nseed-1)
   
    allregions = [GREGION(0,0,0,[], exptimes, bg) for _ in range(nseed)]
    nlab = 0
    for i in range(m * n):
        if outim[i] >= 0:
            allregions[outim[i]].size += 1
            allregions[outim[i]].tmean += obsim[:, :, i]
            nlab += 1
            # print(outim[i],allregions[outim[i]].size,allregions[1].nnbr)

    for i in range(nseed):
        allregions[i].tmean = np.sum(allregions[i].tmean, axis=1) / float(allregions[i].size * sum(exptimes))  # num_band vector

    for r in range(m):
        for c in range(n):
            i = r * n + c
            if outim[i] == -1:
                if r > 0:
                    check_and_fix_nbr(allregions, obsim, outim, i, up)
                if r < m-1:
                    check_and_fix_nbr(allregions, obsim, outim, i, down)
                if c > 0:
                    check_and_fix_nbr(allregions, obsim, outim, i, left)
                if c < n-1:
                    check_and_fix_nbr(allregions, obsim, outim, i, right)
    
    for i in range(nseed):
        allregions[i].sort_n()
    initmdl = 0

    return allregions, nlab, initmdl
    

def add_nbr(region1, obsim, newpos):
    for i in range(region1.nnbr):
        if region1.pxl_lst[i].pos == newpos:
            break
    else:
        region1.pxl_lst.append(PIXEL(newpos,obsim[:, :, newpos]))
        region1.nnbr += 1

        
def standardize_init_seeds(old_label):
    d = {}
    new_label = list(old_label)
    count = 0
    for i,c in enumerate(new_label):
        if c == -1:
            continue
        if c not in d:
            d[c] = count      
            count += 1
        new_label[i] = d[c]
    return new_label
        
        
def rsgrow(obsim, label_im, m, n, exptimes, bg):
    outim = list(standardize_init_seeds(label_im)) # deep copy
    allregions, nlabelled, initmdl = init_allregions(obsim, outim, m, n, exptimes, bg)
    
    working_ls = []
    for i,region in enumerate(allregions):
        if region.nnbr > 0:
            exptimes = np.array(region.exptimes)
            par = np.array([max(a,b) for a,b in zip(region.tmean, region.bg)])
            x = region.pxl_lst[-1].value  # num_band*T np.matrix

            tempscore = -np.sum(np.log(np.dot(par[:,None], exptimes[None,:])) * x) + np.sum(par)*np.sum(exptimes) + \
                    np.sum(np.array(list(map(fac_robust, x.flatten()))))
            working_ls.append((tempscore, i, region.pxl_lst[-1].pos))

    heapq.heapify(working_ls)
    while nlabelled < m * n:
        _, bregion,bpos = heapq.heappop(working_ls)
        while outim[bpos] != -1:
            _, bregion,bpos = heapq.heappop(working_ls)
                         
        outim[bpos] = bregion
        
        allregions[bregion].remove_tail()
        updated_ls = set([bregion])
        if bpos >= n:
            if outim[bpos - n] == -1:
                add_nbr(allregions[bregion], obsim, bpos - n)
            elif outim[bpos - n] >= 0 and outim[bpos - n] != bregion:
                allregions[outim[bpos - n]].remove_neck(bpos)
                updated_ls.add(outim[bpos - n])
                
        if bpos % n != 0:
            if outim[bpos - 1] == -1:
                add_nbr(allregions[bregion], obsim, bpos - 1)
            elif outim[bpos - 1] >= 0 and outim[bpos - 1] != bregion:
                allregions[outim[bpos - 1]].remove_neck(bpos)
                updated_ls.add(outim[bpos - 1])
                
        if bpos % n != n - 1:
            if outim[bpos + 1] == -1:
                add_nbr(allregions[bregion], obsim, bpos + 1)
            elif outim[bpos + 1] >= 0 and outim[bpos + 1] != bregion:
                allregions[outim[bpos + 1]].remove_neck(bpos)
                updated_ls.add(outim[bpos + 1])
                
        if bpos < m * n - n:
            if outim[bpos + n] == -1:
                add_nbr(allregions[bregion], obsim, bpos + n)
            elif outim[bpos + n] >= 0 and outim[bpos + n] != bregion:
                allregions[outim[bpos + n]].remove_neck(bpos)
                updated_ls.add(outim[bpos + n])
                
        
        for region in updated_ls:
            if allregions[region].nnbr > 0:
                allregions[region].sort_n()

                exptimes = np.array(allregions[region].exptimes)
                par = np.array([max(a,b) for a,b in zip(allregions[region].tmean, allregions[region].bg)])
                x = allregions[region].pxl_lst[-1].value  # num_band*T np.matrix

                tempscore = -np.sum(np.log(np.dot(par[:,None], exptimes[None,:])) * x) + np.sum(par)*np.sum(exptimes) + \
                        np.sum(np.array(list(map(fac_robust, x.flatten()))))

                heapq.heappush(working_ls, (tempscore, region, allregions[region].pxl_lst[-1].pos))

        nlabelled += 1
        
    return outim
