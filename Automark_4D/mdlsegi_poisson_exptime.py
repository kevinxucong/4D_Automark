# use priority queue (heapq)

import numpy as np
import collections
import heapq

class TLREGION:
    def __init__(self, lper, npix, spix):
        """
        Multi-band region
        :param lper: perimeter (int)
        :param npix: area of the region (int)
        :param spix: (num_band, T) np.array of summations of pixels within one region
        """
        self.lper = lper
        self.npix = npix
        self.spix = spix
        self.nlist = set([]) # set of indices of neighboring regions

        
class TLGRAPH:
    def __init__(self, observe_im, inim, m, n, exptimes, bg):
        """
        """
        self.region_ls = collections.defaultdict(TLREGION)
        self.boundary_d = collections.defaultdict(int)
        self.exptimes = exptimes
        self.bg = bg
        self.num_band, self.T, self.mn = np.array(observe_im).shape
        
        for i in np.unique(inim):
            self.region_ls[i] = TLREGION(0, 0, np.zeros((self.num_band, self.T)))

        down = n
        right = 1
        for i in range(m*n):
            # initialize boundary, lper and nlist
            if i + down < m*n:
                if inim[i] != inim[i+down]:
                    self.boundary_d[(min(inim[i],inim[i+down]),max(inim[i],inim[i+down]))] += 1
                    self.region_ls[inim[i]].lper += 1
                    self.region_ls[inim[i+down]].lper += 1
                    self.region_ls[inim[i]].nlist.add(inim[i+down])
                    self.region_ls[inim[i+down]].nlist.add(inim[i])
                    
            if i%n < n-1:
                if inim[i] != inim[i+right]:
                    self.boundary_d[min(inim[i],inim[i+right]),max(inim[i],inim[i+right])] += 1
                    self.region_ls[inim[i]].lper += 1
                    self.region_ls[inim[i+right]].lper += 1
                    self.region_ls[inim[i]].nlist.add(inim[i+right])
                    self.region_ls[inim[i+right]].nlist.add(inim[i])
                    
            # initialize npix, spix
            self.region_ls[inim[i]].npix += 1
            self.region_ls[inim[i]].spix += observe_im[:, :, i]
            
    def merge_regions(self, r1, r2):
        if r1 > r2:
            r1,r2 = r2, r1
        
        self.region_ls[r1].lper += self.region_ls[r2].lper - 2*self.boundary_d[r1,r2]
        self.region_ls[r1].npix += self.region_ls[r2].npix
        self.region_ls[r1].spix += self.region_ls[r2].spix
        self.region_ls[r1].nlist.update(self.region_ls[r2].nlist)
        self.region_ls[r1].nlist.discard(r1)
        self.region_ls[r1].nlist.discard(r2)
        for nei in self.region_ls[r1].nlist:
            self.boundary_d[min(r1, nei), max(r1, nei)] += self.boundary_d[min(r2, nei), max(r2, nei)]
            self.region_ls[nei].nlist.add(r1)
            self.region_ls[nei].nlist.discard(r2)
        self.region_ls.pop(r2)
    
    def compute_neg_loglikelihood(self):
        """
        :param graph1:
        :param bg0: lower bound for Poisson process rate
        :return: negative log likelihood
        """
        exptimes = np.array(self.exptimes)
        bg = np.array(self.bg)
        neg_loglikelihood = 0
        for _, tlregion in self.region_ls.items():
            x = tlregion.spix
            a = tlregion.npix
            neg_loglikelihood -= np.sum(np.sum(x, axis=1) * np.log(np.maximum(np.sum(x, axis=1) / (a*np.sum(exptimes)), bg)) + np.dot(x, np.log(exptimes)))
        return float(neg_loglikelihood)
        
    def compute_mdl(self):
        K = 0
        logarea = 0
        perimeter = 0
        
        for  _, tlregion in self.region_ls.items():
            K += 1
            logarea += np.log((tlregion.npix)*self.T)
            perimeter += tlregion.lper
        
        neg_loglikelihood = self.compute_neg_loglikelihood()
        mdlscore = K*np.log(self.mn) + np.log(3)/2.0 * perimeter + self.num_band*logarea/2.0 + neg_loglikelihood
        return(mdlscore)

    def compute_delta_mdl(self, r1, r2):
        """
        delta MDL if r1 and r2 merge together
        """
        if r1 > r2:
            r1,r2 = r2, r1
            
        tlregion1 = self.region_ls[r1]
        x1 = tlregion1.spix
        a1 = tlregion1.npix
        tlregion2 = self.region_ls[r2]
        x2 = tlregion2.spix
        a2 = tlregion2.npix
        exptimes = np.array(self.exptimes)
        bg = np.array(self.bg)
        
        delta_neglogl = np.sum(np.sum(x1, axis=1) * np.log(np.maximum(np.sum(x1, axis=1) / (a1*np.sum(exptimes)), bg)) + np.dot(x1, np.log(exptimes))) + \
                         np.sum(np.sum(x2, axis=1) * np.log(np.maximum(np.sum(x2, axis=1) / (a2*np.sum(exptimes)), bg)) + np.dot(x2, np.log(exptimes))) - \
                         np.sum(np.sum(x1+x2, axis=1) * np.log(np.maximum(np.sum(x1+x2, axis=1) / ((a1+a2)*np.sum(exptimes)), bg)) + np.dot(x1+x2, np.log(exptimes)))
        delta_logarea = np.log(a1 + a2) - np.log(a1) - np.log(a2)- np.log(self.T)
        delta_mdl = - np.log(self.mn) - np.log(3)/2.0 * 2 * self.boundary_d[r1, r2] + self.num_band*delta_logarea/2.0 + delta_neglogl
        return delta_mdl

    
def relabel_im(label_im, modified):
    """
    index [1, K]
    """
    d = {r2:r1 for r2, r1 in modified}
    def helper(val):
        if val not in d:
            return val
        if d[val] != val:
            d[val] = helper(d[val])
        return d[val]
    
    res = [0]*len(label_im)
    count = 0
    visited = {}
    for i, val in enumerate(label_im):
        new_val = helper(val)
        if new_val not in visited:
            count += 1
            visited[new_val] = count
            
        res[i] = visited[new_val]
    return res

    
def segment_mdl_ind(observe_im, label_im, m, n, exptimes, bg):
    graph = TLGRAPH(observe_im, label_im, m, n, exptimes, bg)
    working_ls = []
    for r1, tlregion1 in graph.region_ls.items():
        for r2 in tlregion1.nlist:
            if r1 < r2:
                working_ls.append((graph.compute_delta_mdl(r1, r2), r1, r2, 0, 0))
    heapq.heapify(working_ls)
    
    times_of_merging = collections.Counter() # total number of oversegments 
    
    mdl_ls = [graph.compute_mdl()]
    modified_ls = []
    
    K = len(graph.region_ls)
    for _ in range(K-1):
        delta_mdl, r1, r2, c1, c2 = heapq.heappop(working_ls)
        while r1 not in graph.region_ls or r2 not in graph.region_ls or times_of_merging[r1] > c1 or times_of_merging[r2] > c2:
            delta_mdl, r1, r2, c1, c2 = heapq.heappop(working_ls)

        mdl_ls.append(mdl_ls[-1] + delta_mdl)
        modified_ls.append((r2, r1)) # r2 merge to r1
        times_of_merging[r1] += times_of_merging[r2] + 1
        
        graph.merge_regions(r1, r2)
        for nei in graph.region_ls[r1].nlist:
            heapq.heappush(working_ls, (graph.compute_delta_mdl(r1, nei), min(r1, nei), max(r1,nei), times_of_merging[min(r1, nei)], times_of_merging[max(r1, nei)]))
    
    label_im_merging = relabel_im(label_im, modified_ls[:int(np.argmin(mdl_ls))])
    
    return float(np.min(mdl_ls)), label_im_merging
