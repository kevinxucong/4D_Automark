from __future__ import print_function
from .SetADT import *
from .basic import *


# In[2]:

import math
import sys
import os


# In[3]:

PI = math.pi


# In[4]:

def log(x):
    return(math.log(x))


# In[5]:

MAX_NBR = 17000
MAX_REGION = 17000
MERGED = -167
RSSthresh = 0.000005
SMALLRSS = 0.0


# In[6]:

class TLREGION:
    def __init__(self, ID, lper, npix, spix):
        """
        :param ID: from 1 to nseed
        :param lper: perimeter (int)
        :param npix: area of the region (int)
        :param spix: (T,) np.array of summations of pixels within one image ([s_1,...,s_T])
        """
        self.ID = ID
        self.lper = lper
        self.npix = npix
        self.spix = spix


# In[7]:

class TLGRAPH:
    def __init__(self, nregion, rlist, nlist, exptimes):
        """
        :param nregion:
        :param rlist: list of TLREGION
        :param nlist: list of neighboring regions
        :param exptimes: list of exposure times
        """
        self.nregion = nregion
        self.rlist = rlist
        self.nlist = nlist
        self.exptimes = exptimes


# In[8]:

def test_n_nbr(n):
    if n > MAX_NBR:
        print("Error: maximum number of neighbours exceeded!")
        exit()


# In[9]:

def image_to_graph(observe,inim,permtx,m,n,num_region,exptimes):
    """
    :param observe: one-band of data (T*(m*n) np.array)
    :param inim:
    :param permtx:
    :param m:
    :param n:
    :param num_region:
    :param exptimes: list of exposure times
    :return:
    """
    rlist = [0]*(num_region+1)
    T = len(exptimes)
    for i in range(num_region+1):
        rlist[i] = TLREGION(i,permtx[i,i],0, np.zeros(T))
        
    for i in range(m*n):
        rlist[inim[i]].npix += 1
        rlist[inim[i]].spix += observe[:,i]

    for i in range(1,num_region+1):
        if rlist[i].npix == 0:
            print("Internal error: check if the region in overseg are consecutively labelled")
            print("Try relabel")
            return
        
    nlist = [0]*(num_region+1)
    for i in range(num_region+1):
        nlist[i] = [0]*(MAX_NBR+1)
    
    """
     Add regions next to each other in the same row into the neighbors
    """
    
    for i in range(m):
        old_id = inim[i*n]
        for j in range(1,n):
            new_id = inim[i*n+j]
            if old_id != new_id:
                n_nbr = nlist[old_id][0]
                already_in_nlist = False
                for k in range(1,n_nbr+1):
                    if nlist[old_id][k] == new_id:
                        already_in_nlist = True
                        break
                else:
                    nlist[old_id][n_nbr+1] = new_id
                    nlist[old_id][0] = n_nbr+1
                    test_n_nbr(n_nbr+1)
                    """
                    if old_id == 256:
                        print(old_id,new_id)
                        print(nlist[old_id][0:n_nbr+2])
                    """
                        
                n_nbr = nlist[new_id][0]
                
                if not already_in_nlist:
                    nlist[new_id][n_nbr+1] = old_id
                    nlist[new_id][0] += 1
                    
                    """
                    if new_id == 256:
                        print(new_id,old_id)
                        print(nlist[new_id][0:n_nbr+2])
                    """ 
            old_id = new_id
        
    for j in range(n):
        old_id = inim[j]
        for i in range(1,m):
            new_id = inim[i*n+j]
            if old_id != new_id:
                n_nbr = nlist[old_id][0]
                already_in_nlist = False
                for k in range(1,n_nbr+1):
                    if nlist[old_id][k] == new_id:
                        already_in_nlist = True
                        break
                else:
                    nlist[old_id][n_nbr+1] = new_id
                    nlist[old_id][0] = n_nbr + 1
                    test_n_nbr(n_nbr+1)
                    """
                    if old_id == 256:
                        print(old_id,new_id)
                        print(nlist[old_id][0:n_nbr+2])
                    """
                n_nbr = nlist[new_id][0]
                
                if not already_in_nlist:
                    nlist[new_id][n_nbr+1] = old_id
                    nlist[new_id][0] = n_nbr + 1
                    test_n_nbr(n_nbr+1)
                    
                    """
                    if new_id == 256:
                        print(new_id,old_id)
                        print(nlist[new_id][0:n_nbr+2])
                    """
                    
            old_id = new_id

    graph1 = TLGRAPH(num_region,rlist,nlist,exptimes)
    return(graph1)


# In[10]:
"""
def print_graph(graph1):
    print("number of regions: {}".format(graph1.nregion))
    for i in range(1,graph1.nregion+1):
        print("{} {} {} {}".format(i,graph1.rlist[i].npix,
                                   graph1.rlist[i].spix/graph1.rlist[i].npix,
                                   graph1.rlist[i].s2pix))
    for i in range(1,graph1.nregion+1):
        print("{} {}".format(i,graph1.nlist[i][0]),end = " ")
        for j in range(1,graph1.nlist[i][0]+1):
            print("{}".format(graph1.nlist[i][j]), end = " ")
        print()
"""

# In[11]:

def init_perimeter_matrix(inim, m, n, num_region):
    """
    initiate the perimeter matrix (numpy.array)
    
    permtx[r1,0] == 0: region r1 is merged/removed
    permtx[r1,0] == 1: region r1 is ok for merging
    permtx[r1,r1]: length of region r1's boundary
    permtx[r2,r1]: length of common boundary of r2 and r1 (r1 < r2)
    """
    permtx = np.array([[0]*(num_region+1)]*(num_region+1))
    for i in range(0,num_region+1):
        permtx[i,0] = 1
    
    #four corners
#     l = 0
#     if inim[l] != inim[l+1]:
#         permtx[max(inim[l],inim[l+1]),min(inim[l],inim[l+1])]
            
    up = -n
    down = n
    left = -1
    right = 1
    
    for l in range(m*n):
        if l+up >= 0:
            if inim[l] != inim[l+up]:
                permtx[max(inim[l],inim[l+up]),min(inim[l],inim[l+up])] += 1
        if l+down <= m*n-1:
            if inim[l] != inim[l+down]:
                permtx[max(inim[l],inim[l+down]),min(inim[l],inim[l+down])] += 1
                
        if l%n != 0:
            if inim[l] != inim[l+left]:
                permtx[max(inim[l],inim[l+left]),min(inim[l],inim[l+left])] += 1
                
        if l%n != n-1:
            if inim[l] != inim[l+right]:
                permtx[max(inim[l],inim[l+right]),min(inim[l],inim[l+right])] += 1
                
    for i in range(1,num_region+1):
        for j in range(1,i):
            permtx[i,i] += permtx[i,j]
        for j in range(i+1,num_region+1):
            permtx[i,i] += permtx[j,i]
            
    # each edge is calculated twice
    for i in range(1,num_region+1):
        for j in range(1,num_region+1):
            permtx[i,j] = (permtx[i,j])/2
            
    return(permtx)


# In[12]:

def update_perimeter_matrix(permtx,r1,r2,num_region):
    if permtx[r1,0] <= 0 or permtx[r2,0] <= 0:
        print("Internal error: check if the regions in overseg are consecutively labelled")
        print("Try relabel")
        return
    
    permtx[r2,r2] += permtx[r1,r1] - 2*permtx[r2,r1]
    
    for i in range(1,r1):
        permtx[r2,i] += permtx[r1,i]
    for i in range(r1+1,r2):
        permtx[r2,i] += permtx[i,r1]
    for i in range(r2+1,num_region+1):
        permtx[i,r2] += permtx[i,r1]
        
    for i in range(r1+1):
        permtx[r1,i] = 0
    for i in range(r1+1,num_region+1):
        permtx[i,r1] = 0


# In[13]:

def compute_likelihood(graph1, bg0):
    """

    :param graph1:
    :param bg0: lower bound for Poisson process rate
    :return: negative log likelihood
    """
    exptimes = np.array(graph1.exptimes)
    full_likelihood = 0
    for i in range(1,graph1.nregion+1):
        if graph1.nlist[i][0] != MERGED:
            full_likelihood -= np.sum(graph1.rlist[i].spix)*np.log(max(np.sum(graph1.rlist[i].spix)/(graph1.rlist[i].npix*np.sum(exptimes)), bg0)) \
                               + np.sum(graph1.rlist[i].spix*np.log(exptimes))
    full_likelihood = float(full_likelihood)
    return(full_likelihood)


# In[14]:

def compute_mdl(graphs, permtx, m, n, num_band, bg):
    K = 0
    logarea = 0
    perimeter = 0
    T = len(graphs[0].exptimes)
    for i in range(1, graphs[0].nregion+1):
        if graphs[0].nlist[i][0] != MERGED:
            K += 1
            logarea += math.log((graphs[0].rlist[i].npix)*T)
            perimeter += permtx[i,i]
    
    neg_loglikelihood = 0
    for nb in range(num_band):
        neg_loglikelihood += compute_likelihood(graphs[nb], bg[nb])
    
    mdlscore = K*log(m*n) + math.log(3)/2*perimeter + num_band*logarea/2 + neg_loglikelihood
    return(mdlscore)


# In[15]:

def best_merging_pair(graphs,likeli,mdlscore,permtx,m,n,num_band, bg):
    """
    Return the pair of regions such that when these two regions are merged, the new mdl 
    score is minimum amongst all other possible merges
    
    Input: graph1, permtx, rss and mdlscore of graph1 before merging
    Output: two regions (region1 < region2), new rss and mdlscore
    """
    region1, region2 = [0,0]
    new_likeli = [DBL_MAX]*num_band
    old_likeli = [0]*num_band
    temp_likeli = [0]*num_band
    for i in range(num_band):
        old_likeli[i] = likeli[i]
    new_score = DBL_MAX
    old_score = mdlscore
    mn = m*n
    logmn = math.log(mn)
    cl = math.log(3)/2
    exptimes = np.array(graphs[0].exptimes)
    T = len(graphs[0].exptimes)

    for i in range(1,graphs[0].nregion+1):
        if (graphs[0].nlist[i][0] != MERGED) and (graphs[0].nlist[i][0] != 0):
            for k in range(1,graphs[0].nlist[i][0]+1):
                j = graphs[0].nlist[i][k]
                if (i<j) and (graphs[0].nlist[j][0] != MERGED):
                    for l in range(num_band):
                        temp_likeli[l] = old_likeli[l] +\
                                         np.sum(graphs[l].rlist[i].spix) * np.log(max(np.sum(graphs[l].rlist[i].spix) / (graphs[l].rlist[i].npix * np.sum(exptimes)), bg[l])) + \
                                         np.sum(graphs[l].rlist[i].spix * np.log(exptimes)) + \
                                         np.sum(graphs[l].rlist[j].spix) * np.log(max(np.sum(graphs[l].rlist[j].spix) / (graphs[l].rlist[j].npix * np.sum(exptimes)), bg[l])) + \
                                         np.sum(graphs[l].rlist[j].spix * np.log(exptimes)) - \
                                         np.sum(graphs[l].rlist[i].spix + graphs[l].rlist[j].spix) * np.log(max(np.sum(graphs[l].rlist[i].spix + graphs[l].rlist[j].spix) / ((graphs[l].rlist[i].npix + graphs[l].rlist[j].npix) * np.sum(exptimes)), bg[l])) - \
                                         np.sum((graphs[l].rlist[i].spix+graphs[l].rlist[j].spix) * np.log(exptimes))

                    d_area = (log(graphs[0].rlist[i].npix + graphs[0].rlist[j].npix) -
                             log(graphs[0].rlist[i].npix) - log(graphs[0].rlist[j].npix)-log(T))
                    temp_score = old_score - logmn - cl*2*permtx[j,i] + num_band*d_area/2
                    
                    for l in range(num_band):
                        temp_score += -old_likeli[l] + temp_likeli[l]
                        
                    if temp_score < new_score:
                        new_score = temp_score
                        for l in range(num_band):
                            new_likeli[l] = temp_likeli[l]
                        region1 = i
                        region2 = j
                        
    for l in range(num_band):
        likeli[l] = new_likeli[l]
       
    mdlscore = new_score
    
    return(region1, region2, likeli, mdlscore)

# In[16]:

def merge_regions(graphs,r1,r2,permtx,num_band):
    """
    merge region r1 and r2, and give updated graph and permtx
    """
    update_perimeter_matrix(permtx,r1,r2,graphs[0].nregion)
    for j in range(num_band):
        for i in range(1,graphs[j].nregion+1):
            if (i not in [r1,r2]) and (graphs[j].nlist[i][0] != MERGED):
                if is_an_element(graphs[j].nlist[i], r1):
                    remove_element(graphs[j].nlist[i], r1)
                    add_element(graphs[j].nlist[i], r2, MAX_NBR)
        
        get_union(graphs[j].nlist[r1],graphs[j].nlist[r2],MAX_NBR)
        remove_element(graphs[j].nlist[r2],r2)
        remove_element(graphs[j].nlist[r2],r1)
        
        test_n_nbr(graphs[j].nlist[r2][0])
        
        graphs[j].nlist[r1][0] = MERGED
        graphs[j].rlist[r2].lper = permtx[r2,r2]
        graphs[j].rlist[r2].npix += graphs[j].rlist[r1].npix
        graphs[j].rlist[r2].spix += graphs[j].rlist[r1].spix


# In[17]:

def init_maps(graph1):
    region_map = [0]
    grey_map = [0.0]
    exptimes = graph1.exptimes
    for i in range(1,graph1.nregion+1):
        region_map.append(i)
        grey_map.append(float(np.sum(graph1.rlist[i].spix)/((graph1.rlist[i].npix)*np.sum(exptimes))))
        
    return(region_map, grey_map)


# In[18]:

def update_maps(region_map,grey_map,graph1,region1,region2):
    exptimes = graph1.exptimes
    region_map[region1] = region2
    for i in range(1,graph1.nregion+1):
        if region_map[i] == region1:
            region_map[i] = region2
    for i in range(1,graph1.nregion+1):
        grey_map[i] = float(np.sum(graph1.rlist[region_map[i]].spix)/(graph1.rlist[region_map[i]].npix*np.sum(exptimes)))


# In[19]:

def copy_maps(region_map1,grey_map1,region_map2,grey_map2,graph1):
    for i in range(1,graph1.nregion+1):
        region_map2[i] = region_map1[i]
        grey_map2[i] = grey_map1[i]


# In[20]:

def relabel_region_map(input_region_map,num_region):
    """
    relabel regions into 1,2,...,num_distinct_region
    """
    packed_region_map = [0]*(num_region+1)
    packed_region_map[1] = input_region_map[1]
    num_distinct_region = 1
    
    for i in range(2,num_region+1):
        for j in range(1,num_distinct_region+1):
            if input_region_map[i] == packed_region_map[j]:
                break
        else:
            num_distinct_region += 1
            packed_region_map[num_distinct_region] = input_region_map[i]
            
    for i in range(1,num_region+1):
        for j in range(1,num_distinct_region+1):
            if input_region_map[i] == packed_region_map[j]:
                input_region_map[i] = j


# In[21]:

def segment_mdl_ind(observe_im, label_im,m,n,num_merge,num_band,bg, exptimes,
                    mdl_out_region,mdl_out_grey,mdl_curve):
    
    num_region = max(label_im)
    
    graphs = [0]*num_band
    mdl_region_map = [0]*(num_region+1)
    current_region_map = [0]*(num_region+1)
    mdl_grey_map = [[0.0]*(num_region+1)]*num_band
    current_grey_map = [[0.0]*(num_region+1)]*num_band
    
    permtx = init_perimeter_matrix(label_im,m,n,num_region)
    
    for i in range(num_band):
        graphs[i] = image_to_graph(observe_im[i],label_im,permtx,m,n,num_region, exptimes)
        mdl_region_map, mdl_grey_map[i] = init_maps(graphs[i])
        current_region_map, current_grey_map[i] = init_maps(graphs[i])
    
    likeli = [0.0]*num_band
    for i in range(num_band):
        likeli[i] = compute_likelihood(graphs[i], bg[i])
        
    min_mdl = compute_mdl(graphs, permtx, m,n,num_band,bg)
    mdl = min_mdl
    
    #print(mdl)
    #print(compute_mdl(graphs,permtx,m,n,num_band))
    for i in range(graphs[0].nregion-num_merge):
        region1,region2,likeli,mdl = best_merging_pair(graphs,likeli,mdl,permtx,m,n,num_band,bg)
        merge_regions(graphs,region1,region2,permtx,num_band)
        
        #print(mdl)
        #print(compute_mdl(graphs,permtx,m,n,num_band))
        
        for j in range(num_band):
            update_maps(current_region_map,current_grey_map[j],graphs[j],region1,region2)
            
        if min_mdl > mdl:
            min_mdl = mdl
            for j in range(num_band):
                copy_maps(current_region_map, current_grey_map[j],
                          mdl_region_map, mdl_grey_map[j], graphs[j])
        
        mdl_curve[i*2] = graphs[0].nregion-i-1
        mdl_curve[i*2+1] = mdl/(m*n)
        
    relabel_region_map(mdl_region_map,num_region)
    for i in range(m*n):
        mdl_out_region[i] = mdl_region_map[label_im[i]]
    for j in range(num_band):
        for i in range(m*n):
            mdl_out_grey[j][i] = mdl_grey_map[j][label_im[i]]


# In[148]:

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 3:
        print("Purpose: MDL segmentation for independent noise")
        exit()
        
    num_band = 1
    num_merge = 1
    tempim,m,n = read_2ddata(os.path.join(os.getcwd(),str(argv[1])))
    
    observe_im = [[[0.0]*m*n]*T]*num_band
    for i in range(m):
        for j in range(n):
            observe_im[0][i*n+j] = float(tempim[i,j])
            
    mdl_out_grey = [[0.0]*m*n]*num_band
    
    tempim,tm,tn = read_2ddata(os.path.join(os.getcwd(),str(argv[2])))
    
    if (tm != m) or (tn != n):
        print("Error: imagex and seeds are not of the same size")
    
    label_im = [0]*m*n
    for i in range(m):
        for j in range(n):
            label_im[i*n+j] = int(tempim[i,j])
            
    mdl_out_region = [0]*m*n
    
    num_over = max(label_im)
    
    mdl_curve = [0.0]*2*(num_over-num_merge)
    
    segment_mdl_ind(observe_im, label_im,m,n,num_merge,num_band,
                    mdl_out_region,mdl_out_grey,mdl_curve)
    
    for i in range(m):
        for j in range(n):
            tempim[i,j] = mdl_out_region[i*n+j]
    save_image(tempim, "junk_mdlsegi_region_t", m, n)
    
    for i in range(m):
        for j in range(n):
            tempim[i,j] = mdl_out_grey[0][i*n+j]
    save_image(tempim, "junk_mdlsegi_grey_t", m, n)
    
    for i in range(num_over-num_merge):
        for j in range(2):
            tempim[i,j] = mdl_curve[i*2+j]
    save_image(tempim, "junk_mdlsegi_curve_t",num_over-num_merge,2)
