
# coding: utf-8


from __future__ import print_function

def is_empty_intersection(seta, setb):
    """
    test if the intersection of two sets is empty
    """
    num_seta=seta[0]
    num_setb=setb[0]
    for i in range(1,num_seta+1):
        for j in range(1,num_setb+1):
            if (seta[i]==setb[j]):
                return False
    else:
        return True

def is_an_element(seta, element):
    """
    test if element is in seta
    """
    num_seta = seta[0]
    for i in range(1, num_seta+1):
        if (element == seta[i]):
            return True
    else:
        return False
    
"""
or "element in seta[1:]"
"""


# In[90]:

def add_element(seta, element, max_elmt):
    already_exist = False
    num_seta = seta[0]
    for i in range(1, num_seta+1):
        if element == seta[i]:
            already_exist = True
    if not already_exist:
        if num_seta >= max_elmt:
            seta[0] = -1
        else:
            seta[0] = num_seta+1
            seta[num_seta+1] = element


# In[80]:

def remove_element(seta, element):
    num_seta = seta[0]
    for i in range(1,num_seta+1):
        if element == seta[i]:
            if i != num_seta:
                seta[i] = seta[num_seta]
            seta[0] = num_seta - 1


# In[88]:

def get_union(seta, setb, max_elmt):
    num_seta = seta[0]
    for i in range(1,num_seta+1):
        add_element(setb, seta[i], max_elmt)
        seta[i] = 0
    seta[0] = 0


# In[97]:

def print_setofsets(setofsets, num_set, max_elmt):
    for i in range(num_set):
        print("{} | ".format(setofsets[i][0]), end = "")
        for j in range(1,max_elmt+1):
            print("{} ".format(setofsets[i][j]), end = "")
        print()

