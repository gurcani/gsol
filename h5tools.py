#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:55:08 2025

@author: ogurcan
"""

def save_data(fl,grpname,ext_flag=False,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
        if not l in grp:
            if(not ext_flag):
                grp[l]=m
            else:
                if(not hasattr(m, '__len__')): # np.isscalar(m)
                    grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                else:
                    grp.create_dataset(l,(1,)+m.shape,chunks=(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                lptr=grp[l]
                lptr[-1,]=m
                lptr.flush()
        else:
            lptr=grp[l]
            if(ext_flag):
                lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
                lptr[-1,]=m
            else:
                lptr[...]=m
            lptr.flush()
        fl.flush()