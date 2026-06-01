#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:14:14 2025

@author: ogurcan
"""
#from .etdrk4cp import etdrk4cp as etd
import numpy as np
from importlib import import_module
get = lambda x : x.get() if ('cupy' in str(type(x))) else x

class callbacks:
    def __init__(self,dts,fncbs,tnexts=None):
        self.fncbs=fncbs
        self.dts=dts
        self.ts=[0.0 for l in range(len(dts)) ]
        self.tnexts=dts.copy() if tnexts is None else tnexts
        self.dense_output=None
    def append(self,dt,fncb,t=0):
        self.fncbs.append(fncb)
        self.dts.append(dt)
        self.ts.append(t)
        self.tnexts.append(dt)
    def act(self,t,u):
        for l in range(len(self.dts)):
            if(t>=self.tnexts[l]):
                self.tnexts[l]=t+self.dts[l]
                if(self.dense_output is not None):
                    self.fncbs[l](self.tnexts[l],self.dense_output(self.tnexts[l]).reshape(u.shape))
                else:
                    self.fncbs[l](t,u)

class gsol:
    def __init__(self,fexp,t0,y0,t1,L,dtstep,callbacks=None,sv='scipy.DOP853',tol=1e-8,**kwargs):
        self.t0=float(t0)
        self.t=float(t0)
        self.t1=float(t1)
        self.yshp=y0.shape
        self.dtype=y0.dtype
        self.cbs=callbacks
        self.hlast=dtstep
        valid_bases = ["scipy""scipy_old","cupy_ivp","etdrk4cp"]
        smdef="DOP853"
        smdef_old="vode"
        match sv.split('.'):
            case [sn,sm] if sn in ["scipy","cupy_ivp"]:
                if not (sm in ["DOP853","RK45","RK23"]):
                    print("scipy submodule:",sm,"not implemented.")
                    print("using",smdef,"instead.")
                    sm=smdef
                svmod = getattr(import_module("scipy.integrate" if sn=="scipy" else "gsol."+sn),sm)
                self.run=self.run_scipy
            case ["etdrk4cp"]:
                svmod = getattr(import_module("gsol.etdrk4cp"),"etdrk4cp")
                sn="etdrk4cp"
                self.run=self.run_etdrk4cp
            case [sn,sm] if sn =="scipy_old":
                if not (sm in ["vode","zvode","dop853"]):
                    print("scipy.ode method:",sm,"not implemented.")
                    print("using",smdef_old,"instead.")
                    sm=smdef_old
                odmod = getattr(import_module("scipy.integrate"),"ode")
                self.run=self.run_scipy_old
            case _:
                print("no such module:",sv)

        if ('cupy' in str(type(y0))):
            xp=import_module("cupy")
        else:
            xp=import_module("numpy")
        self.dot = lambda x,y : xp.einsum('ijk,ki->ji',x,y,optimize=True)
        dot=self.dot
        if(sn=="etdrk4cp"):
            self.tol=tol
            if (L.size==y0.size): # means L is diagonal
                self.y=y0
                self.r_=svmod(fexp,L,y0,dtstep,xp,**kwargs)
            else:
                gams,Tk=np.linalg.eig(get(L))
                Tinvk=np.linalg.inv(Tk)
                gams=xp.array(gams)
                Tk=xp.array(Tk)
                Tinvk=xp.array(Tinvk)
                yshp=L.shape[:-1][::-1]
                self.yshp=yshp
                xi0=dot(Tinvk,y0.reshape(yshp)).ravel()
                self.y=xi0
                self.Tk=Tk
                self.Tinvk=Tinvk
                def fexp_dia(t,zk):
                    xik=zk.reshape(yshp)
                    phink=dot(Tk,xik)
                    dphinkdt=fexp(t,phink)
                    dzkdt=dot(Tinvk,dphinkdt)
                    return dzkdt.ravel()
                self.r_=svmod(fexp_dia,gams.T.ravel(),xi0,dtstep,xp,**kwargs)
        elif(sn=="scipy_old"):
            self.y=y0.view(dtype=float).ravel()
            self.r_=odmod(lambda t,x : (dot(L,x.view(dtype=complex).reshape(self.yshp))+fexp(t,x.view(dtype=complex).reshape(self.yshp))).view(dtype=float).ravel()).set_integrator(sm,**kwargs)
            self.r_.set_initial_value(self.y, t0)
            self.dtstep=dtstep
        else:
            self.y=y0.ravel()
            self.r_=svmod(lambda t,x : (dot(L,x.reshape(self.yshp))+fexp(t,x.reshape(self.yshp))).ravel(),t0,self.y,t1,**kwargs)
    def act_callback(self):
        if hasattr(self, 'Tk'):
            self.cbs.act(self.t,self.dot(self.Tk,self.y.view(dtype=self.dtype).reshape(self.yshp)))
        else:
            self.cbs.act(self.t,self.y.view(dtype=self.dtype).reshape(self.yshp))

    def run_scipy_old(self):
        r=self.r_
        while(self.t<self.t1):
            r.integrate(r.t+self.dtstep)
            self.y[:]=r.y
            self.t=r.t
            self.act_callback()

    def run_scipy(self):
        r=self.r_
        while(self.t<self.t1):
            r.step()
            self.y[:]=r.y
            self.t=r.t
            self.cbs.dense_output=self.r_.dense_output()
            self.act_callback()

    def run_etdrk4cp(self):
        r=self.r_
        while(self.t<self.t1):
            uk,err=r.step(self.t,self.y)
            self.y[:]=uk
            self.t+=r.h
            self.act_callback()
            r.recompute_stepsize(err,self.tol)
            self.hlast=r.h
