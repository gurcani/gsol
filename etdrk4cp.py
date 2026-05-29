#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:35:25 2025

@author: ogurcan
"""

class etdrk4cp:
    #Mostly from Nicola Farenga's implementation [https://github.com/farenga/ETDRK4] of Kassam and Lloyd 05
    def __init__(self,Nl,L,v,h,xp,M=64,maxstep=0.25):
        self.Nl=Nl
        self.L=L
        self.h=h
        self.M=M
        self.xp=xp
        self.compute_coeffs()
        self.maxstep=maxstep

    def compute_coeffs(self):
        L,h,M=self.L,self.h,self.M
        xp=self.xp
        sz=L.size
        E = xp.exp(h*L)
        E2 = xp.exp(h*L/2)
        r = xp.exp(1j*xp.pi*(xp.arange(1,M+1)-.5)/M)
        LR = xp.repeat(h*L[:,None],M,1) + xp.repeat(r[None,:],sz,0)
        Q = h*xp.mean(((xp.exp(LR/2)-1)/LR),1).real
        f1 = h*xp.mean(((-4-LR+xp.exp(LR)*(4-3*LR+LR**2))/LR**3),1).real
        f2 = h*xp.mean(((2+LR+xp.exp(LR)*(-2+LR))/LR**3),1).real
        f3 = h*xp.mean(((-4-3*LR-LR**2+xp.exp(LR)*(4-LR))/LR**3),1).real
        self.E=E
        self.E2=E2
        self.Q=Q
        self.f1=f1
        self.f2=f2
        self.f3=f3
        
    def step(self,t,v):
        Nl,E,E2,Q=self.Nl,self.E,self.E2,self.Q
        f1,f2,f3=self.f1,self.f2,self.f3
        xp=self.xp
        Nv = Nl(t,v)
        a = E2*v + Q*Nv
        Na = Nl(t,a)
        b = E2*v + Q*Na
        Nb = Nl(t,b)
        c = E2*a + Q*(2*Nb-Nv)
        Nc = Nl(t,c)
        vp = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        err=xp.linalg.norm(vp-c).item() / vp.size ** 0.5
        return vp,err
        
    def recompute_stepsize(self,err,tol):
        # The idea is from Deka and Einkemmer 22, applied to powers of two so that we don't recompute the coefficients for small changes.
        xp=self.xp
        h=min(self.maxstep,2**xp.floor(xp.log2(self.h*(tol/err)**(1/4))).item())
        if(self.h!=h):
            self.h=h
            self.compute_coeffs()
