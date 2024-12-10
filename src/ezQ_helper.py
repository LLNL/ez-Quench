'''
ezQ_helper.py
R.A. Soltz
helper routines for emcee calibration

created:  2022-dec-06 R.A. Soltz
modified: 2023-jun-30 RAS - insert ezQ_delta_trento (previously ezR_delta)
          2024-may-16 RAS - cleanup vestigal functions: ifunc, ppCostFn, etc.
          2024-aug-07 RAS - add takeLog option
'''

import numpy as np
from scipy.special import gamma

'''
Final fit function for pp-data, assume small (1-MeV) bins so no internal sums
'''
def ppw0(theta,pT):
    (a,b,c,d,e,f) = theta
    w0 = (a/pT)**(b+c*np.log(pT))/(1+d*np.exp((pT-f)/e))
    return w0

'''
Define fx_gamma function to return gamma weights for pT-loss
Inputs:
    pT      = pT binning
    a,b     = 2-parameter for mean pT-loss
    k       = 1-parameter for number of exponentials in gamma-fn

Variables:
    h    = mean relative pT-loss
    x    = 2d array of (delta-pT)_{ij}
            i = init   (axis=0)
            j = quench (axis=1)
    x    = (     0          0          0            0  )
           ( p[2]-p[1]      0          0      ...   0  )
           ( p[3]-p[1]  p[3]-p[2]      0      ...   0  )
           (                     ...                0  )
           ( p[n]-p[1]  p[n]-p[2]  p[n]-p[3]  ...   0  )
    fx   = gamma-function of (x;h,k)

- the following takes pT and parameters and returns fx weights
'''

def fx_gamma(pT,a,b,k):
    h = a*(pT**b)*np.log(pT)
    x = (np.tile(pT,(pT.size,1)).T - pT)
    x[x<0]=0
    fx = (h**(-k)) * (x**(k-1)) * np.exp(-x/h) / gamma(k)
    return fx

def log_like_gamma(theta,pT,w0,denom,pedge,ydata,icov,takeLog=0):

    # insert quench contents to save function call
    (a,b,k)   = theta
    fxg       = fx_gamma(pT,a,b,k)
    w0Q       = np.einsum('i,ij',w0,fxg)
    num,edges = np.histogram(pT,pedge,weights=w0Q)
    if (takeLog):
        if (np.any(num<=0)):
            return -np.inf
        else:
            diff = np.log(num/denom) - np.log(ydata)
    else:
        diff = num/denom - ydata
    loglik = -0.5*np.linalg.multi_dot((diff,icov,diff))
    if np.isfinite(loglik):
        return loglik
    else:
        print ('Error theta,loglik = ',theta,loglik)
        return -np.inf

def log_prior_gamma(theta):
    # for now set priors for a,b,k to be uniform form 0-10
    theta_min = np.array([0.,-1.,1.])
    theta_max = np.array([10.,10.,10.])
    if np.all(theta>theta_min) and np.all(theta<theta_max):
        return 0.0
    return -np.inf

def log_prob_gamma(theta,pT,w0,denom,pedge,ydata,icov,takeLog=0):
    lp = log_prior_gamma(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like_gamma(theta,pT,w0,denom,pedge,ydata,icov,takeLog)

'''No intermediate function is needed for the delta quench'''

def log_like_delta(theta,pT,w0,denom,pedge,ydata,icov,takeLog=0):
    (a,b) = theta
    h = a*(pT**b)*np.log(pT)
    (num,edges) = np.histogram(pT-h,pedge,weights=w0)
    if (takeLog):
        if (np.any(num<=0)):
            return -np.inf
        else:
            diff = np.log(num/denom) - np.log(ydata)
    else:
        diff = num/denom - ydata
    loglik = -0.5*np.linalg.multi_dot((diff,icov,diff))
    if np.isfinite(loglik):
        return loglik
    else:
        print ('Error theta,loglik = ',theta,loglik)
        return 0.0

def log_prior_delta(theta):
    # for now set priors for a,b,k to be uniform form 0-10
    theta_min = np.array([0.,-1.])
    theta_max = np.array([10.,10.])
    if np.all(theta>theta_min) and np.all(theta<theta_max):
        return 0.0
    return -np.inf

def log_prob_delta(theta,pT,w0,denom,pedge,ydata,icov,takeLog=0):
    lp = log_prior_delta(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like_delta(theta,pT,w0,denom,pedge,ydata,icov,takeLog)

'''
    Gridtest function, to return intersections along grid-edges
    - assume angle within [0,pi/2], but should work for [-pi/2,pi/2]
    - fill ygx,xgy with x,y and y,x-gridcrossings
    - then determine if exit from top/right and entrance from bottom/left
    = sort x,y pts by ypts for bottom entrance in case we have vertical line

    Inputs:
    - x0,y0,angle = origin(x,y) and angle [0,pi/2]
    - xe,ye       = x- and y-edges

    Returns:
    - xgridpts, ygridpts

    Other variables
    - eps - set to 100x floating point precision, used for np.argmax and /0
    - tan - np.tan(angle), substitute eps to avoid /0
'''

def gridpts(x0,y0,angle,xe,ye):

    '''determine grid crossing points, yfx = y-from-x, ensure tan!=0'''
    tan  = np.tan(angle)
    if (tan==0.):
        tan = 10*np.finfo(float).eps
    yfx = y0 + (xe-x0)*tan
    xfy = x0 + (ye-y0)/tan

    '''Combine all points, no need to sort yet'''
    xall = np.concatenate((np.array([x0]),xe,xfy))
    yall = np.concatenate((np.array([y0]),yfx,ye))

    '''Remove all points that are out of bounds'''
    xpts = xall[(xall>=xe[0])*(xall<=xe[-1])*(yall>=ye[0])*(yall<=ye[-1])]
    ypts = yall[(xall>=xe[0])*(xall<=xe[-1])*(yall>=ye[0])*(yall<=ye[-1])]

    '''We can always sort by x because np.tan(np.pi/2.)<infinity, ypts first!'''
    ypts = ypts[np.argsort(xpts)]
    xpts = xpts[np.argsort(xpts)]

    return (xpts,ypts)

'''
ezQ_delta_trento

Function Inputs
---------------
theta = parameters for delta/gamma fn
J0    = pp jet distribution in RAA bins
RAA_pedge = pT edges (for any cbin)
tpa   = trento-pair array shape=(ncbins,ntpairs,2)
RAA_pTbins/edges and values
pp_pTbins
winit
    xJ_flag  = to include xJ for 0-10% only (add later)
    xJ_pTbins/edges and values              (add later)
    Cinv = inverted covariance error matrix (needed for loglik)

Internal Arrays --> dimensions must match for histograms
---------------
p0 (ntpair,npT) =  unshifted pT distribution
p1 (ntpair,npT) = shifted by path-1
p2 (ntpair,npT) = shifted by path-2
w  (ntpair,npT) = pp jet weights
* put npT in last index match for broadcasting with pT

Return Array
------------
R (2,ncbin,npTbin) = RAA for p1,p2 for each cbin
  initialize R by fetching ncbin,npT from 

'''

def ezQ_delta_trento(theta,J0,RAA_pedge,pT,w,tpa):
    (ncbin,ntpairs,two) = tpa.shape
    R = np.zeros((ncbin,RAA_pedge.size-1))
    (a,b) = theta
    h = a*(pT**b)*np.log(pT)
    '''Loop over cbins, use einsum for outer products and tiling'''
    for i in np.arange(ncbin):
        p1 = pT - np.einsum('i,j',tpa[i,:,0],h)
        p2 = pT - np.einsum('i,j',tpa[i,:,1],h)
        (R1,edge1) = np.histogram(p1,RAA_pedge,weights=w)
        (R2,edge2) = np.histogram(p2,RAA_pedge,weights=w)
        R[i,:] = (R1+R2)/J0/np.diff(RAA_pedge)/ntpairs/two
    return R

def log_like_delTrento(theta,J0,RAA_pedge,pT,w,tpa,ydata,icov,takeLog=0):
    R = ezQ_delta_trento(theta,J0,RAA_pedge,pT,w,tpa)
    if (takeLog):
        if (np.any(R<=0)):
            return -np.inf
        else:
            diff = (np.log(R) - np.log(ydata)).ravel()
    else:
        diff = (R - ydata).ravel()
    loglik = -0.5*np.linalg.multi_dot((diff,icov,diff))
    if np.isfinite(loglik):
        return loglik
    else:
        print ('Error theta,loglik = ',theta,loglik)
        return 0.0

def log_prior_delTrento(theta):
    # for now set priors for a,b,k to be uniform form 0-10
    theta_min = np.array([0.,-1.])
    theta_max = np.array([10.,10.])
    if np.all(theta>theta_min) and np.all(theta<theta_max):
        return 0.0
    return -np.inf

def log_prob_delTrento(theta,J0,RAA_pedge,pT,w,tpa,ydata,icov,takeLog=0):
    lp = log_prior_delTrento(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like_delTrento(theta,J0,RAA_pedge,pT,w,tpa,ydata,icov,takeLog)
