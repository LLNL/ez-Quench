'''
ezQ_plotTestCov.py
created:  2024-Nov-08 RAS - rename from test_error_fit.py to create figures for Appendix B
modified: 2024-Nov-22 RAS - increase label/legend font size

This is a standalone routine (data & errors included) to fit an exponential function to 
the ATLAS jet-RAA for central 5.02 TeV PbPb collisions for a set 5 error matrices
 1. stat-only
 2. stat+sys
 3. cov-sum
 4. cov-all
 5. cov-20%

Each error treatment above is fit in three ways
 1. standard chi2 minimization
 2. chi2 on log transform 
 3. flip-data about stat-only fit and then refit

'''

# %% 

# Import standard packages and set plots inline
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

'''exponential with takeLog option'''
def myfn(c,X,y,icov,takeLog):
    model = c[0]*(1.-c[1]*np.exp(-X/(1000.*c[2])))
    eps = 100*np.finfo(float).eps
    model[model<eps]=eps
    if (takeLog):
        diff = np.log(model)-np.log(y)
    else:
        diff = model - y
    chi2 = 0.5*np.linalg.multi_dot((diff,icov,diff))
    return chi2

plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

mylin = ('solid','dashed',(0,(5,1)),'dashdot',(0,(5,1,1,1,1,1)))
mycol = ('darkred','darkblue','darkgreen','darkgoldenrod','indigo','indigo','brown')
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'

'''Load data from ATLAS '''
Xedge = np.array([158,178,200,224,251,282,316,355,398,501,631])
Yvals = np.array([0.5114,0.5275,0.5512,0.5676,0.5787,0.601 ,0.6131,0.5996,0.6097,0.6287])
Ystat = np.array([0.001 ,0.0011,0.0015,0.0022,0.003 ,0.0044,0.0064,0.0091,0.0107,0.0243])
YsysT = np.array([0.0287,0.0314,0.0422,0.0444,0.0438,0.0458,0.0484,0.0462,0.0547,0.0562])
YsysA = np.array([[0.0087,0.0084,0.0092,0.0113,0.0124,0.0125,0.0111,0.0081,0.0085,0.0082],
 [0.0065,0.0063,0.0049,0.0027,0.0023,0.0024,0.0016,0.0041,0.0039,0.0037],
 [0.0025,0.0074,0.004 ,0.0044,0.0035,0.0028,0.0023,0.0034,0.0006,0.0051],
 [0.0248,0.0286,0.0407,0.0426,0.0416,0.0438,0.047 ,0.0451,0.0539,0.055 ],
 [0.0092,0.0014,0.002 ,0.0018,0.0036,0.0024,0.0027,0.0032,0.0021,0.0052]])
X = 0.5*(Xedge[:-1]+Xedge[1:])
X_n = X.size
Yvalsqr = np.einsum('i,j',Yvals,Yvals)

# %%

'''Perform 3 fits with 5 sets of errors'''

'''Flip data about stat-only fit'''
cs = np.array([0.64,0.81,0.12])
bestfit = cs[0]*(1.-cs[1]*np.exp(-X/(1.e3*cs[2])))
Yflip = 2*bestfit - Yvals

'''Prep for fits'''
iLog = 1
fit_list = np.array(['standard','takeLog','flipData'])
nfit  = fit_list.size
theta = np.array([0.6,0.8,0.1])
npar  = theta.size

'''Prep Gaussian block'''
Xbin = np.arange(X_n)
d = (np.tile(Xbin,(X_n,1)).T-Xbin)/X_n
G20 = np.exp(-(d*d)/(2.*0.2*0.2))

'''Build covariant error matrices to test'''
cov_list  = np.array(['stat-only','stat+sys','cov-sum','cov-all','cov-20%'])
ncov = cov_list.size
cov    = np.zeros((nfit,ncov,X_n,X_n))
icov   = np.zeros_like(cov)
cov[0,0] = np.diag(Ystat**2)
cov[0,1] = cov[0,0] + np.diag(YsysT**2)
cov[0,2] = cov[0,0]
for i in np.arange(YsysA.shape[0]):
    cov[0,2] +=  np.einsum('j,k',YsysA[i],YsysA[i])
cov[0,3] = cov[0,0] + np.einsum('i,j',YsysT,YsysT)
cov[0,4] = cov[0,0] + G20*np.einsum('i,j',YsysT,YsysT)

'''Divide by y**2 for takeLog and copy for flipData'''
for i in np.arange(ncov):
    cov[1,i] = cov[0,i]/Yvalsqr
    cov[2,i] = cov[0,i]

'''Test covariance inversion before fits'''
iflag = 1
for i in np.arange(nfit):
    for j in np.arange(ncov):
        icov[i,j] = np.linalg.inv(cov[i,j])
        itest = cov[i,j]@icov[i,j]
        if (not np.allclose(itest,np.eye(itest.shape[0]))):
            print('Covariance inversion failed for fit',fit_list[i],'with cov=',cov_list[j])
            iflag = 0
if (iflag): print('Covariance invesion test succesful.')

xx = np.arange(160,575,10)
options = {'maxfun':500}
chi2df  = np.zeros((nfit,ncov))
cpars   = np.zeros((nfit,ncov,npar))
#models  = np.zeros((nfit,ncov,X_n))
models  = np.zeros((nfit,ncov,xx.size))
for i in np.arange(nfit):
    for j in np.arange(ncov):
        print('Fitting',fit_list[i],cov_list[j])
        if (i==2):
            res = optimize.minimize(myfn,theta,(X,Yflip,icov[i,j],i==iLog),jac=None,method='TNC',options=options)
        else:
            res = optimize.minimize(myfn,theta,(X,Yvals,icov[i,j],i==iLog),jac=None,method='TNC',options=options)
        cpars[i,j]  = res.x
        models[i,j] = cpars[i,j,0]*(1.-cpars[i,j,1]*np.exp(-xx/(1000.*cpars[i,j,2])))
        chi2df[i,j] = res.fun/(X.size-npar)
        print(fit_list[i],cov_list[j],
              r'$\chi^2_d$={0:.2e}; pars ='.format(chi2df[i,j]),np.array2string(cpars[i,j],precision=2))

# %%

'''Plot standard and log fits'''

fig,ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.45,0.71])
ax.set_xlim([145.,600.])

box_width = 20
ax.errorbar(X,Yvals,Ystat,marker='s',color='black', label='ATLAS data')
ax.bar(X,2*YsysT,box_width,Yvals-YsysT,color='blue',alpha=0.4,label='systematic errors')
for j in np.arange(ncov):
    ax.plot(xx,models[0,j],linewidth=2,linestyle=mylin[j],color=mycol[j],label=cov_list[j])
    ax.plot(xx,models[1,j],linewidth=2,linestyle=mylin[j],color=mycol[j],marker='*',markersize=8)
ax.plot(xx,models[1,0],linewidth=2,linestyle=mylin[0],color=mycol[0],marker='*',label='log-fit',markersize=8)
ax.legend(loc='lower right',ncol=2,borderaxespad=2,handlelength=3)

saveFig = True
if (saveFig):
    fig.savefig(figdir+'figB_takeLog_compare.pdf')

# %%
'''Plot fits to flipped data'''

fig,ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.45,0.71])
ax.set_xlim([145.,600.])

ax.errorbar(X,Yflip,Ystat,marker='x',linestyle='dotted',color='black', label='reflected data')
ax.bar(X,2*YsysT,box_width,Yflip-YsysT,facecolor='none',edgecolor='black',linestyle='dotted',label='reflected systematics')
ax.errorbar(X,Yvals,Ystat,marker='s',color='black', label='ATLAS data')
ax.bar(X,2*YsysT,box_width,Yvals-YsysT,color='blue',alpha=0.4,label='systematic errors')
for j in np.arange(ncov):
    ax.plot(xx,models[2,j],linewidth=2,linestyle=mylin[j],color=mycol[j],label=cov_list[j],marker='x')
#ax.plot(xx,models[0,0],linewidth=2,linestyle=mylin[0],color=mycol[0],label='unreflected stat-only')
ax.legend(loc='lower right',ncol=2,borderaxespad=2,handlelength=3)

saveFig = True
if (saveFig):
    fig.savefig(figdir+'figB_flipData_compare.pdf')

# %%
