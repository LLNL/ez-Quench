'''
ezQ_plotATLAS.py
created: 2023-apr-07 R.A. Soltz
modified: 2023-apr-11 RAS - Create first heatmaps for covariance error matrices
          2024-may-22 RAS - Rewrite to read errors from hdf5 and make figA_ figures for ezQtex
          2024-jul-03 RAS - Add heatmap for 20% correlation covariance
          2024-jul-10 RAS - add fits to simple exp & log forms
          2024-jul-12 RAS - add CMS and ALICE data for simple fits
          2024-oct-18 RAS - replot fig 4 sys-error contributions as absolute
          2024-nov-08 RAS - fix cov_l20 plot to include stat+lum+TAA and fix range to match cov_complete
          2024-nov-22 RAS - increase label/legend fontsize
          2024-dec-04 RAS - mv figA_cov_lumTAA.pdf back to figdir

This routine reads the ATLAS_data.h5 created by ATLAS_data2h5.py
'''

# %% 

# Import standard packages and set plots inline
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

generate_figures = False
mycol = ['darkblue','darkred','darkgreen','darkgoldenrod','brown','olive','purple']
mylin = ((0,(5,3)),(0,(2,1,2,3)),'solid','dashdot','dashed','dotted',(0,(3,3)),(0,(5,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

'''h5 readout'''
ATLAS_datafile = h5dir + 'ATLAS_data.h5'
with h5py.File(ATLAS_datafile,'r') as f:
    RAA_cedge         = np.array(f['RAA_cedge'])
    RAA_pedge         = np.array(f['RAA_pedge'])
    RAA_vals          = np.array(f['RAA_vals'])
    RAA_sysvals       = np.array(f['RAA_sysvals'])
    cov_pp_lum_E      = np.array(f['cov_pp_lum_E'])
    cov_TAA_E         = np.array(f['cov_TAA_E'])
    RAA_valsqr        = np.array(f['RAA_valsqr'])
    cov_RAA_stat_diag = np.array(f['cov_RAA_stat_diag'])
    cov_RAA_sys_total = np.array(f['cov_RAA_sys_total'])
    cov_RAA_sys_sum   = np.array(f['cov_RAA_sys_sum'])
    cov_RAA_complete  = np.array(f['cov_RAA_complete'])
    cov_RAA_l20pct    = np.array(f['cov_RAA_l20pct'])

ALICE_CMS_datafile = h5dir + 'ALICE_CMS_data.h5'
with h5py.File(ALICE_CMS_datafile,'r') as g:
    ALICE_RAA_cbins = np.array(g['ALICE_RAA_cbins'])
    ALICE_RAA_pT    = np.array(g['ALICE_RAA_pT'   ])
    ALICE_RAA_vals  = np.array(g['ALICE_RAA_vals' ])
    ALICE_RAA_stat  = np.array(g['ALICE_RAA_stat' ])
    ALICE_RAA_corr  = np.array(g['ALICE_RAA_corr' ])
    ALICE_RAA_shap  = np.array(g['ALICE_RAA_shap' ])
    ALICE_RAA_lTAA  = np.array(g['ALICE_RAA_lTAA' ])
    CMS_RAA_cbins   = np.array(g['CMS_RAA_cbins'])
    CMS_RAA_pT      = np.array(g['CMS_RAA_pT'   ])
    CMS_RAA_vals    = np.array(g['CMS_RAA_vals' ])
    CMS_RAA_stat    = np.array(g['CMS_RAA_stat' ])
    CMS_RAA_sys     = np.array(g['CMS_RAA_sys'  ])
    CMS_RAA_lumi    = np.array(g['CMS_RAA_lumi' ])
    CMS_RAA_TAA     = np.array(g['CMS_RAA_TAA'  ])

CMS_pTmid   = 0.5*(CMS_RAA_pT[1:]+CMS_RAA_pT[:-1])
ALICE_pTmid = 0.5*(ALICE_RAA_pT[1:]+ALICE_RAA_pT[:-1])

cbins    = np.array(['0_10','10_30','30_50','50_80'])
ctext    = np.array(['0-10%','10-30%','30-50%','50-80%'])
systypes = np.array(['JER','JES (baseline)','JES (flavor)','JES (quench)','unfolding','Total'])
npT      = RAA_vals[0].size
nc       = RAA_cedge.size - 1
pTmid    = 0.5*(RAA_pedge[1:]+RAA_pedge[:-1])
print('npT=',npT)

'''Test inverse'''
cov = cov_RAA_complete
itest = np.linalg.inv(cov)@cov
print('Complete itest=',np.array2string(itest,precision=2))
print('allclose =',np.allclose(itest,np.eye(itest.shape[0])))

# %%

'''
Loop through unmasked centrality bins and plot covariance error matrices and differences
Use cbin (singular) to specify which elements of cbins to plot
'''

cbin = np.array((0,3))
fig, ax = plt.subplots(cbin.size, 3, figsize=(18, 6*cbin.size),tight_layout=True)
cshrink = 0.70
im = []

for i in np.arange(cbin.size):
    j = cbin[i]
    a = j*npT
    b = (j+1)*npT

    tot = cov_RAA_sys_total[a:b,a:b]
    sum = cov_RAA_sys_sum[a:b,a:b]
    dif = (tot-sum)/sum

    title = 'log(full covariance) '+ctext[j]
    ax[i,0].set(title=title,xlabel='pT-bin',ylabel='pT-bin')
    title = 'log(sum covariance) '+ctext[j]
    ax[i,1].set(title=title,xlabel='pT-bin',ylabel='pT-bin')
    title = '(full-sum)/sum '+ctext[j]
    ax[i,2].set(title=title,xlabel='pT-bin',ylabel='pT-bin')
    im.append(ax[i,0].imshow(np.log(tot),origin='lower',cmap='rainbow'))
    im.append(ax[i,1].imshow(np.log(sum),origin='lower',cmap='rainbow'))
    im.append(ax[i,2].imshow(dif,origin='lower',cmap='rainbow'))
    plt.colorbar(im[i*3+0],ax=ax[i,0],shrink=cshrink)
    plt.colorbar(im[i*3+1],ax=ax[i,1],shrink=cshrink)
    plt.colorbar(im[i*3+2],ax=ax[i,2],shrink=cshrink)

generate_figures = True
if(generate_figures):
    figname = figdir + 'figA_cov_compare'
    fig.savefig(figname+'.pdf')

# %%
    
'''Plot luminosity and TAA'''

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
im = []

axes[0].set(title='log(luminosity covariance)',xlabel='centrality/pT-bin',ylabel='centrality/pT-bin')
axes[1].set(title='log(TAA covariance)',xlabel='centrality/pT-bin',ylabel='centrality/pT-bin')
for ax in axes:
    ax.tick_params('both', length=10, width=1, which='minor')
    ax.tick_params('both', length=0, width=0, which='major')
    ax.set_xticks(np.arange(5,40,10))
    ax.set_xticks(np.arange(-0.5,50,10),minor=True)
    ax.set_xticklabels(ctext)
    ax.set_yticks(np.arange(5,40,10))
    ax.set_yticklabels(ctext)
    ax.set_yticks(np.arange(-0.5,50,10),minor=True)

im.append(axes[0].imshow(np.log(cov_pp_lum_E),origin='lower',cmap='rainbow'))
im.append(axes[1].imshow(np.log(cov_TAA_E),origin='lower',cmap='gist_ncar'))
#im.append(axes[1].imshow(np.log(cov_TAA_E),origin='lower',cmap='prism'))
plt.colorbar(im[0],ax=axes[0],shrink=0.74)
plt.colorbar(im[1],ax=axes[1],shrink=0.74)

generate_figures = True
if(generate_figures):
    figname = figdir + 'figA_cov_lumTAA'
    fig.savefig(figname+'.pdf')

# %%

'''Plot full covariance matrix'''

vmin = -7.9
vmax = -4.6

fig, ax = plt.subplots(1, 1, figsize=(12, 12),tight_layout=True)
title = r'log($R_{AA}$ full covariance)'
ax.set_title(title,fontsize=20,pad=20)
ax.set(xlabel='centrality/pT-bin',ylabel='centrality/pT-bin')
ax.tick_params('both', length=10, width=1, which='minor')
ax.tick_params('both', length=0, width=0, which='major')
ax.set_xticks(np.arange(5,40,10))
ax.set_xticks(np.arange(-0.5,50,10),minor=True)
ax.set_xticklabels(ctext)
ax.set_yticks(np.arange(5,40,10))
ax.set_yticklabels(ctext)
ax.set_yticks(np.arange(-0.5,50,10),minor=True)
im = ax.imshow(np.log(cov_RAA_complete),origin='lower',cmap='rainbow',vmin=vmin,vmax=vmax)
#im = ax.imshow(np.log(cov_RAA_complete),origin='lower',cmap='rainbow')
plt.colorbar(im,ax=ax,shrink=0.73)

generate_figures = True
if(generate_figures):
    figname = figdir + 'figA_cov_complete'
    fig.savefig(figname+'.pdf')

# %%

'''Plot l20 covariance matrix'''

cov_l20_complete = cov_pp_lum_E + cov_TAA_E + cov_RAA_stat_diag + cov_RAA_l20pct
fig, ax = plt.subplots(1, 1, figsize=(12, 12),tight_layout=True)
title = r'log($R_{AA}$ 20% correlation covariance)'
ax.set_title(title,fontsize=20,pad=20)
ax.set(xlabel='centrality/pT-bin',ylabel='centrality/pT-bin')
ax.tick_params('both', length=10, width=1, which='minor')
ax.tick_params('both', length=0, width=0, which='major')
ax.set_xticks(np.arange(5,40,10))
ax.set_xticks(np.arange(-0.5,50,10),minor=True)
ax.set_xticklabels(ctext)
ax.set_yticks(np.arange(5,40,10))
ax.set_yticklabels(ctext)
ax.set_yticks(np.arange(-0.5,50,10),minor=True)
im = ax.imshow(np.log(cov_l20_complete),origin='lower',cmap='rainbow',vmin=vmin,vmax=vmax)
plt.colorbar(im,ax=ax,shrink=0.73)

generate_figures = True
if(generate_figures):
    figname = figdir + 'figA_cov_l20pct'
    fig.savefig(figname+'.pdf')

# %%
'''Make some quick fits and plots to test errors'''

flag  = 0
mycol = ('darkblue','darkred','darkgreen','darkgoldenrod','brown','olive','purple')
mylin = ((0,(5,3)),(0,(2,1,2,3)),'dashdot','solid','dashed','dotted',(0,(3,3)),(0,(5,5)))

def myfn(c,X,y,icov):
    model = c[0]*(1.-c[1]*np.exp(-X/(1000.*c[2])))
    diff = model - y
    chi2 = 0.5*np.linalg.multi_dot((diff,icov,diff))
    return chi2

def myfn1(c,X,y,cov):
    model = c[0]*(1.-c[1]*np.exp(-X/(1000.*c[2])))
    diff = model - y
    chi2 = 0.5*np.dot(diff, np.linalg.solve(cov, diff))
    return chi2

def myfn2(c,X,y,icov):
    model = c[1]*np.log(X/100.) + c[0]
    diff = model - y
    chi2 = 0.5*np.linalg.multi_dot((diff,icov,diff))
    return chi2

'''Stack covariance arrays so we can loop through'''
theta   = np.array([0.6,1.,0.3])
options = {'maxfun':500}
npar    = theta.size
nstack  = 5
models  = np.zeros((nstack,npT))
cpars   = np.zeros((nstack,npar))
chi2s   = np.zeros((nstack))
cov_list  = np.array(['stat-diag','sys-diag','l=20%','*sum','full'])
cov_stack = np.zeros((nstack,npT,npT))
cov_lumTAA   = cov_pp_lum_E[0:npT,0:npT] + cov_TAA_E[0:npT,0:npT]
cov_stack[0] = cov_RAA_stat_diag[0:npT,0:npT]
cov_stack[1] = np.diag(np.diag(cov_RAA_sys_total[0:npT,0:npT]))
cov_stack[2] = cov_RAA_l20pct[0:npT,0:npT] + cov_RAA_stat_diag[0:npT,0:npT] + cov_lumTAA
cov_stack[3] = cov_RAA_sys_sum[0:npT,0:npT] + cov_RAA_stat_diag[0:npT,0:npT] + cov_lumTAA
cov_stack[4] = cov_RAA_sys_total[0:npT,0:npT] + cov_RAA_stat_diag[0:npT,0:npT] + cov_lumTAA

labels = []
icov = np.zeros_like(cov_stack)
for i in np.arange(nstack):
    icov[i] = np.linalg.inv(cov_stack[i])
    itest = icov[i]@cov_stack[i]
    print(cov_list[i],'allclose =',np.allclose(itest,np.eye(itest.shape[0])))
    if (flag==0):
        res = optimize.minimize(myfn,theta,(pTmid,RAA_vals[0],icov[i]),jac=None,method='TNC',options=options)
    elif (flag==1):
        res = optimize.minimize(myfn1,theta,(pTmid,RAA_vals[0],cov_stack[i]),jac=None,method='TNC',options=options)
    elif (flag==2):
        res = optimize.minimize(myfn2,theta,(pTmid,RAA_vals[0],icov[i]),jac=None,method='TNC',options=options)
    chi2s[i] = res.fun/(pTmid.size-npar)
    cpars[i] = res.x
    label     = cov_list[i] + r' $\chi^2$/dof={0:.2e}; pars = '.format(chi2s[i]) + np.array2string(cpars[i],precision=2)
    labels.append(label)
    if (flag<=1):
        models[i] = cpars[i,0]*(1.-cpars[i,1]*np.exp(-pTmid/(1000.*cpars[i,2])))
    elif (flag==2):
        models[i] = cpars[i,1]*np.log(pTmid/100.) + cpars[i,0]

fig,(ax,ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [6,2]})
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.4,0.8])
ax.set_xlim([100.,650.])

'''plot pTlum and pTtaa at beginning and end'''
box_width = 20
LumpT = np.array((pTmid[0]-1.2*box_width,pTmid[-1]+1.2*box_width))
Lumval = np.array((cov_pp_lum_E[0,0]**0.5,cov_pp_lum_E[npT-1,npT-1]**0.5))
Lumbot = np.array((RAA_vals[0,0]-Lumval[0],RAA_vals[0,-1]-Lumval[-1]))
TAApT = np.array((pTmid[0]-2.4*box_width,pTmid[-1]+2.4*box_width))
TAAval = np.array((cov_TAA_E[0,0]**0.5,cov_TAA_E[npT-1,npT-1]**0.5))
TAAbot = np.array((RAA_vals[0,0]-TAAval[0],RAA_vals[0,-1]-TAAval[-1]))
RAA_stat  = np.diag(cov_RAA_stat_diag[0:npT,0:npT])**0.5
RAA_sum   = np.diag(cov_RAA_sys_sum[0:npT,0:npT])**0.5
RAA_total = np.diag(cov_RAA_sys_total[0:npT,0:npT])**0.5
ax.errorbar(pTmid,RAA_vals[0],RAA_stat,marker='s',color='black', label='ATLAS 0-10%')
ax.bar(pTmid,2*RAA_total,box_width,RAA_vals[0]-RAA_total,color='blue',alpha=0.4,label='sys errs (no lum/TAA)')
ax.bar(LumpT,2*Lumval,box_width,Lumbot,color='green',alpha=0.4,label='lum error')
ax.bar(TAApT,2*TAAval,box_width,TAAbot,color='red',alpha=0.4,label='TAA error')
for i in np.arange(nstack):
    print(i,labels[i])
    ax.plot(pTmid,models[i],linewidth=2,linestyle=mylin[i],color=mycol[i],label=labels[i])
ax.legend(loc='upper left',ncol=2)

ax2.set_ylabel(r'$\chi^2$/dof')
ax2.set_xlabel('parameter a')
ax2.set_ylim(-0.2,10.)
for i in np.arange(nstack):
    avals = cpars[i,0]*np.arange(0.8,1.2,0.02)
    chi2ndof = np.zeros_like(avals)
    for j,a in enumerate(avals):
        theta = np.array((a,cpars[i,1],cpars[i,2]))
        if (flag==0):
            chi2ndof[j] =  myfn(theta,pTmid,RAA_vals[0],icov[i])/(pTmid.size-npar)
        elif (flag==1):
            chi2ndof[j] =  myfn1(theta,pTmid,RAA_vals[0],cov_stack[i])/(pTmid.size-npar)
        elif (flag==2):
            chi2ndof[j] =  myfn2(theta,pTmid,RAA_vals[0],icov[i])/(pTmid.size-npar)
    ax2.plot(avals,chi2ndof,linewidth=2,linestyle=mylin[i],color=mycol[i])
    ax2.plot((cpars[i,0],cpars[i,0]),(chi2s[i],chi2s[i]-0.2),linewidth=2,linestyle=mylin[i],color=mycol[i])

if (flag==0):
    ax.text(250.,0.48,r'$fn = a*[1. - b*exp(-pT/(c*1000))]$',fontsize=16)
    ax.text(250.,0.45,r'$\chi^2$=0.5*np.linalg.multi_dot((diff,icov,diff))',fontsize=16)
    fig.savefig(figdir+'err_study_RAA_icov_multidot.pdf')
elif (flag==1):
    ax.text(250.,0.48,r'$fn = a*[1. - b*exp(-pT/(c*1000))]$',fontsize=16)
    ax.text(250.,0.45,r'$\chi^2$=0.5*np.dot(diff, np.linalg.solve(cov, diff))',fontsize=16)
    fig.savefig(figdir+'err_study_RAA_cov_solve.pdf')
elif (flag==2):
    ax.text(250.,0.48,r'$fn = b*log(pT/100) + a$',fontsize=16)
    ax.text(250.,0.45,r'$\chi^2$=0.5*np.linalg.multi_dot((diff,icov,diff))',fontsize=16)
    fig.savefig(figdir+'err_study_RAA_log_func.pdf')

# %%
'''Add CMS and ALICE data'''
flag  = 0
mycol = ('black','darkblue','darkgreen','darkred','darkgoldenrod','brown','olive','purple')
mylin = ('dashed','dotted',(0,(5,3)),(5,(2,1,2,3)),'dashdot','solid','dotted',(0,(3,3)),(0,(5,5)))

'''Stack data for ALICE|ATLAS|CMS, use a,b,c for bin-size shorthand'''
a = ALICE_pTmid.size
b = npT
c = 2
ab  = a + b
abc = a + b + c
ac  = a + c

all_pT   = np.zeros(abc)
all_vals = np.zeros(abc)
all_cov  = np.zeros((abc,abc))

all_pT[:a]         = ALICE_pTmid
all_vals[:a]       = ALICE_RAA_vals
all_cov[:a,:a]     = np.einsum('i,j',ALICE_RAA_corr,ALICE_RAA_corr) + np.diag(ALICE_RAA_stat)
all_pT[a:ab]       = pTmid
all_vals[a:ab]     = RAA_vals[0]
all_cov[a:ab,a:ab] = cov_RAA_sys_sum[:b,:b] + cov_RAA_stat_diag[:b,:b] + cov_lumTAA
all_pT[ab:]        = CMS_pTmid[-c:]
all_vals[ab:]      = CMS_RAA_vals[0,-c:]
all_cov[ab:,ab:]   = np.einsum('i,j',CMS_RAA_sys[0,-c:],CMS_RAA_sys[0,-c:]) + np.diag(CMS_RAA_stat[0,-c:])

'''Set max bin for series of fits, first two for ATLAS with diag and cov errors'''
labels  = []
frange  = np.array([[a,ab],[a,ab],[a,abc],[0,abc], [0,a]])
flab    = ['ATLAS diag', 'ATLAS cov', 'ATLAS+CMS', 'ALICE+ATLAS+CMS','ALICE']
fsiz    = len(flab)
theta   = np.array([0.6,1.,0.3])
options = {'maxfun':500}
npar    = theta.size
models  = np.zeros((fsiz,abc))
cpars   = np.zeros((fsiz,npar))
chi2s   = np.zeros((fsiz))
icov_list = []
for i,(lo,hi) in enumerate(frange):
    print('i=',i,'lo,hi=',lo,hi)
    cov = all_cov[lo:hi,lo:hi]
    if (i==0):
        cov = np.diag(np.diag(all_cov[lo:hi,lo:hi]))
    icov = np.linalg.inv(cov)
    icov_list.append(icov)
    print(flab[i],'allclose =',np.allclose(icov@cov,np.eye(cov.shape[0])))
    if (flag==0):
        res = optimize.minimize(myfn,theta,(all_pT[lo:hi],all_vals[lo:hi],icov),jac=None,method='TNC',options=options)
    elif (flag==2):
        res = optimize.minimize(myfn2,theta,(all_pT[lo:hi],all_vals[lo:hi],icov),jac=None,method='TNC',options=options)
    chi2s[i] = res.fun/(hi-lo-npar)
    cpars[i] = res.x
    label    = flab[i] + r' $\chi^2$/dof={0:.2e}'.format(chi2s[i])
    labels.append(label)
    if (flag==0):        
        models[i] = cpars[i,0]*(1.-cpars[i,1]*np.exp(-all_pT/(1000.*cpars[i,2])))
    elif (flag==2):
        models[i] = cpars[i,1]*np.log(all_pT/100.) + cpars[i,0]

fig,(ax,ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [6,2]})
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.3,0.85])
ax.set_xlim([0.,900.])

box_width = 16
RAA_stat  = np.diag(cov_RAA_stat_diag[0:b,0:b])**0.5
RAA_total = np.diag(cov_RAA_sys_total[0:b,0:b])**0.5
ax.errorbar(pTmid,RAA_vals[0],RAA_stat,marker='s',color='blue', label='ATLAS 0-10%')
ax.bar(pTmid,2*RAA_total,box_width,RAA_vals[0]-RAA_total,color='blue',alpha=0.4,label='sys errs (no lum/TAA)')
ax.errorbar(CMS_pTmid[-2:],CMS_RAA_vals[0,-2:],CMS_RAA_stat[0,-2:],marker='s',color='green', label='CMS 0-10%')
ax.bar(CMS_pTmid[-2:],2*CMS_RAA_sys[0,-2:],box_width,CMS_RAA_vals[0,-2:]-CMS_RAA_sys[0,-2:],color='green',alpha=0.4,label='CMS sys')
ax.errorbar(ALICE_pTmid,ALICE_RAA_vals,ALICE_RAA_stat,marker='v',color='red', label='ALICE 0-10%')
ax.bar(ALICE_pTmid,2*ALICE_RAA_corr,box_width/2.,ALICE_RAA_vals-ALICE_RAA_corr,color='RED',alpha=0.4,label='ALICE corr')
for i,(lo,hi) in enumerate(frange):
    ax.plot(all_pT[lo:hi],models[i,lo:hi],linestyle=mylin[i],color=mycol[i],label=labels[i])
ax.legend(ncol=2)

ax2.set_ylabel(r'$\chi^2$/dof')
ax2.set_xlabel('parameter a')
#ax2.set_ylim(-0.2,10.)
for i in np.arange(fsiz):
    avals = cpars[i,0]*np.arange(0.8,1.2,0.02)
    chi2ndof = np.zeros_like(avals)
    for j,a in enumerate(avals):
        (lo,hi) = frange[i]
        theta = np.array((a,cpars[i,1],cpars[i,2]))
        if (flag==0):
            chi2ndof[j] =  myfn(theta,all_pT[lo:hi],all_vals[lo:hi],icov_list[i])/(hi-lo-npar)
        elif (flag==2):
            chi2ndof[j] =  myfn2(theta,all_pT[lo:hi],all_vals[lo:hi],icov_list[i])/(hi-lo-npar)
    ax2.plot(avals,chi2ndof,linewidth=2,linestyle=mylin[i],color=mycol[i])
    ax2.plot((cpars[i,0],cpars[i,0]),(chi2s[i],chi2s[i]-0.2),linewidth=2,linestyle=mylin[i],color=mycol[i])

if (flag==0):
    ax.text(100.,0.8,r'$fn = a*[1. - b*exp(-pT/(c*1000))]$',fontsize=16)
    fig.savefig(figdir+'err_study_RAA_exp_ALICE_ATLAS_CMS.pdf')
elif (flag==2):
    ax.text(100.,0.8,r'$fn = b*log(pT/100) + a$',fontsize=16)
    fig.savefig(figdir+'err_study_RAA_log_ALICE_ATLAS_CMS.pdf')
   
# %%

'''rename ATLAS fig 4 sys-err contributions for 0-10,50-80'''

mylin = ('dashed','dotted','dashdot',(0,(3,3)),(0,(5,5)),'solid')
mycol = ['darkblue','darkred','darkgreen','purple','brown','black']


fig, axs = plt.subplots(1, 2, figsize=(18, 8),tight_layout=True)

for ax in axs:
    ax.set(xlabel='pT (GeV)',ylabel=r'$R_{AA}$ uncertainty')

for i in np.arange(systypes.size):
    y = np.concatenate((RAA_sysvals[i,0,:],np.zeros(1)))
    axs[0].stairs(RAA_sysvals[i,0,:],RAA_pedge,linestyle=mylin[i],color=mycol[i],label=systypes[i])
    axs[1].stairs(RAA_sysvals[i,3,:],RAA_pedge,linestyle=mylin[i],color=mycol[i],label=systypes[i])

#handles, labels = plt.gca().get_legend_handles_labels()
handles, labels = axs[0].get_legend_handles_labels()
for i,handle in enumerate(handles):
    print('i=',i,'handle=',handles[i],'label=',labels[i])


order = [5,0,1,2,3,4]
h = [handles[i] for i in order]
l = [labels[i] for i in order]
axs[0].legend(h,l,loc='center',title='ATLAS PbPb 0-10%',title_fontsize='large')
axs[1].legend(h,l,loc='upper left',title='ATLAS PbPb 50-80%',title_fontsize='large')

savefig = True
if (savefig):
    fig.savefig(figdir+'figA_ATLAS_syserr.pdf')



# %%
