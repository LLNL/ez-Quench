'''
ezQ_ppJet.py
(originally ppfit2.py --> ezQ_ppfit.py --> ezQ_ppJet.py)
created : 2023-jun-26 R.A. Soltz - redo ppfits with emcee and covariant errors
modified: 2023-jun-28 RAS - finish testing chi2/loglik with cov errors for jets 
                            and start emcee for jet body and tail
          2023-jul-11 RAS - revert to 3-param fit after removing bins < 100 GeV
          2023-jul-12 RAS - test wtih full cov (same result) and save both
                            covdiag and covfull chains and max_theta parames
                            These parameters will now replace ppfit.py for all analyses
          2023-jul-23 RAS - run fit with covsys (no stat) and get same displacment
                            --> something amiss with covariant error matrix !!!
          2023-dec-30 RAS - copy ppfit2.py -> ezQ_ppTest.py to track down cov-error issue
          2023-dec-31 RAS - insert and remove fit in log space (no improvement)
          2024-jan-01 RAS - test Gaussian correlation length, then settle with stat-only fit
          2024-may-14 RAS - add nuisance parameter for fully correlated fits
                            (split to ezQ_ppJet.py to further develop and simplify code)
          2024-may-15 RAS - finalize fit and figure with 3-params and fixed tail
          2024-may-16 RAS - add def_tail parameter for tail, save to hdf5
          2024-aug-01 RAS - switch to stat errors with fixed tail, calculate par-errs and save to h5
          2024-nov-01 RAS - update figure to include plot using theta_nup
          2024-nov-01 RAS - increase legend/label font size
'''

# %%

# Import standard packages and set plots inline
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import h5py
from matplotlib.gridspec import GridSpec

generate_figures = True
mycol = ['darkred','darkgreen','darkorange','darkblue']
mylin = [(0,(5,7)),(6,(2,1,2,7))]
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.spines.top']    = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left']   = True
plt.rcParams['axes.spines.right']  = True
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Tahoma']

'''h5 readout'''
ATLAS_datafile = h5dir + 'ATLAS_data.h5'
with h5py.File(ATLAS_datafile,'r') as f:
    ppJetpT_edge  = np.array(f['ppJetpT_edge'])
    ppJetpT_vals  = np.array(f['ppJetpT_vals'])
    ppJetpT_stat  = np.array(f['ppJetpT_stat'])
    ppJetpT_syshi = np.array(f['ppJetpT_syshi'])
    ppJetpT_syslo = np.array(f['ppJetpT_syslo'])
    ppJetpT_sysav = np.array(f['ppJetpT_sysav'])

'''Trim pp data and errors so lowest bins start at 100 GeV'''
lo = 4
hi = -1
ppJetpT_vals = ppJetpT_vals[lo:hi]
ppJetpT_edge = ppJetpT_edge[lo:hi]
ppJetpT_stat = ppJetpT_stat[lo:hi]
ppJetpT_sysav = ppJetpT_sysav[lo:hi]
print('ppJetpT_edge =',ppJetpT_edge)

pTmid = 0.5*(ppJetpT_edge[:-1]+ppJetpT_edge[1:])
pTones   = np.ones(pTmid.size)

rel_stat = ppJetpT_stat/ppJetpT_vals
rel_sys  = ppJetpT_sysav/ppJetpT_vals

def ppJet(theta,pT_edge,fix_tail):
    (a,b,c) = theta
    (d,e,f) = fix_tail
    pTmin,pTmax,pTbin = 100,1200,1
    pT = np.arange(pTmin,pTmax,pTbin)
    iw = (a/pT)**(b+c*np.log(pT))/(1+d*np.exp((pT-f)/e))
#Fill data histograms
    (ppJ0,edges) = np.histogram(pT,pT_edge,weights=iw)
    ppJ0 = pTbin * ppJ0 / np.diff(edges)
    return ppJ0

def chi2_ppJet(theta,pT_edge,pT_vals,stat,sysd,fix_tail):
    model = ppJet(theta,pT_edge,fix_tail)
    diff = pT_vals - model
#    chi2 = 0.5*np.sum(diff**2/(stat**2+sysd**2))
    chi2 = 0.5*np.sum(diff**2/(stat**2))
    return chi2

'''for chi2_nup, theta has one more (nuisance) parameter'''
def chi2_nup_ppJet(theta,pT_edge,pT_vals,stat,sysd,fix_tail):
    model = ppJet(theta[:-1],pT_edge,fix_tail)
    diff = pT_vals - model - sysd*theta[-1]
    chi2 = 0.5*np.sum(diff**2/stat**2) + 0.5*theta[-1]**2
    return chi2

# %%

'''Test fit to stat and stat+sys errors'''

'''Fit to stat and sysd (diagonal)'''
theta_init = np.array([113.722, 2.720, 0.584])
fix_tail = np.array([1.0,150.,700.])
options = {'maxfun':1000}
res = optimize.minimize(chi2_ppJet,
                        theta_init,
                        (ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail),
                        jac=None,method='L-BFGS-B',options=options)
chi2_stat = res.fun
theta_stat = res.x
#theta_stat = np.array([115.31, 3.74, 0.37, 3.84, 150.02])
#print(res.hess_inv)
#error_stat  = (np.diag(res.hess_inv))**0.5
model_stat = ppJet(theta_stat,ppJetpT_edge,fix_tail)
ratio_stat = model_stat/ppJetpT_vals
print('Cost for theta_stat optimize.minimize: {:.3f}'.format(chi2_stat))
print('theta_stat:',np.array2string(theta_stat,precision=2))
#print('error_stat:',np.array2string(error_stat,max_line_width=120,precision=2))

'''Fit to stat and fully correlated sysav with nuisance parameter'''
res = optimize.minimize(chi2_nup_ppJet,
                        np.append(theta_init,0.),
                        (ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail),
                        jac=None,method='L-BFGS-B',options=options)
#                        method='TNC',options=options)
chi2_nup = res.fun
theta_nup = res.x
model_nup = ppJet(theta_nup[:-1],ppJetpT_edge,fix_tail)
ratio_nup = model_nup/ppJetpT_vals
print('Cost for theta_nup optimize.minimize: {:.3f}'.format(chi2_nup))
print('theta_nup:',np.array2string(theta_nup[:-1],precision=2))
print('ratio_nup=',np.array2string(ratio_nup,precision=2))
fnup = 1.6

'''Plot for ezQtex'''

#fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
#fig.subplots_adjust(hspace=0)
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(hspace=0)
gs = GridSpec(4, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:2,0])
ax2 = fig.add_subplot(gs[2,0],sharex=ax1)

ax1.set(xlabel=r'$p_T$ (GeV)',ylabel=r'$\dfrac{d\sigma^2}{dp_T dy}$ (nb/GeV)')
ax2.set(xlabel=r'$p_T$ (GeV)',ylabel=r'$\dfrac{\rm fit}{\rm data}$',ylim=(0.75,1.25))
ax1.tick_params(axis='x',which='both',top=True,direction='in')
ax1.tick_params(axis='y',which='both',right=True,direction='in')
ax2.tick_params(axis='x',which='both',top=True,direction='in')
ax2.tick_params(axis='y',which='both',right=True,direction='in')
ax1.loglog()
ax2.semilogx()

mycol = ['darkred','darkgreen','darkorange','darkblue']
mylin = [(0,(5,7)),(6,(2,1,2,7))]
mylw  = 2
ax1.errorbar(pTmid,ppJetpT_vals,ppJetpT_stat,marker='s',linestyle='None',color='black',label='ATLAS pp 5.02 TeV')
ax1.plot(pTmid,model_stat,'r',linestyle=mylin[0],linewidth=mylw,label='fit with statistical errors')
ax1.plot(pTmid,model_nup*fnup,'b',linestyle=mylin[1],linewidth=mylw,label='fit with combined errors')
ax1.legend(loc='lower left',borderaxespad=2,handlelength=3,fontsize='large')

ax2.errorbar(pTmid,pTones,rel_stat,marker='s',color='black',linestyle='dotted',label='ATLAS pp')
ax2.bar(pTmid,2*rel_sys,0.1*pTmid,pTones-rel_sys,color='b',alpha=0.4,label='sys-err')
ax2.plot(pTmid,ratio_stat,'r',linestyle=mylin[0],linewidth=mylw,label='ratio stat')
ax2.plot(pTmid,ratio_nup*fnup,'b',linestyle=mylin[1],linewidth=mylw,label='ratio stat-sys-nup')

generate_figures = True
if (generate_figures):
    fig.savefig(figdir+'fig_ppJet.pdf',bbox_inches='tight')

# %%
'''Calculate 1-sigma parameter errors with chi2 +/-1 test because res.hess_inv results were not making sense'''

err_pos = np.zeros_like(theta_stat)
err_neg = np.zeros_like(theta_stat)
rel_step = 1.e-5
for i in np.arange(theta_stat.size):
    theta_pos = np.array(theta_stat)
    theta_neg = np.array(theta_stat)
    del_pos = 0
    del_neg = 0
    ipos = 0
    ineg = 0
    while(del_pos<1.):
        ipos += 1
        theta_pos[i] += theta_stat[i]*rel_step
        del_pos = chi2_ppJet(theta_pos,ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail) - chi2_stat
    while(del_neg<1.):
        ineg += 1
        theta_neg[i] -= theta_stat[i]*rel_step
        del_neg = chi2_ppJet(theta_neg,ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail) - chi2_stat
    print('ipos =',ipos,' ineg =',ineg)
    err_pos[i] = theta_pos[i]-theta_stat[i]
    err_neg[i] = theta_stat[i]-theta_neg[i]

print('Cost for theta_stat optimize.minimize: {:.3f}'.format(chi2_stat))
print('theta_stat:',np.array2string(theta_stat,precision=5))
print('err_pos =',np.array2string(err_pos,precision=5))
print('err_neg =',np.array2string(err_neg,precision=5))

# %%
'''Repeat for 1-sigma error calculation for theta_nup'''

err_pos = np.zeros_like(theta_nup)
err_neg = np.zeros_like(theta_nup)
rel_step = 1.e-5
for i in np.arange(theta_nup.size):
    theta_pos = np.array(theta_nup)
    theta_neg = np.array(theta_nup)
    del_pos = 0
    del_neg = 0
    ipos = 0
    ineg = 0
    while(del_pos<1.):
        ipos += 1
        theta_pos[i] += theta_nup[i]*rel_step
        del_pos = chi2_nup_ppJet(theta_pos,ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail) - chi2_stat
    while(del_neg<1.):
        ineg += 1
        theta_neg[i] -= theta_nup[i]*rel_step
        del_neg = chi2_nup_ppJet(theta_neg,ppJetpT_edge,ppJetpT_vals,ppJetpT_stat,ppJetpT_sysav,fix_tail) - chi2_stat
    print('ipos =',ipos,' ineg =',ineg)
    err_pos[i] = theta_pos[i]-theta_nup[i]
    err_neg[i] = theta_nup[i]-theta_neg[i]

print('Cost for theta_nup optimize.minimize: {:.3f}'.format(chi2_stat))
print('theta_nup:',np.array2string(theta_nup,precision=5))
print('err_pos =',np.array2string(err_pos,precision=5))
print('err_neg =',np.array2string(err_neg,precision=5))
# %%
'''Save parameters to hdf5 file'''

fitfile = h5dir + 'fitpars_ppJet.h5'
with h5py.File(fitfile,'a') as g:
    try:
        dset = g.create_dataset('theta_stat',data=theta_stat)
        dset = g.create_dataset('theta_nup',data=theta_nup)
        dset = g.create_dataset('fix_tail',data=fix_tail)
        dset = g.create_dataset('staterr_pos',data=err_pos)
        dset = g.create_dataset('staterr_neg',data=err_neg)
        print('Creating datasets for theta')
    except:
        print('Rewriting datasets for theta')
        g['theta_stat'][...]=theta_stat
        g['theta_nup'][...]=theta_nup
        g['theta_nup'][...]=fix_tail
        g['staterr_pos'][...]=err_pos
        g['staterr_neg'][...]=err_neg

# %%