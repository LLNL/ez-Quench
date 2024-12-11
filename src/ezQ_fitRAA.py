# %%

'''
ezQ_fitRAA.py
R.A. Soltz
run emcee for gamma RAA calculation
created: 2022-nov-11 R.A. Soltz
modified: 2022-dec-28 RAS to parse old and new ATLAS RAA data
          2024-may-31 RAS - Switch to hdf5 data & use covariance matrices for errors
          2024-aug-06 RAS - convert back to generic ezQ_RAAcent_fit.py with variable fit and error type
          2024-aug-07 RAS - add takeLog option, and calculate mean/median RAA and meanest/medianest draws
          2024-nov-01 RAS - switch theta (for pp) to use theta_nup

'''
# Import standard packages and set plots inline
import os
import sys
import h5py
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from ezQ_helper import ppw0,log_like_gamma,log_prob_gamma, fx_gamma, log_like_delta,log_prob_delta

'''
Set default arguments, then check for command-line overwrite
Batch usage = python ezQ_fitRAA.py [run_emcee] [ifit] [ierr]
Script will overwrite defaults only if all 3 arguments are given
'''
run_emcee = False
ifit = 0
ierr = 0
nsamples = 10000
batch_mode = False
if (len(sys.argv)==5):
    batch_mode = True
    run_emcee = int(sys.argv[1])
    ifit = int(sys.argv[2])
    ierr = int(sys.argv[3])
    nsamples = int(sys.argv[4])

fit_type  = np.array(['gamma','delta'])
fit_pTbin = np.array([1.,0.03])
err_type  = np.array(['stat','sysd','systot','syssum','syslog','sysTlumlog'])
fiterr    = 'RAAcent_' + fit_type[ifit] + '_' + err_type[ierr]
print('Executing ',fiterr,'with run_emcee =',run_emcee)

# Use this flag to remake figures
#generate_figures = False
generate_figures = True
mycol = ('darkgreen','darkorange','darkred','darkblue','darkgoldenrod','brown','olive','grey')
mylin = ('dashed','dashdot','dotted',(0,(2,2)),(0,(3,3)),(0,(4,4)),(0,(5,5)),(0,(3,5,1,5,1,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

'''Read hdf5 inputs for ppJet parameters and ATLAS data'''
ATLAS_datafile = h5dir + 'ATLAS_data.h5'
with h5py.File(ATLAS_datafile,'r') as f:
    RAA_cedge     = np.array(f['RAA_cedge'])
    RAA_pedge     = np.array(f['RAA_pedge'])
    RAA_vals      = np.array(f['RAA_vals'])
    RAA_stat      = np.array(f['RAA_stat'])
    RAA_syst      = np.array(f['RAA_syst'])
    cov_RAA_stat_diag = np.array(f['cov_RAA_stat_diag'])
    cov_RAA_sys_sum   = np.array(f['cov_RAA_sys_sum'])
    cov_RAA_sys_tot   = np.array(f['cov_RAA_sys_total'])
    cov_RAA_complete  = np.array(f['cov_RAA_complete'])

ppJet_fitfile = h5dir + 'fitpars_ppJet.h5'
with h5py.File(ppJet_fitfile,'r') as g:
        theta_stat = np.array(g['theta_stat'])
        theta_nup  = np.array(g['theta_nup'])
        fix_tail   = np.array(g['fix_tail'])

'''Fill in pT jet weights and RAA denominator'''
pTlo  = 150  # minimum for pTquench and pT
pThi  = 800 # top value for pT
pTbin = fit_pTbin[ifit]   # bin width
pT    = np.arange(pTlo,pThi,pTbin)
#theta = np.concatenate((theta_stat,fix_tail))
theta = np.concatenate((theta_nup[:-1],fix_tail))
w0    = ppw0(theta,pT)
(denom,edges) = np.histogram(pT,RAA_pedge,weights=w0)

'''Restrict data to first cbin=0-10% and invert covariance matrix'''
cov_RAA_0010 = cov_RAA_complete[0:RAA_vals.shape[1],0:RAA_vals.shape[1]]
RAA_vals_0010 = RAA_vals[0]
RAA_stat_0010 = RAA_stat[0]
RAA_syst_0010 = RAA_syst[0]
RAA_pmid      = 0.5*(RAA_pedge[:-1]+RAA_pedge[1:])
#print('RAA_vals.shape=',RAA_vals.shape)
#print('RAA_pedge=',RAA_pedge)
RAA_vals_sqr = np.einsum('i,j',RAA_vals_0010,RAA_vals_0010)

takeLog = 0
n = RAA_vals.shape[1]
if (ierr==0):
    cov_RAA_0010 = cov_RAA_stat_diag[0:n,0:n]
elif (ierr==1):
    cov_RAA_0010 = cov_RAA_stat_diag[0:n,0:n] + np.diag(np.diag(cov_RAA_sys_tot[0:n,0:n]))
elif (ierr==2):
    cov_RAA_0010 = cov_RAA_sys_tot[0:n,0:n] + cov_RAA_stat_diag[0:n,0:n]
elif (ierr==3):
    cov_RAA_0010 = cov_RAA_sys_sum[0:n,0:n] + cov_RAA_stat_diag[0:n,0:n]
elif (ierr==4):
    cov_RAA_0010 = (cov_RAA_sys_sum[0:n,0:n]+cov_RAA_stat_diag[0:n,0:n])/RAA_vals_sqr
    takeLog = 1
elif (ierr==5):
    cov_RAA_0010 = cov_RAA_complete[0:n,0:n]/RAA_vals_sqr
    takeLog = 1
icov_RAA_0010 = np.linalg.inv(cov_RAA_0010)

# %%

''' Loop through some a-coefficient values, plot RAA, and test log-like'''
fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.set_xlim(100.,650.)
ax.set(title=fiterr,xlabel='pT (GeV)',ylabel='RAA 0-10%')
ax.errorbar(RAA_pmid,RAA_vals_0010,RAA_stat_0010,marker='s',color='k',label='ATLAS RAA 5.02 TeV')

if (ifit==0):
    theta = np.array([0.7,0.2,2.5])
    a_loop = np.arange(0.4,0.85,0.07)
    for i,a in enumerate(a_loop):
        theta[0]  = a
        (b,k)     = theta[1:]
        fxg       = fx_gamma(pT,a,theta[1],theta[2])
        w0Q       = np.einsum('i,ij',w0,fxg)
        smoothe_RAA = w0Q/w0
        num,edges = np.histogram(pT,RAA_pedge,weights=w0Q)
        test_RAA  = num/denom
        test_loglik = log_like_gamma(theta,pT, w0, denom, RAA_pedge,RAA_vals_0010,icov_RAA_0010,takeLog)
        label = 'a = {0:.2f}, b = {1:.2f}, k = {2:.2f},loglike= {3:.2f}'.format(a,b,k,test_loglik)
        ax.plot(pT,smoothe_RAA,color=mycol[i],linestyle='solid',label=label)
        ax.plot(RAA_pmid,test_RAA,color=mycol[i],linestyle=mylin[i],label=label)
elif (ifit==1):
    theta = np.array([1.,0.24])
    a_loop = np.arange(0.96,1.35,0.05)
    for i,a in enumerate(a_loop):
        theta[0] = a
        b = theta[1]
        h = a*(pT**b)*np.log(pT)
        (num,edges) = np.histogram(pT-h,RAA_pedge,weights=w0)
        test_RAA = num/denom
        test_loglik = log_like_delta(theta,pT, w0, denom, RAA_pedge,RAA_vals_0010,icov_RAA_0010,takeLog)
        label = 'a = {0:.2f}, b = {1:.2f}, loglike= {2:.2f}'.format(a,b,test_loglik)
        ax.plot(RAA_pmid,test_RAA,color=mycol[i],linestyle=mylin[i],label=label)
ax.legend()

generate_figures = False
if(generate_figures):
    fig.savefig(figdir+fiterr+'_loop.pdf')
generate_figures = True

# %%

'''Set variables for emcee generation and readback'''
nwalkers = 8
sampletxt = '{:.0e}'.format(nsamples)
emcee_filename = h5dir + fiterr + '_' + sampletxt + '.h5'
if (ifit==0):
    theta = np.array([0.65,0.2,2.5])
elif (ifit==1):
    theta = np.array([0.8,0.3])

ndim = theta.size
pos = theta + 1e-4 * np.random.randn(nwalkers,ndim)

'''Set run_emcee=True to run backend and save to file'''
if (run_emcee):
    if (os.path.exists(emcee_filename)):
        os.remove(emcee_filename)
    backend = emcee.backends.HDFBackend(emcee_filename)
    backend.reset(nwalkers, ndim)
    if (ifit==0):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_gamma,
                args=(pT,w0,denom,RAA_pedge,RAA_vals_0010,icov_RAA_0010,takeLog),backend=backend)
    elif (ifit==1):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_delta,
                args=(pT,w0,denom,RAA_pedge,RAA_vals_0010,icov_RAA_0010,takeLog),backend=backend)
    sampler.run_mcmc(pos, nsamples, progress=True);

# %%

'''Read back emcee output and plot diagnostics (works for gamma and delta)'''

labels = ['a','b','k']
emcee_filename = h5dir + fiterr + '_' + sampletxt + '.h5'
reader = emcee.backends.HDFBackend(emcee_filename)
tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
all_samples = reader.get_chain()
flat_samples = reader.get_chain(flat=True)
afterburn_samples = reader.get_chain(discard=burnin,flat=True)
thin_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

print('tau: {0}'.format(tau))
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
#print("all chain shape: {0}".format(all_samples.shape))
#print("flat chain shape: {0}".format(flat_samples.shape))
#print("thin chain shape: {0}".format(thin_samples.shape))
#print("flat log prob shape: {0}".format(log_prob_samples.shape))

fig, axes = plt.subplots(ndim, figsize=(12, 6), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(all_samples[:, :, i], 'k', alpha=0.1)
    ax.set_xlim(0, len(all_samples))
    ax.set_ylabel(labels[i]+'\n'+r'$\tau$ = {0:.0f}'.format(tau[i]))
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('step number')

if (not batch_mode):
    corner.corner(afterburn_samples,labels=labels[:ndim])

#%%

'''Plot most probable and some draws'''

fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.set(title=fiterr,xlabel='pT (GeV)',ylabel='RAA')
ax.errorbar(RAA_pmid,RAA_vals_0010,RAA_stat_0010,marker='s',color='k',label='ATLAS RAA 5.02 TeV')

max_ind = np.argmax(reader.get_log_prob(flat=True))
max_pars = flat_samples[max_ind]
ave_pars = np.mean(flat_samples,axis=0)
rms_pars = np.std(flat_samples,axis=0)

if (ifit==0):
    (a,b,k)   = max_pars
    fxg       = fx_gamma(pT,a,b,k)
    w0Q       = np.einsum('i,ij',w0,fxg)
    num,edges   = np.histogram(pT,RAA_pedge,weights=w0Q)
elif (ifit==1):
    (a,b)   = max_pars
    h = a*(pT**b)*np.log(pT)
    (num,edges) = np.histogram(pT-h,RAA_pedge,weights=w0)

RAA_max  = num/denom
ax.plot(RAA_pmid,RAA_max,color=mycol[ifit],linestyle=mylin[ifit],linewidth=2,label=fit_type[ifit]+'-most-probable')

print('max-index = ',max_ind)
print('fit max  =',max_pars)
print('fit ave =',ave_pars)
print('fit rms =',rms_pars)

ndraws = 100
inds = np.random.randint(len(thin_samples), size=ndraws)
RAA_draw = np.zeros((ndraws,RAA_pmid.size))
par_draw = np.zeros((ndraws,ndim))
for i,ind in enumerate(inds):
    par_draw[i] = thin_samples[ind]
    if (ifit==0):
        (a,b,k) = par_draw[i]
        fxg  = fx_gamma(pT,a,b,k)
        w0Q  = np.einsum('i,ij',w0,fxg)
        num,edges   = np.histogram(pT,RAA_pedge,weights=w0Q)
    elif (ifit==1):
        (a,b) = par_draw[i]
        h = a*(pT**b)*np.log(pT)
        (num,edges) = np.histogram(pT-h,RAA_pedge,weights=w0)

    RAA_draw[i] = num/denom
    ax.plot(RAA_pmid,RAA_draw[i],color=mycol[ifit],alpha=0.1)

'''find draw closest to average'''

RAA_ave = np.mean(RAA_draw,axis=0)
RAA_med = np.median(RAA_draw,axis=0)

i_ave = np.argmin(np.sum((RAA_draw - RAA_ave)**2,axis=1))
i_med = np.argmin(np.sum((RAA_draw - RAA_med)**2,axis=1))
RAA_meanest = RAA_draw[i_ave]
RAA_medianest = RAA_draw[i_med]
par_meanest = par_draw[i_ave]
par_medianest = par_draw[i_med]

ax.plot(RAA_pmid,RAA_ave,color=mycol[2],linestyle=mylin[2],linewidth=2,label='RAA-mean')
ax.plot(RAA_pmid,RAA_med,color=mycol[3],linestyle=mylin[3],linewidth=2,label='RAA-median')
ax.plot(RAA_pmid,RAA_meanest,color=mycol[2],linestyle=mylin[4],linewidth=2,label='RAA-mean-est')
ax.plot(RAA_pmid,RAA_medianest,color=mycol[3],linestyle=mylin[5],linewidth=2,label='RAA-median-est')

ax.legend()

if(generate_figures):
    fig.savefig(figdir+fiterr+'_'+sampletxt+'_draws.pdf')

# %%
'''Save parameters and draws hdf5 file'''

h5file = h5dir + 'ezQ_RAAcent_' + sampletxt + '_draws.h5'
with h5py.File(h5file,'a') as g:
    try:
        dset = g.create_dataset(fiterr+'_max_pars',data=max_pars)
        dset = g.create_dataset(fiterr+'_ave_pars',data=ave_pars)
        dset = g.create_dataset(fiterr+'_rms_pars',data=rms_pars)
        dset = g.create_dataset(fiterr+'_RAA_pmid',data=RAA_pmid)
        dset = g.create_dataset(fiterr+'_RAA_maxp',data=RAA_max)
        dset = g.create_dataset(fiterr+'_RAA_draw',data=RAA_draw)
        dset = g.create_dataset(fiterr+'_par_draw',data=par_draw)
        dset = g.create_dataset(fiterr+'_RAA_ave',data=RAA_ave)
        dset = g.create_dataset(fiterr+'_RAA_med',data=RAA_med)
        dset = g.create_dataset(fiterr+'_RAA_meanest',data=RAA_meanest)
        dset = g.create_dataset(fiterr+'_RAA_medianest',data=RAA_medianest)
        dset = g.create_dataset(fiterr+'_par_meanest',data=par_meanest)
        dset = g.create_dataset(fiterr+'_par_medianest',data=par_medianest)
        print('Creating h5 datasets for',fiterr)
    except:
        g[fiterr+'_max_pars'][...]=max_pars
        g[fiterr+'_ave_pars'][...]=ave_pars
        g[fiterr+'_rms_pars'][...]=rms_pars
        g[fiterr+'_RAA_pmid'][...]=RAA_pmid
        g[fiterr+'_RAA_maxp'][...]=RAA_max # type: ignore
        g[fiterr+'_RAA_draw'][...]=RAA_draw # type: ignore
        g[fiterr+'_par_draw'][...]=par_draw
        g[fiterr+'_RAA_ave'][...]=RAA_ave
        g[fiterr+'_RAA_med'][...]=RAA_med
        g[fiterr+'_RAA_meanest'][...]=RAA_meanest
        g[fiterr+'_RAA_medianest'][...]=RAA_medianest
        g[fiterr+'_par_meanest'][...]=par_meanest
        g[fiterr+'_par_medianest'][...]=par_medianest
        print('Rewriting h5 datasets for',fiterr)

# %%
