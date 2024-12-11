'''
ezQ_fitTrentoRAA.py
created:  2024-jun-11 R.A. Soltz (by adding emcee to ezQ_trentoTest.py)
modified: 2024-jul-17 RAS complete tau and error studies
          2024-oct-17 RAS add takeLog option
          2024-oct-18 RAS revise for batch_mode running
          2024-nov-23 RAS extend most-prob plot into ALICE,CMS ranges
'''

# %%

# Import standard packages and set plots inline
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import emcee
import corner
from ezQ_helper import ppw0, ezQ_delta_trento, log_prob_delTrento, log_like_delTrento

'''
Set default arguments, then check for command-line overwrite
Batch usage = python ezQ_fitTrentoRAA.py [run_emcee] [lf] [tau] [ierr] [nsamples]
Script will overwrite defaults only if all 3 arguments are given
'''
run_emcee = False
lf = 0
tau = 0.5
ierr = 0
nsamples = 10000
takeLog = 0
batch_mode = False
run_emcee  = False
if (len(sys.argv)==6):
    batch_mode = True
    run_emcee = int(sys.argv[1])
    lf   = int(sys.argv[2])
    tau  = float(sys.argv[3])
    ierr = int(sys.argv[4])
    nsamples = int(sys.argv[5])
err_type  = np.array(['stat','dsys','dslT','comp','clog'])
print('Executing ezQ_fitTrentoRAA,py with lf={0:d}, tau={1:.2f}, err = '.format(lf,tau) + err_type[ierr] )
if (ierr == 4):
    takeLog = 1
    print('Setting takeLog = 1')

generate_figures = True
mycol = ('darkred','darkgreen','darkblue','darkorange','darkgoldenrod','brown','olive')
mylin = ((0,(5,7)),(6,(2,1,2,7)),'dotted','dashed','dashdot',(0,(2,2)),(0,(3,3)),(0,(4,4)),(0,(5,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + 'fig/'
h5dir   = workdir + 'h5/'
base    = 'RAA_Pb5_ATL_del'

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 16

'''Read hdf5 inputs for ppJet parameters and ATLAS data'''
ATLAS_datafile = h5dir + 'ATLAS_data.h5'
with h5py.File(ATLAS_datafile,'r') as f:
    RAA_cedge     = np.array(f['RAA_cedge'])
    RAA_pedge     = np.array(f['RAA_pedge'])
    RAA_vals      = np.array(f['RAA_vals'])
    RAA_stat      = np.array(f['RAA_stat'])
    RAA_syst      = np.array(f['RAA_syst'])
    cov_pp_lum_E      = np.array(f['cov_pp_lum_E'])
    cov_TAA_E         = np.array(f['cov_TAA_E'])
    cov_RAA_stat_diag = np.array(f['cov_RAA_stat_diag'])
    cov_RAA_sys_total = np.array(f['cov_RAA_sys_total'])
    cov_RAA_sys_sum   = np.array(f['cov_RAA_sys_sum'])
    cov_RAA_l20pct    = np.array(f['cov_RAA_l20pct'])
    cov_RAA_complete  = np.array(f['cov_RAA_complete'])

ALICE_CMS_datafile = h5dir + 'ALICE_CMS_data.h5'
with h5py.File(ALICE_CMS_datafile,'r') as g:
    ALICE_RAA_pT    = np.array(g['ALICE_RAA_pT'])
    CMS_RAA_pT      = np.array(g['CMS_RAA_pT'])[3:]
RAA_pedge_x = np.concatenate((np.array((100,120,140)),RAA_pedge))
pTmid_x     = 0.5*(RAA_pedge_x[1:]+RAA_pedge_x[:-1])
pTmid_CMS   = 0.5*(CMS_RAA_pT[1:]+CMS_RAA_pT[:-1])
pTmid_ALICE = 0.5*(ALICE_RAA_pT[1:]+ALICE_RAA_pT[:-1])
print('ATLAS_pT=',RAA_pedge)
print('ALICE_RAA_pT =',ALICE_RAA_pT)
print('CMS_RAA_pT =',CMS_RAA_pT)

ppJet_fitfile = h5dir + 'fitpars_ppJet.h5'
with h5py.File(ppJet_fitfile,'r') as g:
        theta_stat = np.array(g['theta_stat'])
        theta_nup  = np.array(g['theta_nup'])
        fix_tail   = np.array(g['fix_tail'])

'''Add some short-hand'''
nc    = RAA_cedge.size - 1
npT   = RAA_pedge.size - 1
pTmid = 0.5*(RAA_pedge[1:]+RAA_pedge[:-1])
RAA_sys_err = (np.diag(cov_RAA_sys_total)**0.5).reshape(nc,npT)
RAA_lum_err = (np.diag(cov_pp_lum_E)**0.5).reshape(nc,npT)
RAA_taa_err = (np.diag(cov_TAA_E)**0.5).reshape(nc,npT)
#print('RAA_taa_err=',np.array2string(RAA_taa_err,precision=3))

diagsys = np.diag(np.diag(cov_RAA_sys_total))
diagsys_lum_Taa = diagsys + cov_pp_lum_E + cov_TAA_E
cov_comp_log = cov_RAA_complete/np.einsum('i,j',RAA_vals.ravel(),RAA_vals.ravel())
                
istat = np.linalg.inv(cov_RAA_stat_diag)
idsys = np.linalg.inv(diagsys)
idslT = np.linalg.inv(diagsys_lum_Taa)
icomp = np.linalg.inv(cov_RAA_complete)
iclog = np.linalg.inv(cov_comp_log)

ierr_stack = np.vstack((istat,idsys,idslT,icomp,iclog)).reshape(err_type.size,nc*npT,nc*npT)
RAA_err   = ierr_stack[ierr]
print('ierr_stack.shape =',ierr_stack.shape)
print('RAA_err.shape =',RAA_err.shape)

ctxt = []
for i in np.arange(nc):
    ctxt.append(str(RAA_cedge[i])+'-'+str(RAA_cedge[i+1]))
print('ctxt =',ctxt)

'''
Reading in trento pairs is a multi-step process
1. open h5 file containing one data set per centrality
2. check that tpair count is the same for all cbins
3. size up array with dimensions [cbins,pair#,entries]
4. read desired centrailties into pair_array
5. use lf==2 for l-weighted tpairs (in entries [2:4])
'''

trento_base = 'PbPb_10k'
h5_tpair = h5dir + 't-pairs_' + trento_base + '_tau{:0.1f}.h5'.format(tau)
print(h5_tpair)
with h5py.File(h5_tpair,'r') as h:
    nprev = -1
    for k in h.keys():
    #    print(k, len(h[k]))
        npairs = len(h[k])
        if ((nprev>0) and (nprev != npairs)):
            print('t-pair mismatch: Regenerate with trentoCent.py')
        nprev  = npairs
    tpair_array = np.zeros((nc,npairs,2))
    for i in np.arange(nc):
        tpair_key = trento_base + '_' + ctxt[i] + '_tau{:0.1f}'.format(tau)
        if (lf==2):
            tpair_array[i,:,:] = np.array(h[tpair_key])[:,2:4]
        else:
            tpair_array[i,:,:] = np.array(h[tpair_key])[:,0:2]
    #    print(tpair_key,len(h[tpair_key]))
tpair_sort = np.sort(tpair_array)

#for i in np.arange(nc):
#    tpair_key = trento_base + '_' + ctxt[i] + '_tau{:0.1f}'.format(tau)
#    print(tpair_key,'(first 10 pairs):\n',tpair_array[i,:10])
#    print(tpair_key,'(10 pairs sorted):\n',tpair_sort[i,:10])

# %%

'''
    Test smootheness and normalization of R0,R1,R2 histograms
    ppJet weights exhibit percent-scale binning effects for pTbin >= 0.3 and above, set to 0.1
    RAA ratios show binning effects for pTbin >=3, set to 1 GeV
'''

if (not batch_mode):
    pTlo   = 150   # minimum for pTquench and pT
    pThi   = 800   # top value for pT
    pTbin1 = 0.01  # bin width
    pTbin2 = 0.1   # bin width
    pTfine = np.arange(pTlo,pThi,pTbin1)
    pTcors = np.arange(pTlo,pThi,pTbin2)
    theta_pp = np.concatenate((theta_nup[:-1],fix_tail))
    wfine = ppw0(theta_pp,pTfine)
    wcors = ppw0(theta_pp,pTcors)

    (Jf,edges) = np.histogram(pTfine,RAA_pedge,weights=wfine)
    Jf = Jf*pTbin1/np.diff(edges)
    (Jc,edge) = np.histogram(pTcors,RAA_pedge,weights=wcors)
    Jc = Jc*pTbin2/np.diff(edges)
    Rcf = Jc/Jf

    '''Calculate TRENTO weights'''
    ntpair = 500
    pTbin = 1
    pT    = np.arange(pTlo,pThi,pTbin)
    w     = ppw0(theta_pp,pT)
    wt    = np.einsum('i,j',np.ones(ntpair),w)
    (a,b) = (0.62,0.24)
    h = a*(pT**b)*np.log(pT)
    for i in np.arange(1):
        p1 = pT - np.einsum('i,j',tpair_sort[i,:ntpair,0],h)
        p2 = pT - np.einsum('i,j',tpair_sort[i,:ntpair,1],h)
        J1,edges = np.histogram(p1,RAA_pedge,weights=wt)
        J1 = J1*pTbin/np.diff(edges)/ntpair
        R1 = J1/Jf
        J2,edges = np.histogram(p2,RAA_pedge,weights=wt)
        J2 = J2*pTbin/np.diff(edges)/ntpair
        R2 = J2/Jf

    (fig,ax) = plt.subplots(2, 1, figsize=(12, 8),sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].set(xlabel='pT (GeV)',ylabel='dN/dJet')
    ax[1].set(xlabel='pT (GeV)',ylabel='ratios',ylim=(0.2,1.2))
    for a in ax:
        a.tick_params(axis='x',which='both',top=True,direction='in')
        a.tick_params(axis='y',which='both',right=True,direction='in')
    ax[0].loglog()
    ax[1].semilogx()

    ax[0].plot(pTfine,wfine,'k',linestyle='dashed',linewidth=0.5,label='smoothe')
    ax[0].plot(pTmid,Jf,'r',linestyle=mylin[0],label='fine = {0:0.2f} MeV'.format(pTbin1))
    ax[0].plot(pTmid,Jc,'g',linestyle=mylin[1],label='coarse = {0:0.2f} MeV'.format(pTbin2))
    ax[0].plot(pTmid,J1,color=mycol[1],label='Jet-1 TRENTO {0:0.1f} MeV'.format(pTbin))
    ax[0].plot(pTmid,J2,color=mycol[2],label='Jet-2 TRENTO {0:0.1f} MeV'.format(pTbin))
    ax[0].legend()

    ax[1].plot(pTmid,np.ones(pTmid.size),'k--')
    ax[1].plot(pTmid,Rcf,'k-',label='Ratio coarse/fine')
    ax[1].plot(pTmid,R1,color=mycol[1],label='R1')
    ax[1].plot(pTmid,R2,color=mycol[2],label='R2')
    ax[1].legend()

# %%

'''  Run loop test for ezQ_fitTrentoRAA'''

'''Set initial theta for loop and emcee'''
beta = 0.2
if (lf==0):
    alpha = 1.2 - 0.8*tau
elif (lf==2):
    alpha = 1.2 - tau
theta = np.array((alpha,beta))

title = 'RAA_delTrento_l' + str(lf) + 'loop_' + 'tau{:0.1f}_'.format(tau) + err_type[ierr]

# Set weights for initial jet using finer pT bins
#pTlo   = 150   # minimum for pTquench and pT
#pThi   = 800   # top value for pT
pTlo   = 100   # minimum for pT extension
pThi   = 800   # top value for pT extension
pTbin  = 0.01  # bin width
pTJ0 = np.arange(pTlo,pThi,pTbin)
thetaJ0 = np.concatenate((theta_nup[:-1],fix_tail))
wJ0 = ppw0(thetaJ0,pTJ0)
(J0,edges) = np.histogram(pTJ0,RAA_pedge,weights=wJ0)
J0 = J0*pTbin/np.diff(edges)
(J0_x,edges) = np.histogram(pTJ0,RAA_pedge_x,weights=wJ0)
J0_x = J0_x*pTbin/np.diff(edges)


# Set weights for initial jet using courser pT bins for Trento RAA
pTbin  = 1  # bin width
pT = np.arange(pTlo,pThi,pTbin)
w0 = ppw0(thetaJ0,pT)

#ntpmax = 1000
ntpmax = 500
tpa = tpair_sort[:,:ntpmax,:]
w  = np.einsum('i,j',np.ones(ntpmax),w0)

ncent = 4
fig,ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
#ax.set_ylim((0.4,1.2))
ax.set_xlim((100.,625.))
ax.set_title(title,fontsize=16)

# plot data outside parameter loop
box_width=20
for i in np.arange(ncent):
    ax.errorbar(pTmid+(2*i),RAA_vals[i],RAA_stat[i],marker='s',color=mycol[i], label='ATLAS ' + ctxt[i] + '%')
    ax.bar(pTmid+(2*i),2*RAA_sys_err[i],box_width,RAA_vals[i]-RAA_sys_err[i],color=mycol[i],alpha=0.4)
    ax.bar(pTmid[0]-1.2*box_width,2*(RAA_lum_err[i,0]),box_width,RAA_vals[i,0]-RAA_lum_err[i,0],color='grey',alpha=0.4)
    ax.bar(pTmid[0]-2.4*box_width,2*RAA_taa_err[i,0],box_width,RAA_vals[i,0]-RAA_taa_err[i,0],color='black',alpha=0.4)
    # print(i,'RAA_taa_err=',np.array2string(RAA_taa_err[i,0],precision=3))

f_loop = np.arange(0.75,1.4,0.2)
for j,f in enumerate(f_loop):
    th_loop = (f*theta[0],theta[1])

    R = ezQ_delta_trento(th_loop,J0,RAA_pedge,pT,w,tpa)
    log_like = log_like_delTrento(th_loop,J0,RAA_pedge,pT,w,tpa,RAA_vals,RAA_err,takeLog)
    print('j =',j,'log_like =',log_like)

    for i in np.arange(ncent):
        if (i==0):
            ax.plot(pTmid,R[0,:],color=mycol[i],linestyle=mylin[j],label='{0:s} log_lik = {1:.2e}'.format(err_type[ierr],log_like))
        else:
            ax.plot(pTmid,R[i,:],color=mycol[i],linestyle=mylin[j])
ax.legend(ncol=2)

fig.savefig(figdir+title+'.pdf')

# %%

'''Run emcee and save backend'''

sampletxt = '{:.0e}'.format(nsamples)
nwalkers = 8
ndim = theta.size
pos = theta + 1e-4 * np.random.randn(nwalkers,ndim)
filebase = 'RAA_delTrento_l' + str(lf) + '_tau{:0.1f}_'.format(tau) + err_type[ierr]  + sampletxt 
emcee_filename = h5dir + filebase + '.h5'

'''Set run_emcee=True to run backend and save to file'''
#run_emcee = True
if (run_emcee):
    if (os.path.exists(emcee_filename)):
        os.remove(emcee_filename)
    backend = emcee.backends.HDFBackend(emcee_filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_delTrento,
            args=(J0,RAA_pedge,pT,w,tpa,RAA_vals,RAA_err,takeLog),backend=backend)
    sampler.run_mcmc(pos, nsamples, progress=True);

# %%

'''Read back emcee output and plot diagnostics'''

'''Reset settings if not in batch_mode'''
if (not batch_mode):
    print('Not in batch mode')
    nsamples = 10000
    tau = 0.1
    lf  = 2
    ierr = 2
#    ierr = 4

h5_tpair = h5dir + 't-pairs_' + trento_base + '_tau{:0.1f}.h5'.format(tau)
with h5py.File(h5_tpair,'r') as h:
    nprev = -1
    for k in h.keys():
        npairs = len(h[k])
        if ((nprev>0) and (nprev != npairs)):
            print('t-pair mismatch: Regenerate with trentoCent.py')
        nprev  = npairs
    tpair_array = np.zeros((nc,npairs,2))
    for i in np.arange(nc):
        tpair_key = trento_base + '_' + ctxt[i] + '_tau{:0.1f}'.format(tau)
        if (lf==2):
            tpair_array[i,:,:] = np.array(h[tpair_key])[:,2:4]
        else:
            tpair_array[i,:,:] = np.array(h[tpair_key])[:,0:2]

tpair_sort = np.sort(tpair_array)
ntpmax = 500
tpa = tpair_sort[:,:ntpmax,:]

sampletxt = '{:.0e}'.format(nsamples)
dlabels = ['a','b']
ddim = len(dlabels)
filebase = 'RAA_delTrento_l' + str(lf) + '_tau{:0.1f}_'.format(tau) + err_type[ierr]  + sampletxt 
demcee_filename = h5dir + filebase + '.h5'
dreader = emcee.backends.HDFBackend(demcee_filename)
dtau = dreader.get_autocorr_time(tol=0)
dburnin = int(2 * np.max(dtau))
dthin = int(0.5 * np.min(dtau))
dall_samples = dreader.get_chain()
dflat_samples = dreader.get_chain(flat=True)
dthin_samples = dreader.get_chain(discard=dburnin, flat=True, thin=dthin)
dlog_prob_samples = dreader.get_log_prob()
dlog_prior_samples = dreader.get_blobs()

print('tau: {0}'.format(dtau))
print("burn-in: {0}".format(dburnin))
print("thin: {0}".format(dthin))
print("all chain shape: {0}".format(dall_samples.shape))
print("flat chain shape: {0}".format(dflat_samples.shape))
print("thin chain shape: {0}".format(dthin_samples.shape))
print("flat log prob shape: {0}".format(dlog_prob_samples.shape))

corner.corner(dthin_samples,labels=dlabels)

fig, axes = plt.subplots(ndim, figsize=(12, 6), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(dall_samples[:, :, i], 'k', alpha=0.1)
    ax.set_xlim(0, len(dall_samples))
    ax.set_ylabel(dlabels[i]+'\n'+r'$\tau$ = {0:.0f}'.format(dtau[i]))
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('step number')

'''Plot most probable and some draws'''

ncent = 4
fig,ax = plt.subplots(1, 1, figsize=(12, 12))
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.4,1.01])
#ax.set_xlim([100.,625.])
ax.set_xlim([0.,1000.])
ax.set_title(filebase,fontsize=16)

# Fetch most probable
max_ind = np.argmax(dreader.get_log_prob(flat=True))
delta_max_prob = dreader.get_log_prob(flat=True)[max_ind]
delta_max_pars = dflat_samples[max_ind]
(a,b) = delta_max_pars
mylabel = 'delta '
labels  = ['a','b']
for i,val in enumerate((a,b)):
    mylabel += labels[i]+'={0:.02f} '.format(val)
R = ezQ_delta_trento(delta_max_pars,J0,RAA_pedge,pT,w,tpa)
R_x = ezQ_delta_trento(delta_max_pars,J0_x,RAA_pedge_x,pT,w,tpa)


for i in np.arange(ncent):
    ax.errorbar(pTmid+(2*i),RAA_vals[i],RAA_stat[i],marker='s',color=mycol[i], label='ATLAS ' + ctxt[i] + '%')
    ax.bar(pTmid+(2*i),2*RAA_sys_err[i],box_width,RAA_vals[i]-RAA_sys_err[i],color=mycol[i],alpha=0.4)
    ax.bar(pTmid[0]-1.2*box_width,2*(RAA_lum_err[i,0]),box_width,RAA_vals[i,0]-RAA_lum_err[i,0],color='grey',alpha=0.4)
    ax.bar(pTmid[0]-2.4*box_width,2*RAA_taa_err[i,0],box_width,RAA_vals[i,0]-RAA_taa_err[i,0],color='black',alpha=0.4)
    if (i==0):
        ax.plot(pTmid,R[0,:],color=mycol[i],label=mylabel)
        ax.plot(pTmid_x,R_x[0,:],color=mycol[i],linestyle='dashed')
    else:
        ax.plot(pTmid,R[i,:],color=mycol[i])
ax.legend()

delta_ave_pars = np.mean(dflat_samples,axis=0)
delta_rms_pars = np.std(dflat_samples,axis=0)
print('\n')
print('delta means =',delta_ave_pars)
print('delta   rms =',delta_rms_pars)
print(mylabel,'max-index = ',max_ind)

'''Plot some draws'''
# ndraws = 50
# inds = np.random.randint(len(dthin_samples), size=ndraws)
# RAA_draw = np.zeros((ndraws,RAA_pmid.size))
# for i,ind in enumerate(inds):
#     (a,b) = dthin_samples[ind]
#     h = a*(pT**b)*np.log(pT)
#     (num,edges) = np.histogram(pT-h,RAA_pedge,weights=w0)
#     RAA_draw[i] = num/denom
#     ax.plot(RAA_pmid,RAA_draw[i],color=mycol[1],alpha=0.1)

generate_figures = True
if (generate_figures):
    fig.savefig(figdir+filebase+'.pdf')

'''Save parameters to hdf5 file'''

h5file = h5dir + 'ezQ_RAA_delTrento_pars.h5'
base = 'RAA_pb5_ATL_0080_l' + str(lf) + '_tau{:0.1f}_'.format(tau) + err_type[ierr]  + '_' + sampletxt
with h5py.File(h5file,'a') as g:
    try:
        dset = g.create_dataset(base + '_delRAA_x',data=R_x)
        dset = g.create_dataset(base + '_pTmid_x',data=pTmid_x)
        dset = g.create_dataset(base + '_max_prob',data=delta_max_prob)
        dset = g.create_dataset(base + '_max_pars',data=delta_max_pars)
        dset = g.create_dataset(base + '_ave_pars',data=delta_ave_pars)
        dset = g.create_dataset(base + '_rms_pars',data=delta_rms_pars)
        dset = g.create_dataset(base + '_pTmid'   ,data=pTmid)
        dset = g.create_dataset(base + '_delRAA'  ,data=R)
        print('Creating h5 datasets for delTrento fits')
    except:
        g[base + '_delRAA_x'][...]=R_x
        g[base + '_pTmid_x' ][...]=pTmid_x
        g[base + '_max_prob'][...]=delta_max_prob
        g[base + '_max_pars'][...]=delta_max_pars
        g[base + '_ave_pars'][...]=delta_ave_pars
        g[base + '_rms_pars'][...]=delta_rms_pars
        g[base + '_pTmid'   ][...]=pTmid
        g[base + '_delRAA'  ][...]=R
        print('Rewriting h5 datasets for delTrento fits')


# %%
