# %%

'''
ezQ_plotRAA.py
plot Bayesian draws for delta and gamma fits to ATLAS data
created:  2024-may-31 R.A. Soltz
modified: 2024-jun-01 RAS finish and make publication-worthy plots
          2024-nov-21 RAS increase legend/label font size
  
'''

# Import standard packages and set plots inline
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ezQ_helper import ppw0

# Use this flag to remake figures
#generate_figures = False
generate_figures = True
mycol = ('darkgreen','darkorange','indigo','darkred','darkgoldenrod','brown','olive','darkblue','grey')
mylin = ('dashed','dashdot','dotted',(0,(2,2)),(0,(3,3)),(0,(4,4)),(0,(5,5)),(0,(3,5,1,5,1,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

'''Read hdf5 inputs for ppJet parameters and ATLAS data'''
ATLAS_datafile = h5dir + 'ATLAS_data.h5'
print('Reading from',ATLAS_datafile)
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

'''Use these variables for plotting data'''
X = 0.5*(RAA_pedge[:-1]+RAA_pedge[1:])
Y = RAA_vals[0]
Ystat = RAA_stat[0]
Ysys  = np.sqrt(np.diag(cov_RAA_sys_tot)[:X.size])

RAA_drawfile = h5dir + 'ezQ_RAAcent_1e+04_draws.h5'
print('Reading from',RAA_drawfile)
with h5py.File(RAA_drawfile,'r') as f:
        delta_stat_max       = np.array(f['RAAcent_delta_stat_RAA_maxp'])
        delta_sysd_max       = np.array(f['RAAcent_delta_sysd_RAA_maxp'])
        delta_systot_max     = np.array(f['RAAcent_delta_systot_RAA_maxp'])
        delta_syssum_max     = np.array(f['RAAcent_delta_syssum_RAA_maxp'])
        delta_syslog_max     = np.array(f['RAAcent_delta_syslog_RAA_maxp'])
        delta_stat_meanest   = np.array(f['RAAcent_delta_stat_RAA_meanest'])
        delta_sysd_meanest   = np.array(f['RAAcent_delta_sysd_RAA_meanest'])
        delta_systot_meanest = np.array(f['RAAcent_delta_systot_RAA_meanest'])
        delta_syssum_meanest = np.array(f['RAAcent_delta_syssum_RAA_meanest'])
        delta_syslog_meanest = np.array(f['RAAcent_delta_syslog_RAA_meanest'])
        delta_stat_draw      = np.array(f['RAAcent_delta_stat_RAA_draw'])
        delta_sysd_draw      = np.array(f['RAAcent_delta_sysd_RAA_draw'])
        delta_syslog_draw    = np.array(f['RAAcent_delta_syslog_RAA_draw'])

        gamma_stat_max       = np.array(f['RAAcent_gamma_stat_RAA_maxp'])
        gamma_sysd_max       = np.array(f['RAAcent_gamma_sysd_RAA_maxp'])
        gamma_systot_max     = np.array(f['RAAcent_gamma_systot_RAA_maxp'])
        gamma_syssum_max     = np.array(f['RAAcent_gamma_syssum_RAA_maxp'])
        gamma_syslog_max     = np.array(f['RAAcent_gamma_syslog_RAA_maxp'])
        gamma_stat_meanest   = np.array(f['RAAcent_gamma_stat_RAA_meanest'])
        gamma_sysd_meanest   = np.array(f['RAAcent_gamma_sysd_RAA_meanest'])
        gamma_systot_meanest = np.array(f['RAAcent_gamma_systot_RAA_meanest'])
        gamma_syssum_meanest = np.array(f['RAAcent_gamma_syssum_RAA_meanest'])
        gamma_syslog_meanest = np.array(f['RAAcent_gamma_syslog_RAA_meanest'])
        gamma_stat_draw      = np.array(f['RAAcent_gamma_stat_RAA_draw'])
        gamma_sysd_draw      = np.array(f['RAAcent_gamma_sysd_RAA_draw'])
        gamma_syslog_draw    = np.array(f['RAAcent_gamma_syslog_RAA_draw'])


# %%

'''make plots'''
box_width = 20.

'''prep axis and plot data with errors in all 4 panels'''
fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)
fig.subplots_adjust(left=0.06,right=0.99,top=0.98,bottom=0.08,hspace=0,wspace=0)
mylw = 2.5
ndraw = 30
for i,r in enumerate(axs):
    for j,a in enumerate(r):
        if (i==1):
            a.set(xlabel=r'$p_T$ (GeV)')
        if (j==0):
            a.set(ylabel=r'$R_{AA}$')
        a.set_ylim(0.35,1.)
        a.tick_params(axis='x',which='both',top=True,direction='in')
        a.tick_params(axis='y',which='both',right=True,direction='in')
        if (i+j==0):
            a.errorbar(X,Y,Ystat,marker='s',color='k',linestyle='',label=r'ATLAS PbPb 0-10%')
            a.bar(X,2*Ysys,box_width,Y-Ysys,color='b',alpha=0.4,label='systematic errors')
            for k in np.arange(ndraw):
                a.plot(X,gamma_sysd_draw[k],color='darkorange',alpha=0.2)
            a.plot(X,gamma_sysd_draw[k],color='darkorange',alpha=0.3,label='stat+sys posterior')
            a.plot(X,gamma_sysd_max,color='darkred',linewidth=mylw,label='stat+sys probable')
            a.plot(X,gamma_stat_max,color='indigo',linestyle='dashed',linewidth=mylw,label='stat-only probable')
            a.legend(title='gamma jet-quench',loc='upper left',ncol=2)
        if (i==0 and j==1):
            a.errorbar(X,Y,Ystat,marker='s',color='k',linestyle='')
            a.bar(X,2*Ysys,box_width,Y-Ysys,color='b',alpha=0.4)
            for k in np.arange(ndraw):
                a.plot(X,delta_sysd_draw[k],color='darkorange',alpha=0.2)
            a.plot(X,delta_sysd_draw[k],color='darkorange',alpha=0.3,label='stat+sys posterior')
            a.plot(X,delta_sysd_max,color='darkred',linewidth=mylw,label='stat+sys')
            a.plot(X,delta_stat_max,color='indigo',linestyle='dashed',linewidth=mylw,label='delta stat only')
            a.legend(title='delta jet-quench',loc='upper left')
        if (i==1 and j==0):
            a.errorbar(X,Y,Ystat,marker='s',color='k',linestyle='')
            a.bar(X,2*Ysys,box_width,Y-Ysys,color='b',alpha=0.4)
            for k in np.arange(ndraw):
                a.plot(X,gamma_syslog_draw[k],color='green',alpha=0.1)
            a.plot(X,gamma_syslog_draw[k],color='green',alpha=0.2,label='cov-err log-fit posterior')
            a.plot(X,gamma_syslog_max,color='darkslategrey',linewidth=mylw,label='cov-err log-fit probable')
            a.plot(X,gamma_syssum_max,color='midnightblue',linestyle='dashed',linewidth=mylw,label='sys-cov probable')
            a.legend(title='gamma jet-quench',loc='upper left')
        if (i+j==2):
            a.errorbar(X,Y,Ystat,marker='s',color='k',linestyle='')
            a.bar(X,2*Ysys,box_width,Y-Ysys,color='b',alpha=0.4)
            for k in np.arange(ndraw):
                a.plot(X,delta_syslog_draw[k],color='green',alpha=0.1)
            a.plot(X,delta_syslog_draw[k],color='green',alpha=0.2,label='cov-err log-fit posterior')
            a.plot(X,delta_syslog_max,color='darkslategrey',linewidth=mylw,label='cov-err log-fit probable')
            a.plot(X,delta_syssum_max,color='midnightblue',linestyle='dashed',linewidth=mylw,label='sys-cov probable')
            a.legend(title='delta jet-quench',loc='upper left')

generate_figures = True
if (generate_figures):
    fig.savefig(figdir+'fig_RAAcent_draws.pdf')


# %%