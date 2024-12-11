# %%

'''
ezQ_plotQuench.py
plot delta-pT heatmap with mean pT-loss for delta/gamma and plot gamma-distributions fixed pT values
created:  2024-may-31 R.A. Soltz
modified: 2024-jun-04 RAS finish heatmap and dist figures
          2024-aug-09 RAS update to plot heatmaps for stat and syslog emcee fits
          2024-nov-02 RAS switch stat to diag and increase legend fontsize
          2024-nov-21 RAS increase legend/label fontsize further
  
'''

# Import standard packages and set plots inline
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Use this flag to remake figures
generate_figures = True
mycol = ('darkblue','darkorange','darkred','darkgreen','darkgoldenrod','brown','olive','grey')
mylin = ('dotted','dashed','dashdot',(0,(2,2)),(0,(3,3)),(0,(4,4)),(0,(5,5)),(0,(3,5,1,5,1,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

'''Read most probable values for delta/gamma fits'''
RAA_drawfile = h5dir + 'ezQ_RAAcent_1e+04_draws.h5'
print('Reading from',RAA_drawfile)
with h5py.File(RAA_drawfile,'r') as f:
#        delta_stat_par   = np.array(f['RAAcent_delta_stat_max_pars'])
#        gamma_stat_par   = np.array(f['RAAcent_gamma_stat_max_pars'])
        delta_sysd_par   = np.array(f['RAAcent_delta_sysd_max_pars'])
        gamma_sysd_par   = np.array(f['RAAcent_gamma_sysd_max_pars'])
        delta_syslog_par = np.array(f['RAAcent_delta_syslog_max_pars'])
        gamma_syslog_par = np.array(f['RAAcent_gamma_syslog_max_pars'])


# %%

'''Make the heatmap first'''
pTlo  = 150  # minimum for pTquench and pT
pThi  = 650 # top value for pT
pTbin = 1.   # bin width
pT    = np.arange(pTlo,pThi,pTbin)

'''prep axis and plot data with errors in both panels'''
mycol = ['darkred','darkgreen']
mylin = [(0,(5,7)),(6,(2,1,2,7))]
mylw = 2.5
fig, axs = plt.subplots(2, 1, figsize=(10,4),sharex=True,tight_layout=True)
fig.subplots_adjust(hspace=0)

for ax in axs:
    ax.set_xlim(160.,650.)
    ax.tick_params(axis='x',which='both',top=True,direction='in',width=3)
    ax.tick_params(axis='y',which='both',right=True,direction='in',width=2)


'''heatmap pT quench figure for stat-fit'''
del_pT_max  = 90
#(a,b,k)     = gamma_stat_par
(a,b,k)     = gamma_sysd_par
h           = a*(pT**b)*np.log(pT)
x   = np.tile(np.arange(del_pT_max),(pT.size,1)).T
fx  = (h**(-k)) * (x**(k-1)) * np.exp(-x/h) / gamma(k)
mean_gamma  = h*k
#(c,d)       = delta_stat_par
(c,d)       = delta_sysd_par
mean_delta  = c*(pT**d)*np.log(pT)

ax = axs[0]
ax.set(ylabel=r'$\Delta p_T$')
extent = (pTlo,pThi,del_pT_max,0)
im1 = ax.imshow(fx,origin='upper',extent=extent,cmap='rainbow')
ax.plot(pT,mean_delta,color=mycol[0],linestyle=mylin[0],linewidth=mylw,label=r'delta quench <$\Delta p_T>$ (diag-fit)')
ax.plot(pT,mean_gamma,color=mycol[1],linestyle=mylin[1],linewidth=mylw,label=r'gamma quench <$\Delta p_T>$ (diag-fit)')
ax.legend(loc='lower left',framealpha=1.,borderaxespad=0.8,handlelength=3.5)

'''heatmap pT quench figure for syslog-fit'''
(a,b,k)     = gamma_syslog_par
h           = a*(pT**b)*np.log(pT)
x   = np.tile(np.arange(del_pT_max),(pT.size,1)).T
fx  = (h**(-k)) * (x**(k-1)) * np.exp(-x/h) / gamma(k)
mean_gamma  = h*k
(c,d)       = delta_syslog_par
mean_delta  = c*(pT**d)*np.log(pT)

ax = axs[1]
ax.set(xlabel=r'$p_T$ (GeV)',ylabel=r'$\Delta p_T$')
extent = (pTlo,pThi,del_pT_max,0)
im1 = ax.imshow(fx,origin='upper',extent=extent,cmap='rainbow')
ax.plot(pT,mean_delta,color=mycol[0],linestyle=mylin[0],linewidth=mylw,label=r'delta quench <$\Delta p_T>$ (covlog-fit)')
ax.plot(pT,mean_gamma,color=mycol[1],linestyle=mylin[1],linewidth=mylw,label=r'gamma quench <$\Delta p_T>$ (covlog-fit)')
ax.legend(loc='lower left',framealpha=1.,borderaxespad=0.8,handlelength=3.5)

if (generate_figures):
    fig.savefig(figdir+'fig_pTquench_heatmap.pdf')
# %%

'''distribution pT quench figure with gamma distribution'''

mycol = ['darkred','darkgreen','darkorange','darkblue']
mylin = [(0,(5,7)),(6,(2,1,2,7))]
mylw = 2.5
fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.set_xlim(140.,630.)
ax.set_ylim(0.,0.065)
ax.tick_params(axis='x',which='both',top=True,direction='in')
ax.tick_params(axis='y',which='both',right=True,direction='in')

'''Re-load syslog parameters'''
(a,b,k)     = gamma_syslog_par
(c,d)       = delta_syslog_par

for pTjet in np.arange(200,620,50):
    pT = np.arange(pTjet)
    h  = a*(pT**b)*np.log(pT)
    x  = pTjet - pT
    fx = (h**(-k)) * (x**(k-1)) * np.exp(-x/h) / gamma(k)
    ax.plot(pT,fx,color=mycol[1])
    x_gamma = pTjet - k * a * (pTjet**b)*np.log(pTjet) 
    x_delta = pTjet - c*(pTjet**d)*np.log(pTjet)
    ax.plot((x_delta,x_delta),(0.0,fx[int(x_delta)]),color=mycol[0],linestyle=mylin[0],linewidth=mylw)
    ax.plot((x_gamma,x_gamma),(0.0,fx[int(x_gamma)]),color=mycol[1],linestyle=mylin[1],linewidth=mylw)

y_top = 0.038
ax.plot((600.,600.),(0.,y_top),color='black',linestyle='dashdot',linewidth=2)
ax.annotate(' ',(575.,0.03),xytext=(600.,0.03),
            arrowprops=dict(facecolor=mycol[1],shrink=0.02),verticalalignment='top')
ax.annotate(r'$p_T$ init',(600.,y_top+0.001),horizontalalignment='center',fontsize=16)
ax.plot((585.,615),(y_top,y_top),color='black')

ax.plot((0.),(10.), color=mycol[1],label='gamma distribution (syslog-fit)')
ax.plot((0.),(10.),color=mycol[0],linestyle=mylin[0],linewidth=mylw,label=r'delta $p_T = <p_T>$ (syslog-fit)')
ax.plot((0.),(10.),color=mycol[1],linestyle=mylin[1],linewidth=mylw,label=r'gamma $<p_T>$ (syslog-fit)')

'''Re-load stat parameters'''
(a,b,k)     = gamma_syslog_par
(c,d)       = delta_syslog_par

for pTjet in np.arange(200,620,50):
    pT = np.arange(pTjet)
    h  = a*(pT**b)*np.log(pT)
    x  = pTjet - pT
    fx = (h**(-k)) * (x**(k-1)) * np.exp(-x/h) / gamma(k)
    ax.plot(pT,fx,color=mycol[3],linestyle='dashed')
    x_gamma = pTjet - k * a * (pTjet**b)*np.log(pTjet) 
    x_delta = pTjet - c*(pTjet**d)*np.log(pTjet)
    ax.plot((x_delta,x_delta),(0.0,fx[int(x_delta)]),color=mycol[2],linestyle=mylin[0],linewidth=mylw)
    ax.plot((x_gamma,x_gamma),(0.0,fx[int(x_gamma)]),color=mycol[3],linestyle=mylin[1],linewidth=mylw)

ax.plot((0.),(10.), color=mycol[3],linestyle='dashed',label='gamma distribution (stat-fit)')
ax.plot((0.),(10.),color=mycol[2],linestyle=mylin[0],linewidth=mylw,label=r'delta $p_T = <p_T>$ (syslog-fit)')
ax.plot((0.),(10.),color=mycol[3],linestyle=mylin[1],linewidth=mylw,label=r'gamma $<p_T>$ (syslog-fit)')

ax.legend(loc='upper right',borderaxespad=1,handlelength=4.2)

if (generate_figures):
    fig.savefig(figdir+'fig_pTquench_dist.pdf')

# %%
