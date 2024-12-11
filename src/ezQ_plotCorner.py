# %%

'''
ezQ_plotCorner.py
Plot corner figures for RAAcent fits with delta/gamma and stat/syslog fits
created:  2024-aug-09 R.A. Soltz
modified: 2024-nov-04 RAS add extra fit type for gamma with alpha*k
          2024-nov-21 RAS increase font size, add diagonal/covariant legend/labels
  
'''

# Import standard packages and set plots inline
import os
import sys
import emcee
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Use this flag to remake figures
saveFig = True
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
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

ifit = 2
ierrs = [1,4]
if (ifit==0):
    labels    = [r'$\alpha$',r'$\beta$',r'$k$']
elif (ifit==1):
    labels    = [r'$\alpha$',r'$\beta$']
elif (ifit==2):
    labels    = [r'$\alpha$ k',r'$\beta$']

limits    = np.array([[-0.2,11.],[-0.6,0.6],[1,9.9]])
ticks     = [[0,5,10],[-0.5,0.0,0.5],[2,4,6,8]]
sampletxt = '1e+04'
fit_type  = np.array(['gamma','delta','gamma'])
fit_name  = np.array(['gamma','delta','alphak'])
err_type  = np.array(['stat','sysd','systot','syssum','syslog','sysTlumlog'])

'''Read most probable values for delta/gamma fits'''
RAA_drawfile = h5dir + 'ezQ_RAAcent_1e+04_draws.h5'
par_list = []
with h5py.File(RAA_drawfile,'r') as f:
        for i in ierrs:
            dname = 'RAAcent_'+fit_type[ifit]+'_'+err_type[i]+'_max_pars'
            pars = np.array(f[dname])
            par_list.append(pars)
            print('Fetching',dname,'=',pars)

samples = []
for i in ierrs:
    fiterr    = 'RAAcent_' + fit_type[ifit] + '_' + err_type[i]
    print('Fetching samples for',fiterr)
    emcee_filename = h5dir + fiterr + '_' + sampletxt + '.h5'
    reader = emcee.backends.HDFBackend(emcee_filename)
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    samples.append(reader.get_chain(discard=burnin,flat=True))

'''ifit==2, multiply alpha-entries by k-enties'''
if (ifit==2):
    for i in np.arange(len(ierrs)):
        par_list[i][0] = par_list[i][0]*par_list[i][2]
        samples[i][:,0] = samples[i][:,0]*samples[i][:,2]

'''Fill histograms for plotting'''
nd = len(labels)
nb = 40
h1d = np.zeros((2,nd,nb))
e1d = np.zeros((2,nd,nb+1))
h2d = np.zeros((nd,nd,nb,nb))
x2d = np.zeros((nd,nd,nb+1))
y2d = np.zeros((nd,nd,nb+1))
for i in np.arange(nd):
    for j in np.arange(nd):
        if (i==j):
            h1d[0,i],e1d[0,j] = np.histogram(samples[0][:,i],bins=nb,range=(limits[i]))
            h1d[1,i],e1d[1,j] = np.histogram(samples[1][:,i],bins=nb,range=(limits[i]))
        elif (i>j):
            h2d[i,j],x2d[i,j],y2d[i,j] = np.histogram2d(samples[0][:,i],samples[0][:,j],bins=nb,range=(limits[i],limits[j]))
        else:
            h2d[i,j],x2d[i,j],y2d[i,j] = np.histogram2d(samples[1][:,i],samples[1][:,j],bins=nb,range=(limits[i],limits[j]))

fig, axs = plt.subplots(nd,nd, figsize=(nd*3, nd*3))
fig.align_ylabels()
fig.subplots_adjust(hspace=0,wspace=0)
for i in np.arange(nd):
    for j in np.arange(nd):
        axs[i,j].tick_params(axis='x',which='both',top=True,direction='in')
        axs[i,j].tick_params(axis='y',which='both',right=True,direction='in')
        axs[0,0].tick_params(axis='y',which='both',right=False,direction='in')
        axs[i,j].set_xticks(ticks[j])
        axs[i,j].set_yticks(ticks[i])
        if (i==nd-1):
            axs[i,j].set(xlabel=labels[j])
        if (j==0):
            axs[i,j].set(ylabel=labels[i])
        else:
            axs[i,j].get_yaxis().set_ticklabels([])
        if (i==j):
            axs[i,i].get_yaxis().set_visible(False)
            axs[i,j].set_xlim(limits[i])
            axs[i,j].plot(0.5*(e1d[0,i,1:]+e1d[0,i,:-1]),h1d[0,i],'-',color='blue',label='diagonal')
            axs[i,j].plot(0.5*(e1d[1,i,1:]+e1d[1,i,:-1]),h1d[1,i],'--',color='red',label='covariant')
            if ((i==0)or(i==nd-1)):
                axs[i,j].legend()
        elif (i>j):
            axs[i,j].set_xlim(limits[j])
            axs[i,j].set_ylim(limits[i])
            X,Y = np.meshgrid(0.5*(x2d[i,j,1:]+x2d[i,j,:-1]),0.5*(y2d[i,j,1:]+y2d[i,j,:-1]),indexing='ij')
            axs[i,j].contour(Y,X,h2d[i,j])
            axs[i,j].plot(par_list[0][j],par_list[0][i],marker='*',color='red',markersize=10)
        else:
            axs[i,j].set_xlim(limits[j])
            axs[i,j].set_ylim(limits[i])
            X,Y = np.meshgrid(0.5*(x2d[i,j,1:]+x2d[i,j,:-1]),0.5*(y2d[i,j,1:]+y2d[i,j,:-1]),indexing='ij')
            axs[i,j].contour(Y,X,h2d[i,j])
            axs[i,j].plot(par_list[1][j],par_list[1][i],marker='*',color='red',markersize=10)
            print('pars',i,j,':',par_list[1][i],par_list[1][j])
            axs[i,j].set(ylabel=labels[i])
            axs[i,j].yaxis.set_label_position('right')
        axs[nd-1,0].text(0.9,0.9,'diagonal',transform=axs[nd-1,0].transAxes,fontsize=14,ha='right',va='top')
        axs[0,nd-1].text(0.9,0.9,'covariant',transform=axs[0,nd-1].transAxes,fontsize=14,ha='right',va='top')

saveFig = True
if (saveFig):
    figname = 'fig_corner_'+fit_name[ifit]+'_'+err_type[ierrs[0]]+'_'+err_type[ierrs[1]]
    fig.savefig(figdir+figname+'.pdf',bbox_inches='tight')

# %%
