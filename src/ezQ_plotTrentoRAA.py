'''
ezQ_plotTrentoRAA.py
created:  2024-oct-16 R.A. Soltz (separate figure plot from ezQ_delTrentoRAA.py)
modified: 2024-oct-25 RAS settle on tau=0.1,0.5,0.9 plots for paper
          2024-nov-21 RAS increase legend/label fontsize further

'''

# %%

# Import standard packages and set plots inline
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

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
ALICE_RAA_sys = (ALICE_RAA_corr**2 + ALICE_RAA_shap**2)**0.5

'''Add some short-hand'''
nc    = RAA_cedge.size - 1
npT   = RAA_pedge.size - 1
pTmid = 0.5*(RAA_pedge[1:]+RAA_pedge[:-1])
RAA_sys_err = (np.diag(cov_RAA_sys_total)**0.5).reshape(nc,npT)
RAA_lum_err = (np.diag(cov_pp_lum_E)**0.5).reshape(nc,npT)
RAA_taa_err = (np.diag(cov_TAA_E)**0.5).reshape(nc,npT)

ctxt = []
for i in np.arange(nc):
    ctxt.append(str(RAA_cedge[i])+'-'+str(RAA_cedge[i+1]))
cmstxt = ['0-10%','10-30%','30-50%','50-90%']

# %%

'''Make comparison plots for l-tau'''

nlwt = 2
ntau = 3
npar = 2
errsample = 'dslT_1e+04_'
axtitle   = 'semi-diagonal errors'
#errsample = 'clog_1e+04_'
#axtitle   = 'fully covariant errors'
line_list = [[(0,(5,1)),(0,(5,4)),(0,(5,8))],
             [(0,(3,1,1,1)),(0,(3,3,1,3)),(0,(3,5,1,5)),]]
comp_list = [['RAA_pb5_ATL_0080_l0_tau0.1_','RAA_pb5_ATL_0080_l0_tau0.5_','RAA_pb5_ATL_0080_l0_tau0.9_'],
             ['RAA_pb5_ATL_0080_l2_tau0.1_','RAA_pb5_ATL_0080_l2_tau0.5_','RAA_pb5_ATL_0080_l2_tau0.9_']]
label_list = [[r'linear,$\tau_f$=0.1 fm/c',r'linear,$\tau_f$=0.5 fm/c',r'linear,$\tau_f$=0.9 fm/c'],
              [r'quadratic,$\tau_f$=0.1 fm/c',r'quadratic,$\tau_f$=0.5 fm/c',r'quadratic,$\tau_f$=0.9 fm/c']]
delRAA_vals = np.zeros((nlwt,ntau,nc,npT))
delRAA_pars = np.zeros((nlwt,ntau,npar))
delRAA_prob = np.zeros((nlwt,ntau))
delRAA_vals_x = np.zeros((nlwt,ntau,nc,npT))

h5file = h5dir + 'ezQ_RAA_delTrento_pars.h5'
with h5py.File(h5file,'r') as g:
    for i in np.arange(nlwt):
        for j in np.arange(ntau):
            base = comp_list[i][j] + errsample
            pTmid  = np.array(g[base+'pTmid'])
            delRAA_vals[i,j] = np.array(g[base+'delRAA'])
            delRAA_pars[i,j] = np.array(g[base+'max_pars'])
            delRAA_prob[i,j] = np.array(g[base+'max_prob'])
delRAA_chi2dof = -2*delRAA_prob/((npT*nc)-npar)

for i in np.arange(nlwt):
    for j in np.arange(ntau):
        print(comp_list[i][j],np.array2string(delRAA_pars[i,j],precision=2),
                'chi2dof =',np.array2string(delRAA_chi2dof[i,j],precision=2))

box_width = 20
fig,axs = plt.subplots(1, 2, figsize=(16, 8),tight_layout=True)
for ax in axs:
    ax.set_ylabel(r'$R_{AA}$')
    ax.set_xlabel(r'$p_T (GeV)$')
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.set_ylim([0.21,1.02])
    ax.set_xlim([90.,610.])

for i in np.arange(nlwt):
    ax = axs[i]
    for k in np.arange(nc):
        ax.errorbar(pTmid+(2*k),RAA_vals[k],RAA_stat[k],marker='s',color=mycol[k], label='ATLAS ' + ctxt[k] + '%')
        ax.bar(pTmid+(2*k),2*RAA_sys_err[k],box_width,RAA_vals[k]-RAA_sys_err[k],color=mycol[k],alpha=0.4)
        ax.bar(pTmid[0]-1.2*box_width,2*(RAA_lum_err[k,0]),box_width,RAA_vals[k,0]-RAA_lum_err[k,0],color='grey',alpha=0.4)
        ax.bar(pTmid[0]-2.4*box_width,2*RAA_taa_err[k,0],box_width,RAA_vals[k,0]-RAA_taa_err[k,0],color='black',alpha=0.4)
        for j in np.arange(ntau):
            if (k==0):
                ax.plot(pTmid,delRAA_vals[i,j,k,:],color=mycol[k],linestyle=line_list[i][j],label=label_list[i][j])
                ax.plot(pTmid,delRAA_vals_x[i,j,k,:],color=mycol[k],linestyle=line_list[i][j])
            else:
                ax.plot(pTmid,delRAA_vals[i,j,k,:],color=mycol[k],linestyle=line_list[i][j])

    handles, labels = ax.get_legend_handles_labels()
#    for i,handle in enumerate(handles): print('i=',i,'handle=',handles[i],'label=',labels[i])
    order = [6,5,4,3,2,1,0]
    h = [handles[i] for i in order]
    l = [labels[i] for i in order]
    ax.legend(h,l,title=axtitle,ncol=2,loc='lower center')

generate_figures = True
if (generate_figures):
    figname = figdir + 'fig_TrentoRAA_Ltau_' + errsample[:-1] + '.pdf'
    fig.savefig(figname)

# %%

'''add CMS,ALICE data to ATLAS error comparison for l-squared tau=0.9'''

h5file = h5dir + 'ezQ_RAA_delTrento_pars.h5'
with h5py.File(h5file,'r') as g:
    pTmid = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_dslT_1e+04_pTmid'])
    TrentoRAA_dslT = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_dslT_1e+04_delRAA'])
    TrentoRAA_clog = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_clog_1e+04_delRAA'])
    pTmid_x = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_dslT_1e+04_pTmid_x'])
    TrentoRAA_dslT_x = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_dslT_1e+04_delRAA_x'])
    TrentoRAA_clog_x = np.array(g['RAA_pb5_ATL_0080_l2_tau0.1_clog_1e+04_delRAA_x'])

dash1d = (0,(3,1))
dash2d = (0,(3,1,1,1))

box_width = 20
small_box = 6
fig,ax = plt.subplots(1, 1, figsize=(8, 8),tight_layout=True)
ax.set_ylabel(r'$R_{AA}$')
ax.set_xlabel(r'$p_T (GeV)$')
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.set_ylim([0.275,1.02])
ax.set_xlim([0.,800.])

for k in np.arange(nc):
    '''ATLAS data'''
    ax.errorbar(pTmid+(2*k),RAA_vals[k],RAA_stat[k],marker='s',color=mycol[k], label='ATLAS ' + ctxt[k] + '%')
    ax.bar(pTmid+(2*k),2*RAA_sys_err[k],box_width,RAA_vals[k]-RAA_sys_err[k],color=mycol[k],alpha=0.4)
#    ax.bar(pTmid[0]-1.2*box_width,2*(RAA_lum_err[k,0]),box_width,RAA_vals[k,0]-RAA_lum_err[k,0],color='grey',alpha=0.4)
#    ax.bar(pTmid[0]-2.4*box_width,2*RAA_taa_err[k,0],box_width,RAA_vals[k,0]-RAA_taa_err[k,0],color='black',alpha=0.4)
    '''CMS data omit 50-90% for clarity'''
    if (k<=2):
        ax.errorbar(CMS_pTmid+(2*k),CMS_RAA_vals[k],CMS_RAA_stat[k],marker='o',ls='none',
                    color=mycol[k],fillstyle='none',label='CMS ' + cmstxt[k])
        ax.bar(CMS_pTmid+(2*k),2*CMS_RAA_sys[k],box_width,CMS_RAA_vals[k]-CMS_RAA_sys[k],color=mycol[k],
               edgecolor='k',alpha=0.2)
#    ax.bar(pTmid[0]-1.2*box_width,2*(RAA_lum_err[k,0]),box_width,RAA_vals[k,0]-RAA_lum_err[k,0],color='grey',alpha=0.4)
#    ax.bar(pTmid[0]-2.4*box_width,2*RAA_taa_err[k,0],box_width,RAA_vals[k,0]-RAA_taa_err[k,0],color='black',alpha=0.4)
    '''ALICE data'''
    if (k==0):
        ax.errorbar(ALICE_pTmid,ALICE_RAA_vals,ALICE_RAA_stat,marker='v',ls='none',color=mycol[k],
                    fillstyle='none',label='ALICE ' + ctxt[k])
        ax.bar(ALICE_pTmid,2*ALICE_RAA_sys,small_box,ALICE_RAA_vals-ALICE_RAA_sys,color=mycol[k],alpha=0.2,edgecolor='k')
        ax.plot(pTmid_x,TrentoRAA_dslT_x[k],color=mycol[k],linestyle='dashed',label='semi-diagonal')
        ax.plot(pTmid_x,TrentoRAA_clog_x[k],color=mycol[k],linestyle='dashdot',label='fully covariant')
    else:
        ax.plot(pTmid,TrentoRAA_dslT[k],color=mycol[k],linestyle='dashed')
        ax.plot(pTmid,TrentoRAA_clog[k],color=mycol[k],linestyle='dashdot')


handles, labels = ax.get_legend_handles_labels()
for i,handle in enumerate(handles): print('i=',i,'label=',labels[i])
order = [0,8,6,3,4,1,9,7,5,2]
h = [handles[i] for i in order]
l = [labels[i] for i in order]
ax.legend(h,l,ncol=2,loc='lower right')

generate_figures = True
if (generate_figures):
    figname = figdir + 'fig_TrentoRAA_allexp' + '.pdf'
    fig.savefig(figname)

# %%
