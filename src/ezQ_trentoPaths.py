# %%

'''
ezQ_trentoPaths.py
Read trento h5 output and perform the following:
 - determine centrality bins using Npart + epsilon*multiplicity (trentoCent.pdf)
 - sample jet paths through medium and save to h5
   - save_tpath saves density for all jet pairs, one file for each centrality
   - save_tpair saves only the integral, all centralities to one file
 - plot trento_heatmaps for jet path pairs

created: 2023-mar-30 R.A. Soltz
modified: 2023-apr-19 RAS add isample_cbin counters to unify number of trento pairs per cbin
                          (nominally set to 10k per cbin for 10k trento output)
          2024-jun-06 RAS add tpexp dataset to tpath h5 to store expansion weighted path-integrals
                          also add tau formation time, implemented as hard-cut on rmid (no interp)
          2024-jul-17 RAS add l-squared factor for posint2 negint2 and save as tpair_list[2:4]
          2024-oct-18 RAS set tau=0.5,0.9 to generate new tpairs
'''

# Import standard packages and set plots inline
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from ezQ_helper import gridpts

# Default fonts for pyplot
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Use this flag to remake figures
mycol = ('darkblue','darkorange','darkred','darkgreen','darkgoldenrod','brown','olive')
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + '/fig/'
h5dir   = workdir + '/h5/'

# Initialize random numbers, set grid to match trento run
rng    = np.random.default_rng(12345)
gridstep = 0.2
gridmax  = 10.
xedge = np.arange(-gridmax,gridmax+gridstep/2.,gridstep)
yedge = np.arange(-gridmax,gridmax+gridstep/2.,gridstep)
extent = [-gridmax,gridmax,-gridmax,gridmax]

'''
   March centrality bins for 
   ATLAS https://arxiv.org/abs/2205.00682 (accepted by PRC)
   4 bins centrality = 0-10%, 10-30%, 30-50%, 50-80%
   Subtract edges from 1. so that bin-0 is most central 0-10%
   Sort by npart then mult using npart+mult/(maxmult+1)
   Find npmult_edges, then fill list of cent_events[centbin][eventlist]
'''
cent_range = np.array([.0,.1,.3,.5,.8])
cent_edges = 1. - cent_range
cent_nbins = cent_edges.size - 1
cent_label = []
for i in np.arange(cent_nbins):
    cent_label.append('{0:.0f}-{1:.0f}%'.format(100*cent_range[i],100*cent_range[i+1]))

# Use to normalize samples_per_event by ncoll/max_or_max for each centrality bin
cent_min_ncoll = np.zeros(cent_nbins)
cent_max_ncoll = np.zeros(cent_nbins)
cent_med_ncoll = np.zeros(cent_nbins)

# %%
'''
Read in hdf5 trento file and do some simple checks
- need to loop to extract metadata mult
- here 10k refers to total number of events in all centrality bins
'''

#trento_base = 'PbPb_1k'
trento_base = 'PbPb_10k'
h5file = h5dir + 'trento_' + trento_base + '.h5'
generate_figures = False
f      = h5py.File(h5file,'r')
events = list(f.keys())

'''Fill arrays for multiplicity, npart, npmult (npart+fractional_mult), and ncoll'''
nev    = len(events)
print(nev, ' Events')
mult   = np.zeros(nev)
npart  = np.zeros(nev)
npmult = np.zeros(nev)
ncoll  = np.zeros(nev)
for i in np.arange(nev):
    mult[i]  = f[events[i]].attrs['mult']
    npart[i] = f[events[i]].attrs['npart']
    grid     = np.array(f[events[i]])
    TaTb     = grid*grid
    ncoll[i] = np.sum(TaTb)

npmult = npart + (mult/(mult[np.argmax(mult)]+1.))
npmult_edges = np.quantile(npmult,cent_edges)
cent_events = []
for i in np.arange(cent_nbins):
    cent_events.append(np.intersect1d(
    np.where(npmult<=npmult_edges[i]),np.where(npmult>npmult_edges[i+1])))
    '''Use new cent_events 2d list to calculate max ncoll'''
    cent_max_ncoll[i] = np.max(ncoll[cent_events[i]])
    cent_min_ncoll[i] = np.min(ncoll[cent_events[i]])
    cent_med_ncoll[i] = np.median(ncoll[cent_events[i]])
print('cent_edges:',cent_edges)
print('npmult_edges:',npmult_edges)
print('cent_min_ncoll:',cent_min_ncoll)
print('cent_max_ncoll:',cent_max_ncoll)
print('cent_med_ncoll:',cent_med_ncoll)
# print(npmult[cent_events[0]])
# print(cent_events[0])

'''Centrality histogram'''
#hist_max   = 255.
hist_max   = 420.
hist_width = 10
hist_edges = np.arange(0.,hist_max,hist_width)
hist_nbins = hist_edges.size - 1
''' counts2d has one extra row of zeros for bottom of bar plot'''
counts2d = np.zeros((cent_nbins+1,hist_nbins))
for i in np.arange(cent_nbins):
    counts2d[i][:],edges = np.histogram(npart[cent_events[i]],hist_edges)

f1 = plt.figure(1,figsize=(12,6))
f1p1 = f1.add_subplot(121)
f1p1.set(xlabel='Npart',ylabel='Counts',title='Centrality Bins')
for i in np.arange(cent_nbins):
    f1p1.bar(hist_edges[:-1]+hist_width/2.,counts2d[i][:],
             hist_width,counts2d[i+1][:],label=cent_label[i])
f1p1.legend(loc=1,fontsize=14)

f1p2 = f1.add_subplot(122)
f1p2.tick_params(which='both',direction='in',top=True,right=True)
f1p2.set(xlabel='Npart',ylabel='Multiplicity',title='Mult vs. Npart')
for i in np.arange(cent_nbins):
    f1p2.plot(npart[cent_events[i]],mult[cent_events[i]],'*')

generate_figures = False
if (generate_figures):
    f1.savefig(figdir+'trentoCent.pdf')

# %%

'''Set tau and pick one cbin for heatmap figures'''
tau  = 0.9
cbin = 0
myfigs = []

'''
Random numbers needed:
- sample cell from TaTb binary collision distribution
- randomize x,y location within cell
- randomize angle [0,pi/2]
'''

for evt in cent_events[cbin]:
    myran  = rng.random(4)
    grid = np.array(f[events[evt]])
    TaTb = grid*grid
    '''select cell from flattened TaTb deviate'''
    cell = np.where(np.cumsum(TaTb)/np.sum(TaTb)>myran[0])[0][0]
    xcell = cell%100
    ycell = cell//100

    '''randomize position within cell, set angle, and call gridpts()'''
    x = (xcell+myran[1])*gridstep - gridmax
    y = (ycell+myran[2])*gridstep - gridmax
    angle = myran[3]*np.pi

    (xgrid,ygrid) = gridpts(x,y,angle,xedge,yedge)

    xdif = np.diff(xgrid)
    ydif = np.diff(ygrid)
    xmid = (xgrid[1:]+xgrid[:-1])/2.
    ymid = (ygrid[1:]+ygrid[:-1])/2.
    xrel = xmid - x
    yrel = ymid - y
    xbin = np.floor((xmid-xedge[0])/gridstep).astype(int)
    ybin = np.floor((ymid-yedge[0])/gridstep).astype(int)
    mbin = xbin + ybin*(xedge.size-1)
    dens = grid.flatten()[mbin]

    '''Take sign from yrel if all xrel == 0'''
    if (np.all(xrel == 0)):
        sign = np.sign(yrel)
    else:
        sign = np.sign(xrel)
    rmid = ((xrel**2 + yrel**2)**0.5)*sign
    rdif = (xdif**2 + ydif**2)**0.5
    '''calculate positive and negative path lengths for arrows'''
    arrowpos = np.sum(rdif[np.where(sign*dens>0.05*np.amax(dens))])
    arrowneg = np.sum(rdif[np.where(sign*dens<-0.05*np.amax(dens))])

    '''sort into pb,nb (for pos/neg bins) and apply formation time and longitudinal expansion'''
    '''no need to worry about rmid==0, as edges are on zero'''
    pb = rmid>tau
    nb = rmid<-tau
    pmid = rmid[pb]
    nmid = rmid[nb]
    pdif = rdif[pb]
    ndif = rdif[nb]
    pdns = dens[pb]*pmid[0]/pmid
    ndns = dens[nb]*nmid[-1]/nmid
    pdl2 = dens[pb]*pmid[0]*(pmid-tau)/pmid
    ndl2 = dens[nb]*nmid[-1]*(-nmid-tau)/nmid

    ''' plot heatmaps in top two panels'''
    figname = figdir+'trento_heatmap_'+cent_label[cbin][:-1]+'_tau{0:0.1f}_evt{1:d}'.format(tau,evt)
    label = 'Event ' + str(evt) + ' (' + cent_label[cbin] + ')'
    myfigs.append(plt.figure(figsize=(18,6)))
    p1 = myfigs[-1].add_subplot(131)
    p2 = myfigs[-1].add_subplot(132)
    p3 = myfigs[-1].add_subplot(133)
    p1.set(xlabel='x (fm)',ylabel='y (fm)')
    p2.set(xlabel='x (fm)',ylabel='y (fm)')
    p1.set_title(r'$T_A T_B$')
    p2.set_title(r'$\sqrt{T_A T_B}$')
    im1 = p1.imshow(TaTb,origin='lower',extent=extent,cmap='rainbow')
    im2 = p2.imshow(grid,origin='lower',extent=extent,cmap='rainbow')

    p1.plot(x,y,'*')
    p1.text(-0.8*gridmax,0.8*gridmax,label,fontsize='xx-large',color='w')
    p2.plot(x,y,'*')
    p2.text(-0.8*gridmax,0.8*gridmax,label,fontsize='xx-large',color='w')
    p2.arrow(x,y,arrowpos*np.cos(angle)/2.,arrowpos*np.sin(angle)/2.,ls='--',head_width=0.5,head_length=0.5)
    p2.arrow(x,y,-arrowneg*np.cos(angle)/2.,-arrowneg*np.sin(angle)/2.,ls='--',head_width=0.5,head_length=0.5)

    p3.set(xlabel='r (fm)',ylabel=r'$\sqrt{T_A T_B}$')
    p3.set_title(r'Path Integral Steps')
    p3.bar(rmid,dens,rdif,color='white',edgecolor='black',linewidth=1,label=r'TRENTO')
    p3.bar(pmid,pdns,pdif,color='red',alpha=0.8,edgecolor='black',linewidth=1,label=r'Expanded ($\tau={:0.1f}$ fm) '.format(tau))
    p3.bar(nmid,ndns,ndif,color='red',alpha=0.8,edgecolor='black',linewidth=1)
    p3.bar(pmid,pdl2,pdif,color='blue',alpha=0.3,edgecolor='black',hatch='xx',linewidth=1,label='Path-weighted')
    p3.bar(nmid,ndl2,ndif,color='blue',alpha=0.3,edgecolor='black',hatch='xx',linewidth=1)
    p3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    reply = input('Enter to (s)ave figure, (e)xit or (c)ontinue loop:')
    if (reply=='e'):
        break
    elif (reply=='s'):
        myfigs[-1].savefig(figname+'.jpg')
#        myfigs[-1].savefig(figname+'.pdf')
    else:
        ConnectionRefusedError

print('Exit loop or last event')

# %%
'''
    Use h5-file and centrality bins calculated in last cell to caculate path-length pairs for each bin
    Set samples_per_event to adjust event sampling

    - Loop over centrality bins
      - Loop over events per centrality
        - Loop over samples_per_event (not yet weighted with TaTb per centrality class)

    Outputs - 2 kinds of h5 outputs are created, depending on save_tpath/tpair/values
    1. h5_tpath = rmid,rdif,dens path arrays and x0,y0,angle attributes, one for each centrality
    2. h5_tpair = pairs of path lengths for each centrality written to a single file
'''

'''At least one of the save flags should be set True to both running this cell'''
save_tpath = False
save_tpair = True
if ((save_tpath+save_tpair)==0):
    print('At least one of save_tpath/save_tpair should be set True, exitting ...')
    exit(1)

'''Use tau to adjust jet formation time in gradations of 0.1 fm'''
tau = 0.9
sample_factor = 11
sample_cbin   = 10000
eps = 100*np.finfo(float).eps
angles = np.arange(0.,np.pi+eps,np.pi/12)

if (save_tpair):
    h5_tpair = h5dir + 't-pairs_' + trento_base + '_tau{:0.1f}.h5'.format(tau)
    h = h5py.File(h5_tpair,'w')

for i in np.arange(cent_nbins):
    isample_cbin  = 0
    if (save_tpath):
        h5_tpath = h5dir + 't-paths_' + trento_base + '_' + cent_label[i][:-1] + '.h5'
        if os.path.exists(h5_tpath):
            os.remove(h5_tpath)
        g = h5py.File(h5_tpath,'w')

    print('Processing Centrality',cent_label[i])
    tpair_list = []
    tpexp_list = []
    # for evt in cent_events[i][:10]: '''to limit to 10 events'''
    for evt in cent_events[i]:
        grid = np.array(f[events[evt]])
        TaTb = grid*grid
        sumTaTb = np.sum(TaTb)
        sample_evt = int(sample_factor*sumTaTb/cent_med_ncoll[i])
#        print('sumTaTb,samples:',sumTaTb,samples)

        isample_evt = 0
        while (isample_evt<sample_evt and isample_cbin<sample_cbin):
            isample_evt += 1
            isample_cbin += 1
            myran  = rng.random(4)
            cell = np.where((np.cumsum(TaTb)/sumTaTb)>myran[0])[0][0]
            xcell = cell%100
            ycell = cell//100
            '''randomize position within cell, set angle, and call gridpts()'''
            x = (xcell+myran[1])*gridstep - gridmax
            y = (ycell+myran[2])*gridstep - gridmax
            angle = myran[3]*np.pi/2.

            (xgrid,ygrid) = gridpts(x,y,angle,xedge,yedge)

            xdif = np.diff(xgrid)
            ydif = np.diff(ygrid)
            xmid = (xgrid[1:]+xgrid[:-1])/2.
            ymid = (ygrid[1:]+ygrid[:-1])/2.
            xrel = xmid - x
            yrel = ymid - y
            xbin = np.floor((xmid-xedge[0])/gridstep).astype(int)
            ybin = np.floor((ymid-yedge[0])/gridstep).astype(int)
            mbin = xbin + ybin*(xedge.size-1)
            dens = grid.flatten()[mbin]
            if (angle>np.pi/4.):
                sign = np.sign(yrel)
            else:
                sign = np.sign(xrel)
            rmid = ((xrel**2 + yrel**2)**0.5)*sign
            rdif = (xdif**2 + ydif**2)**0.5

            '''Calculate p1,p2 path integrals, apply expansion of 1./rmid for tau>0'''
            if (save_tpair):
                pb = rmid>tau
                nb = rmid<-tau
                pmid = rmid[pb]
                nmid = rmid[nb]
                posint  = np.sum(rdif[pb]*dens[pb]*pmid[0]/pmid)
                negint  = np.sum(rdif[nb]*dens[nb]*nmid[-1]/nmid)
                posint2 = np.sum(rdif[pb]*dens[pb]*pmid[0]*(pmid-tau)/pmid)
                negint2 = np.sum(rdif[nb]*dens[nb]*nmid[-1]*(-nmid-tau)/nmid)
                tpair_list.append([posint,negint,posint2,negint2])

#                if (tau==0):
#                    posindx = np.where(rmid>=0.)[0][::-1]
#                    negindx = np.where(rmid<0.)[0]
#                    posint  = np.sum(rdif[posindx]*dens[posindx])
#                    negint  = np.sum(rdif[negindx]*dens[negindx])
#                    posint2 = np.sum(rdif[posindx]*dens[posindx]*(np.abs(rmid[posindx])-tau))
#                    negint2 = np.sum(rdif[negindx]*dens[negindx]*(np.abs(rmid[negindx])-tau))
#                    tpair_list.append([posint,negint,posint2,negint2])
#                else:
#                    posindx = np.where(rmid>=tau)[0][::-1]
#                    negindx = np.where(rmid<-tau)[0]
#                    posint  = np.sum(rdif[posindx]*dens[posindx]*tau/np.abs(rmid[posindx]))
#                    negint  = np.sum(rdif[negindx]*dens[negindx]*tau/np.abs(rmid[negindx]))
#                    posint2 = np.sum(rdif[posindx]*dens[posindx]*tau*(np.abs(rmid[posindx])-tau)/np.abs(rmid[posindx]))
#                    negint2 = np.sum(rdif[negindx]*dens[negindx]*tau*(np.abs(rmid[negindx])-tau)/np.abs(rmid[negindx]))
#                    tpair_list.append([posint,negint,posint2,negint2])


            if (save_tpath):
                evtname  = 'event_' + str(evt) + '_' + str(isample_evt)
                pathinfo = np.vstack((rmid,rdif,dens))
                dset     = g.create_dataset(evtname, data=pathinfo)
                dset.attrs['x0']=x
                dset.attrs['y0']=y
                dset.attrs['angle']=angle
                # print('Processing Centrality',cent_label[i],evtname,'sumTaTb=',sumTaTb)
    if (save_tpath):
        g.close()
    if (save_tpair):
        tpair_name = trento_base + '_' + cent_label[i][:-1] + '_tau{:0.1f}'.format(tau)
#        print(tpair_name,'(first 10 pairs):\n',tpair_list[:10])
        dset = h.create_dataset(tpair_name,data=tpair_list)
        print(isample_cbin,'trento pairs and saved to',tpair_name)

if (save_tpair):
    h.close()

# %%
'''Read h5_tpair files and plot path1 vs. path2 for each cbin'''
h = h5py.File(h5_tpair,'r')
tpath_cbins = list(h.keys())

'''Assuming 4-cbins for RAA tpaths, but Xj has 6'''
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
im = []

n = 20
for i in np.arange(4):
    tpaths = np.array(h[tpath_cbins[i]])
#    for j in np.arange(10): print('Event',j,tpaths[j,0],tpaths[j,1])
    print(tpath_cbins[i], 'pair-count =',tpaths[:,0].size)

    myax = ax[int(i/2),i%2]
    vals,e1,e2 = np.histogram2d(tpaths[:,0],tpaths[:,1],n)
    myax.set(title=tpath_cbins[i],xlabel='tpath-2',ylabel='tpath-1')
    im.append(myax.imshow(vals,origin='lower',extent=[e1[0],e1[-1],e2[0],e2[-1]],cmap='rainbow'))

plt.tight_layout()

if(generate_figures):
    figname = figdir + 't-paths_heatmap_' + trento_base
    fig.savefig(figname+'.pdf')

h.close()
# %%