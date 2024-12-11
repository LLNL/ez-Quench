'''
ezQ_plotPPP.py
created: 2024-nov-06 R.A. Soltz 
modified: 2023-apr-11 RAS - 

Remake figure 1a from https://doi.org/10.1063/1.1945011 and project to log-space
Also confirm chi2 minimization results

'''

# %% 

# Import standard packages and set plots inline
import numpy as np
import matplotlib.pyplot as plt

generate_figures = False
mycol = ['darkblue','darkred','darkgreen','darkgoldenrod','brown','olive','purple']
mylin = ((0,(5,3)),(0,(2,1,2,3)),'solid','dashdot','dashed','dotted',(0,(3,3)),(0,(5,5)))
workdir = os.environ.get('ezQuench_workdir')
if (workdir==None):
    print('source setup.sh from ../ to define ezQuench_workdir')
    sys.exit()
figdir  = workdir + 'fig/'
h5dir   = workdir + 'h5/'

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# %%

'''Solve for most probable x using notation from paper'''

m1 = 1.5
m2 = 1.0
e1 = 0.1
e2 = 0.1
e12 = 0.2
C = np.array([[m1*m1*(e1*e1+e12*e12), m1*m2*e12*e12],[m1*m2*e12*e12, m2*m2*(e2*e2+e12*e12)]])
Clog = np.array([[(e1*e1+e12*e12), e12*e12],[e12*e12, (e2*e2+e12*e12)]])
G = np.array((1,1))
m = np.array((m1,m2))
Ci = np.linalg.inv(C)
Cli = np.linalg.inv(Clog)
step1 = np.linalg.multi_dot((G,Ci,G))
step2 = np.linalg.multi_dot((G,Ci,m))
xmin  = step2/step1
logstep1 = np.linalg.multi_dot((G,Cli,G))
logstep2 = np.linalg.multi_dot((G,Cli,m))
lmin     = logstep2/logstep1
print('step1 = {0:.3f}, step2 = {1:0.3f}'.format(step1,step2))
print('logstep1 = {0:.3f}, logstep2 = {1:0.3f}'.format(logstep1,logstep2))
print('x= {0:0.3f}, xlog={1:0.3f}'.format(xmin,lmin))

'''Plot joint probability distribution'''

x = np.arange(0.25,2.5,0.01)
n = x.size
x1m = x-m1
x2m = x-m2
'''set 2D x1,x2 array with x1 in column-j position'''
expo = Ci[0,0]*np.tile(x1m*x1m,(n,1)) + (Ci[0,1]+Ci[1,0])*np.outer(x2m,x1m) + Ci[1,1]*np.tile(x2m*x2m,(n,1)).T
prob = np.exp(-expo/2.)
'''redo for log transformation'''
l1 = np.log(x/m1)
l2 = np.log(x/m2)
elog = Cli[0,0]*np.tile(l1*l1,(n,1)) + (Cli[0,1]+Cli[1,0])*np.outer(l2,l1) + Cli[1,1]*np.tile(l2*l2,(n,1)).T
plog = np.exp(-elog/2.)

#print(np.array2string(Ci,precision=2))
#print(np.array2string(x1m,precision=2))
#print(np.array2string(x2m,precision=2))
#print(np.array2string(expo,precision=2))

fig,ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_ylabel(r'$x_2$')
ax.set_xlabel(r'$x_1$')
ax.tick_params(which='both',direction='in',top=True,right=True,pad=10)
lo,hi = 0., 2.5
ax.set_xlim([lo,hi])
ax.set_ylim([lo,hi])

lvl1 = np.arange(0.06,1.,0.15)
lvl2 = np.arange(0.02,1.0,0.15)
x1,x2 = np.meshgrid(x,x)
ax.contour(x1,x2,prob,cmap='winter',levels=lvl1)
ax.contour(x1,x2,plog,cmap='autumn',linestyles='dashdot',levels=lvl2)
ax.plot([lo,hi],[lo,hi],color='k',linestyle='dotted')
ax.plot((0,lmin),(lmin,lmin),color='r',linestyle='dashdot',label=r'log transform contour')
ax.plot((0,xmin),(xmin,xmin),color='b',label=r'standard $\chi^2$ contour')
ax.text(0.2,xmin+0.02,'0.88',color='b',fontsize=14)
ax.text(0.2,lmin+0.02,'1.25',color='r',fontsize=14)

ax.legend(loc='lower right',borderaxespad=2,fontsize='large')

save_fig = True
if(save_fig):
    figname = figdir + 'figB_PPP_contour'
    fig.savefig(figname+'.pdf')


# %%
