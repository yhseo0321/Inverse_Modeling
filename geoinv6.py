import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import forward6 
from pyPCGA import PCGA
import math

# model domain and discretization
nx = ny = 100
m = nx*ny
N = np.array([nx,ny])
xmin = np.array([0,0])
xmax = np.array([2000.0,2000.0])
pts = None # for regular grids, you don't need to specify pts. 

# covairance kernel and scale parameters
prior_std = 1.0
prior_cov_scale = np.array([500.0,200.0])

def kernel(r): return (prior_std ** 2) * np.exp(-r**2)


params_pre = {'R': (0.1) ** 2, 'n_pc': 150,
        'maxiter': 5, 'restol': 0.1,
        'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
        'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
        'kernel': kernel, 'post_cov': "diag",
        'precond': True, 'LM': True,
        'parallel': True, 'linesearch': True,
        'forward_model_verbose': False, 'verbose': True,
        'iter_save': True, 'precision':1.E-4}

prob_pre = PCGA(forward_model=None, s_init = np.ones((m,1)), pts=pts, params=params_pre)

#prob_pre = PCGA(forward_model=None, s_init = np.ones(m,1), pts = None, params = params_pre, s_true=None, obs=None)

prob_pre.ConstructCovariance(method = prob_pre.matvec, kernel = prob_pre.kernel, xmin = prob_pre.xmin, xmax = prob_pre.xmax, N= prob_pre.N, theta = prob_pre.prior_cov_scale)
prob_pre.ComputePriorEig()

d = prob_pre.priord
U = prob_pre.priorU

# augmentation 
d = np.vstack((d,d))

U1 = np.vstack((U,np.zeros(U.shape)))
U2 = np.vstack((np.zeros(U.shape),U))
U = np.hstack((U1,U2))    

#print(d.shape)
#print(U.shape)

#import sys
#sys.exit(0)
########################################################
## Restart the problem
########################################################

# model domain and discretization for two unknown fields

dx = np.array([20., 20.])

x = np.linspace(0. + dx[0]/2., 2000. - dx[0]/2., N[0])
y = np.linspace(0. + dx[1]/2., 2000. - dx[1]/2., N[1])

XX, YY = np.meshgrid(x, y)
pts = np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))
pts = np.vstack((pts,pts))

m = 2*nx*ny

# forward model wrapper for pyPCGA
s_true = np.loadtxt('./Results/s_true.txt')
obs = np.loadtxt('./Results/obs.txt')
#s_true = np.loadtxt('true.txt')
#obs = np.loadtxt('obs.txt')



nwell = 20 #15
nsp = 100+ 16 #12
nmt = 5248
wellidx = nwell
logidx = wellidx+ nwell
spidx = logidx + nsp 
mtidx = spidx + nmt

std_obs = np.ones_like(obs)
std_obs[:wellidx]= 0.3 # Head
std_obs[wellidx:logidx]= 0.05 # Core
std_obs[logidx:spidx]=1.e-5 # SP
std_obs[spidx::2]=0.1 # phase 
std_obs[spidx+1::2]=2.0 # magnitude 

assert(obs.shape[0]==mtidx)

params = {'R': (std_obs) ** 2, 'n_pc': 300,
        'maxiter': 10, 'restol': 0.01,
        'matvec': 'hmatrix', 'priord' : d, 'priorU' : U, 'kernel': kernel,       
        'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
        'post_cov': "diag", 'post_diag_direct': True,
        'precision':1.E-4, 'direct': True, #'precond': True, 
        'LM': True, 'alphamax_LM': 1000.,
        'parallel': True, 'linesearch': True,
        'forward_model_verbose': False, 'verbose': True,
        'iter_save': True,'LM_smin':-20,'LM_smax':13.}

#'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified
params['ncores'] = 8 # use 6 cores for mare2dem 
    
m_ = int(m/2)

s_true1 = s_true[:m_]
s_true2 = s_true[m_:]

#np.random.seed(0)

s_init = np.zeros((m,1))

s_init[:m_,:] = np.mean(s_true1)*np.ones((m_, 1)) 
s_init[m_:,:] = np.mean(s_true2)*np.ones((m_, 1))
# s_init = np.copy(s_true) # you can try with s_true!

#s_init = np.loadtxt('/home/harry/Dropbox/Work/pyPCGA_SP/test1/result_new05/shat10.txt')
#s_init = np.loadtxt('/home/harry/Dropbox/Work/pyPCGA_SP/test1/poster_joint_pc150/shat10.txt')
#s_init = np.copy(s_true).reshape(m,1)

#s_init = np.copy(s_true)

X = np.zeros((m,2))

X[:m_,0] = 1./np.sqrt(m_)
X[m_:,1] = 1./np.sqrt(m_)

#X[:m_,0] = 1.
#X[m_:,1] = 1.

# initialize
# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    forward_params = {'nx':nx,'ny':ny}
    model = forward6.Model(forward_params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs

#prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()


s_hat1 = s_hat[:m_,:]
s_hat2 = s_hat[m_:,:]

post_std = np.sqrt(post_diagv)

post_std1 = post_std[:m_,:]
post_std2 = post_std[m_:,:]


s1min = s_true1.min()
s1max = s_true1.max()
s2min = s_true2.min()
s2max = s_true2.max()

fig, ax = plt.subplots(nrows=1,ncols=2,sharey=True)
im0 = ax[0].imshow(s_true1.reshape(ny,nx),vmin=s1min,vmax=s1max, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[0].set_title(r'true ln K')
ax[0].set_aspect('equal','box')
ax[1].imshow(s_hat1.reshape(ny,nx),vmin=s1min,vmax=s1max, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[1].set_title(r'estimated ln K')
ax[1].set_aspect('equal','box')
fig.colorbar(im0, ax=ax.ravel().tolist(),shrink=0.4)
fig.savefig('./Results/est_K.png',bbox_inches = 'tight')
plt.show()
plt.close(fig)

fig = plt.figure()
plt.title(r'Uncertainty (posterior std) in lnK estimate)')
im0 = plt.imshow(post_std1.reshape(ny,nx), cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
plt.gca().set_aspect('equal','box')
fig.colorbar(im0)
fig.savefig('./Results/std_K.png')
plt.show()
plt.close(fig)

fig, ax = plt.subplots(nrows=1,ncols=2,sharey=True)
im0 = ax[0].imshow(s_true2.reshape(ny,nx),vmin=s2min,vmax=s2max, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[0].set_title(r'true $\ln \rho$')
ax[0].set_aspect('equal','box')
ax[1].imshow(s_hat2.reshape(ny,nx),vmin=s2min,vmax=s2max, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[1].set_title(r'estimated $\ln \rho$')
ax[1].set_aspect('equal','box')
fig.colorbar(im0, ax=ax.ravel().tolist(),shrink=0.4)
fig.savefig('./Results/est_res.png',bbox_inches = 'tight')
plt.show()
plt.close(fig)

fig = plt.figure()
plt.title(r'Uncertainty (posterior std) in $\ln \sigma$ estimate)')
im0 = plt.imshow(post_std2.reshape(ny,nx), cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
plt.gca().set_aspect('equal','box')
fig.colorbar(im0)
fig.savefig('./Results/std_res.png')
plt.show()
plt.close(fig)

nobs = prob.obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(prob.obs, simul_obs, '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs, simul_obs)).min(0)
maxobs = np.vstack((prob.obs, simul_obs)).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('./Results/obs.png')
plt.close(fig)

fig = plt.figure()
plt.title('obs. vs simul. (pressure)')
plt.plot(prob.obs[0:20], simul_obs[0:20], '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs[0:20], simul_obs[0:20])).min(0)
maxobs = np.vstack((prob.obs[0:20], simul_obs[0:20])).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('./Results/obs_pressure.png')
plt.close(fig)

fig = plt.figure()
plt.title('obs. vs simul. (core)')
plt.plot(prob.obs[20:40], simul_obs[20:40], '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs[20:40], simul_obs[20:40])).min(0)
maxobs = np.vstack((prob.obs[20:40], simul_obs[20:40])).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('./Results/obs_core.png')
plt.close(fig)

fig = plt.figure()
plt.title('obs. vs simul. (sp)')
plt.plot(prob.obs[40:156], simul_obs[40:156], '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs[40:156], simul_obs[40:156])).min(0)
maxobs = np.vstack((prob.obs[40:156], simul_obs[40:156])).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('./Results/obs_sp.png')
plt.close(fig)

fig = plt.figure()
plt.title('obs. vs simul. (MT)')
plt.plot(prob.obs[156:], simul_obs[156:], '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs[156:], simul_obs[156:])).min(0)
maxobs = np.vstack((prob.obs[156:], simul_obs[156:])).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('./Results/obs_MT.png')
plt.close(fig)

fig = plt.figure()
plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('./Results/obj.png')
plt.close(fig)
