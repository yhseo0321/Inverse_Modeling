import numpy as np
import flopy
import time
import h5py
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

##Foward Model Using Geophygical relation

# Folder = './SP/FW1'  

WS = './SP/FW2'
Folder = 'SP/FW2/Mod6'  

##########Load RES###################
# filepath= './%s/relz_res.mat' %(Folder)
filepath= './%s/relz_res.mat' %(WS)
f2 = h5py.File(filepath)
mat2 = list(f2['relz'])
data = np.asarray(mat2)
EC = np.exp(data[0,:,:].T)/1500   #FW1 FW2 FW3
# EC = np.exp(data[1,:,:].T)/100   #FW3 xx

resistivity = 1/EC
# resistivity = (np.exp(data[1,:,:]).T)*10+100 # for i = 0  
# EC = 1/resistivity

##porosity
sigma_w = 0.02
m = 1.64
aa = 0.47
poro = aa*(EC/sigma_w)**(1/m)  #if correct equation >> (aa*EC/0.2)**(1/m) and change m and aa


##permeability
# #volcanic
Cc = 3.87*10**(-12)
# kp = Cc*(poro**m)/(1-poro)

# kp = (31*poro + 7463*(poro**2)+191*(10*poro)**10)/(10**18)
kp = (303*(100*poro)**3.05)/(10**18)

#-----------permeability test--------------------
# ps = np.linspace(poro.min(),poro.max(),20)
# kp1 = Cc*(ps**m)/(1-ps)
# kp2 = (31*ps + 7463*(ps**2)+191*(10*ps)**10)/(10**18)
# kp3 = (303*(100*ps)**3.05)/(10**18)
# kp11 = kp3.std()*((kp1-kp1.mean())/kp1.std())+kp3.mean()

# plt.plot(ps,kp1,'-o',label = '1')
# plt.plot(ps,kp2,'-x',label = '2')
# plt.plot(ps,kp3,'--',label = '3')
# plt.plot(ps,np.log10(86400*kp1*rho*grav/mu),'-o',label = '1')
# plt.plot(ps,np.log10(86400*kp2*rho*grav/mu),'-x',label = '2')
# plt.plot(ps,np.log10(86400*kp3*rho*grav/mu),'--',label = '3')
# plt.legend()
# plt.show()

# kp1 = Cc*(poro**m)/(1-poro)
# kp2 = (31*poro + 7463*(poro**2)+191*(10*poro)**10)/(10**18)
# kp3 = (303*(100*poro)**3.05)/(10**18)
# kp11 = kp3.std()*((kp1-kp1.mean())/kp1.std())+kp3.mean()
#-------------------------------------------------
##Hydraulic conductivity
rho = 1000
grav = 9.8
mu = 8.9/(10**4) #>>>>>>>>>>>>>>>>>>>>>>check error

K_ms = kp*rho*grav/mu
Kdata = K_ms*86400

Qv = (10**(-9.2))*(kp**(-0.82))


##grid
N = int(Kdata.shape[1])
n= int(N/2)
kk = np.asarray(list(range(0,n+1))+list(range(-n+1,0)))
kk = kk*(2*np.pi/2000)
xk, yk = np.meshgrid(kk,kk)

##########GW MODEL########################

# Kdata = np.ones((100,100))
# Kdata[:,0:10]= 100
# Kdata[:,90:100] = 100

def GWmodel(K):
    exe_path = 'C:/UHM/Research/Upscaling/mf6'
    sim_path = Folder

    sim = flopy.mf6.MFSimulation(sim_name='fine6', version='mf6', exe_name= exe_path, 
                            sim_ws=sim_path+'/MOD6_fine')

    tdis = flopy.mf6.ModflowTdis(sim,pname='tdis', time_units='DAYS', nper=1,
                            perioddata=[(1.0,1,1.0)])

    model_name='upscaling'
    model = flopy.mf6.ModflowGwf(sim, modelname=model_name,
                            model_nam_file='{}.nam'.format(model_name))

    # ims_package = flopy.mf6.ModflowIms(sim, pname='ims', print_option='ALL',csv_output_filerecord = 'ALL',
    #                                 complexity='SIMPLE', outer_hclose= 1e-8, outer_rclosebnd = 1e-8, 
    #                                 outer_maximum=300, under_relaxation='NONE', rcloserecord =[1e-8, 'STRICT'],
    #                                 inner_maximum=1000, inner_hclose=1e-8,
    #                                 linear_acceleration='BICGSTAB',
    #                                 preconditioner_levels=7,
    #                                 preconditioner_drop_tolerance=0.0001,
    #                                 number_orthogonalizations=2)

    ims_package = flopy.mf6.ModflowIms(sim, pname='ims', print_option='ALL',
                                complexity='SIMPLE', outer_hclose=0.0001,
                                outer_maximum=100, under_relaxation='NONE',
                                inner_maximum=80, inner_hclose=0.0001,
                                linear_acceleration='BICGSTAB',
                                preconditioner_levels=7,
                                preconditioner_drop_tolerance=0.01,
                                number_orthogonalizations=2)


    sim.register_ims_package(ims_package, [model_name])

    dis_package = flopy.mf6.ModflowGwfdis(model, pname='dis', length_units='METERS',
                                    nlay=1,
                                    nrow=100, ncol=100, delr=20,
                                    delc=20,
                                    top=0.0, botm = -20.0,
                                    filename='{}.dis'.format(model_name))
    # set the nocheck property in the simulation namefile
    sim.name_file.nocheck = True
    # set the print_input option in the model namefile
    model.name_file.print_input = True

    #Hydraulic conductivity
    # layer_storage_types = [flopy.mf6.data.mfdatastorage.DataStorageType.internal_array]
    # k_template = flopy.mf6.ModflowGwfnpf.k.empty(model, False, layer_storage_types, 100.0)
    # k_template['data'] = K11

    # k_template2 = flopy.mf6.ModflowGwfnpf.k.empty(model, False, layer_storage_types, 100.0)
    # k_template2['data'] = K22
    # #Angle
    # angle_template = flopy.mf6.ModflowGwfnpf.angle1.empty(model,False,layer_storage_types,0)
    # angle_template['data'] = -1.0*Angle1

    # angle_template2 = flopy.mf6.ModflowGwfnpf.angle1.empty(model,False,layer_storage_types,0)

    # angle_template3 = flopy.mf6.ModflowGwfnpf.angle1.empty(model,False,layer_storage_types,0)
    # angle_template3['data'] = Angle3
    #print(k_template)
    # create npf package using the k template to define k
    npf_package = flopy.mf6.ModflowGwfnpf(model, pname='npf', save_flows=True, save_specific_discharge=True ,icelltype=1, k = K)
                                    #  , angle2 = angle_template2, angle3 = angle_template3)

    strt=[20]
    ic_package = flopy.mf6.ModflowGwfic(model, pname='ic', strt=strt,
                                    filename='{}.ic'.format(model_name))
    # #define initial heads for model to 20
    # icPackage = flopy.mf6.ModflowGwfic(model,strt=20)

    H_init = []
    for i in range(100):
        a = [[0,i,0],20]
        H_init.append(a)
        b = [[0,i,99],10]
        H_init.append(b) 
    # print(H_init)
    stress_period_data1 = {0:H_init}
    chdPackage = flopy.mf6.ModflowGwfchd(model, maxbound = len(H_init), stress_period_data=stress_period_data1)

    # set up Output Control Package
    printrec_tuple_list = [('HEAD', 'ALL'), ('BUDGET', 'ALL')]
    saverec_dict = {0:[('HEAD', 'ALL'), ('BUDGET', 'ALL')]}
    oc_package = flopy.mf6.ModflowGwfoc(model, pname='oc', 
                                    budget_filerecord=[('{}.cbc'.format(model_name),)],
                                    head_filerecord=[('{}.hds'.format(model_name),)],
                                    saverecord=saverec_dict,
                                    printrecord=printrec_tuple_list)
    # write simulation to new location 
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    #OUPUT
    keys = sim.simulation_data.mfdata.output_keys()

    import matplotlib.pyplot as plt

    # get all head data
    head = sim.simulation_data.mfdata['upscaling', 'HDS', 'HEAD']
    qxqy = sim.simulation_data.mfdata['upscaling', 'CBC', 'DATA-SPDIS']
    # get the head data from the end of the model run
    head_end = head[-1]
    Head = head_end[0,:,:]-10



    # dum = (head_avg1-head_up)**2
    # RMSE[dl]= np.sqrt(dum.mean())

    spq1 = np.zeros((100,100))
    spq2 = np.zeros((100,100))
    spq = np.zeros((2,100,100))
    w = 0
    for i in range(100):
        for j in range(100):
            spq1[i,j] = qxqy[0,w][3]
            spq2[i,j] = qxqy[0,w][4]
            w +=1
    spq[0] = spq1
    spq[1] = spq2
    return Head, spq, qxqy
#########Self-potential####################
def spec2D(xx):
    xx = xx.reshape(-1,1)
    Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(xk.reshape(-1,1)*xx,(N,N))))-yk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(yk.reshape(-1,1)*xx,(N,N))))
    # Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(xk*np.fft.fftn(xx.reshape(N,N))))\
    #     -yk*np.fft.fftn(EC*np.fft.ifftn(yk*np.fft.fftn(xx.reshape(N,N))))
    return Ax.reshape(-1)


def SPforward(K,spq):
    # poro = 0.3
    rho = 1000
    grav = 9.8
    mu = 8.9/(10**4)
    
    K_ms = K/86400   ##input K
    # Keff= K_ms/poro
    Keff = K_ms
    keff = Keff*mu/(rho*grav)
    Qv = (10**(-9.2))*(keff**(-0.82))

    # b1 = 1j*xk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*xk*np.fft.fftn(head[0,:,:])))+1j*yk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*yk*np.fft.fftn(head[0,:,:])))
    b2 = 1j*xk*np.fft.fftn(Qv*spq[0]/86400)+1j*yk*np.fft.fftn(Qv*spq[1]/86400)
    A = LinearOperator((N**2,N**2), matvec = spec2D)

    # sp1, exitcode1 = cg(A,b1.reshape(-1))
    # if exitcode1 != 0:
    #     print("cg not converged: %d" % (exitcode1))
    #     sp1, exitcode1 = gmres(A,b1.reshape(-1),x0=sp1)
    
    sp2, exitcode1 = cg(A,b2.reshape(-1), atol =1e-08 )
    if exitcode1 != 0:
        print("cg not converged: %d" % (exitcode1))
        sp2, exitcode1 = gmres(A,b2.reshape(-1),x0=sp2)        

    # SP1 = np.fft.ifftn(sp1.reshape(N,N))
    SP2 = np.fft.ifftn(sp2.reshape(N,N))
    # SP1 = sp1
    return SP2, exitcode1




[head, spq, qxqy] = GWmodel(Kdata)
[sp, code]  = SPforward(Kdata,spq)

# plt.figure(1)
# plt.imshow(np.real(t1))
# plt.colorbar()

# plt.figure(2)
# plt.imshow(np.real(t2[0,:,:]))
# plt.colorbar()
# plt.show()

# qTest = -1*Kdata[:,0:99]*(head[0,:,1:]-head[0,:,0:99])/20
# plt.figure(1)
# plt.imshow(qTest)
# plt.colorbar()

# plt.figure(2)
# plt.imshow(spq[0][0,:,:])
# plt.colorbar()
# plt.show()
############################[Save]#################################################
np.save(WS+'/EC.npy',EC)
np.save(WS+'/K.npy',Kdata)
np.save(WS+'/sp.npy',sp)
np.save(WS+'/Head.npy',head[:,:])
np.save(WS+'/res.npy',resistivity)
np.save(WS+'/spq.npy',spq)

#############################[figure]###############################################
plt.figure(1)
plt.title('Hydraulic head [m]')
im0=plt.imshow(head[:,:], cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
# plt.gca().set_aspect('equal','box-forced')
# cbar = plt.colorbar(im0, ticks=[np.log(1),np.log(10), np.log(100),np.log(1000)])
# cbar.ax.set_yticklabels(['1','10','100','1000'])
plt.tight_layout()
plt.savefig('./%s/Head.png' %(Folder)) 

plt.figure(2)
plt.title(r'Hydraulic conductivity '+ r'$[\log_{10}$' + '(m/d)]')
im1 = plt.imshow(np.log10(Kdata), cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/K.png' %(Folder))

plt.figure(3)
plt.title('Self-Potential [V]')
im3 = plt.imshow(np.real(sp), cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Sp.png' %(Folder)) 

plt.figure(4)
plt.title(r'Resistivity ' + r'$[\Omega \cdot m]$')
im4 = plt.imshow(resistivity, cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Resistivity.png' %(Folder)) 

plt.figure(5)
plt.title('Electrical Conductivity [S/m]')
im5 = plt.imshow(EC, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/EC.png' %(Folder)) 

plt.figure(6)
plt.title('Porosity')
im5 = plt.imshow(poro, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Porosity.png' %(Folder)) 

plt.figure(7)
plt.title('Permeability' +r'$ [m^2]$')
im5 = plt.imshow(kp, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Permeability.png' %(Folder)) 

# plt.figure(8)
# plt.title(r'$ q_{x} [m/d]$')
# im5 = plt.imshow(spq1, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
# plt.xticks(np.linspace(0,2000,3))
# plt.yticks(np.linspace(0,2000,3))
# plt.xlabel('Max = %f & Min = %f' %(spq1.max(),spq1.min()))
# plt.ylabel('depth [m]')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('./%s/spq1.png' %(Folder)) 

#######################[boundary graph]################################################
px = np.linspace(1,100,100)
plt.figure(8)
plt.title('SP on Boundary')
plt.plot(px,sp[:,0], label = 'Left')
plt.plot(px,sp[:,-1], label = 'Right')
plt.xlabel('Depth [m]')
plt.ylabel('SP [V]')
plt.legend()
plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4),useMathText=True)
plt.savefig('./%s/Boundary.png' %(Folder))


plt.close('all')
