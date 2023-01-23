 
import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np
import flopy 
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import cg


from shutil import copy2, rmtree
import subprocess
#from subprocess import call
from time import time
#import pdb; pdb.set_trace()

'''
three operations
1. write inputs
2. run simul
3. read input
'''

class Model:
    def __init__(self,params = None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
        self.deletedir = True
        self.outputdir = None
        self.parallel = False
        self.record_cobs = False

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)

        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
                self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
            if 'inputdir' in params:
                self.inputdir = params['inputdir']
            if 'ncores' in params:
                self.ncores = params['ncores']
            if 'outputdir' in params:
                # note that outputdir is not used for now; pyPCGA forces outputdir in ./simul/simul0000
                self.outputdir = params['outputdir']
            if 'parallel' in params:
                self.parallel = params['parallel']
            if 'nx' in params:
                self.nx = params['nx']
            else:
                raise NameError('nx is not defined')
            
            if 'ny' in params:
                self.ny = params['ny']
            else:
                raise NameError('ny is not defined')

    def create_dir(self,idx=None):
        
        mydirbase = "./simul/simul"
        if idx is None:
            idx = self.idx
        
        mydir = mydirbase + "{0:04d}".format(idx)
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        
        for filename in os.listdir(self.inputdir):
            copy2(os.path.join(self.inputdir,filename),mydir)
        
        return mydir

    def cleanup(self,outputdir=None):
        """
        Removes outputdir if specified. Otherwise removes all output files
        in the current working directory.
        """
        import shutil
        import glob
        log = "dummy.log"
        if os.path.exists(log):
            os.remove(log)
        if outputdir is not None and outputdir != os.getcwd():
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
        else:
            filelist = glob.glob("*.out")
            filelist += glob.glob("*.sim")
            
            for file in filelist:
                os.remove(file)

    def write_input(self,s):
        
        return True

    def run_model(self,s,idx=0):
        m = s.shape[0]
        m_ = int(m/2) # half point
        s1 = s[:m_] # lnK
        s2 = s[m_:] # lnres
        #s1 = s1 - 3. 

        head, q = self.run_model_GW(s1,idx)
        sp = self.run_model_SP(s1,s2,head,q,idx)
        simul_obs4 = self.run_model_MT(s2,idx)
        
        # observations
        
        simul_obs1 = head[0,0:5,12:-1:25].reshape(-1) # 20 head 
        simul_obs2 = (s1.reshape(100,100)[0:5,12:-1:25]).reshape(-1) # 20 well log
        
        simul_obs3_1 = sp[0,:].reshape(-1) # 100 surface SP
        simul_obs3_2 = sp[1:5,12:-1:25].reshape(-1) # 16 well SP
        
        #simul_obs1 = head[0,0:5,25::25].reshape(-1) # 15 head 
        #simul_obs2 = (s1.reshape(100,100)[0:5,25::25]).reshape(-1) # 15 well log
        
        #simul_obs3_1 = sp[0,:].reshape(-1) # 100 surface SP
        #simul_obs3_2 = sp[1:5,25::25].reshape(-1) # 12 well SP
        
        simul_obs = np.hstack((simul_obs1,simul_obs2,simul_obs3_1,simul_obs3_2,simul_obs4))  
        #simul_obs = np.hstack((simul_obs1,simul_obs2,simul_obs3_1,simul_obs4)) 
        return simul_obs


    def run_model_GW(self,s,idx=0):
        # Assign name and create modflow model object
        #1) MT
        sim_dir = self.create_dir(idx)
        os.chdir(sim_dir)
        
        K = np.exp(s).reshape(100,100) # m/day 
    
        sim = flopy.mf6.MFSimulation(sim_name='fine6', version='mf6', exe_name= './mf6', 
                                sim_ws= './')

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
        Head = head_end-10



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

        os.chdir(self.homedir)
        
        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)


        return Head, spq

    #########Self-potential####################
    def run_model_SP(self,s1,s2,head,spq,idx=0):
        #1,s2,idx=0,head,spq
        #EC,K
        print('SP model start')
        # res = np.exp(s1) [Omega m]
        
        K = np.exp(s1).reshape(100,100) # [m/d]
        EC = 1./(np.exp(s2).reshape(100,100)) # [S/m] need to check the unit consistent with MARE2DEM
        ##porosity
        # poro = 0.3
        rho = 1000. # [Kg/m^3]
        grav = 9.8 # gravity [m/s^2]
        mu = 8.9E-4 # Pa s 

        ##permeability
        K_ms = K/86400.   ##input K [m/s] => need to check later
        # Keff= K_ms/poro
        Keff = K_ms
        keff = Keff*mu/(rho*grav) # m^2 => m/s * Kg/m/s^2 * s / Kg * m^3 /m * s^2
        Qv = (10.**(-9.2))*(keff**(-0.82)) # [C/m^3]
        
        ##grid
        N = int(K.shape[1])
        n= int(N/2)
        kk = np.asarray(list(range(0,n+1))+list(range(-n+1,0)))
        kk = kk*(2.*np.pi/2000.)
        xk, yk = np.meshgrid(kk,kk)

        def spec2D(xx):
            xx = xx.reshape(-1,1)
            Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(xk.reshape(-1,1)*xx,(N,N))))-yk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(yk.reshape(-1,1)*xx,(N,N))))
            # Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(xk*np.fft.fftn(xx.reshape(N,N))))\
            #     -yk*np.fft.fftn(EC*np.fft.ifftn(yk*np.fft.fftn(xx.reshape(N,N))))
            return Ax.reshape(-1)

        # b1 = 1j*xk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*xk*np.fft.fftn(head[0,:,:])))+1j*yk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*yk*np.fft.fftn(head[0,:,:])))
        #b2 = 1j*xk*np.fft.fftn(Qv*spq[0]/86400.)+1j*yk*np.fft.fftn(Qv*spq[1]/86400.)
        b2 = 1j*xk*np.fft.fftn(Qv*spq[0].reshape(100,100)/86400.)+1j*yk*np.fft.fftn(Qv*spq[1].reshape(100,100)/86400.)
        A = LinearOperator((N**2,N**2), matvec = spec2D)

        # sp1, exitcode1 = cg(A,b1.reshape(-1))
        # if exitcode1 != 0:
        #     print("cg not converged: %d" % (exitcode1))
        #     sp1, exitcode1 = gmres(A,b1.reshape(-1),x0=sp1)
        
        sp2, exitcode1 = cg(A,b2.reshape(-1))
        if exitcode1 != 0:
            print("cg not converged: %d, gmres starts" % (exitcode1))
            sp2, exitcode1 = gmres(A,b2.reshape(-1),x0=sp2)        
        # SP1 = np.fft.ifftn(sp1.reshape(N,N))
        SP = np.fft.ifftn(sp2.reshape(N,N))
        # SP1 = sp1

        simul_obs = np.real(SP)
        #simul_obs = sp_all[:,] 
        # observation
        print('SP end')

        return simul_obs

    def run_model_MT(self,s,idx=0):
        '''
        MT 2 core 114 

        '''

        sim_dir = self.create_dir(idx)
        os.chdir(sim_dir)
        
        nx, ny = self.nx, self.ny
        m = nx*ny
        
        with open("fwd_model.0.resistivity_head","r") as f:
            lines = f.readlines()


        with open("fwd_model.0.resistivity","w") as f:
            for line in lines:
                f.write(line)

            f.write("Number of regions:          %d                               ! input \n" % (m+2))
            f.write("!#       Rho          Param    Lower        Upper        Prej         Weight        \n")
            f.write("1        500          0        0            0            0            0       \n")  
            f.write("2        1e+12        0        0            0            0            0       \n")
            for num, sval in enumerate(s):
                f.write("%d       %9.5f      0        0        0        0        0        \n" % (num+3,np.exp(sval)))

        #subprocess.call(["mpirun","-n","6","--bind-to","none","MARE2DEM","-f","fwd_model.0.resistivity"], stdout=subprocess.PIPE)
        #subprocess.call(["mpirun","-n","6","--bind-to","none","MARE2DEM","-f","fwd_model.0.resistivity"], stdout=open('/dev/null','w'))
        subprocess.call(["mpirun","-n","6","--bind-to","none","./MARE2DEM","-f","fwd_model.0.resistivity"], stdout=open('/dev/null','w'))
        # read results
        with open("fwd_model.0.resp","r") as f:
            lines = f.readlines()

            # Next three lines should be read from the files
            nobs = 5248
            simul_obs = np.zeros(nobs,)    
            nskips = 1 + 1 + 32 + 1 + 1+ 41 + 1 + 1 
            obslines = lines[nskips:]
            
            for num, obsline in enumerate(obslines):
                simul_obs[num] = float(obsline.split()[6])

        
        os.chdir(self.homedir)
        
        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)
            # self.cleanup(sim_dir)

        return simul_obs

    def run(self,s,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(s.shape[1])
        args_map = [(s[:, arg:arg + 1], arg) for arg in method_args]

        if par:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs =[]
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T

        #pool.close()
        #pool.join()

    def __call__(self,args):
        return self.run_model(args[0],args[1])


if __name__ == '__main__':
    import numpy as np
    from time import time
    import forward6

    s1 = np.load("./Input_yh/lnK1.npy").reshape(-1, 1)
    s2 = np.load("./Input_yh/lnres1.npy").reshape(-1, 1)
    
    s = np.vstack((s1,s2))
    #np.savetxt('true.txt',s)
    #resistivity = (np.exp(s).T)*10.+100.
    #EC = 1./resistivity
    #s = np.loadtxt('shat0.txt').reshape(-1,1)

    #m_ = int(s.shape[0]/2)
    #s[:m_,:] = s[:m_,:] + 0.05*np.random.randn(m_, 1)

    nx = ny = 100
    m = nx*ny

    params = {'nx':nx,'ny':ny, 'deletedir':False}

    par = False # parallelization false

    mymodel = forward6.Model(params)
    print('(1) single run')

    from time import time
    stime = time()
    simul_obs = mymodel.run(s,par)
    print('simulation run: %f sec' % (time() - stime))
    #obs = simul_obs + 0.1*np.random.randn(simul_obs.shape[0],simul_obs.shape[1])

    obs = simul_obs
    
    nwell = 20 #15
    nsp = 100+16 #12
    nmt = 5248
    wellidx = nwell
    logidx = wellidx+ nwell
    spidx = logidx + nsp 
    mtidx = spidx + nmt
    assert(obs.shape[0]==mtidx)

    obs[:wellidx] = obs[:wellidx] +  0.3*np.random.randn(nwell,1)
    obs[wellidx:logidx] = obs[wellidx:logidx] +  0.05*np.random.randn(nwell,1)
    obs[logidx:spidx] = obs[logidx:spidx] +  1.e-5*np.random.randn(nsp,1)
    obs[spidx::2] = obs[spidx::2] + 0.1*np.random.randn(int(nmt/2),1)
    obs[spidx+1::2] = obs[spidx+1::2] + 2.0*np.random.randn(int(nmt/2),1)
    np.savetxt('./Results/obs.txt',obs)
    np.savetxt('./Results/obs_mf.txt',obs[:logidx])
    np.savetxt('./Results/obs_MT.txt', obs[spidx:])
    np.savetxt('./Results/s_true.txt',s)


    import sys
    sys.exit(0)

    ncores = 8
    nrelzs = 8
    
    print('(2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    srelz = np.zeros((np.size(s,0),nrelzs),'d')
    for i in range(nrelzs):
        srelz[:,i:i+1] = s + 0.1*np.random.randn(np.size(s,0),1)
    
    simul_obs_all = mymodel.run(srelz,par,ncores = ncores)

    print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('(3) parallel run with all the physical cores')
    #simul_obs_all = mymodel.run(srelz,par)
    #print(simul_obs_all)
