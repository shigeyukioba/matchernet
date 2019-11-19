import numpy as np

from matchernet_py_001 import fn
from matchernet_py_001 import utils

printDebug = True
class StateSpaceModel2Dim(object):
    def __init__(self,dt):
        n=2
        self.n=n
        #self.w=N(0,sigma)
        self.A=np.array([[-0.1,2],[-2,-0.1]],dtype=np.float32)
        #self.f=utils.MyLinear(n,n,self.A);
        self.F=utils.calc_matrix_F(self.A,dt)
        self.g=fn.LinearFn(n,n)
        self.g.params["A"] = self.A
        self.sigma_w=0
        self.sigma_z=0.1
        self.x=np.array([0,1],dtype=np.float32)
        self.y=utils.zeros((1,2))

    def simulation(self,dt, numSteps):
        x0=self.x
        for i in range(numSteps):
            (x, y)=self.onestep(dt)
            if i==0:
                self.xrec = x
                self.yrec = y
            else:
                self.xrec=np.concatenate((self.xrec,x),axis=0)
                self.yrec=np.concatenate((self.yrec,y),axis=0)

        return (self.xrec, self.yrec)

    def onestep(self,dt):
        F=self.F
        self.w = self.sigma_w * np.random.standard_normal(size=(1,self.n))
        #pdb.set_trace()
        self.x = np.dot( self.x, F )+self.w
        self.z = self.sigma_z * np.random.standard_normal(size=(1,self.n))
        self.y = self.x+self.z
        #Sigma = dt * Q + np.dot( np.dot( F.T,  Sigma), F )
        return (self.x, self.y)
