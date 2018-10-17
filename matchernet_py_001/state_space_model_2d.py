import numpy as np
import utils
import fn
import matplotlib.pyplot as plt

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


def test_state_space_model_2Dim(numSteps):
    dt = 0.1
    ssm = StateSpaceModel2Dim(dt)
    xrec, yrec = ssm.simulation(dt, numSteps)
    if printDebug is True:
        print('x: ',xrec)
        print('y: ',yrec)

    plt.subplot(211)
    timestamp=np.array(range(0,numSteps))
    plt.plot(timestamp,xrec[:,0])
    plt.scatter(timestamp,yrec[:,0],s=1)
    plt.title("test_state_space_model_2Dim")
    plt.ylabel("X")
    plt.subplot(212)
    plt.plot(timestamp,xrec[:,1])
    plt.scatter(timestamp,yrec[:,1],s=1)
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':

    test_state_space_model_2Dim(500)
