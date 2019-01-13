import numpy as np
import utils
from utils import print1, print2, print3, print4, print5
import state
import matchernet


class Observer(matchernet.Bundle):
    '''Observer works as a Bundle that provides vector data at each time step of the MatcherNet simulation.

    Usage:
    Construct an observer by

    >> o1 = Observer("Observer1", buffer)

    where buffer is an numpy.ndarray of shape (length, dim).
    Then, call it for each step by

    >> result = o1( dummy_input )

    and you get the vector data of the current time-stamp

     results.state.data["mu"]

    with corresponding observation error covariance matrix

     results.state.data["Sigma"].

    Note:
    When the vector data in buffer included  NaN  entries, they are regarded as missing entries and the Observer outpus a zero vector  mu  with covariance matrix  cov  of large eigen values. (See the function  missing_handler001()  for a default setting to construct the corresponding output. )
    '''
    def __init__(self, name, buff):
        self.name = name
        self.buffer = buff
        self.counter = -1
        self.length = buff.shape[0]
        self.dim = buff.shape[1]
        self.state = state.StateMuSigma(self.dim)
        self.obs_noise_covariance = 1000 * np.eye(self.dim,dtype=np.float32)
        # self.missing_handler = missing_handler001
        self.missing_handler = 0 # a dummy value
            # default setting of missing value handler function
        self.set_results()
            # set the first value with large obs_noise_covariance
            # for an initial value
        super(Observer,self).__init__(self.name, self.state)

    def __call__(self, inputs):
        """ The main routine that is called from brica.
        """
        #for key in inputs: # key is one of the matcher names
        #    if inputs[key] is not None:
        #        self.accept_feedback(inputs[key]) # Doing nothing

        print2("=== In Bundle {}".format(self.name))
        self.count_up()
        self.set_results()
        return self.results

    def count_up(self):
        self.counter = (self.counter + 1) % self.length

    def set_buffer(self, buff):
        self.buffer = buff
        self.counter = -1
        self.length = buff.shape[0]
        self.dim = buff.shape[1]

    def get_buffer(self):
        return self.buffer

    def get_state(self):
        b = self.get_buffer()
        z = b[self.counter].copy()
        return [z] # returning [[1,2,3,...]] rather than [1,2,3,...]

    def set_results(self):
        q = self.get_state()
        mu, Sigma = missing_handler001( np.array(q,np.float32), self.obs_noise_covariance )
        self.state.data["mu"]=mu
        self.state.data["Sigma"]=Sigma
        self.state.data["time_stamp"]=self.counter
        self.results = {"state": self.state}
          # === Note: We may regard  "time_stamp"  as a real time rather than a counter in a future version.

    def print_state(self):
        '''Prints the state of the self.'''
        pt = self.ports_to_matchers[0]
        print("self.results[]={c}".format(c=self.results[pt]))

def missing_handler001(mu,Sigma):
    ''' A missing value handler function.
    It receives a vector data  mu  with a default covariance matrix  Sigma, find NaN in the vector  mu, and outputs a modified set of a vector  mu  and a covariance  cov.
    '''
    if np.any(np.isnan(mu)):
        print3("Missing!")
        cov = Sigma * 1000
        mu = np.zeros(mu.shape)
    else:
        cov = Sigma
    return mu, cov

def test_missing_handler001():
    mu = np.array([1,2,np.nan])
    Sigma = np.eye(3)
    mu1, Sigma1 = missing_handler001(mu, Sigma)
    print("mu",mu)
    print("Sigma",Sigma)
    print("mu1",mu1)
    print("Sigma1",Sigma1)



class ObserverMultiple(Observer):
    '''A bundle that provides sequencial data'''
    def __init__(self, name, buff, mul):
        self.name = name
        self.mul = mul
        super().__init__(self.name, buff)

    def get_state(self):
        b = self.get_buffer()
        z = b[self.counter].copy()
        for i in range(1,self.mul):
            j = (self.counter + i ) % self.length
            z = np.concatenate( (z, b[j].copy()) )
        self.state.data["mu"] = z
        return z

def test_observer():
    print("=== Unit test of class Observer ===")
    dim = 4
    buffersize = 5
    x = np.zeros((buffersize,dim),dtype=np.float32)
    for i in range(buffersize):
        x[i][0]=i

    b0 = Observer("b0",x)
    for i in range(buffersize*2):
        b0.count_up()
        print("get_state()=",b0.get_state() )

    print("=== Unit test of class Observer without BriCA ===")
    for i in range(buffersize*2):
        b0(b0.state)
        print("b0.results[state].data[mu]=",b0.results["state"].data["mu"])

    print("=== Unit test of class ObserverMultiple ===")
    mul = 3
    b0 = ObserverMultiple("b0",x,mul)
    for i in range(buffersize*2):
        b0.count_up()
        print("get_state()=",b0.get_state() )

def test_ObserverWithMissing():
    print("=== Unit test of class ObserverWithMissing ===")
    dim = 4
    buffersize = 2
    x = np.zeros((buffersize,dim),dtype=np.float32)
    for i in range(buffersize):
        x[i][0]=i

    x[1][2] = np.nan
    b0 = ObserverWithMissing("b0",x)
    for i in range(buffersize*2):
        b0.count_up()
        b0.set_results()
        print4(b0.state.data)


if __name__ == '__main__':
    utils._print_level = 5
    test_observer()
    test_missing_handler001()
    test_ObserverWithMissing()
