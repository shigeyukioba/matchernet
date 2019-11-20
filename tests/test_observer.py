import numpy as np
from matchernet_py_001.observer import Observer, ObserverMultiple, missing_handler001

def test_missing_handler001():
    mu = np.array([1,2,np.nan])
    Sigma = np.eye(3)
    mu1, Sigma1 = missing_handler001(mu, Sigma)
    print("mu",mu)
    print("Sigma",Sigma)
    print("mu1",mu1)
    print("Sigma1",Sigma1)

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

if __name__ == '__main__':
    test_observer()
    test_missing_handler001()
