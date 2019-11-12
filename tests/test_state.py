import numpy as np
from matchernet_py_001.state import StatePlain, StateMuSigma, StateMuSigmaDiag

if __name__ == '__main__':
    print("===Testing StatePlain===")
    s = StatePlain(4)
    s.data["mu"][0]=np.array([1,2,3,4],dtype=np.float32)
    print("mu",s.data["mu"])

    print("===Testing StateMuSigma===")
    s = StateMuSigma(4)
    s.data["mu"][0]=np.array([1,2,3,4],dtype=np.float32)
    s.data["Sigma"]=np.eye(4,dtype=np.float32)
    print("mu", s.data["mu"])
    print("Sigma", s.data["Sigma"])

    print("===Testing StateMuSigmaDiag===")
    s = StateMuSigmaDiag(4)
    s.data["mu"][0]=np.array([1,2,3,4],dtype=np.float32)
    s.data["sigma"][0]=np.array([5,5,5,5],dtype=np.float32)
    print("mu",s.data["mu"])
    print("sigma",s.data["sigma"])