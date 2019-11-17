import numpy as np
from matchernet_py_001.fn import LinearFn, LinearFnXU

def test_LinearFn():
    print("=== testLinearFn() ===")
    matA = np.array([[5,1,1],[1,2,3]],dtype=np.float32)
    x = np.array([[3,1]],dtype=np.float32)
    y = np.dot(x,matA)
    f = LinearFn(3,2)
    f.params["A"] = matA
    yout = f.value(x)
    print("matA:",matA)
    print("x:",x)
    print("y:",y)
    print("f(x):",yout)
    print("f_x(x):",f.x(x))
    print("f_xx(x):",f.xx(x))

def test_LinearFnXU():
    print("=== testLinearFnXU ===")
    matA = np.array([[5,1,1],[1,2,3]],dtype=np.float32)
    matB = np.array([[1,0,0]],dtype=np.float32)
    x = np.array([[3,1]],dtype=np.float32)
    u = np.array([[1]])
    y = np.dot(x,matA) + np.dot(u,matB)
    f = LinearFnXU(3,1,2)
    f.params["A"] = matA
    f.params["B"] = matB
    yout = f.value(x,u)
    print("matA:",matA)
    print("matB:",matB)
    print("x:",x)
    print("u:",u)
    print("y:",y)
    print("f(x,u):",yout)
    print("f_x(x,u):",f.x(x,u))
    print("f_xx(x,u):",f.xx(x,u))
    print("f_u(x,u):",f.u(x,u))
    print("f_xu(x,u):",f.xu(x,u))
    print("f_uu(x,u):",f.uu(x,u))

def test_all():
    test_LinearFn()
    test_LinearFnXU()

if __name__ == '__main__':
    test_all()
