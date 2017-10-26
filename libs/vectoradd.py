'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-05-26
'''
import numpy as np
from timeit import default_timer as timer
from libs.cVectorAdd import cVectorAdd_f


def numpyVectorAdd(a, b):
    return a + b

def cycleVectorAdd(a, b, c):
    for i in range(a.size):
        c[i] = a[i] + b[i]

def main():
    N = int(32e6)

    A = np.ones(N)
    B = np.ones(N)
    C = np.zeros(N, dtype=np.float64)

    start = timer()
    C = numpyVectorAdd(A, B)
    vectorAdd_time = timer() - start

    print("VectorAdd took {0:f} seconds".format(vectorAdd_time))

    C = np.zeros(N, dtype=np.float64)

    start = timer()
    C = cVectorAdd_f(A, B)
    vectorAdd_time = timer() - start

    print("cythonVectorAdd took {0:f} seconds".format(vectorAdd_time))

    C = np.zeros(N, dtype=np.float64)

    start = timer()
    cycleVectorAdd(A, B, C)
    vectorAdd_time = timer() - start

    print("VectorAdd took {0:f} seconds".format(vectorAdd_time))
if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    main()
