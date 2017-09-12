from joblib import Parallel, delayed
import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import time

def f(name):
    print ('hello', name)
    time.sleep(2)
    print ('hello', name)

class A():
    _idx = 0
    # def __init__(self):
    #     t = 0

    def get_idx(self):
        return A._idx
    def set_idx(self, i):
        A._idx = i

# def runInParallel(*fns):
#   proc = []
#   for fn in fns:
#     p = Process(target=fn)
#     p.start()
#     proc.append(p)
#   for p in proc:
#     p.join()


if __name__ == '__main__':
    # # p = Process(target=f, args=('bob',))
    # # p.start()
    # # p.join()
    # num_cores = multiprocessing.cpu_count()
    # # for i in range(3):
    # #     print(f(i))
    # ((i) for i in range(3))
    # print( Parallel(n_jobs=3)(delayed(f)(i) for i in range(3)) )
    # # runInParallel(f('1'), f('2'),f('3'))

    #
    #
    # plt.switch_backend('QT4Agg') #default on my system
    # print('Backend: {}'.format(plt.get_backend()))
    #
    # fig = plt.figure()
    # ax = fig.add_axes([0,0, 1,1])
    # ax.axis([0,10, 0,10])
    # ax.plot(5, 5, 'ro')
    #
    # mng = plt._pylab_helpers.Gcf.figs.get(fig.number, None)
    #
    # mng.window.showMaximized() #maximize the figure
    # time.sleep(3)
    # mng.window.showMinimized() #minimize the figure
    # time.sleep(3)
    # mng.window.showNormal() #normal figure
    # time.sleep(3)
    # mng.window.hide() #hide the figure
    # time.sleep(3)
    # fig.show() #show the previously hidden figure
    #
    # ax.plot(6,6, 'bo') #just to check that everything is ok
    # plt.show()

    a = A()
    b = A()
    print(a.get_idx())
    print(b.get_idx())
    a.set_idx(5)
    print(a.get_idx())
    print(b.get_idx())

