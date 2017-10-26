'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-10-13
'''

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

g_e = 2.0023 # G-factor Lande
mu_Bohr = 927.4e-26 # J/T
Navagadro = 6.02214e23 #1/mol
k_b = 1.38065e-23 #J/K

def f_Langevin(x=1):
    #  The Langevin function

    # x=g*mu_bohr*J*B/(k*T)
    return ( 1/np.tanh(x) - 1/x )
# print(f_Langevin(x=1000))
# print('---')

def f_Brillouin(x, J=2.5):
    #  The Brillouin function
    # x=g*mu_bohr*J*B/(k*T)
    b = 0.5/J
    a = (2*J+1)*b
    return ( a/np.tanh(a*x) - b/np.tanh(b*x) )
# print(f_Brillouin(x=1000))
# print('---')

def f_PM(x, n, J=2.5, g_factor=g_e):
    # n = N/V
    return n*1e27*J*g_factor*mu_Bohr*f_Brillouin(x, J)

#  fit to a global function
# def func(x, A, B, x0, sigma):
#     return A+B*np.tanh((x-x0)/sigma)

def f_diff_PM_for_2_T (B, n, J=2.5, T1=2, T2=5, g_factor=g_e):
    # diff of 2 f_PM function for substrate FM component
    x1 = (g_factor*J*mu_Bohr*B)/k_b/T1
    x2 = (g_factor*J*mu_Bohr*B)/k_b/T2
    return f_PM(x1, n, J=J, g_factor=g_factor) - f_PM(x2, n, J=J, g_factor=g_factor)

def f_PM_with_T (B, n, J=2.5, T=50, g_factor=g_e):
    # PM component by using a Brillouin function
    x = (g_factor*J*mu_Bohr*B)/k_b/T
    return f_PM(x, n, J=J, g_factor=g_factor)

def f_SPM_with_T (B, n, J=2.5, T=50, g_factor=g_e):
    # PM component by using a Langevin function
    x = (g_factor*J*mu_Bohr*B)/k_b/T
    return n*1e27*J*g_factor*mu_Bohr*f_Langevin(x)

def func(x,n):
    return f_PM(x, n)
    # return x+n*x

def return_fit_param(x,y):
    popt, pcov = curve_fit(func, x, y)
    return popt

def linearFunc(x, k, b):
    return k*x + b

if __name__ == '__main__':
    print ('-> you run ',  __file__, ' file in a main mode' )
    # B = np.array((3, 4.5, 5.5, 6))
    J = 2.5
    T = 2
    # x = (g_e*J*mu_Bohr*B)/k_b/T
    x = np.array(( 5.12259078,  5.29071205,  5.45883331,  5.62695458,  5.79507585,  5.96319712,  6.13131838,  6.29943965,  6.46756092,  6.63568218 , 6.80380345,  6.97192472, 7.14004599,  7.30816725,  7.47628852,  7.64440979,  7.81253105,  7.98065232, 8.14877359,  8.31689486,  8.48501612,  8.65313739,  8.82125866,  8.98937992,  9.15750119 , 9.32562246,  9.49374373 , 9.66186499,  9.82998626 , 9.99810753))
    y = np.array((36562.255,  36704.364,  36846.472,  36959.829,  37071.074,  37182.319,  37288.874,  37394.099,  37499.323,  37575.449,  37634.214,  37692.98 ,  37769.696,  37866.358,  37963.021,  38026.567,  38020.08 ,  38013.593,  38025.271,  38126.641,  38228.011,  38328.095,  38347.998,  38367.901,  38387.804,  38459.712,  38539.864,  38620.017,  38658.663,  38680.4 ))
    M = y
    B = x*k_b*T/(g_e*J*mu_Bohr)
    plt.plot(B,func(x, n=return_fit_param(x, y)), 'r-', B,M,'.-', label='classic')
    B = x * k_b * T / (5.82 * mu_Bohr)
    plt.plot(B,func(x, n=return_fit_param(x, y)), 'r-', B,M,'.-', label='5.82')
    B = x * k_b * T / (4.82 * mu_Bohr)
    plt.plot(B, func(x, n=return_fit_param(x, y)), 'r-', B, M, '.-', label='4.82')
    plt.legend()
    plt.draw()
    plt.show()
    popt, pcov = curve_fit(func, x, y)
    print('finished')