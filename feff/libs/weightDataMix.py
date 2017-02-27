import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from feff.libs.feff_processing import xftf
from scipy.optimize import minimize
import os
from feff.libs.dir_and_file_operations import get_folder_name, runningScriptDir
from feff.libs.load_chi_data_file import load_and_apply_xftf, load_chi_data

root = tk.Tk()
root.withdraw()
# file_path1 = filedialog.askopenfilename(filetypes = [("FT(R) file",'*.dat')],
#                                        initialdir=r'C:\wien2k_ver14\VirtualBoxShare\results_Pasha')
# file_path2 = filedialog.askopenfilename(filetypes = [("FT(R) file",'*.dat')],
#                                        initialdir=r'C:\wien2k_ver14\VirtualBoxShare\results_Pasha')
file_path1 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000002_00001.dat'
file_path2 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000131_00130.dat'
# file_path1 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/350.txt'
# file_path2 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/350.txt'
exp_data_path2 = os.path.join(get_folder_name(runningScriptDir), 'data', '350.chik')

k=1

fr1 = load_and_apply_xftf(file_path1)
fr2 = load_and_apply_xftf(file_path2)
fr3 = load_and_apply_xftf(exp_data_path2)
rex  = fr3[0]
ftex = fr3[2]


r1  = fr1[0]
ft1 = fr1[2]

r2  = fr2[0]
ft2 = fr2[2]

#
# plt.plot(r1, ft1,c='r', lw=2, label='model for 450')
# plt.plot(r2, ft2,c='g', lw=2, label='model with 1VGa')
# plt.plot(rex, ftex,c='k', lw = 2, label='exp. data 350')
# plt.legend()
# plt.show()

index1 = (np.abs(r1 - 1)).argmin()
index2 = (np.abs(r1 - 5)).argmin()

yaxis = max(max(ftex),max(ft1),max(ft2))
yMax = (yaxis+0.01*yaxis)
#maassive of values for function weighting like f(x1)*1-x and f(x2)*x
step = np.arange(0.001, 1, 0.001)

def rFactor(tPoint, exPoint):
    r_factor = (np.sum(tPoint)/np.sum(exPoint))#/len(exPoint)
    return r_factor

def deltaToPower(weigtedFunc, expdataPoint):
    # deltaPow = np.zeros(len(weigtedFunc))
    # expPow = np.zeros(len(weigtedFunc))
    deltaPow = np.power(expdataPoint - weigtedFunc, 2)
    expPow = np.power(expdataPoint, 2)
    return deltaPow[index1: index2], expPow[index1: index2]

def weightMix(x, fun1, fun2):
    wft = fun1*(1-x) + fun2*x
    inputForR = deltaToPower(wft, ftex)
    r_val = rFactor(inputForR[0], inputForR[1])
    return r_val, x

def calc(stepArray, x1, x2 ):
    rvalue = np.zeros((len(stepArray), 2))
    for i in range(len(stepArray)):
        rvalue [i, 0], rvalue[i, 1] = weightMix(stepArray[i], x1, x2)
    return rvalue

def func(x):
    out = weightMix(x, ft1, ft2)
    return out[0]

fullArra = calc(step, ft1, ft2)
minR = np.amin(fullArra, axis=0)
aminX=np.argmin(fullArra, axis=0)[0]

res = minimize(func, x0=0, options={'gtol': 1e-6, 'disp': True})
print(res)

print('R-factor = {0}, x = {1}'.format(round(minR[0],4), round(fullArra[aminX,1],4)))
# for i in range(len(step)):
#     plt.plot(r1, ft1*(1-fullArra[i,1]) + ft2*fullArra[i,1])

# plt.plot(fullArra[:, 1], fullArra[:,0])
# plt.yscale('log')

plt.plot(r1, ft1,c='r', lw=2, label='model for 450')
plt.plot(r2, ft2,c='g', lw=2, label='model with 1VGa')
plt.plot(rex, ftex,c='k', lw = 2, label='exp. data 350')
plt.plot(r1, ft1*(1-fullArra[aminX,1]) + ft2*fullArra[aminX,1], ls='-', marker='o', c='c', label='best fit')
plt.text(3, 0.18, '$R$-$factor$ = {0}, $x$ = {1}'.format(round(minR[0],4), round(fullArra[aminX,1],4)), fontdict={'size': 20})
plt.axis([1, 5, 0, yMax])
plt.legend()
plt.show()




