#!/usr/bin/env python
import numpy as np
from numpy import (pi, arange, zeros, ones, sin, cos,
                   exp, log, sqrt, where, interp, linspace)
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from feff.libs.dir_and_file_operations import get_folder_name, runningScriptDir
plt.rcParams.update({'font.size': 14})
WindowName = 'kaiser'



def fourierTransform(filePath =
    r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/result_1mono1SR2VasVga2_6.txt'):
    # expData = r'C:\Users\melikhov\Desktop\GAMNAS_CWT\450.chik'
    expData = os.path.join(get_folder_name(runningScriptDir), 'data', '450.chik')
    f2 = open(expData)
    file2 = np.loadtxt(f2, skiprows=37)
    kex2 = file2[:, 0]
    chiex2 = file2[:, 1]

    expData2 = os.path.join(get_folder_name(runningScriptDir), 'data', '350.chik')
    f3 = open(expData2)
    file3 = np.loadtxt(f3, skiprows=37)
    kex3 = file3[:, 0]
    chiex3 = file3[:, 1]

    f = open(filePath)
    title = os.path.basename(os.path.splitext(filePath)[0])
    expDataLegend = os.path.basename(os.path.splitext(expData)[0])
    expDataLegend2 = os.path.basename(os.path.splitext(expData2)[0])
    file = np.loadtxt(f, skiprows=1)
    kex = file[:, 0]
    chiex = file[:, 1]
    a = xftf(kex, chiex)
    b = xftf(kex2, chiex2)
    c = xftf(kex3, chiex3)

    arrayFTout = np.zeros((len(a[0]),2))
    arrayFTout[:,0] = a[0]
    arrayFTout[:,1] = a[2]
    saveFTdir = os.path.dirname(os.path.abspath(filePath))
    makeFTdir = os.path.join(saveFTdir, 'FT')
    if not os.path.isdir(makeFTdir):
        os.mkdir(makeFTdir)
        filenameFT = os.path.join(makeFTdir, title + '.dat')
        if not os.path.exists(filenameFT):
            fOut = np.savetxt(filenameFT, arrayFTout,delimiter='\t',
                              header=('R (angstrom)\tFT(R)'), comments='#window function {0}\n'.format(WindowName))
        else:
            pass
    else:
        filenameFT = os.path.join(makeFTdir, title + '.dat')
        if not os.path.exists(filenameFT):
            fOut = np.savetxt(filenameFT, arrayFTout, delimiter='\t',
                              header=('R (angstrom)\tFT(R)'), comments='#window function {0}\n'.format(WindowName))
        else:
            pass

    maxAxisX = max(max(a[2]), max(b[2]))
    xMax = (maxAxisX+0.05*maxAxisX)
    # plt.switch_backend('TkAgg')
    plt.figure(figsize=(12, 8))
    plt.plot(b[0], b[2], lw=2, c='k', label='exp {0}'.format(expDataLegend))
    plt.plot(c[0], c[2], lw=2, c='r', label='exp {0}'.format(expDataLegend2))
    plt.plot(a[0], a[2], lw=2, ls='--', label='theory')
    plt.plot(a[0], a[2] * 0.84, lw=1, ls='-', label='theory * S02 = 0.84')
    plt.plot(a[0], a[2] * 0.81, lw=1, c='m', ls='-', label='theory * S02 = 0.81')
    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.1, right=0.99)
    plt.xlabel(r'$R \, (\AA) $')
    plt.ylabel(r'$FT(R)$')
    plt.title(title)
    # plt.text(3, 0.18, '${}$ $window$'.format(WindowName), fontdict={'size': 24})
    plt.text(3, 0.18, '{} $model$'.format(title), fontdict={'size': 20})
    # plt.text(4.8, 0.16,'$Kaiser-bessel$ $window$',fontdict ={'size': 24})
    plt.axis([1, 6, 0, xMax])
    plt.grid(True)
    plt.legend()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # DPI = plt.figure.get_dpi()
    # plt.figure.set_size_inches(1920.0 / DPI, 1080.0 / DPI)
    # figname = os.path.join(
    #                         os.path.dirname(filePath),
    #                        'FT(R)_' + os.path.split(os.path.split(os.path.dirname(filePath))[0])[1] + '.png'
    #                         )
    # plt.savefig(figname,dpi=200)
    # plt.draw()
    plt.show()

def ftwindow(n, windname=WindowName):
    k_wind = np.zeros(len(n))
    dx = 1

    kmin = 3.9
    kmax = 12
    eps = 0.01

    # kmin_ind = np.where(n == kmin)[0][0]
    # kmin_ind1 = np.where(n == kmin + dx)[0][0]
    # kmax_ind = np.where(n == kmax)[0][0]
    # kmax_ind1 = np.where(n == kmax - dx)[0][0]

    kmin_ind = np.where(np.abs(n - kmin) < eps)[0][0]
    kmin_ind1 = np.where(np.abs(n - (kmin + dx)) < eps)[0][0]
    kmax_ind = np.where(np.abs(n - kmax) < eps)[0][0]
    kmax_ind1 = np.where(np.abs(n - (kmax - dx)) < eps)[0][0]



    wind_point = len(n[kmin_ind:kmin_ind1 + 1])
    if windname == 'kaiser':
        init_window = np.kaiser(2 * wind_point, 3)
    else:
        init_window = np.hanning(2 * wind_point)

    max1 = np.where(init_window == max(init_window))[0][0]
    max2 = np.where(init_window == max(init_window))[0][1]

    dx1 = [init_window[0:max2]][0]
    dx2 = [init_window[max2:]][0]

    win_shift = int(len(dx1) / 2)

    k_wind[kmin_ind - win_shift:kmin_ind1 - win_shift + 1] = dx1
    k_wind[kmax_ind1 + win_shift:kmax_ind + win_shift + 1] = dx2
    k_wind[kmin_ind1 - win_shift:kmax_ind1 + win_shift] = max(init_window)

    return k_wind
# def ftwindow(x):
#     """
#     Kaiser-Bessel window function
#     :param x: array of k-value
#     :return: array of window value
#     """
#     dx=1
#     dx1 = dx
#     dx2 = dx1
#     xmin = 3.9#min(x)
#     xmax = 12#max(x)
#     xstep = (x[-1] - x[0]) / (len(x)-1)
#     xeps  = 1.e-4 * xstep
#     x1 = max(min(x), xmin - dx1 / 2.0)
#     x2 = xmin + dx1 / 2.0  + xeps
#     x3 = xmax - dx2 / 2.0  - xeps
#     x4 = min(max(x), xmax + dx2 / 2.0)
#     def asint(val):
#         return int((val+xeps)/xstep)
#     i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
#     i1, i2 = max(0, i1), max(0, i2)
#     i3, i4 = min(len(x)-1, i3), min(len(x)-1, i4)
#     if i2 == i1: i1 = max(0, i2-1)
#     if i4 == i3: i3 = max(i2, i4-1)
#     x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
#     if x1 == x2: x2 = x2+xeps
#     if x3 == x4: x4 = x4+xeps
#
#     # initial window
#     fwin =  zeros(len(x))
#     cen  = (x4+x1)/2
#     wid  = (x4-x1)/2
#     arg  = 1 - (x-cen)**2 / (wid**2)
#     arg[where(arg<0)] = 0
#
#     # fwin = bessel_i0(dx* sqrt(arg)) / bessel_i0(dx)
#     # fwin[where(x<=x1)] = 0
#     # fwin[where(x>=x4)] = 0
#
#     scale = max(1.e-10, bessel_i0(dx)-1)
#     fwin = (bessel_i0(dx*100*np.pi * sqrt(arg)) - 1) / scale
#     return fwin

# win = ftwindow(kex)


def xftf(k, chi):
    win = ftwindow(k)
    rmax_out = 10
    kstep=np.round(1000.*(k[1]-k[0]))/1000.0
    # kstep=0.05
    nfft = 2048
    # k = k
    # chi = chi
    cchi, win  = xftf_prep(k, chi)
    out = xftf_fast(cchi*win)
    rstep = pi/(kstep*nfft)
    irmax = min(nfft/2, int(1.01 + rmax_out/rstep))

    r   = rstep * arange(irmax)
    mag = sqrt(out.real**2 + out.imag**2)
    kwin =  win[:len(chi)]
    r    =  r[:irmax]
    chir =  out[:irmax]
    chir_mag =  mag[:irmax]
    chir_re  =  out.real[:irmax]
    chir_im  =  out.imag[:irmax]
    return r, chir, chir_mag, chir_re, chir_im

def xftf_prep(k, chi):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the
    ft window.

    Returns weighted chi, window function which can easily be multiplied
    and used in xftf_fast.
    """

    # kmin = 0
    kmax = 20
    kweight=1
    dk=1
    dk2 = dk
    # nfft = 2048
    # kstep = 0.05
    kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    npts = int(1.01 + max(k)/kstep)
    k_max = max(max(k), kmax+dk2)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    chi_ = interp(k_, k, chi)
    win  = ftwindow(np.asarray(np.asarray(k_)))

    return ((chi_[:npts] *k_[:npts]**kweight), win[:npts])

def xftf_fast(chi, nfft=2048, kstep=0.05):

    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chi)] = chi
    return (kstep / sqrt(pi)) * fft(cchi)[:int(nfft/2)]

if __name__=='__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("result_chi", '*.txt')],
                                           initialdir=r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001')

    fourierTransform(filePath=file_path)


