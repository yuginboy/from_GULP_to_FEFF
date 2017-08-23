#!/usr/bin/env python
import numba
import timeit
import numpy as np
from numpy import (pi, arange, zeros, ones, sin, cos,
                   exp, log, sqrt, where, interp, linspace)
from scipy.fftpack import fft, ifft
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import filedialog
import os
from feff.libs.dir_and_file_operations import get_folder_name, runningScriptDir
# plt.rcParams.update({'font.size': 14})
# WindowName = 'kaiser'



@numba.jit('f8[:](f8[:], int32)', cache=True)
def ftwindow_main_numba(n,  user):
    k_wind = np.zeros(len(n))

    eps = 0.01
    kmin = 3.5
    kmax = 11.5
    dx = 2
    # kmin_ind = np.where(n == kmin)[0][0]
    # kmin_ind1 = np.where(n == kmin + dx)[0][0]
    # kmax_ind = np.where(n == kmax)[0][0]
    # kmax_ind1 = np.where(n == kmax - dx)[0][0]

    kmin_ind = np.where(np.abs(n - kmin) < eps)[0][0]
    kmin_ind1 = np.where(np.abs(n - (kmin + dx)) < eps)[0][0]
    kmax_ind = np.where(np.abs(n - kmax) < eps)[0][0]
    kmax_ind1 = np.where(np.abs(n - (kmax - dx)) < eps)[0][0]



    wind_point = len(n[kmin_ind:kmin_ind1 + 1])

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

def ftwindow_main_raw(n,  user):
    k_wind = np.zeros(len(n))
    dx = 1

    kmin = 3.9
    kmax = 12
    windname = 'hanning'

    if user == 1:
        #'PK'
        windname = 'kaiser'
        kmin = 3.9
        kmax = 12
        dx = 1
    elif user == 2:
        #'PK_test'
        windname = 'kaiser'
        kmin = 3.9
        kmax = 12
        dx = 3
    elif user == 3:
        #'ID_test'
        windname = 'hanning'
        kmin = 3.9
        kmax = 12
        dx = 3
    elif user == 4:
        #'ID':
        windname = 'hanning'
        kmin = 3.5
        kmax = 11.5
        dx = 2
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
    elif windname == 'hanning':
        init_window = np.hanning(2 * wind_point)
    # elif windname == 'blackman':
    #     init_window = np.blackman(2 * wind_point)
    # elif windname == 'hamming':
    #     init_window = np.hamming(2 * wind_point)
    # elif windname == 'chebwin':
    #     init_window = signal.chebwin(2 * wind_point, at=100)
    # elif windname == 'bartlett':
    #     init_window = np.bartlett(2 * wind_point)

    max1 = np.where(init_window == max(init_window))[0][0]
    max2 = np.where(init_window == max(init_window))[0][1]

    dx1 = [init_window[0:max2]][0]
    dx2 = [init_window[max2:]][0]

    win_shift = int(len(dx1) / 2)

    k_wind[kmin_ind - win_shift:kmin_ind1 - win_shift + 1] = dx1
    k_wind[kmax_ind1 + win_shift:kmax_ind + win_shift + 1] = dx2
    k_wind[kmin_ind1 - win_shift:kmax_ind1 + win_shift] = max(init_window)

    return k_wind

# @numba.jit(cache=True)
def ftwindow(n,  user='PK'):

    i = 1
    if user == 'PK':
        i = 1
    elif user == 'PK_test':
        i = 2
    elif user == 'ID_test':
        i = 3
    elif user == 'ID':
        i = 4

    k_wind = ftwindow_main_numba(n,  user=i)

    return k_wind

# @numba.jit(cache=True)
def xftf(k, chi, user='PK'):
    rmax_out = 10
    kstep=np.round(1000.*(k[1]-k[0]))/1000.0
    nfft = 2048
    cchi, win  = xftf_prep(k, chi, user)
    out = xftf_fast(cchi*win, nfft=nfft, kstep=kstep)
    rstep = pi/(kstep*nfft)
    irmax = min(nfft/2, int(1.01 + rmax_out/rstep))

    r   = rstep * np.arange(irmax)
    mag = np.sqrt(out.real**2 + out.imag**2)
    kwin =  win[:len(chi)]
    r    =  r[:irmax]
    chir =  out[:irmax]
    chir_mag =  mag[:irmax]
    chir_re  =  out.real[:irmax]
    chir_im  =  out.imag[:irmax]
    return r, chir, chir_mag, chir_re, chir_im

# @numba.jit('Tuple((f8[:], f8[:]))(f8[:], f8[:], int32, int32, int32)', cache=True)
def xftf_prep_numba(k, chi, kmax=20, kweight=1, dk=3):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the
    ft window.

    Returns weighted chi, window function which can easily be multiplied
    and used in xftf_fast.
    """
    kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    npts = int(1.01 + max(k)/kstep)
    k_max = max(max(k), kmax+dk)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    chi_ = interp(k_, k, chi)

    res_first = chi_[:npts] *k_[:npts]**kweight
    return res_first, k_

# @numba.jit(cache=True)
def xftf_prep(k, chi, user='PK'):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the
    ft window.

    Returns weighted chi, window function which can easily be multiplied
    and used in xftf_fast.
    """

    # kmin = 0
    kmax = 20
    kweight=1
    dk=3
    # if user == 'PK':
    #     dk = 1
    # elif user == 'ID':
    #     dk = 2
    # elif user == 'ID_test':
    #     dk = 1
    # elif user == 'PK_test':
    #     dk = 1


    # dk2 = dk
    # # nfft = 2048
    # # kstep = 0.05
    kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    npts = int(1.01 + np.max(k)/kstep)
    # k_max = max(max(k), kmax+dk2)
    # k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    # chi_ = interp(k_, k, chi)

    res1, k_ = xftf_prep_numba(k, chi, kmax=kmax, kweight=kweight, dk=dk)
    win  = ftwindow(np.asarray(np.asarray(k_)), user=user)

    return ((res1), win[:npts])


# @numba.jit('c8[:](f8[:], int32, f8)')
def xftf_fast(chi, nfft, kstep):
    cchi = np.zeros(nfft)
    cchi[0:len(chi)] = chi
    res_fft = np.fft.fft(cchi, nfft)
    res = np.empty_like(res_fft)
    res = (kstep / sqrt(pi)) * res_fft[:int(nfft/2)]
    return res

def xftf_fast_raw(chi, nfft, kstep):
    cchi = np.zeros(nfft)
    cchi[0:len(chi)] = chi
    res_fft = np.fft.fft(cchi, nfft)
    return (kstep / sqrt(pi)) * res_fft[:int(nfft/2)]

if __name__=='__main__':

    k = np.arange(0.5, 15, 0.05)
    chi = np.sin(5 * k + np.pi * 0.3)
    print(timeit.timeit('xftf_fast(chi, 2048, 0.05)', number=10000, globals=globals()))
    print(timeit.timeit('xftf_fast_raw(chi, 2048, 0.05)', number=10000, globals=globals()))

    print(timeit.timeit('ftwindow_main_raw(k, 4)', number=10000, globals=globals()))
    print(timeit.timeit('ftwindow_main_numba(k, 4)', number=10000, globals=globals()))


    # chi = np.sin(7 * k) + np.sin(5 * k + np.pi * 0.3)




