#!/usr/bin/env python
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
plt.rcParams.update({'font.size': 14})
WindowName = 'kaiser'

class GraphElement():
    # base graph elements class
    def __init__(self):
        self.axes = []

class SimpleSpectrum():
    def __init__(self):
        self.k_vector = []
        self.chi_vector = []
        self.r_vector = []
        self.ftr_vector = []
        self.user = ''
        self.label_latex = ''

    def updateInfo(self):
        self.label_latex = f'user: {self.user}'

    def calcFTRtransform(self):
        ftr_out = xftf(self.k_vector, self.chi_vector, user=self.user)
        self.r_vector = ftr_out[0]
        self.ftr_vector = ftr_out[2]

    def plotOneSpectrum_chi_k(self):
        plt.plot(self.k_vector, self.chi_vector, lw=2, label=self.label_latex)
        plt.plot(self.k_vector, ftwindow(self.k_vector, user=self.user), lw=2, label=f'window: {self.user}')

        plt.ylabel('$\chi(k)$', fontsize=20, fontweight='bold')
        plt.xlabel('$k$ $[\AA^{-1}]$', fontsize=20, fontweight='bold')

    def plotOneSpectrum_FTR_r(self):
        plt.plot(self.r_vector, self.ftr_vector, lw=2, label=self.label_latex)
        plt.ylabel('$FT(r)$', fontsize=20, fontweight='bold')
        plt.xlabel('$r$ $[\AA]$', fontsize=20, fontweight='bold')


class CompareUserPresets():
    def __init__(self):
        self.dictOfSpectra = {}
        self.showFigs = True
        self.fig = []
        self.FTR   = GraphElement()
        self.Chi_k = GraphElement()
        self.suptitle_fontsize = 18
        self.graph_title_txt = 'Compare two user xftf presets'


    def addSpectraToDict(self, currentSpectra):
        num = len(self.dictOfSpectra)
        if isinstance(currentSpectra, SimpleSpectrum):
            self.dictOfSpectra[num] = dict({'data' : currentSpectra})

    def setupAxes(self):
        if self.showFigs:
            # create figure with axes:

            pylab.ion()  # Force interactive
            plt.close('all')
            ### for 'Qt4Agg' backend maximize figure
            plt.switch_backend('QT5Agg')

            # plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            self.fig = plt.figure()
            # gs1 = gridspec.GridSpec(1, 2)
            # fig.show()
            # fig.set_tight_layout(True)
            self.figManager = plt.get_current_fig_manager()
            DPI = self.fig.get_dpi()
            self.fig.set_size_inches(800.0 / DPI, 600.0 / DPI)

            gs = gridspec.GridSpec(1, 2)

            self.fig.clf()

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            self.FTR.axes.invert_xaxis()
            self.FTR.axes.set_title('$FT(r)$')
            self.FTR.axes.grid(True)

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            self.Chi_k.axes.invert_xaxis()
            self.Chi_k.axes.set_title('$\chi(k)$')
            self.Chi_k.axes.grid(True)

            self.FTR.axes.set_ylabel('Reletive Intensity (a.u.)', fontsize=16, fontweight='bold')
            self.FTR.axes.set_xlabel('$r$ $[\AA]$', fontsize=16, fontweight='bold')
            self.Chi_k.axes.set_ylabel('Reletive Intensity (a.u.)', fontsize=16, fontweight='bold')
            self.Chi_k.axes.set_xlabel('$k$ $[\AA^{-1}]$', fontsize=16, fontweight='bold')

            # Change the axes border width
            for axis in ['top', 'bottom', 'left', 'right']:
                self.FTR.axes.spines[axis].set_linewidth(2)
                self.Chi_k.axes.spines[axis].set_linewidth(2)

            # plt.subplots_adjust(top=0.85)
            # gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
            self.fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)

            # put window to the second monitor
            # figManager.window.setGeometry(1923, 23, 640, 529)
            self.figManager.window.setGeometry(1920, 20, 1920, 1180)

            plt.show()

            self.fig.suptitle(self.graph_title_txt, fontsize=self.suptitle_fontsize, fontweight='normal')

            # put window to the second monitor
            # figManager.window.setGeometry(1923, 23, 640, 529)
            # self.figManager.window.setGeometry(780, 20, 800, 600)
            self.figManager.window.setWindowTitle('Compare xftf')
            self.figManager.window.showMinimized()


            # save to the PNG file:
            # out_file_name = '%s_' % (case) + "%05d.png" % (numOfIter)
            # fig.savefig(os.path.join(out_dir, out_file_name))

    def setAxesLimits_FTR(self):
        plt.axis([1, 5, plt.ylim()[0], plt.ylim()[1]])

    def setAxesLimits_Chi(self):
        pass
        # plt.axis([3.5, 13, plt.ylim()[0], plt.ylim()[1]])
        # plt.axis([3.5, 13, -1, 1])

    def plotSpectra_FTR_r_All(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_FTR_r()
        plt.title(f'FTR transformation')
        plt.legend()
        plt.show()

    def plotSpectra_chi_k_All(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_chi_k()
        plt.title('$\chi(k)$ transformation')
        plt.legend()
        plt.show()

    def updatePlot(self, saveFigs=False):

        if self.showFigs:

            self.fig.clf()
            gs = gridspec.GridSpec(1, 2)

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            plt.axes(self.FTR.axes)
            self.plotSpectra_FTR_r_All()
            self.FTR.axes.grid(True)
            self.setAxesLimits_FTR()

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.plotSpectra_chi_k_All()
            # self.Chi_k.axes.invert_xaxis()
            self.Chi_k.axes.grid(True)
            self.setAxesLimits_Chi()

            # The formatting of tick labels is controlled by a Formatter object,
            # which assuming you haven't done anything fancy will be a ScalerFormatterby default.
            # This formatter will use a constant shift if the fractional change of the values visible is very small.
            # To avoid this, simply turn it off:
            self.FTR.axes.get_xaxis().get_major_formatter().set_scientific(False)
            self.Chi_k.axes.get_xaxis().get_major_formatter().set_scientific(False)

            self.FTR.axes.get_xaxis().get_major_formatter().set_useOffset(False)
            self.Chi_k.axes.get_xaxis().get_major_formatter().set_useOffset(False)

            # plt.subplots_adjust(top=0.85)
            # gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
            self.fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)

            # put window to the second monitor
            # figManager.window.setGeometry(1923, 23, 640, 529)
            self.figManager.window.setGeometry(1920, 20, 1920, 1180)

            # plt.show()
            plt.draw()
            self.fig.suptitle(self.graph_title_txt, fontsize=self.suptitle_fontsize, fontweight='normal')

            # put window to the second monitor
            # figManager.window.setGeometry(1923, 23, 640, 529)
            # self.figManager.window.setGeometry(780, 20, 800, 600)



            self.figManager.window.setWindowTitle('Search the minimum and find the coordinates')
            # self.figManager.window.showMinimized()

        # if saveFigs and self.showFigs:
        #     # save to the PNG file:
        #     # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
        #     modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        #     if self.scale_theory_factor_FTR == 1:
        #         out_file_name =  snapNumberStr + '_R={0:1.4}.png'.format(self.minimum.Rtot)
        #     else:
        #         out_file_name =  snapNumberStr + '_So={1:1.3f}_R={0:1.4}.png'.format(self.minimum.Rtot,
        #                                                                             self.scale_theory_factor_FTR)
        #     self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))

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

def ftwindow(n,  user='PK'):
    k_wind = np.zeros(len(n))
    dx = 1

    kmin = 3.9
    kmax = 12
    windname = WindowName

    if user == 'PK':
        windname = 'kaiser'
        kmin = 3.9
        kmax = 12
        dx = 1
    elif user == 'PK_test':
        windname = 'kaiser'
        kmin = 3.9
        kmax = 12
        dx = 3
    elif user == 'ID_test':
        windname = 'hanning'
        kmin = 3.9
        kmax = 12
        dx = 3
    elif user == 'ID':
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
    elif windname == 'blackman':
        init_window = np.blackman(2 * wind_point)
    elif windname == 'hamming':
        init_window = np.hamming(2 * wind_point)
    elif windname == 'chebwin':
        init_window = signal.chebwin(2 * wind_point, at=100)
    elif windname == 'bartlett':
        init_window = np.bartlett(2 * wind_point)

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


def xftf(k, chi, user='PK'):
    rmax_out = 10
    kstep=np.round(1000.*(k[1]-k[0]))/1000.0
    nfft = 2048
    cchi, win  = xftf_prep(k, chi, user)
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


    dk2 = dk
    # nfft = 2048
    # kstep = 0.05
    kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    npts = int(1.01 + max(k)/kstep)
    k_max = max(max(k), kmax+dk2)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    chi_ = interp(k_, k, chi)
    win  = ftwindow(np.asarray(np.asarray(k_)), user=user)

    return ((chi_[:npts] *k_[:npts]**kweight), win[:npts])



def xftf_fast(chi, nfft=2048, kstep=0.05):

    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chi)] = chi
    return (kstep / sqrt(pi)) * fft(cchi)[:int(nfft/2)]

if __name__=='__main__':
    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilename(filetypes=[("result_chi", '*.txt')],
    #                                        initialdir=r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001')
    #
    # fourierTransform(filePath=file_path)
    k = np.arange(0.5, 15, 0.05)
    chi = np.sin(7 * k) + np.sin(5 * k + np.pi * 0.3)
    obj = CompareUserPresets()


    a = SimpleSpectrum()
    a.k_vector = k
    a.chi_vector = chi
    a.user = 'ID'
    a.calcFTRtransform()
    a.updateInfo()
    obj.addSpectraToDict(a)

    # b = SimpleSpectrum()
    # b.k_vector = k
    # b.chi_vector = chi
    # b.user = 'ID'
    # b.calcFTRtransform()
    # b.updateInfo()
    # obj.addSpectraToDict(b)

    c = SimpleSpectrum()
    c.k_vector = k
    c.chi_vector = chi
    c.user = 'ID_test'
    c.calcFTRtransform()
    c.updateInfo()
    obj.addSpectraToDict(c)

    obj.setupAxes()
    obj.updatePlot()
    plt.show()



