'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-05-31
'''
import numpy as np
import scipy as sp
import re
import matplotlib.pyplot as plt
import os

class OUTdataFile():
    '''
    data manipulation with out-data file of GULP MD simulations
    '''
    def __init__(self):
        self.filepath = '/mnt/soliddrive/yugin/1mono1SR2VasVga2_6/out'
        self.kineticEnergy = [] # eV
        self.potentialEnergy = [] # eV
        self.totalEnergy = [] # eV
        self.temperature = [] # K
        self.pressure = [] # GPa
        self.time = [] # ps

        # start index on time scale for autocorrelation:
        self.start_idx = 500

    def loadData(self):
        file = open(self.filepath)
        pattern_1 = 'Molecular dynamics equilibration :'
        pattern_2 = 'Molecular dynamics production :'
        i = 0
        start_inner_cycles = False
        # do cycle between to lines:
        for line in file:
            if pattern_1 in line:
                start_inner_cycles = True
            if pattern_2 in line:
                start_inner_cycles = False

            if start_inner_cycles:
                i = i + 1

                if '** Time :' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[0]
                    self.time.append(tmp)

                if 'Kinetic energy    (eV) =' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[1]
                    self.kineticEnergy.append(tmp)
                if 'Potential energy  (eV) =' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[1]
                    self.potentialEnergy.append(tmp)
                if 'Total energy      (eV) =' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[1]
                    self.totalEnergy.append(tmp)
                if 'Temperature       (K)  =' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[1]
                    self.temperature.append(tmp)
                if 'Pressure         (GPa) =' in line:
                    tmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[1]
                    self.pressure.append(tmp)

        self.time = np.array(self.time, dtype=float)
        self.kineticEnergy = np.array(self.kineticEnergy, dtype=float)
        self.potentialEnergy = np.array(self.potentialEnergy, dtype=float)
        self.totalEnergy = np.array(self.totalEnergy, dtype=float)
        self.temperature = np.array(self.temperature, dtype=float)
        self.pressure = np.array(self.pressure, dtype=float)

    def plotEnergy(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.time, self.totalEnergy, '-', label='$E_{{tot}}(file:${0}$)$'.format(os.path.basename(self.filepath)))
        plt.legend(loc='best')
        ax.set_ylabel('$E_{tot}[eV]$', fontsize=14, fontweight='bold')
        ax.set_xlabel('$time, [ps]$', fontsize=14, fontweight='bold')
        plt.show()

    def plotAutocorrelation(self):


        z = obj.totalEnergy[self.start_idx:]
        t = obj.time[self.start_idx:]
        x = t[int(t.size / 2):]
        acr = obj.calc_autocorrelation(z)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, acr, '-', label='$R_{{EE}}(file:${0}$)$'.format(os.path.basename(self.filepath)))
        plt.legend(loc='best')
        ax.set_ylabel('$R_{EE}$', fontsize=14, fontweight='bold')
        ax.set_xlabel('$time, [ps]$', fontsize=14, fontweight='bold')
        plt.show()

    def loadDataByGUI(self):
        import tkinter as tk
        from tkinter import filedialog
        from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars

        # open GUI filedialog to select snapshot file:
        a = StoreAndLoadVars()
        print('last used: {}'.format(a.getLastUsedFilePath()))
        # openfile dialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("history files", '*')],
                                               initialdir=os.path.dirname(a.getLastUsedFilePath()))
        if os.path.isfile(file_path):
            a.lastUsedFilePath = file_path
            a.saveLastUsedFilePath()

            self.filepath = file_path
            self.loadData()

    def calc_autocorrelation(self, y):
        yunbiased = y - np.mean(y)
        ynorm = np.sum(yunbiased ** 2)
        acor = np.correlate(yunbiased, yunbiased, "same") / ynorm
        # use only second half
        acor = acor[int(len(acor) / 2):]
        return acor

    def autocorrelation(self, x):
        """
        Compute the autocorrelation of the signal, based on the properties of the
        power spectral density of the signal.

        the autocorrelation may be computed in the following way:

        1. subtract the mean from the signal and obtain an unbiased signal

        2. compute the Fourier transform of the unbiased signal

        3. compute the power spectral density of the signal, by taking the square norm of each value of the Fourier
        transform of the unbiased signal

        4. compute the inverse Fourier transform of the power spectral density

        5. normalize the inverse Fourier transform of the power spectral density by the sum of the squares of the
        unbiased signal, and take only half of the resulting vector

        """
        xp = x - np.mean(x)
        f = np.fft.fft(xp)
        p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
        pi = np.fft.ifft(p)
        # return np.real(pi)[:int(x.size / 2)] / np.sum(xp ** 2)
        return np.real(pi)[int(x.size / 2):] / np.sum(xp ** 2)






if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    obj = OUTdataFile()
    obj.loadDataByGUI()

    obj.plotAutocorrelation()
    print('finish')

    # # generate some data
    # x = np.arange(0., 6.12, 0.01)
    # y = np.sin(x)
    # y = np.random.uniform(size=300)
    # yunbiased = y - np.mean(y)
    # ynorm = np.sum(yunbiased ** 2)
    # acor = np.correlate(yunbiased, yunbiased, "same") / ynorm
    # # use only second half
    # acor = acor[int(len(acor) / 2):]
    #
    # plt.plot(acor)
    # plt.show()
