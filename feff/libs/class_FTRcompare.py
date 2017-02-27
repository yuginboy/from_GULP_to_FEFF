'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-27
'''
import os
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name
from feff.libs.load_chi_data_file import load_and_apply_xftf, load_chi_data
import matplotlib.pyplot as plt
import numpy as np

class Spectrum():
    # base spectrum class
    def __init__(self):
        # Chi intensities vector:
        self.chi_vector = []
        # k(1/A)
        self.k_vector = []

        # R(Angstrom):
        self.r_vector = []
        # Furie transform of chi(k) intensities vector:
        self.ftr_vector = []

        self.label = 'xperiment T=350 C'
        self.label_latex = 'experiment $T=350^{\circ}$ '

        # path to the stored ASCII data file:
        self.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data', '350.chik')

        # region for R-factor calculation in Angstroms:
        self.r_factor_region = np.array([1, 5])
        # array of ideal values for R-factor calculation:
        self.baseline_curve = []
        # array of calculated values for R-factor calculation:
        self.target_curve = []

    def loadSpectrumData(self):
        # load data from ASCII file:
        data = load_chi_data(self.pathToLoadDataFile)
        self.k_vector   = data[:, 0]
        self.chi_vector = data[:, 1]

        data = load_and_apply_xftf(self.pathToLoadDataFile)
        self.r_vector   = data[0]
        self.ftr_vector = data[2] # get only the Real-part of values

    def plotSpectrum_chi_k(self):
        plt.plot(self.k_vector, self.chi_vector, lw=2, label=self.label_latex)

    def plotSpectrum_FTR_r(self):
        plt.plot(self.r_vector, self.ftr_vector, lw=2, label=self.label_latex)

    def selectPointsInRegion(self):


    def get_R_factor(self):
        if (len(self.target_curve) > 1) and (len(self.baseline_curve) > 1):
            if (len(self.baseline_curve) == len(self.target_curve)):
                # calc only when the size of arrays are eq and consist more then 1 elements:
                indexMin = (np.abs(self.baseline_curve - self.r_factor_region[0])).argmin()
                indexMax = (np.abs(self.baseline_curve - self.r_factor_region[1])).argmin()

            else:
                print('the sizes of compared curves are not equal')


class FTR_gulp_to_feff():
    def __init__(self):
        self.experiment = Spectrum()
        self.theory_one = Spectrum()
        self.theory_two = Spectrum()

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    a = Spectrum()
    a.loadSpectrumData()

    a.plotSpectrum_chi_k()
    plt.legend()
    plt.show()

    a.plotSpectrum_FTR_r()
    plt.legend()
    plt.show()
