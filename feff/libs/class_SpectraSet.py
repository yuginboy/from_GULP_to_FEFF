'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-03-14
'''
from feff.libs.class_Spectrum import Spectrum
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SpectraSet():
    '''
    class to combine serial snapshots spectra and find the optimum
    '''
    def __init__(self):
        self.dictOfSpectra = {}

        # weights for calc tota R-factor in minimization procedure
        # (Rtot = (Rchi * w1 + Rrtf * w2) / (w1 + w2)):
        self.weight_R_factor_FTR = 1
        self.weight_R_factor_chi = 1

        # object to store target spectrum:
        self.target = Spectrum()
        self.target.label_latex = 'target'

        self.result = Spectrum()
        self.result.label_latex = 'composition'

        self.result_simple = Spectrum()
        self.result_simple.label_latex = 'simple composition'

        self.coefficient_vector = []

    def set_ideal_curve_params(self):
        self.result.ideal_curve_ftr = self.target.ftr_vector
        self.result.ideal_curve_r =   self.target.r_vector
        self.result.ideal_curve_k =   self.target.k_vector
        self.result.ideal_curve_chi = self.target.chi_vector
        self.result.k_vector =        self.target.k_vector
        self.result.r_vector =        self.target.r_vector


        self.result_simple.ideal_curve_ftr = self.target.ftr_vector
        self.result_simple.ideal_curve_r =   self.target.r_vector
        self.result_simple.ideal_curve_k =   self.target.k_vector
        self.result_simple.ideal_curve_chi = self.target.chi_vector
        self.result_simple.k_vector = self.target.k_vector
        self.result_simple.r_vector = self.target.r_vector



    def addSpectraToDict(self, currentSpectra):
        num = len(self.dictOfSpectra)
        if isinstance(currentSpectra, Spectrum):
            self.dictOfSpectra[num] = dict({'data' : currentSpectra})

    def flushDictOfSpectra(self):
        self.dictOfSpectra = {}

    def funcForOptimize(self, x):
        # create function of snapshots linear composition Sum[x_i*F_i]
        num = len(self.dictOfSpectra)
        tmp_chi_vector = []
        tmp_ftr_vector = []
        sum_x = np.sum(x)

        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]

            if abs(sum_x) > 0:
                k = x[i] / sum_x
            else:
                k = x[i]

            if i < 1:
                tmp_chi_vector = k * val['data'].chi_vector
                tmp_ftr_vector = k * val['data'].ftr_vector

                # tmp_chi_vector2 = k * val['data'].chi_vector
                # tmp_ftr_vector2 = k * val['data'].ftr_vector
            else:
                tmp_chi_vector = tmp_chi_vector + k * val['data'].chi_vector
                tmp_ftr_vector = tmp_ftr_vector + k * val['data'].ftr_vector

        return tmp_chi_vector, tmp_ftr_vector

    def calcSimpleSpectraComposition(self):
        # only calc the mean of spectra
        num = len(self.dictOfSpectra)
        x0 = np.ones(num)
        tmp_chi_vector, tmp_ftr_vector = self.funcForOptimize(x0)

        self.result_simple.chi_vector = tmp_chi_vector
        self.result_simple.ftr_vector = tmp_ftr_vector
        self.coefficient_vector = x0/np.sum(x0)

    def calcLinearSpectraComposition(self):
        # calc the minimum of Rfactros minimum R_chi+R_ftr
        num = len(self.dictOfSpectra)
        x0 = np.zeros(num)
        def func(x):
            # print(x)
            self.result.chi_vector, self.result.ftr_vector = self.funcForOptimize(x)
            R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition()
            return R_tot

        # res_tmp = func(x0)

        res = minimize(func, x0=x0, options={'gtol': 1e-6, 'disp': True})

        self.coefficient_vector = res.x / np.sum(res.x)
        self.result.chi_vector = []
        self.result.ftr_vector = []

        tmp_chi_vector, tmp_ftr_vector = self.funcForOptimize(self.coefficient_vector)
        self.result.chi_vector = tmp_chi_vector
        self.result.ftr_vector = tmp_ftr_vector

    def get_R_factor_SimpleComposition(self):
        R_chi = self.result_simple.get_chi_R_factor()
        R_ftr = self.result_simple.get_FTR_R_factor()

        R_tot = (self.weight_R_factor_FTR * R_ftr + self.weight_R_factor_chi * R_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return R_tot, R_ftr, R_chi

    def get_R_factor_LinearComposition(self):
        R_chi = self.result.get_chi_R_factor()
        R_ftr = self.result.get_FTR_R_factor()

        R_tot = (self.weight_R_factor_FTR * R_ftr + self.weight_R_factor_chi * R_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return R_tot, R_ftr, R_chi

    def updateInfo_SimpleComposition(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_SimpleComposition()
        txt = 'simple R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            txt = txt + '{0}*'.format(round(self.coefficient_vector[i-1], 4)) + val['data'].label
            if i < num-1:
                txt = txt + ' + '
        self.result_simple.label_latex = txt
        self.result_simple.label = txt.replace(':', '_').replace(' ', '_')

    def updateInfo_LinearComposition(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition()
        txt = 'linear R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            txt = txt + '{0}*'.format(round(self.coefficient_vector[i-1], 4)) + val['data'].label
            if i < num-1:
                txt = txt + ' + '
        self.result.label_latex = txt
        self.result.label = txt.replace(':', '_').replace(' ', '_')

    def plotSpectra_FTR_r_SimpleComposition(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_FTR_r()
        self.result_simple.plotTwoSpectrum_FTR_r()
        plt.title('$R_{{FT(r)}}$  = {0}'.format(round(self.get_R_factor_SimpleComposition()[1], 4)))
        plt.legend()
        plt.show()


    def plotSpectra_chi_k_SimpleComposition(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_chi_k()
        self.result_simple.plotTwoSpectrum_chi_k()
        plt.title('$R_{{\chi(k)}}$ = {0}'.format(round(self.get_R_factor_SimpleComposition()[2], 4)))
        plt.legend()
        plt.show()

    def plotSpectra_FTR_r_LinearComposition(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_FTR_r()
        self.result.plotTwoSpectrum_FTR_r()
        plt.title('$R_{{FT(r)}}$  = {0}'.format(round(self.get_R_factor_LinearComposition()[1], 4)))
        plt.legend()
        plt.show()

    def plotSpectra_chi_k_LinearComposition(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_chi_k()
        self.result.plotTwoSpectrum_chi_k()
        plt.title('$R_{{\chi(k)}}$ = {0}'.format(round(self.get_R_factor_LinearComposition()[2], 4)))
        plt.legend()
        plt.show()




if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')