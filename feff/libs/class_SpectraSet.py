'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-03-14
'''
from feff.libs.class_Spectrum import Spectrum
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from feff.libs.feff_processing import xftf

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

        # user's parameters for xftf preparation ['PK'- Pavel Konstantinov, 'ID' - Iraida Demchenko]:
        self.user = 'PK'

        # object to store target spectrum:
        self.target = Spectrum()
        self.target.label_latex = 'target'

        self.result = Spectrum()
        self.result.label_latex = 'composition'

        self.result_simple = Spectrum()
        self.result_simple.label_latex = 'simple composition'

        self.result_FTR_from_linear_Chi_k = Spectrum()
        self.result_FTR_from_linear_Chi_k.label_latex = 'FTR from Chi(k) composition'

        self.coefficient_vector = []
        self.coefficient_vector_FTR_from_linear_Chi_k = []



    def set_ideal_curve_params(self):
        self.result.user = self.user
        self.result.ideal_curve_ftr = self.target.ftr_vector
        self.result.ideal_curve_r =   self.target.r_vector
        self.result.ideal_curve_k =   self.target.k_vector
        self.result.ideal_curve_chi = self.target.chi_vector
        self.result.k_vector =        self.target.k_vector
        self.result.r_vector =        self.target.r_vector

        self.result_simple.user = self.user
        self.result_simple.ideal_curve_ftr = self.target.ftr_vector
        self.result_simple.ideal_curve_r =   self.target.r_vector
        self.result_simple.ideal_curve_k =   self.target.k_vector
        self.result_simple.ideal_curve_chi = self.target.chi_vector
        self.result_simple.k_vector = self.target.k_vector
        self.result_simple.r_vector = self.target.r_vector

        self.result_FTR_from_linear_Chi_k.user = self.user
        self.result_FTR_from_linear_Chi_k.ideal_curve_ftr = self.target.ftr_vector
        self.result_FTR_from_linear_Chi_k.ideal_curve_r =   self.target.r_vector
        self.result_FTR_from_linear_Chi_k.ideal_curve_k =   self.target.k_vector
        self.result_FTR_from_linear_Chi_k.ideal_curve_chi = self.target.chi_vector
        self.result_FTR_from_linear_Chi_k.k_vector = self.target.k_vector
        self.result_FTR_from_linear_Chi_k.r_vector = self.target.r_vector


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
    def func_FTR_from_linear_Chi_k(self, x):
        # create function of snapshots linear composition coeff[i]*Chi(k)[i] and then do xftf transformation to
        # FT(r) space
        val = self.dictOfSpectra[0]
        tmp_k_vector = val['data'].k_vector

        tmp_chi_vector, tmp_ftr_vector = self.funcForOptimize(x)
        tmp_ftr_vector = xftf(tmp_k_vector, tmp_chi_vector, user=self.user)[2]

        return tmp_chi_vector, tmp_ftr_vector

    def calcSimpleSpectraComposition(self):
        # only calc the mean of spectra
        num = len(self.dictOfSpectra)
        x0 = np.ones(num)
        tmp_chi_vector, tmp_ftr_vector = self.funcForOptimize(x0)

        self.result_simple.chi_vector = tmp_chi_vector
        self.result_simple.ftr_vector = tmp_ftr_vector
        self.coefficient_vector = x0/np.sum(x0)

    def calcLinearSpectraComposition(self, method='differential_evolution'):
        # calc the minimum of Rfactros minimum R_chi+R_ftr
        num = len(self.dictOfSpectra)
        x0 = np.zeros(num)
        def func(x):
            # print(x)
            self.result.chi_vector, self.result.ftr_vector = self.funcForOptimize(x)
            R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition()
            return R_tot

        # create bounds:
        bounds = []
        for i in x0:
            bounds.append((0, 1))

        # res_tmp = func(x0)
        if method == 'minimize':
            res = minimize(func, x0=x0, bounds=bounds, options={'gtol': 1e-6, 'disp': True})
        elif method == 'differential_evolution':
            res = differential_evolution(func, bounds)


        if np.sum(res.x) > 0:
            self.coefficient_vector = res.x / np.sum(res.x)
        else:
            self.coefficient_vector = res.x

        self.result.chi_vector = []
        self.result.ftr_vector = []

        tmp_chi_vector, tmp_ftr_vector = self.funcForOptimize(self.coefficient_vector)
        self.result.chi_vector = tmp_chi_vector
        self.result.ftr_vector = tmp_ftr_vector

    def calcLinearSpectraComposition_FTR_from_linear_Chi_k(self, method='differential_evolution'):
        # calc the minimum of Rfactros minimum R_chi+R_ftr
        num = len(self.dictOfSpectra)
        x0 = np.zeros(num)
        def func(x):
            # print(x)
            self.result_FTR_from_linear_Chi_k.chi_vector, self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)
            self.result_FTR_from_linear_Chi_k.ftr_vector = self.result_FTR_from_linear_Chi_k.ftr_vector \
                                                           * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
            R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
            return R_tot

        # create bounds:
        bounds = []
        for i in x0:
            bounds.append((0, 1))

        # res_tmp = func(x0)
        if method == 'minimize':
            res = minimize(func, x0=x0, bounds=bounds, options={'gtol': 1e-6, 'disp': True})
        elif method == 'differential_evolution':
            res = differential_evolution(func, bounds)


        if np.sum(res.x) > 0:
            self.coefficient_vector_FTR_from_linear_Chi_k = res.x / np.sum(res.x)
        else:
            self.coefficient_vector_FTR_from_linear_Chi_k = res.x

        self.result_FTR_from_linear_Chi_k.chi_vector = []
        self.result_FTR_from_linear_Chi_k.ftr_vector = []

        tmp_chi_vector, tmp_ftr_vector = self.func_FTR_from_linear_Chi_k(self.coefficient_vector_FTR_from_linear_Chi_k)
        self.result_FTR_from_linear_Chi_k.chi_vector = tmp_chi_vector
        self.result_FTR_from_linear_Chi_k.ftr_vector = tmp_ftr_vector * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR

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

    def get_R_factor_LinearComposition_FTR_from_linear_Chi_k(self):
        R_chi = self.result_FTR_from_linear_Chi_k.get_chi_R_factor()
        R_ftr = self.result_FTR_from_linear_Chi_k.get_FTR_R_factor()

        R_tot = (self.weight_R_factor_FTR * R_ftr + self.weight_R_factor_chi * R_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return R_tot, R_ftr, R_chi

    def updateInfo_SimpleComposition(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_SimpleComposition()
        txt = 'simple R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            txt = txt + '{0}x'.format(round(self.coefficient_vector[i], 4)) + val['data'].label
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
            txt = txt + '{0}x'.format(round(self.coefficient_vector[i], 4)) + val['data'].label
            if i < num-1:
                txt = txt + ' + '
        self.result.label_latex = txt
        self.result.label = txt.replace(':', '_').replace(' ', '_')

    def updateInfo_LinearComposition_FTR_from_linear_Chi_k(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
        txt = 'FTR(linear $\chi$) R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            txt = txt + '{0}x'.format(round(self.coefficient_vector_FTR_from_linear_Chi_k[i], 4)) + val['data'].label
            if i < num-1:
                txt = txt + ' + '
        self.result_FTR_from_linear_Chi_k.label = txt.replace(':', '_').replace(' ', '_').replace('$', '').replace('\\', '')
        txt = 'complex spectra ' + 'FTR(linear $\chi$) R={0:1.4f}'.format(R_tot)
        self.result_FTR_from_linear_Chi_k.label_latex = txt



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

    def plotSpectra_FTR_r_LinearComposition_FTR_from_linear_Chi_k(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_FTR_r()
        self.result_FTR_from_linear_Chi_k.plotTwoSpectrum_FTR_r()
        plt.title('$R_{{FT(r)\leftarrow\chi(k)}}$  = {0}'.format(round(self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()[1], 4)))
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

    def plotSpectra_chi_k_LinearComposition_FTR_from_linear_Chi_k(self):
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            val['data'].plotOneSpectrum_chi_k()
        self.result_FTR_from_linear_Chi_k.plotTwoSpectrum_chi_k()
        plt.title('$R_{{\chi(k)}}$ = {0}'.format(round(self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()[2], 4)))
        plt.legend()
        plt.show()




if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')