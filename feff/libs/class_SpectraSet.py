'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-03-14
'''
from feff.libs.class_Spectrum import Spectrum
import os
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from feff.libs.feff_processing import xftf
from feff.libs.math_libs import sigma_squared, approx_errors

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
        self.coefficient_vector_FTR_from_linear_Chi_k_std = []
        self.coefficient_vector_FTR_from_linear_Chi_k_s2 = []



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
        x = np.abs(x, type=float)
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

    def calcLinearSpectraComposition_FTR_from_linear_Chi_k(self, method='minimize'):
        # calc the minimum of Rfactros minimum R_chi+R_ftr
        num = len(self.dictOfSpectra)
        res = []
        x0 = []

        # ==========================================================================================================
        # ===== NEW algorithm with a reduce dimensions option:
        if num == 2:
            x0 = np.zeros(num-1, dtype=float) + 0.5
            # for debug:
            # plt.figure()
            # plt.plot(self.dictOfSpectra[0]['data'].plotTwoSpectrum_FTR_r())
            # plt.plot(self.dictOfSpectra[1]['data'].plotTwoSpectrum_FTR_r())
            # plt.show()

            # def func(x):
            #     # print(x)
            #     self.result_FTR_from_linear_Chi_k.chi_vector, \
            #     self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)
            #
            #     self.result_FTR_from_linear_Chi_k.ftr_vector = self.result_FTR_from_linear_Chi_k.ftr_vector \
            #                                                    * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
            #     R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
            #     return R_tot

            def func(t):
                # print(x)
                # increment dimention by 1 position:
                x = np.zeros(num, dtype=float)
                if np.size(t) > 1:
                    # some times (it's a fucking understending thing for me) after for example 20 iterations with
                    # len(t)= num-1 something was crushed in minimization algorithm and len(t) takes the value: num
                    x[0] = t[0]
                    x[1] = 1 - t[0]
                else:
                    x[0] = t
                    x[1] = 1 - t

                self.result_FTR_from_linear_Chi_k.chi_vector, \
                self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)

                self.result_FTR_from_linear_Chi_k.ftr_vector = self.result_FTR_from_linear_Chi_k.ftr_vector \
                                                               * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
                R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
                return R_tot

            res = minimize_scalar(func, bounds=(0, 1), method='bounded',
                           options={'xatol': 1e-5, 'disp': False})


            res_arr_x = np.zeros(num, dtype=float)
            if np.size(res.x) > 1:
                res_arr_x[0] = res.x[0]
                res_arr_x[1] = 1 - res.x[0]
            else:
                res_arr_x[0] = res.x
                res_arr_x[1] = 1 - res.x

            if np.sum(res_arr_x) > 1:
                self.coefficient_vector_FTR_from_linear_Chi_k = res_arr_x / np.sum(res_arr_x)
            else:
                self.coefficient_vector_FTR_from_linear_Chi_k = res_arr_x

        elif num > 2:
            x0 = np.zeros(num-1, dtype=float) + 0.5
            def func(t):
                # print(x)
                # decrement dimention by 1 position:
                x = np.zeros(num, dtype=float)
                if np.size(t) == num-1:
                    x[0:num - 1] = t[0:num - 1]
                    x[num - 1] = 1 - np.sum(t[0:num - 1])
                else:
                    # some times (it's a fucking understending thing for me) after for example 20 iterations with
                    # len(t)= num-1 something was crushed in minimization algorithm and len(t) takes the value: num
                    x = t

                self.result_FTR_from_linear_Chi_k.chi_vector, \
                self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)

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
                # res = minimize(func, x0=x0, bounds=bounds, method='nelder-mead',
                #                options={'xtol': 1e-8, 'disp': False})
                res = minimize(func, x0=x0, bounds=bounds, method='TNC',
                               options={'gtol': 1e-4, 'disp': False})
            elif method == 'differential_evolution':
                res = differential_evolution(func, bounds)

            # if np.sum(res.x) > 0:
            #     self.coefficient_vector_FTR_from_linear_Chi_k = res.x / np.sum(res.x)
            # else:
            #     self.coefficient_vector_FTR_from_linear_Chi_k = res.x

            res_arr_x = np.zeros(num, dtype=float)
            res_arr_x[0:num-1] = res.x[0:num-1]
            res_arr_x[num-1] = 1 - np.sum(res.x[0:num-1])

            if np.sum(res_arr_x) > 1:
                self.coefficient_vector_FTR_from_linear_Chi_k = res_arr_x / np.sum(res_arr_x)
            else:
                self.coefficient_vector_FTR_from_linear_Chi_k = res_arr_x
        # ====== END of NEW algorithm =============
        # ==========================================================================================================


        # # ==========================================================================================================
        # # ====== Old version of the allgorithm:
        # x0 = np.zeros(num, dtype=float) + 0.5
        # #     x0[0]=1
        # #     x0[1]=0
        #
        # # create bounds:
        # bounds = []
        # for i in x0:
        #     bounds.append((0, 1))
        #
        # # define the function for minimization:
        # def func(x):
        #     # print(x)
        #
        #     self.result_FTR_from_linear_Chi_k.chi_vector, \
        #     self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)
        #
        #     self.result_FTR_from_linear_Chi_k.ftr_vector = self.result_FTR_from_linear_Chi_k.ftr_vector \
        #                                                    * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
        #     R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
        #     return R_tot
        #
        # # res_tmp = func(x0)
        # if method == 'minimize':
        #     # res = minimize(func, x0=x0, bounds=bounds, method='nelder-mead',
        #     #                options={'xtol': 1e-8, 'disp': False})
        #     res = minimize(func, x0=x0, bounds=bounds, method='TNC',
        #                    options={'gtol': 1e-4, 'disp': False})
        # elif method == 'differential_evolution':
        #     res = differential_evolution(func, bounds)
        #
        # if np.sum(res.x) > 0:
        #     self.coefficient_vector_FTR_from_linear_Chi_k = res.x / np.sum(res.x)
        # else:
        #     self.coefficient_vector_FTR_from_linear_Chi_k = res.x
        # # ++++++++++++++++++ END of Old algorithm ++++++++++++++++++++
        # # ==========================================================================================================


        self.result_FTR_from_linear_Chi_k.chi_vector = []
        self.result_FTR_from_linear_Chi_k.ftr_vector = []

        tmp_chi_vector, tmp_ftr_vector = self.func_FTR_from_linear_Chi_k(self.coefficient_vector_FTR_from_linear_Chi_k)
        self.result_FTR_from_linear_Chi_k.chi_vector = tmp_chi_vector
        self.result_FTR_from_linear_Chi_k.ftr_vector = tmp_ftr_vector * \
                                                       self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR

        # for debug:
        # self.coefficient_vector_FTR_from_linear_Chi_k = [1, 0]
        # tmp_chi_vector, tmp_ftr_vector = self.func_FTR_from_linear_Chi_k(self.coefficient_vector_FTR_from_linear_Chi_k)
        # self.result_FTR_from_linear_Chi_k.chi_vector = tmp_chi_vector
        # self.result_FTR_from_linear_Chi_k.ftr_vector = tmp_ftr_vector * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
        # plt.figure()
        # self.plotSpectra_FTR_r_LinearComposition_FTR_from_linear_Chi_k()
        # plt.figure()
        # self.plotSpectra_chi_k_LinearComposition_FTR_from_linear_Chi_k()

        # estimate the errors of the coefficients:
        s2 = self.result_FTR_from_linear_Chi_k.get_FTR_sigma_squared()

        # def func(x):
        #     self.result_FTR_from_linear_Chi_k.chi_vector, self.result_FTR_from_linear_Chi_k.ftr_vector = self.func_FTR_from_linear_Chi_k(x)
        #     self.result_FTR_from_linear_Chi_k.ftr_vector = self.result_FTR_from_linear_Chi_k.ftr_vector \
        #                                                    * self.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR
        #     s2_tot, s2_ftr, s2_chi = self.get_Sigma_Squared_LinearComposition_FTR_from_linear_Chi_k()
        #     return s2_tot

        se = approx_errors(func, self.coefficient_vector_FTR_from_linear_Chi_k)
        std = np.sqrt(s2)*se
        # print('----se:')
        # print (se)
        # print('----s2:')
        # print(s2)
        # print('----std:')
        # print(std)
        # print('==> p0:')
        # print(self.coefficient_vector_FTR_from_linear_Chi_k)
        # print('=='*15)
        self.coefficient_vector_FTR_from_linear_Chi_k_std = std
        self.coefficient_vector_FTR_from_linear_Chi_k_s2 = s2




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

    def get_Sigma_Squared_LinearComposition_FTR_from_linear_Chi_k(self):
        s2_chi = self.result_FTR_from_linear_Chi_k.get_chi_sigma_squared()
        s2_ftr = self.result_FTR_from_linear_Chi_k.get_FTR_sigma_squared()

        s2_tot = (self.weight_R_factor_FTR * s2_ftr + self.weight_R_factor_chi * s2_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return s2_tot, s2_ftr, s2_chi

    def updateInfo_SimpleComposition(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_SimpleComposition()
        txt = 'simple R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        if num < 4:
            for i in self.dictOfSpectra:
                val = self.dictOfSpectra[i]
                txt = txt + '{0}x'.format(round(self.coefficient_vector[i], 4)) + val['data'].label
                if i < num-1:
                    txt = txt + ' + '
        else:
            val_first = self.dictOfSpectra[0]
            val_last = self.dictOfSpectra[num - 1]
            txt = txt + 'sum of N={n} snapshots, [from {f} to {l}]'. \
                format(n=num, f=val_first['data'].label, l=val_last['data'].label)

        self.result_simple.label_latex = txt
        self.result_simple.label = txt.replace(':', '_').replace(' ', '_')

    def getInfo_SimpleComposition(self):
        txt = ''
        num = len(self.dictOfSpectra)
        if num < 4:
            for i in self.dictOfSpectra:
                val = self.dictOfSpectra[i]
                txt = txt + '{0}x'.format(round(self.coefficient_vector[i], 4)) + val['data'].label
                if i < num-1:
                    txt = txt + ' + '
        else:
            val_first = self.dictOfSpectra[0]
            val_last = self.dictOfSpectra[num - 1]
            txt = txt + 'sum of N={n} snapshots, [from {f} to {l}]'. \
                format(n=num, f=val_first['data'].label, l=val_last['data'].label)
        return txt

    def updateInfo_LinearComposition(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition()
        txt = 'linear R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        if num < 4:
            for i in self.dictOfSpectra:
                val = self.dictOfSpectra[i]
                txt = txt + '{0}x'.format(round(self.coefficient_vector[i], 4)) + val['data'].label
                if i < num-1:
                    txt = txt + ' + '
        else:
            val_first = self.dictOfSpectra[0]
            val_last = self.dictOfSpectra[num - 1]
            txt = txt + 'sum of N={n} snapshots, [from {f} to {l}]'.\
                format(n=num, f=val_first['data'].label, l=val_last['data'].label)

        self.result.label_latex = txt
        self.result.label = txt.replace(':', '_').replace(' ', '_')

    def updateInfo_LinearComposition_FTR_from_linear_Chi_k(self):
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
        txt = 'FTR(linear $\chi$) R={0:1.4f}: '.format(R_tot)
        num = len(self.dictOfSpectra)
        if num < 4:
            for i in self.dictOfSpectra:
                val = self.dictOfSpectra[i]
                txt = txt + '{0}x'.format(round(self.coefficient_vector_FTR_from_linear_Chi_k[i], 3)) + val['data'].label
                if i < num-1:
                    txt = txt + ' + '
        else:
            val_first = self.dictOfSpectra[0]
            val_last = self.dictOfSpectra[num - 1]
            txt = txt + 'sum of N={n} snapshots, [from {f} to {l}]'. \
                format(n=num, f=val_first['data'].label, l=val_last['data'].label)

        self.result_FTR_from_linear_Chi_k.label = txt.replace(':', '_').replace(' ', '_').\
            replace('$', '').replace('\\', '')
        txt = 'complex spectra ' + 'FTR(linear $\chi$) R={0:1.4f}'.format(R_tot)
        self.result_FTR_from_linear_Chi_k.label_latex = txt

    def getInfo_LinearComposition_FTR_from_linear_Chi_k(self):
        # return only the formula of snapshot name and coefficient
        txt = ''
        num = len(self.dictOfSpectra)
        if num < 4:
            for i in self.dictOfSpectra:
                val = self.dictOfSpectra[i]
                txt = txt + '{0}x[{1}]'.format( round(self.coefficient_vector_FTR_from_linear_Chi_k[i], 3),
                                               val['data'].label )
                if i < num-1:
                    txt = txt + ' + '
        else:
            val_first = self.dictOfSpectra[0]
            val_last = self.dictOfSpectra[num - 1]
            txt = txt + 'sum of N={n} snapshots, [from {f} to {l}]'. \
                format(n=num, f=val_first['data'].label, l=val_last['data'].label)
        return txt


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

        txt_coeff = '['
        txt_sep = ' ,'
        for idx, k in enumerate(self.coefficient_vector_FTR_from_linear_Chi_k):
            if idx == (len(self.coefficient_vector_FTR_from_linear_Chi_k) - 1):
                txt_sep = ''
            txt_coeff = txt_coeff + '{}'.format(round(k,4)) + txt_sep
        txt_coeff = txt_coeff + ']'

        self.result_FTR_from_linear_Chi_k.plotTwoSpectrum_FTR_r()
        plt.title('$R_{{FT(r)\leftarrow\chi(k)}}$  = {0}, k = {1}'.format(
            round(self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()[1], 4), txt_coeff))
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

        txt_coeff = '['
        txt_sep = ' ,'
        for idx, k in enumerate(self.coefficient_vector_FTR_from_linear_Chi_k):
            if idx == (len(self.coefficient_vector_FTR_from_linear_Chi_k) - 1):
                txt_sep = ''
            txt_coeff = txt_coeff + '{}'.format(round(k, 4)) + txt_sep
        txt_coeff = txt_coeff + ']'

        self.result_FTR_from_linear_Chi_k.plotTwoSpectrum_chi_k()
        plt.title('$R_{{\chi(k)}}$ = {0}, k = {1}'.format(
            round(self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()[2], 4), txt_coeff))
        plt.legend()
        plt.show()

    def saveSpectra_LinearComposition_FTR_from_linear_Chi_k(self, output_dir=''):
        # save Spectra to ASCII column file:
        num = len(self.dictOfSpectra)
        numOfRows = len(self.result_FTR_from_linear_Chi_k.r_vector)
        out_array_ftr = np.zeros((numOfRows, num + 3))
        numOfRows = len(self.result_FTR_from_linear_Chi_k.chi_vector)
        out_array_chi = np.zeros((numOfRows, num + 3))
        out_array_ftr[:, 0] = self.result_FTR_from_linear_Chi_k.r_vector
        out_array_chi[:, 0] = self.result_FTR_from_linear_Chi_k.k_vector
        headerTxt_ftr = 'r_vector[AA]\t'
        headerTxt_chi = 'k_vector[1/AA]\t'
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
        txt = 'FTR(linear chi composition) Rtot={0:1.5f}, Rftr={1:1.5f}, Rchi={2:1.5f}: \n sqrt of sigma squared is: {3:1.5f} \n formula is:\n'\
            .format(R_tot, R_ftr, R_chi, np.sqrt(self.coefficient_vector_FTR_from_linear_Chi_k_s2))
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            out_array_ftr[:, i+1] = val['data'].ftr_vector
            out_array_chi[:, i+1] = val['data'].chi_vector
            headerTxt_ftr = headerTxt_ftr + 'sanpshot:' + val['data'].label
            headerTxt_chi = headerTxt_chi + 'sanpshot:' + val['data'].label
            txt = txt + '({0} +/- {1})*[ '.format(round(self.coefficient_vector_FTR_from_linear_Chi_k[i], 4),
                                              round(self.coefficient_vector_FTR_from_linear_Chi_k_std[i], 4)) \
                  + val['data'].label + ' ]'
            if i < num - 1:
                txt = txt + ' + '
            headerTxt_ftr = headerTxt_ftr + '\t'
            headerTxt_chi = headerTxt_chi + '\t'


        i = i + 2
        headerTxt_ftr = headerTxt_ftr + 'complex spectra'
        headerTxt_chi = headerTxt_chi + 'complex spectra'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result_FTR_from_linear_Chi_k.ftr_vector
        out_array_chi[:, i] = self.result_FTR_from_linear_Chi_k.chi_vector

        i = i + 1
        headerTxt_ftr = headerTxt_ftr + 'ideal curve'
        headerTxt_chi = headerTxt_chi + 'ideal curve'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result_FTR_from_linear_Chi_k.ideal_curve_ftr
        out_array_chi[:, i] = self.result_FTR_from_linear_Chi_k.ideal_curve_chi

        headerTxt = txt + '\n' + headerTxt_ftr
        out_array = out_array_ftr
        np.savetxt(os.path.join(output_dir, f'{self.result_FTR_from_linear_Chi_k.label}__ftr(r).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result_FTR_from_linear_Chi_k.label}__ftr(r).txt')
        # print(f'=========== file: {txt_out} has been saved')

        headerTxt = txt + '\n' + headerTxt_chi
        out_array = out_array_chi
        np.savetxt(os.path.join(output_dir, f'{self.result_FTR_from_linear_Chi_k.label}__chi(k).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result_FTR_from_linear_Chi_k.label}__chi(k).txt')
        # print(f'=========== file: {txt_out} has been saved')

    def saveSpectra_LinearComposition(self, output_dir=''):
        # save Spectra to ASCII column file:
        num = len(self.dictOfSpectra)
        numOfRows = len(self.result.r_vector)
        out_array_ftr = np.zeros((numOfRows, num + 3))
        numOfRows = len(self.result.chi_vector)
        out_array_chi = np.zeros((numOfRows, num + 3))
        out_array_ftr[:, 0] = self.result.r_vector
        out_array_chi[:, 0] = self.result.k_vector
        headerTxt_ftr = 'r_vector[AA]\t'
        headerTxt_chi = 'k_vector[1/AA]\t'
        R_tot, R_ftr, R_chi = self.get_R_factor_LinearComposition()
        txt = '(Linear chi composition) Rtot={0:1.5f}, Rftr={1:1.5f}, Rchi={2:1.5f}: \n formula is:\n'.format(R_tot, R_ftr, R_chi)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            out_array_ftr[:, i+1] = val['data'].ftr_vector
            out_array_chi[:, i+1] = val['data'].chi_vector
            headerTxt_ftr = headerTxt_ftr + 'sanpshot:' + val['data'].label
            headerTxt_chi = headerTxt_chi + 'sanpshot:' + val['data'].label
            txt = txt + '{0}*[ '.format(round(self.coefficient_vector[i], 4)) + val['data'].label + ' ]'
            if i < num - 1:
                txt = txt + ' + '
            headerTxt_ftr = headerTxt_ftr + '\t'
            headerTxt_chi = headerTxt_chi + '\t'


        i = i + 2
        headerTxt_ftr = headerTxt_ftr + 'complex spectra'
        headerTxt_chi = headerTxt_chi + 'complex spectra'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result.ftr_vector
        out_array_chi[:, i] = self.result.chi_vector

        i = i + 1
        headerTxt_ftr = headerTxt_ftr + 'ideal curve'
        headerTxt_chi = headerTxt_chi + 'ideal curve'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result.ideal_curve_ftr
        out_array_chi[:, i] = self.result.ideal_curve_chi

        headerTxt = txt + '\n' + headerTxt_ftr
        out_array = out_array_ftr
        np.savetxt(os.path.join(output_dir, f'{self.result.label}__ftr(r).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result.label}__ftr(r).txt')
        print(f'=========== file: {txt_out} has been saved')

        headerTxt = txt + '\n' + headerTxt_chi
        out_array = out_array_chi
        np.savetxt(os.path.join(output_dir, f'{self.result.label}__chi(k).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result.label}__chi(k).txt')
        print(f'=========== file: {txt_out} has been saved')

    def saveSpectra_SimpleComposition(self, output_dir=''):
        # save Spectra to ASCII column file:
        num = len(self.dictOfSpectra)
        numOfRows = len(self.result_simple.r_vector)
        out_array_ftr = np.zeros((numOfRows, num + 3))
        numOfRows = len(self.result_simple.chi_vector)
        out_array_chi = np.zeros((numOfRows, num + 3))
        out_array_ftr[:, 0] = self.result_simple.r_vector
        out_array_chi[:, 0] = self.result_simple.k_vector
        headerTxt_ftr = 'r_vector[AA]\t'
        headerTxt_chi = 'k_vector[1/AA]\t'
        R_tot, R_ftr, R_chi = self.get_R_factor_SimpleComposition()
        txt = '(simple chi composition) Rtot={0:1.5f}, Rftr={1:1.5f}, Rchi={2:1.5f}: \n formula is:\n'.format(R_tot, R_ftr, R_chi)
        for i in self.dictOfSpectra:
            val = self.dictOfSpectra[i]
            out_array_ftr[:, i+1] = val['data'].ftr_vector
            out_array_chi[:, i+1] = val['data'].chi_vector
            headerTxt_ftr = headerTxt_ftr + 'sanpshot:' + val['data'].label
            headerTxt_chi = headerTxt_chi + 'sanpshot:' + val['data'].label
            txt = txt + '{0}*[ '.format(round(self.coefficient_vector[i]*0+0.5, 1)) + val['data'].label + ' ]'
            if i < num - 1:
                txt = txt + ' + '
            headerTxt_ftr = headerTxt_ftr + '\t'
            headerTxt_chi = headerTxt_chi + '\t'


        i = i + 2
        headerTxt_ftr = headerTxt_ftr + 'complex spectra'
        headerTxt_chi = headerTxt_chi + 'complex spectra'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result_simple.ftr_vector
        out_array_chi[:, i] = self.result_simple.chi_vector

        i = i + 1
        headerTxt_ftr = headerTxt_ftr + 'ideal curve'
        headerTxt_chi = headerTxt_chi + 'ideal curve'
        headerTxt_ftr = headerTxt_ftr + '\t'
        headerTxt_chi = headerTxt_chi + '\t'
        out_array_ftr[:, i] = self.result_simple.ideal_curve_ftr
        out_array_chi[:, i] = self.result_simple.ideal_curve_chi

        headerTxt = txt + '\n' + headerTxt_ftr
        out_array = out_array_ftr
        np.savetxt(os.path.join(output_dir, f'{self.result_simple.label}__ftr(r).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result_simple.label}__ftr(r).txt')
        # print(f'=========== file: {txt_out} has been saved')

        headerTxt = txt + '\n' + headerTxt_chi
        out_array = out_array_chi
        np.savetxt(os.path.join(output_dir, f'{self.result_simple.label}__chi(k).txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        txt_out = os.path.join(output_dir, f'{self.result_simple.label}__chi(k).txt')
        # print(f'=========== file: {txt_out} has been saved')



if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')