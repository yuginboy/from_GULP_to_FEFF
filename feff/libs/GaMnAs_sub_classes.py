'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-10-18
'''
import sys
import os
from copy import deepcopy
from collections import OrderedDict as odict
from itertools import cycle
from io import StringIO
import inspect
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.gridspec as gridspec
from matplotlib import pylab
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, splrep, splev, splprep
import re
from shutil import copyfile
from libs.dir_and_file_operations import listOfFilesFN, listOfFiles, listOfFilesFN_with_selected_ext
from feff.libs.numpy_group_by_ep_second_draft import group_by
from scipy.signal import savgol_filter
from feff.libs.fit_current_curve import return_fit_param, func, f_PM, f_diff_PM_for_2_T, \
    linearFunc, f_PM_with_T, f_SPM_with_T
from scipy.optimize import curve_fit, leastsq
from feff.libs.math_libs import approx_errors


g_J_Mn2_plus = 5.92
g_J_Mn3_plus = 4.82
g_e = 2.0023 # G-factor Lande
mu_Bohr = 927.4e-26 # J/T
Navagadro = 6.02214e23 #1/mol
k_b = 1.38065e-23 #J/K

rho_GaAs = 5.3176e3 #kg/m3
mass_Molar_kg_GaAs = 144.645e-3 #kg/mol

mass_Molar_kg_Diamond = 12.011e-3 # diamond
rho_Diamond = 3.515e3

testX1 = [0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038, 0.03, 0.0325]
testY1 = [0.6, 0.527361, 0.564139, 0.602, 0.640714, 0.676684, 0.713159, 0.7505, 0.9, 0.662469]
testArray = np.array([testX1, testY1])

def fromRowToColumn (Array = testArray):
    # if num of columns bigger then num of rows then transpoze that marix
    n,m = Array.shape
    if n < m:
        return Array.T
    else:
        return Array

def sortMatrixByFirstColumn(Array = fromRowToColumn(testArray), colnum = 0):
    # return sorted by selected column number the matrix
    return Array[Array[:, colnum].argsort()]

out = sortMatrixByFirstColumn()
# print('sorted out:')
# print(out)
# print('--')
def deleteNonUniqueElements(key = out[:, 0], val = out[:, 1]):
    # calc the val.mean value for non-uniq key values
    # u, idx = np.unique(Array[:, key_colnum], return_index=True)
    return fromRowToColumn(np.array(group_by(key).mean(val)))

# print('mean :')
# print(deleteNonUniqueElements())
# print('--')
def from_EMU_cm3_to_A_by_m(moment_emu = 2300e-8, V_cm3 = 3e-6):
    # return value of Magnetization in SI (A/m)
    return (moment_emu / V_cm3)*1000
def concentration_from_Ms(Ms = 7667, J=2.5):
    # return concentration from Ms = n*gj*mu_Bohr*J = n*p_exp*mu_Bohr
    return Ms/mu_Bohr/J/g_e
def number_density(rho = rho_GaAs, M = mass_Molar_kg_GaAs):
    # return concentration from Molar mass
    return Navagadro*rho/M


class MagneticPropertiesOfPhases:
    def __init__(self):
        self.concentration_ParaMagnetic = None
        self.concentration_ParaMagnetic_error = None
        self.J_total_momentum = 2.5
        self.Mn_type = 'Mn2+'  # Mn2+ or Mn3+
        self.spin_type_cfg = 'high'  # low or high
        self.g_factor = g_e
        self.mu_eff = g_J_Mn2_plus
        self.info = ''

class MagneticData:
    '''
    base class for store spectra
    '''
    COLOR_CYCOL = cycle('bgrcmk')
    def __init__(self):
        self.magnetic_field = []
        self.magnetic_moment = []
        self.magnetic_moment_raw = []
        self.label = []
        self.do_plot = True

        self.line_style = '-'
        self.line_color = 'cornflowerblue'
        self.line_width = 2
        self.line_alpha = 1.0
        self.line_marker_style = 'o'
        self.line_marker_size = 4
        self.line_marker_face_color = 'blue'
        self.line_marker_edge_color = 'darkblue'

        self.temperature = []
        self.magnetic_field_shift = []
        self.magnetic_moment_shift = []
        # for intersection of region for two models:
        self.accepted_indices = []
        self.history_log = odict()

    def append_history_log(self, case=''):
        num = len(self.history_log)
        num += 1
        self.history_log[num] = case

    def plot(self, ax=plt.gca()):
        if self.do_plot:
            ax.plot(self.magnetic_field, self.magnetic_moment,
                    linestyle=self.line_style,
                    color=self.line_color,
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                    label=self.label,
                    marker=self.line_marker_style,
                    markersize=self.line_marker_size,
                    markerfacecolor=self.line_marker_face_color,
                    markeredgecolor=self.line_marker_edge_color
                    )

class StructBase:
    '''
    Describe structure for a data
    '''
    def __init__(self):
        self.raw = MagneticData()
        self.prepared_raw = MagneticData()
        self.for_fit = MagneticData()
        self.line = MagneticData()
        self.fit = MagneticData()

        self.magnetic_field_inflection_point = 30
        self.magnetic_field_step = 0.1 #[T]
        self.magnetic_field_minimum = 0
        self.magnetic_field_maximum = 0

        # obj for saving params about unique phases in current material:
        self.current_magnetic_phase_data = MagneticPropertiesOfPhases()

        # main magnetic params for a structure which could be saved in unique phases:
        self.J_total_momentum = 2.5
        self.Mn_type = 'Mn2+' # Mn2+ or Mn3+
        self.spin_type_cfg = 'high' # low or high
        self.g_factor = g_e
        self.mu_eff = g_J_Mn2_plus
        self.mu_eff_min = g_J_Mn2_plus - 0.1
        self.mu_eff_max = g_J_Mn2_plus + 0.1
        self.volumeOfTheFilm_GaMnAs = 0 #[m^3]
        self.fit.magnetic_moment = []
        self.forFit_y = []
        self.forFit_x = []
        self.zeroIndex = []
        # point of magnetic field which define a outside region where we fit the functions
        self.magnetic_field_value_for_fit = 3 # [T]

        # number of density for PM fit only for the current temperature:
        self.concentration_ParaMagnetic = 0
        self.concentration_ParaMagnetic_error = 10

        # linear coefficient from the line_subtracted procedure:
        self.linear_coefficient = 0


        # corrections for curve:
        self.y_shift = 0
        self.x_shift = 0

        # 'IUS' - Interpolation using univariate spline
        # 'RBF' - Interpolation using Radial basis functions
        # Interpolation using RBF - multiquadrics
        # 'Spline'
        # 'Cubic'
        # 'Linear'
        self.typeOfFiltering = 'IUS'

        self.R_factor = 100
        self.std = 100

        self.label_summary = ''
        self.title = ''
        self.font_size = 18
        self.y_label = '$M(A/m)$'
        self.x_label = '$B(T)$'

        self.dict_of_magnetic_phases = odict()

    def addDataToDict(self, current_data):
        num = len(self.dict_of_magnetic_phases)
        if isinstance(current_data, MagneticPropertiesOfPhases):
            self.dict_of_magnetic_phases[num] = odict({'data': current_data})

    def flushDictOfSpectra(self):
        self.dict_of_magnetic_phases = odict()

    def define_Mn_type_variables(self):
        '''
                              # unpaired electrons            examples
                    d-count    high spin   low spin
                    d 4           4          2              Cr 2+ , Mn 3+
                    d 5           5          1              Fe 3+ , Mn 2+
                    d 6           4          0              Fe 2+ , Co 3+
                    d 7           3          1              Co 2+
                    Table: High and low spin octahedral transition metal complexes.
                    '''
        # ===================================================
        # Mn2 +
        # 5.916,  3d5 4s0, 5 unpaired e-, observed: 5.7 - 6.0 in [muB]
        # self.mu_spin_only = np.sqrt(5*(5+2))
        # Mn3 +
        # 5.916, 3d4 4s0, 4 unpaired e-, observed: 4.8 - 4.9 in [muB]
        # self.mu_spin_only = np.sqrt(4*(4+2))
        if self.Mn_type == 'Mn2+':
            if self.spin_type_cfg == 'high':
                self.J_total_momentum = 2.5 # high spin
            elif self.spin_type_cfg == 'low':
                self.J_total_momentum = 1.5 # low spin ?
            self.mu_eff = g_J_Mn2_plus
            self.mu_eff_min = 5.7
            self.mu_eff_max = 6.0

        elif self.Mn_type == 'Mn3+':
            if self.spin_type_cfg == 'low':
                self.J_total_momentum = 2.0 # ? low-spin, probably because mu_eff is 4.82 from the experiment
            elif self.spin_type_cfg == 'high':
                self.J_total_momentum = 0.0 # high-spin
            self.mu_eff = g_J_Mn3_plus
            self.mu_eff_min = 4.8
            self.mu_eff_max = 4.9

        self.g_factor = self.mu_eff / self.J_total_momentum

    def set_Mn2_plus_high(self):
        self.Mn_type = 'Mn2+'
        self.spin_type_cfg = 'high'
        self.define_Mn_type_variables()

    def set_Mn2_plus_low(self):
        self.Mn_type = 'Mn2+'
        self.spin_type_cfg = 'low'
        self.define_Mn_type_variables()

    def set_Mn3_plus_low(self):
        self.Mn_type = 'Mn3+'
        self.spin_type_cfg = 'low'
        self.define_Mn_type_variables()

    def save_magnetic_params_to_current_phase_obj(self):
        self.current_magnetic_phase_data.J_total_momentum = self.J_total_momentum
        self.current_magnetic_phase_data.Mn_type = self.Mn_type
        self.current_magnetic_phase_data.g_factor = self.g_factor
        self.current_magnetic_phase_data.spin_type_cfg = self.spin_type_cfg
        self.current_magnetic_phase_data.concentration_ParaMagnetic = self.concentration_ParaMagnetic
        self.current_magnetic_phase_data.concentration_ParaMagnetic_error = self.concentration_ParaMagnetic_error
        self.current_magnetic_phase_data.mu_eff = self.mu_eff


    def interpolate_data(self):

        x = np.array(self.raw.magnetic_field)
        y = np.array(self.raw.magnetic_moment)

        if self.magnetic_field_minimum == self.magnetic_field_maximum:
            self.magnetic_field_minimum = np.fix(10 * self.raw.magnetic_field.min()) / 10
            self.magnetic_field_maximum = np.fix(10 * self.raw.magnetic_field.max()) / 10

        if self.magnetic_field_minimum < self.raw.magnetic_field.min():
            self.magnetic_field_minimum = np.fix(10 * self.raw.magnetic_field.min()) / 10

        if self.magnetic_field_maximum > self.raw.magnetic_field.max():
            self.magnetic_field_maximum = np.fix(10 * self.raw.magnetic_field.max()) / 10

        self.fit.magnetic_field = \
            np.r_[self.magnetic_field_minimum: self.magnetic_field_maximum: self.magnetic_field_step]

        if self.typeOfFiltering == 'Linear':
            f = interp1d(self.raw.magnetic_field, self.raw.magnetic_moment)
            self.fit.magnetic_moment = f(self.fit.magnetic_field)
        if self.typeOfFiltering == 'Cubic':
            f = interp1d(self.raw.magnetic_field, self.raw.magnetic_moment, kind='cubic')
            self.fit.magnetic_moment = f(self.fit.magnetic_field)
        if self.typeOfFiltering == 'Spline':
            tck = splrep(x, y, s=0)
            self.fit.magnetic_moment = splev(self.fit.magnetic_field, tck, der=0)
        if self.typeOfFiltering == 'IUS':
            f = InterpolatedUnivariateSpline(self.raw.magnetic_field, self.raw.magnetic_moment)
            self.fit.magnetic_moment = f(self.fit.magnetic_field)
        if self.typeOfFiltering == 'RBF':
            f = Rbf(self.raw.magnetic_field, self.raw.magnetic_moment, function = 'linear')
            self.fit.magnetic_moment = f(self.fit.magnetic_field)

        if abs(self.magnetic_field_minimum) == abs(self.magnetic_field_maximum):
            self.y_shift = self.fit.magnetic_moment[-1] - abs(self.fit.magnetic_moment[0])
            self.fit.magnetic_moment = self.fit.magnetic_moment - self.y_shift
            self.fit.magnetic_moment_shift = self.y_shift

            yy_0 = np.r_[0:self.fit.magnetic_moment[-1]:self.fit.magnetic_moment[-1]/100]
            f_0 = interp1d(self.fit.magnetic_moment, self.fit.magnetic_field)
            xx_0 = f_0(yy_0)

            self.x_shift = xx_0[0]
            self.fit.magnetic_field = self.fit.magnetic_field - self.x_shift
            self.fit.magnetic_field_shift = self.x_shift


        # we need to adjust new self.fit.magnetic_field values to a good precision:
        if self.magnetic_field_minimum < self.fit.magnetic_field.min():
            self.magnetic_field_minimum = np.fix(10 * self.fit.magnetic_field.min()) / 10

        if self.magnetic_field_maximum > self.fit.magnetic_field.max():
            self.magnetic_field_maximum = np.fix(10 * self.fit.magnetic_field.max()) / 10

        xx = np.r_[self.magnetic_field_minimum: self.magnetic_field_maximum: self.magnetic_field_step]
        self.zeroIndex = np.nonzero((np.abs(xx) < self.magnetic_field_step*1e-2))
        xx[self.zeroIndex] = 0
        f = interp1d(self.fit.magnetic_field, self.fit.magnetic_moment)
        self.fit.magnetic_moment = f(xx)
        self.fit.magnetic_field = xx
        self.fit.append_history_log(
            case='do interpolation with type of filtering: {}'.format(self.typeOfFiltering))

        # store interpolated data of raw data in spacial object:
        self.prepared_raw = deepcopy(self.fit)
        self.for_fit = deepcopy(self.fit)
        self.line = deepcopy(self.fit)

        self.raw.label = 'raw: T={0}K'.format(self.raw.temperature)
        self.prepared_raw.label = 'prep raw: T={0}K {1}'.format(self.prepared_raw.temperature, self.typeOfFiltering)
        self.line.label = 'subtracted line: T={0}K {1}'.format(self.prepared_raw.temperature, self.typeOfFiltering)
        self.for_fit.label = 'selected points for fit: T={0}K {1}'.format(self.fit.temperature, self.typeOfFiltering)
        self.fit.label = 'fit: T={0}K {1}'.format(self.fit.temperature, self.typeOfFiltering)

    def filtering(self):
        # do some filtering operations under data:
        if self.magnetic_field_minimum == self.magnetic_field_maximum:
            self.magnetic_field_minimum = self.raw.magnetic_field.min()
            self.magnetic_field_maximum = self.raw.magnetic_field.max()
        self.fit.magnetic_field = np.r_[self.magnetic_field_minimum: self.magnetic_field_maximum: self.magnetic_field_step]
        window_size, poly_order = 101, 3
        self.fit.magnetic_moment = savgol_filter(self.fit.magnetic_moment, window_size, poly_order)
        self.fit.append_history_log(
            case='apply savgol filter: window = {}, poly_order = {}'.format(window_size, poly_order))

    def line_subtracting(self):
        indx_plus = (self.prepared_raw.magnetic_field >= self.magnetic_field_value_for_fit)
        indx_minus = (self.prepared_raw.magnetic_field <= -self.magnetic_field_value_for_fit)

        # > 0:
        indx = indx_plus
        self.for_fit.magnetic_field = self.prepared_raw.magnetic_field[indx]
        self.for_fit.magnetic_moment = self.prepared_raw.magnetic_moment[indx]
        # fit the data:
        par_plus, pcov = curve_fit(linearFunc,
                              self.for_fit.magnetic_field,
                              self.for_fit.magnetic_moment
                              )
        # < 0:
        indx = indx_minus
        self.for_fit.magnetic_field = self.prepared_raw.magnetic_field[indx]
        self.for_fit.magnetic_moment = self.prepared_raw.magnetic_moment[indx]
        # fit the data:
        par_minus, pcov = curve_fit(linearFunc,
                              self.for_fit.magnetic_field,
                              self.for_fit.magnetic_moment
                              )

        self.linear_coefficient = 0.5*(par_plus[0] + par_minus[0])
        self.line.magnetic_moment = (self.linear_coefficient * self.prepared_raw.magnetic_field)
        # store to for_fit object:
        indx = np.logical_or(indx_minus, indx_plus)
        self.for_fit.magnetic_field = self.prepared_raw.magnetic_field[indx]
        self.for_fit.magnetic_moment = self.prepared_raw.magnetic_moment[indx]

        self.prepared_raw.magnetic_moment -= (self.linear_coefficient * self.prepared_raw.magnetic_field)
        self.prepared_raw.append_history_log('line k={coef} * B was subtracted'.format(coef=self.linear_coefficient))
        self.line.label = 'subtracted line $M-\\mathbf{{k}} \\ast B$: $\\mathbf{{k}}={:1.5}$ '.format(self.linear_coefficient)
        # self.for_fit = deepcopy(self.prepared_raw)
        self.line.do_plot = True

    def fit_PM_single_phase(self):
        # do a fit procedure:
        # indx = np.argwhere(self.fit.magnetic_field >= 3)
        indx = (np.abs(self.prepared_raw.magnetic_field) >= self.magnetic_field_value_for_fit)
        # indx = ((self.prepared_raw.magnetic_field) >= self.magnetic_field_value_for_fit)
        self.for_fit.magnetic_field = self.prepared_raw.magnetic_field[indx]
        self.for_fit.magnetic_moment = self.prepared_raw.magnetic_moment[indx]
        self.forFit_x = (self.g_factor * self.J_total_momentum * mu_Bohr * self.for_fit.magnetic_field) \
                        / k_b / self.fit.temperature
        self.forFit_y = self.for_fit.magnetic_moment

        # try to fit concentration of Mn atoms n[1/m^3*1e27]
        def fun_tmp(x, n):
            return f_PM(x, n, J=self.J_total_momentum, g_factor=self.g_factor)

        popt, pcov = curve_fit(fun_tmp,
                               xdata=self.forFit_x,
                               ydata=self.forFit_y,
                               )
        self.concentration_ParaMagnetic = popt[0] #[1/m^3*1e27]
        self.concentration_ParaMagnetic_error = np.sqrt(np.diag(pcov[0]))

        # calc x values for all values of magnetic_field range:
        xx = (self.g_factor * self.J_total_momentum * mu_Bohr * self.fit.magnetic_field) \
             / k_b / self.fit.temperature

        self.fit.magnetic_moment = fun_tmp(xx, self.concentration_ParaMagnetic)
        # fight with uncertainty in 0 vicinity:
        self.fit.magnetic_moment[self.zeroIndex] = 0

        self.calc_R_factor(raw=self.prepared_raw.magnetic_moment, fit=self.fit.magnetic_moment)
        self.fit.label = \
        '\nfit [$R=\\mathbf{{{R:1.3}}}\%$, $\sigma=\\mathbf{{{std:1.3}}}$] ' \
        '\n$g_{{factor}}=\\mathbf{{{g_f:1.3}}}$, T={temper:2.1g}K\n'\
        '$J({Mn_type}$, ${spin_type})=\\mathbf{{{J:1.3}}}$ $[\mu_{{Bohr}}]$'\
        '\n$n_{{{Mn_type}}}=({conc:1.4g}\\pm{conc_error:1.4g})\\ast10^{{27}} [1/m^3]$' \
        '\n or $\\mathbf{{{conc_GaAs:1.3g}}}\%$ of $n(GaAs)$'.format(
                g_f=float(self.g_factor),
                R=float(self.R_factor),
                std=float(self.std),
                temper=float(self.fit.temperature),
                Mn_type=self.Mn_type,
                spin_type=self.spin_type_cfg,
                J=float(self.J_total_momentum),
                conc=float(self.concentration_ParaMagnetic),
                conc_error=float(np.round(self.concentration_ParaMagnetic_error,4)),
                conc_GaAs=float(self.concentration_ParaMagnetic / 22.139136 * 100),
        )

        print('->> fit PM (single PM phase) have been done. '
              'For T = {0} K obtained n = {1:1.3g} *1e27 [1/m^3] or {2:1.3g} % of the n(GaAs)' \
              .format(self.raw.temperature,
                      self.concentration_ParaMagnetic,
                      self.concentration_ParaMagnetic / 22.139136 * 100))
        print('->> R = {R_f:1.5g} %'.format(R_f=self.R_factor))
        print('->> J[{Mn_type}, {spin_type}] = {J:1.3} [mu(Bohr)]'.format(
            Mn_type=self.Mn_type,
            spin_type=self.spin_type_cfg,
            J=float(self.J_total_momentum)))

    def multi_phase_PM_func(self, n_concentration, magnetic_field, temperature, zero_index=None):
        # calc Brillouin function for multi-phase sample
        num = len(self.dict_of_magnetic_phases)
        len_of_x = len(magnetic_field)
        # vector for magnetic calculation:
        vec_x = np.zeros(len_of_x)
        #concentration of Mn atoms n[1/m^3*1e27]
        out = np.zeros(len_of_x)
        for i in self.dict_of_magnetic_phases:
            val = self.dict_of_magnetic_phases[i]
            n = n_concentration[i]
            J = val['data'].J_total_momentum
            g = val['data'].g_factor
            Mn_type = val['data'].Mn_type
            spin_type = val['data'].spin_type_cfg
            tmp = np.zeros(len_of_x)
            # create unique x-vector for Brillouin function:
            vec_x = (g * J * mu_Bohr * magnetic_field) \
                        / k_b / temperature
            # tmp = f_PM(x=vec_x, n=n, J=J, g_factor=g)
            tmp = f_PM_with_T(B=magnetic_field, n=n, J=J, T=temperature, g_factor=g)
            # fight with uncertainty in 0 vicinity:
            if zero_index is not None:
                tmp[zero_index] = 0
            out += tmp
        return out

    def fit_PM_multi_phase(self):
        # do a fit procedure for the multi-phases magnetic material:
        # indx = np.argwhere(self.fit.magnetic_field >= 3)
        indx = (np.abs(self.prepared_raw.magnetic_field) >= self.magnetic_field_value_for_fit)
        # indx = ((self.prepared_raw.magnetic_field) >= self.magnetic_field_value_for_fit)
        self.for_fit.magnetic_field = self.prepared_raw.magnetic_field[indx]
        self.for_fit.magnetic_moment = self.prepared_raw.magnetic_moment[indx]

        self.forFit_y = self.for_fit.magnetic_moment
        len_of_vec = len(self.forFit_y)
        self.forFit_x = self.for_fit.magnetic_field

        num = len(self.dict_of_magnetic_phases)

        # try to fit concentration of Mn atoms n[1/m^3*1e27]

        # construct tmp function for a minimization procedure:
        def fun_tmp(n_concentration):
            out = np.zeros(len_of_vec)
            out = self.multi_phase_PM_func(n_concentration,
                                           magnetic_field=self.forFit_x,
                                           temperature=self.fit.temperature)
            return self.get_R_factor(raw=self.forFit_y, fit=out)

        # create bounds:
        bounds = []
        for i in range(num):
            bounds.append((0, 10))

        res = differential_evolution(fun_tmp, bounds)

        self.concentration_ParaMagnetic = res.x #[1/m^3*1e27]
        s2 = self.get_std(raw=self.forFit_y,
                          fit=self.multi_phase_PM_func(self.concentration_ParaMagnetic,
                                                       magnetic_field=self.forFit_x,
                                                       temperature=self.fit.temperature
                                                       )
                          )
        se = approx_errors(fun_tmp, self.concentration_ParaMagnetic, epsilon=0.01*np.min(self.concentration_ParaMagnetic))
        std = np.sqrt(s2) * se
        self.concentration_ParaMagnetic_error = std

        self.fit.magnetic_moment = self.multi_phase_PM_func(
            self.concentration_ParaMagnetic,
            magnetic_field=self.fit.magnetic_field,
            temperature=self.fit.temperature,
            zero_index=self.zeroIndex
        )
        # fight with uncertainty in 0 vicinity:
        # self.fit.magnetic_moment[self.zeroIndex] = 0

        self.calc_R_factor(raw=self.prepared_raw.magnetic_moment, fit=self.fit.magnetic_moment)

        # write label for plotting:
        self.fit.label = \
            '\nFit [$R=\\mathbf{{{R:1.3}}}\%$, $\sigma=\\mathbf{{{std:1.3}}}$], T={temper:2.1g}K\n\n'.format(
                R=float(self.R_factor),
                std=float(self.std),
                temper=float(self.fit.temperature),
            )
        tmp_txt = ''
        for i in self.dict_of_magnetic_phases:
            val = self.dict_of_magnetic_phases[i]
            n = self.concentration_ParaMagnetic[i]
            n_std = self.concentration_ParaMagnetic_error[i]
            J = val['data'].J_total_momentum
            g = val['data'].g_factor
            Mn_type = val['data'].Mn_type
            spin_type = val['data'].spin_type_cfg
            tmp_txt += 'phase $\\mathbf{{{num_of_phase:}}}$:\n'\
            '$g_{{factor}}=\\mathbf{{{g_f:1.3}}}$, $J({Mn_type}$, ${spin_type})=\\mathbf{{{J:1.3}}}$ $[\mu_{{Bohr}}]$'\
                .format(
                num_of_phase = i,
                g_f=float(g),
                Mn_type=Mn_type,
                J=float(J),
                spin_type=spin_type,
            )
            tmp_txt += '\n$n_{{{Mn_type}}}=({conc:1.4g}\\pm{conc_error:1.4g})\\ast10^{{27}} [1/m^3]$' \
                '\n or $\\mathbf{{{conc_GaAs:1.3g}}}\%$ of $n(GaAs)$\n'.format(
                        Mn_type=Mn_type,
                        spin_type=spin_type,
                        conc=float(n),
                        conc_error=float(np.round(n_std, 4)),
                        conc_GaAs=float(n / 22.139136 * 100),
                )

        self.fit.label += tmp_txt
        print('==='*15)
        print('    fit PM (multi PM phases) have been done. '
              'For T = {0} K obtained:' \
              .format(self.raw.temperature,)
              )
        print('    R = {R_f:1.5g} %'.format(R_f=self.R_factor))

        for i in self.dict_of_magnetic_phases:
            val = self.dict_of_magnetic_phases[i]
            n = self.concentration_ParaMagnetic[i]
            n_std = self.concentration_ParaMagnetic_error[i]
            J = val['data'].J_total_momentum
            g = val['data'].g_factor
            Mn_type = val['data'].Mn_type
            spin_type = val['data'].spin_type_cfg
            print('-------   phases #{}:'.format(i))
            print(' n = ( {n:1.3g} +/- {err:1.4g} )*1e27 [1/m^3] or {n_2:1.3g} % of the n(GaAs)'.format(
                          n=n,
                          n_2=n / 22.139136 * 100,
                          err=n_std
                        )
                 )

            print(' J[{Mn_type}, {spin_type}] = {J:1.3} [mu(Bohr)]'.format(
                Mn_type=Mn_type,
                spin_type=spin_type,
                J=float(J)))

        print('===' * 15)

    def set_default_line_params(self):
        self.raw.line_style = 'None'
        self.raw.line_marker_size = 6
        self.raw.line_alpha = 0.2
        self.raw.line_marker_face_color = next(MagneticData.COLOR_CYCOL)

        self.line.line_style = '-'
        self.line.do_plot = False
        self.line.line_width = 3
        self.line.line_color = 'r'
        self.line.line_alpha = 0.3
        self.line.line_marker_style = 'None'

        self.prepared_raw.line_style = 'None'
        self.prepared_raw.line_marker_size = 6
        self.prepared_raw.line_marker_style = 'v'
        self.prepared_raw.line_alpha = 0.3
        self.prepared_raw.line_marker_face_color = next(MagneticData.COLOR_CYCOL)

        self.for_fit.line_style = 'None'
        self.for_fit.line_marker_size = 12
        self.for_fit.line_marker_style = 'D'
        self.for_fit.line_alpha = 0.2
        self.for_fit.line_marker_face_color = 'g'
        self.for_fit.line_marker_edge_color = next(MagneticData.COLOR_CYCOL)

        self.fit.line_style = 'None'
        self.fit.line_marker_size = 9
        self.fit.line_alpha = 0.3
        self.fit.line_marker_face_color = next(MagneticData.COLOR_CYCOL)

    def plot(self, ax=plt.gca()):

        self.raw.plot(ax)

        self.line.plot(ax)

        self.prepared_raw.plot(ax)

        self.for_fit.plot(ax)

        self.fit.plot(ax)

        ax.set_ylabel(self.y_label, fontsize=16, fontweight='bold')
        ax.set_xlabel(self.x_label, fontsize=16, fontweight='bold')
        ax.grid(True)
        ax.set_title(self.title, fontsize=self.font_size)
        # ax.legend(shadow=True, fancybox=True, loc='best')
        ax.legend(shadow=False, fancybox=True, loc='best')
        # ax.fill_between(x, y - error, y + error,
        #                 alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        #                 linewidth=4, linestyle='dashdot', antialiased=True, label='$\chi(k)$')

    def calc_R_factor(self, raw=[], fit=[]):
        # eval R-factor
        denominator = np.sum(np.abs(raw))
        if (len(raw) == len(fit)) and (denominator != 0):
            self.R_factor = 100 * np.sum(np.abs(raw - fit))/denominator
            self.std = np.sqrt(
                np.sum(
                    (raw - fit)**2
                )  / ( len(raw) -1 )
            )
        else:
            print('raw = {} and fit = {}'.format(len(raw), len(fit)))

    def get_R_factor(self, raw=[], fit=[]):
        self.calc_R_factor(raw, fit)
        return self.R_factor

    def get_std(self, raw=[], fit=[]):
        self.calc_R_factor(raw, fit)
        return self.std


class StructComplex(StructBase):
    def __init__(self):
        super().__init__()
        self.model_A = StructBase()
        self.model_B = StructBase()

    def set_global_Mn2_plus_high(self):
        self.model_A.set_Mn2_plus_high()
        self.model_B.set_Mn2_plus_high()

    def set_global_Mn2_plus_low(self):
        self.model_A.set_Mn2_plus_low()
        self.model_B.set_Mn2_plus_low()

    def set_global_Mn3_plus_low(self):
        self.model_A.set_Mn3_plus_low()
        self.model_B.set_Mn3_plus_low()

    def find_common_region_for_fit(self):
        if len(self.model_A.prepared_raw.magnetic_field) != len(self.model_B.prepared_raw.magnetic_field):
            print('len(T={T1}K)={L1} but len(T={T2}K)={L2}'.format(
                T1=self.model_A.prepared_raw.temperature,
                L1=len(self.model_A.prepared_raw.magnetic_field),
                T2=self.model_B.prepared_raw.temperature,
                L2=len(self.model_B.prepared_raw.magnetic_field)
            ))

        # Find the intersection of two arrays to avoid conflict with numbers of elements.
        self.model_A.prepared_raw.accepted_indices = np.nonzero(
            np.isin(self.model_A.prepared_raw.magnetic_field,
                    self.model_B.prepared_raw.magnetic_field))

        self.model_B.prepared_raw.accepted_indices = np.nonzero(
            np.isin(self.model_B.prepared_raw.magnetic_field,
                    self.model_A.prepared_raw.magnetic_field))

    def prepare_data_for_diff_calc(self):
        self.raw.magnetic_field = self.model_A.prepared_raw.magnetic_field[self.model_A.prepared_raw.accepted_indices]
        # for calculating diff_PM we need 2 different Temperature data for ex: m(T=2K) - m(T=5K)
        # select only common points in two models:
        self.raw.magnetic_moment = \
            self.model_A.prepared_raw.magnetic_moment[self.model_A.prepared_raw.accepted_indices] - \
            self.model_B.prepared_raw.magnetic_moment[self.model_B.prepared_raw.accepted_indices]

        self.prepared_raw = deepcopy(self.raw)

        # shift spectra to the center line:
        if len(self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field > 0)]) \
                != \
                len(self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field < 0)]):
            # reduce a noise:
            negVal = abs(np.min(self.prepared_raw.magnetic_field))
            pozVal = np.max(self.prepared_raw.magnetic_field)

            if pozVal >= negVal:
                limitVal = negVal
            else:
                limitVal = pozVal

            eps = 0.001 * abs(abs(self.prepared_raw.magnetic_field[0]) - abs(self.prepared_raw.magnetic_field[1]))
            self.prepared_raw.magnetic_field = self.prepared_raw.magnetic_field[
                np.logical_or((np.abs(self.prepared_raw.magnetic_field) <= limitVal + eps),
                              (np.abs(self.prepared_raw.magnetic_field) <= eps))
            ]

            Mp = []
            Mn = []
            Mp = self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field > 0)]
            Mn = self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field < 0)]

        if len(self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field > 0)]) \
                == \
                len(self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field < 0)]):
            # reduce a noise:
            Mp = self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field > 0)]
            Mn = self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field < 0)]

        self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field > 0)] = \
            0.5 * (Mp + np.abs(Mn[::-1]))
        self.prepared_raw.magnetic_moment[np.where(self.prepared_raw.magnetic_field < 0)] = \
            0.5 * (Mn - np.abs(Mp[::-1]))
        # M_for_fit[(B > 0)] = 0.5*(Mp + np.abs(Mn))
        # M_for_fit[(B < 0)] = 0.5*(Mn - np.abs(Mp))

        self.for_fit = deepcopy(self.prepared_raw)
        self.fit = deepcopy(self.prepared_raw)



if __name__ == '__main__':
    tmp_obj = StructComplex()

    print('-> you run ', __file__, ' file in a main mode')