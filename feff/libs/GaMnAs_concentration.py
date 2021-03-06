import sys
import os
from io import StringIO
import inspect
import numpy as np
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

g_J_Mn2_plus = 5.82
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





class Struct:
    '''
    Describe structure for a data
    '''
    def __init__(self):
        self.H = []
        self.M = []
        self.Mraw = [] # raw data
        self.T = 2
        self.H_inflection = 30
        self.Hstep = 0.1 #[T]
        self.Hmin = 0
        self.Hmax = 0
        self.J_total_momentum = 2.5
        self.Mn_type = 'Mn2+' # Mn2+ or Mn3+



        self.volumeOfTheFilm_GaMnAs = 0 #[m^3]
        self.yy_fit = []
        self.forFit_y = []
        self.forFit_x = []
        self.zeroIndex = []

        # number of density for PM fit only for the current temperature:
        self.n_PM = 0


        # corrections for curve:
        self.y_shift = 0
        self.x_shift = 0

        # fit does not calculated:
        self.fitWasDone = False

        # 'IUS' - Interpolation using univariate spline
        # 'RBF' - Interpolation using Radial basis functions
        # Interpolation using RBF - multiquadrics
        # 'Spline'
        # 'Cubic'
        # 'Linear'
        self.typeOfFiltering = 'IUS'

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
            self.J_total_momentum = 2.5 # high spin
            self.mu_eff = g_J_Mn2_plus
        elif self.Mn_type == 'Mn3+':
            self.J_total_momentum = 2 # low-spin
            self.mu_eff = g_J_Mn3_plus

    def interpolate(self):

        if self.Hmin == self.Hmax:
            self.Hmin = np.fix(10*self.H.min())/10
            self.Hmax = np.fix(10*self.H.max())/10

        if self.Hmin < self.H.min():
            self.Hmin = np.fix(10*self.H.min())/10

        if self.Hmax > self.H.max():
            self.Hmax = np.fix(10*self.H.max())/10

        self.xx = np.r_[self.Hmin: self.Hmax: self.Hstep]
        x = np.array(self.H)
        y = np.array(self.M)

        if self.typeOfFiltering == 'Linear':
            f = interp1d(self.H, self.M)
            self.yy = f(self.xx)
        if self.typeOfFiltering == 'Cubic':
            f = interp1d(self.H, self.M, kind='cubic')
            self.yy = f(self.xx)
        if self.typeOfFiltering == 'Spline':
            tck = splrep(x, y, s=0)
            self.yy = splev(self.xx, tck, der=0)
        if self.typeOfFiltering == 'IUS':
            f = InterpolatedUnivariateSpline(self.H, self.M)
            self.yy = f(self.xx)
        if self.typeOfFiltering == 'RBF':
            f = Rbf(self.H, self.M, function = 'linear')
            self.yy = f(self.xx)

        if abs(self.Hmin) == abs(self.Hmax):
            self.y_shift = self.yy[-1] - abs(self.yy[0])
            self.yy = self.yy - self.y_shift

            yy_0 = np.r_[0:self.yy[-1]:self.yy[-1]/100]
            f_0 = interp1d(self.yy, self.xx)
            xx_0 = f_0(yy_0)

            self.x_shift = xx_0[0]
            self.xx = self.xx - self.x_shift


        # we need to adjust new self.xx values to a good precision:
        if self.Hmin < self.xx.min():
            self.Hmin = np.fix(10*self.xx.min())/10

        if self.Hmax > self.xx.max():
            self.Hmax = np.fix(10*self.xx.max())/10

        xx = np.r_[self.Hmin: self.Hmax: self.Hstep]
        self.zeroIndex = np.nonzero((np.abs(xx) < self.Hstep*1e-2))
        xx[self.zeroIndex] = 0
        f = interp1d(self.xx, self.yy)
        self.yy = f(xx)
        self.xx = xx





    def filtering(self):
        # do some filtering operations under data:
        if self.Hmin == self.Hmax:
            self.Hmin = self.H.min()
            self.Hmax = self.H.max()
        self.xx = np.r_[self.Hmin: self.Hmax: self.Hstep]
        window_size, poly_order = 101, 3
        self.yy = savgol_filter(self.yy, window_size, poly_order)

    def fit_PM(self):
        # do a fit procedure:
        # indx = np.argwhere(self.xx >= 3)
        indx = (self.xx >= 3)
        B = self.xx[indx]
        M = self.yy[indx]
        self.forFit_x = (g_e * self.J_total_momentum * mu_Bohr * B) / k_b / self.T
        self.forFit_y = M
        n = return_fit_param(self.forFit_x, self.forFit_y) # [1/m^3*1e27]

        # try to fit by using 2 params: n and J
        # initial_guess = [0.01, 2.5]
        # popt, pcov = curve_fit(f_PM, (g_e*self.J*mu_Bohr*self.xx[(self.xx >= 0)])/k_b/self.T, self.yy[(self.xx >= 0)],
        #                        p0= initial_guess, bounds=([0, 0], [np.inf, np.inf]))
        # yy = f_PM((g_e*self.J*mu_Bohr*self.xx)/k_b/self.T, popt[0], popt[1])

        # plt.plot(B, func(self.forFit_x, n), 'r-', B, M, '.-')

        self.yy_fit = func((g_e * self.J_total_momentum * mu_Bohr * self.xx) / k_b / self.T, n)

        self.yy_fit[self.zeroIndex] = 0

        # plt.plot(self.xx, self.yy_fit, 'r-', self.xx, self.yy, '.-', self.xx, yy, 'x')

        self.n_PM = n[0]
        print('->> fit PM have been done. For T = {0} K obtained n = {1:1.3g} *1e27 [1/m^3] or {2:1.3g} % of the n(GaAs)'\
              .format(self.T, self.n_PM, self.n_PM/22.139136*100))
        self.fitWasDone = True



    def plot(self, ax):
        ax.plot(self.xx, self.yy, 'k-', label='T={0}K {1}'.format(self.T, self.typeOfFiltering))
        ax.plot(self.H, self.M, 'x', label='T={0}K raw'.format(self.T))
        if self.fitWasDone:
            ax.plot(self.xx, self.yy_fit,  'r-', label='T={0}K fit_PM_single_phase'.format(self.T))
        ax.set_ylabel('$Moment (A/m)$', fontsize=20, fontweight='bold')
        ax.set_xlabel('$B (T)$', fontsize=20, fontweight='bold')
        ax.grid(True)
        # ax.fill_between(x, y - error, y + error,
        #                 alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        #                 linewidth=4, linestyle='dashdot', antialiased=True, label='$\chi(k)$')

    def plotLogT(self, ax, H1 = 10, H2 = 300):
        # Plot graph log(1/rho) vs 1/T to define the range with a different behavior of charge currents
        a = self.xx
        # find indeces for elements inside the region [T1, T2]:
        ind = np.where(np.logical_and(a >= H1, a <= H2))
        x = 1/(self.xx[ind]**1)
        # y = np.log(self.yy[ind])
        y = self.yy[ind]**(-1)
        ax.plot(x, y, 'o-', label='T={0}K'.format(self.T))
        # ax.set_ylabel('$ln(1/\\rho)$', fontsize=20, fontweight='bold')
        ax.set_ylabel('$(\sigma)$', fontsize=20, fontweight='bold')
        ax.set_xlabel('$1/T^{1}$', fontsize=20, fontweight='bold')
        ax.grid(True)






m_B180v = 14.3 #mg
m_B180c = 12.2 #mg
m_B180b = 13.5 #mg
m_B180a = 17.2 #mg

t = np.array([14.3, 12.2, 13.5, 17.2])*1e-6/(1.5e-3*5e-3*rho_GaAs)
# to = t.mean()*1000 #mm


class ConcentrationOfMagneticIons:
    def __init__(self):
        self.selectCase = 'B180v'

        self.mass_kg = 0.0
        self.magnetic_moment_saturation_experiment = 0.0
        self.how_many_Mn_in_percent = 2.3 #[%]
        self.dataFolderSorceBase = '/home/yugin/VirtualboxShare/FEFF/Origin_Sawicki_measur (B180)'
        self.dataFolderSorce = 'B180v[s]m(H)-dia'

        # We suppose that the weight measurements has more accuracy then spatial measurements in the typical lab conditions.
        # There for we calculate Area size for each sample from the density of pure GaAs and from the information about
        #  typical substrate thickness (Y.Syryanyy measured some pieces of the old samples and obtained t0=0.6mm)
        self.t0 = 0.6e-3 # thickness of the GaAs substrate 0.6mm [m]

        # thickness of the investigated film on the sample:
        self.h_film = 400e-9 #[m] 400nm

        # magnetic field region for calculation:
        self.Hmin = 0
        self.Hmax = 0

        self.struct_of_data = {}

        # for calculating diff_PM we need 2 different Temperature data for ex: m(T=2K) - m(T=5K)
        self.diff_PM_keyName1 = '2'
        self.diff_PM_keyName2 = '5'

        # Select temperature for Langevin fit procedure (SuperParaMagnetic state):
        self.SPM_keyName = '50'
        self.SPM_field_region_for_fit = np.array([[0, 6]]) # [T]

        self.diff_PM_field_region_for_fit = np.array([[-6, -3], [3, 6]]) # [T]

        self.FM_field_region_for_fit = np.array([[-6, -2], [2, 6]]) # [T]

    def prepareData(self):
        # load and prepare data to the next calculation

        self.mass_Molar_kg = ( (69.723 + 74.9216)*(100-self.how_many_Mn_in_percent)/100
                               + 54.938*self.how_many_Mn_in_percent/100 )/1000

        if self.selectCase == 'B180v':
            self.mass_kg = m_B180v/1e6 #[kg]
            # when para and dia was subtracted:
            self.magnetic_moment_saturation_experiment = 5150e-8 # emu
            # when only dia was subtracted:
            self.magnetic_moment_saturation_experiment = 6950e-8 # emu
            self.filmArea = 8.72925e-6 #[m^2]

            self.Hmin = -6#[T]
            self.Hmax = 8 #[T]

            self.dataFolderSorce = os.path.join(self.dataFolderSorceBase, 'B180v[s]m(H)-dia')

        if self.selectCase == 'B180c':
            self.mass_kg = m_B180c/1e6 #[kg]
            self.magnetic_moment_saturation_experiment = 2300e-8 # emu
            self.filmArea = 7.2024361e-6 #[m^2]

            self.Hmin = -6#[T]
            self.Hmax = 6 #[T]

            self.dataFolderSorce = os.path.join(self.dataFolderSorceBase, 'B180c[s]m(H)-dia')

        if self.selectCase == 'B180b':
            self.mass_kg = m_B180b/1e6 #[kg]
            self.magnetic_moment_saturation_experiment = 5150e-8 # emu
            self.filmArea = 8.3036995e-6 #[m^2]
            self.dataFolderSorce = os.path.join(self.dataFolderSorceBase, 'B180b[s]m(H)-dia')

            self.Hmin = -6#[T]
            self.Hmax = 6 #[T]
            self.diff_PM_field_region_for_fit = np.array([[-6, -0.1], [0.1, 6]]) # [T]

        if self.selectCase == 'B180a':
            self.mass_kg = m_B180a/1e6 #[kg]
            self.magnetic_moment_saturation_experiment = 2300e-8 # emu
            self.filmArea = 10.111943e-6 #[m^2]
            self.dataFolderSorce = os.path.join(self.dataFolderSorceBase, 'B180a[s]m(H)-dia')

            self.Hmin = -6#[T]
            self.Hmax = 6 #[T]


        self.area = self.mass_kg/rho_GaAs/self.t0 #[m^2]

        self.area = self.filmArea #[m^2]

        self.volumeOfTheFilm_GaMnAs = self.area * self.h_film #[m^3]

        files_lsFN = listOfFilesFN_with_selected_ext(self.dataFolderSorce, ext = 'dat')
        files_ls = list((os.path.basename(i) for i in files_lsFN))

        # setup variables for plot:
        self.suptitle_txt = 'case: {}, '.format(self.selectCase) + '$Ga_{1-x}Mn_xAs$,' \
                                + ' where $x = {0:1.2g}\%$ '.format(self.how_many_Mn_in_percent)
        self.ylabel_txt = 'M (A/m)'
        self.xlabel_txt = '$B (T)$'


        self.setupAxes()

        indx = 0
        parameter = list(int(float(re.findall("\d+", k)[0]) * 1) for k in files_ls)
        sortedlist = [i[0] for i in sorted(zip(files_lsFN, parameter), key=lambda l: l[1], reverse=False)]
        for i in sortedlist:
            tmp_data = np.loadtxt(i, float, skiprows=1)
            tmp_data = sortMatrixByFirstColumn(Array = fromRowToColumn(tmp_data), colnum = 0)
            case_name = re.findall('[\d\.\d]+', os.path.basename(i).split('.')[0])
            # create a new instance for each iteration step otherwise struct_of_data has one object for all keys
            data = Struct()
            data.T = float(case_name[0])
            data.H = tmp_data[:, 0]/10000 #[T]
            data.Mraw = tmp_data[:, 1]
            data.M = from_EMU_cm3_to_A_by_m(moment_emu = data.Mraw, V_cm3 = self.volumeOfTheFilm_GaMnAs*1e6)

            data.Hmin = self.Hmin
            data.Hmax = self.Hmax
            data.volumeOfTheFilm_GaMnAs = self.volumeOfTheFilm_GaMnAs

            # 'IUS' - Interpolation using univariate spline
            # 'RBF' - Interpolation using Radial basis functions
            # Interpolation using RBF - multiquadrics
            # 'Spline'
            # 'Cubic'
            # 'Linear'
            data.typeOfFiltering = 'Linear'
            data.interpolate()
            data.fit_PM()

            data.plot(self.ax)
            plt.draw()
            self.ax.legend(shadow=True, fancybox=True, loc='best')


            self.struct_of_data[case_name[0]] = data

            print('T = ', self.struct_of_data[case_name[0]].T, ' K')
            indx = indx+1

        sortKeys = sorted(self.struct_of_data, key=lambda key: self.struct_of_data[key].T)
        for i in sortKeys:
            self.ax.cla()
            # struct_of_data[i].plotLogT(self.ax)
            self.struct_of_data[i].plot(self.ax)

            self.ax.legend(shadow=True, fancybox=True, loc='upper left')
            plt.draw()
            # print(list(struct_of_data.items())[i])

    def calc_diff_PM(self):
        if len(self.struct_of_data) > 1:

            T1 = self.struct_of_data[self.diff_PM_keyName1].T
            T2 = self.struct_of_data[self.diff_PM_keyName2].T
            def fun_diff(B, n):
                return f_diff_PM_for_2_T (B, n, J=2.5, T1=T1, T2=T2)


            initial_guess = [0.01]

            if len(self.struct_of_data[self.diff_PM_keyName1].xx) != len(self.struct_of_data[self.diff_PM_keyName2].xx):
                print('len(T={0}K)={1} but len(T={2}K)={3}'.format(T1, len(self.struct_of_data[self.diff_PM_keyName1].xx),
                                                                   T2, len(self.struct_of_data[self.diff_PM_keyName2].xx)) )

            # Find the intersection of two arrays to avoid conflict with numbers of elements.
            indices1 = np.nonzero(np.in1d(self.struct_of_data[self.diff_PM_keyName1].xx,
                               self.struct_of_data[self.diff_PM_keyName2].xx))

            indices2 = np.nonzero(np.in1d(self.struct_of_data[self.diff_PM_keyName2].xx,
                               self.struct_of_data[self.diff_PM_keyName1].xx))

            B = self.struct_of_data[self.diff_PM_keyName1].xx[indices1]
            # for calculating diff_PM we need 2 different Temperature data for ex: m(T=2K) - m(T=5K)
            M = self.struct_of_data[self.diff_PM_keyName1].yy[indices1] - \
                self.struct_of_data[self.diff_PM_keyName2].yy[indices2]

            M_for_fit = np.copy(M)

            if abs(self.struct_of_data[self.diff_PM_keyName1].Hmin) == self.struct_of_data[self.diff_PM_keyName1].Hmax:
                if len(M[np.where(B > 0)]) != len(M[np.where(B < 0)]):
                    # reduce a noise:
                    negVal = abs(np.min(B))
                    pozVal = np.max(B)

                    if pozVal >= negVal:
                        limitVal = negVal
                    else:
                        limitVal = pozVal

                    eps = 0.001*abs(abs(B[0])-abs(B[1]))
                    B = B[np.logical_or( (np.abs(B) <= limitVal + eps), (np.abs(B) <= eps) )]

                    Mp = M[np.where(B > 0)]
                    Mn = M[np.where(B < 0)]

                if len(M[np.where(B > 0)]) == len(M[np.where(B < 0)]):
                    # reduce a noise:
                    Mp = M[np.where(B > 0)]
                    Mn = M[np.where(B < 0)]

                M_for_fit[np.where(B > 0)] = 0.5*(Mp + np.abs(Mn[::-1]))
                M_for_fit[np.where(B < 0)] = 0.5*(Mn - np.abs(Mp[::-1]))
                # M_for_fit[(B > 0)] = 0.5*(Mp + np.abs(Mn))
                # M_for_fit[(B < 0)] = 0.5*(Mn - np.abs(Mp))

            if self.diff_PM_field_region_for_fit[0, 1] < B.min():
                self.diff_PM_field_region_for_fit[0, 1] = np.fix(10*B.min())/10 + \
                                                          5*self.struct_of_data[self.diff_PM_keyName1].Hstep

            condlist1 = (self.diff_PM_field_region_for_fit[0, 0] <= B) & (self.diff_PM_field_region_for_fit[0, 1] >= B)
            condlist2 = (self.diff_PM_field_region_for_fit[1, 0] <= B) & (self.diff_PM_field_region_for_fit[1, 1] >= B)
            # fit the data:
            n, pcov = curve_fit(fun_diff, B[np.logical_or(condlist1, condlist2)],
                                M_for_fit[np.where(np.logical_or(condlist1, condlist2))],
                                p0=initial_guess, bounds=([0, ], [np.inf, ]))

            self.n_diffPM = n[0]
            print('->> fit diff PM have been done. For m(T={0}K) - m(T={1}K) obtained n = {2:1.3g} *1e27 [1/m^3] or {3:1.3g} % of the n(GaAs)'\
              .format(self.diff_PM_keyName1, self.diff_PM_keyName2, self.n_diffPM, self.n_diffPM/22.139136*100))

            # setup variables for plot:
            self.ax.cla()
            self.suptitle_txt = 'case: {}, '.format(self.selectCase) + '$Ga_{1-x}Mn_xAs$,' \
                                + ' where $x = {0:1.2g}\%$ '.format(self.how_many_Mn_in_percent)
            self.ylabel_txt = '$M_{{T={0}K}}\,-\,M_{{T={1}K}}$ $(A/m)$'.format(T1, T2)
            self.xlabel_txt = '$B$ $(T)$'
            self.setupAxes()

            inx_condlist = np.logical_or(condlist1, condlist2)
            inx = B.nonzero()
            self.ax.scatter(B[inx_condlist],
                            M_for_fit[np.where(inx_condlist)],
                            label='region for $fit$',
                            facecolor='none',
                            alpha=.25,
                            s=200, marker='s', edgecolors='b',
                            )
            self.ax.plot(B, fun_diff(B, n), 'r-',
                         label='$fit\,M_S*\left[B_J(T={0}K)\,-\,B_J(T={1}K)\\right]$,\n $n_{{PM}}={2:1.3g}\%,\; n_{{[PM,\;x={3:1.3g}\%]}}={4:1.3g}\%$'.
                         format(T1, T2, self.n_diffPM/22.139136*100, self.how_many_Mn_in_percent,
                                self.n_diffPM/22.139136*100/self.how_many_Mn_in_percent*100))
            self.ax.scatter(B[inx], M[inx],  label='raw $M(T={0}K)\,-\,M(T={1}K)$'.format(T1, T2), alpha=.4, color='g', marker='x', s=100)
            self.ax.scatter(B[inx], M_for_fit[inx], label='for $fit$ $\\frac{M_++M_-}{2}$', color='k', alpha=.2, s=70)
            self.ax.legend(shadow=True, fancybox=True, loc='upper left')
            self.ax.grid(True)
            plt.draw()
            plt.draw()

    def subtract_PM_after_calc_diff_PM(self):
        if not(self.n_diffPM == 0):
            B = self.struct_of_data[self.diff_PM_keyName1].xx
            M = self.struct_of_data[self.diff_PM_keyName1].yy
            T = self.struct_of_data[self.diff_PM_keyName1].T
            J = 2.5
            M_sub_PM = M - f_PM_with_T(B=B, n=self.n_diffPM, J=J, T=T)

            self.ax.cla()

            print('->> sub PM after diff PM have been done. For m(T={0}K) - m(T={1}K) obtained n = {2:1.3g} *1e27 [1/m^3] or {3:1.3g} % of the n(GaAs)'\
                  .format(self.diff_PM_keyName1, self.diff_PM_keyName2, self.n_diffPM, self.n_diffPM/22.139136*100))

            # setup variables for plot:
            self.suptitle_txt = 'case: {}, '.format(self.selectCase) + '$Ga_{1-x}Mn_xAs$,' \
                                + ' where $x = {0:1.2g}\%$ '.format(self.how_many_Mn_in_percent)
            self.ylabel_txt = '$M\,(A/m)$'
            self.xlabel_txt = '$B\,(T)$'
            self.setupAxes()

            self.ax.plot(B, f_PM_with_T(B=B, n=self.n_diffPM, J=J, T=T), 'r-',
                         label='PM component $\left(T={0:1.2g}K \\right)$, '.format(T) +\
                        '$n_{{PM}}={:1.3g}\%$,'.format(self.n_diffPM / 22.139136 * 100) + \
                             '\n$n_{{[PM,\;x={0:1.3g}\%]}}={1:1.3g}\%$'.format(self.how_many_Mn_in_percent,
                            self.n_diffPM / 22.139136 * 100/self.how_many_Mn_in_percent*100) + \
                        ' $J={:1.2g}(Mn^{{2+}})$'.format(J/2.5)
                         )
            self.ax.scatter(B, M,  label='raw $\left(T={0:1.2g}K \\right)$'.format(T), alpha=.2, color='g')
            self.ax.scatter(B, M_sub_PM, label='$M_{raw}-M_{PM}$'+' $\left(T={0:1.2g}K \\right)$'.format(T), color='k', alpha=.2)
            self.ax.legend(shadow=True, fancybox=True, loc='upper left')
            self.ax.grid(True)
            plt.draw()

            plt.draw()

    def subtract_Line(self):
        # try to subtract BG line: y=kx+b, which may be correspond to anisotropy effect
        # and then calculate FM phase concentration
        if not(self.n_diffPM == 0):
            B = self.struct_of_data[self.diff_PM_keyName1].xx
            M = self.struct_of_data[self.diff_PM_keyName1].yy
            T = self.struct_of_data[self.diff_PM_keyName1].T
            J = 2.5
            M_sub_PM = M - f_PM_with_T(B=B, n=self.n_diffPM, J=J, T=T)

            condlist1 = (self.FM_field_region_for_fit[0, 0] < B) & (self.FM_field_region_for_fit[0, 1] > B)
            condlist2 = (self.FM_field_region_for_fit[1, 0] < B) & (self.FM_field_region_for_fit[1, 1] > B)
            inx_condlist = np.logical_or(condlist2, condlist2)
            # fit the data:
            par, pcov = curve_fit(linearFunc, B[np.logical_or(condlist2, condlist2)],
                                M_sub_PM[np.logical_or(condlist2, condlist2)],)
            k = par[0]
            b = par[1]
            M_sub_linear = M_sub_PM - (k*B)

            # calc M-saturation and FM phase concentration:
            M_FM_saturation = 0.5 * (abs(np.mean(M_sub_linear[condlist2])) + abs(np.mean(M_sub_linear[condlist1])))
            self.n_FM_phase = concentration_from_Ms(Ms=M_FM_saturation, J=2.5) * (1e-27)
            print('->> subtract_Line have been done.')
            print(
                'For Ms, which we take from m(T={0}K)-PM-k*B we obtained n = {1:1.3g} *1e27 [1/m^3] or {2:1.3g} % of the n(GaAs)' \
                .format(self.diff_PM_keyName1, self.n_FM_phase, self.n_FM_phase / 22.139136 * 100))

            self.ax.cla()

            # setup variables for plot:
            self.suptitle_txt = 'case: {}, '.format(self.selectCase) + '$Ga_{1-x}Mn_xAs$,' \
                                + ' where $x = {0:1.2g}\%$ '.format(self.how_many_Mn_in_percent)
            self.ylabel_txt = '$M\, (A/m)$'
            self.xlabel_txt = '$B\, (T)$'
            self.setupAxes()

            self.ax.plot(B, f_PM_with_T(B=B, n=self.n_diffPM, J=J, T=T), 'r-',
                         label='PM component $\left(T={0:1.2g}K \\right)$, '.format(T) + \
                               '$n_{{PM}}={:1.3g}\%$,'.format(self.n_diffPM / 22.139136 * 100) + \
                             '\n$n_{{[PM,\;x={0:1.3g}\%]}}={1:1.3g}\%$'.format(self.how_many_Mn_in_percent,
                            self.n_diffPM / 22.139136 * 100/self.how_many_Mn_in_percent*100) + \
                               ' $J={:1.2g}(Mn^{{2+}})$'.format(J / 2.5)
                         )
            self.ax.plot(B, (k*B), 'b-', label='line')
            self.ax.scatter(B, M,  label='raw', alpha=.2, color='g')
            self.ax.scatter(B, M_sub_PM,
                            label='$M_{raw}-M_{PM}$'+' $\left(T={0:1.2g}K \\right)$'.format(T),
                            color='k', alpha=.2)
            self.ax.scatter(B, M_sub_linear, label='$M_{PM}-(kx+b)$, '+\
                                                   '$n_{{FM}}={:1.3g}\%$,\n'.format(self.n_FM_phase / 22.139136 * 100) + \
                '$n_{{[FM,\;x={0:1.3g}\%]}}={1:1.3g}\%$'.format(self.how_many_Mn_in_percent,
                            self.n_FM_phase / 22.139136 * 100/self.how_many_Mn_in_percent*100) + \
                                                   ' $J={:1.2g}(Mn^{{2+}})$'.format(J / 2.5),
                            color='b', s=100,
                            alpha=.2)
            self.ax.scatter(B[inx_condlist], M_sub_PM[inx_condlist],
                            label='for $linear\,fit$ $M_{raw}-M_{PM}$'+' $\left(T={0:1.2g}K \\right)$'.format(T),
                            color='k', alpha=.2, s=70)
            self.ax.legend(shadow=True, fancybox=True, loc='upper left')
            self.ax.grid(True)
            plt.draw()



    def calc_SPM_Langevin(self):
        # fit data by using a Langevin function:
        if self.n_diffPM <= 0:
            self.calc_diff_PM()


        T = self.struct_of_data[self.SPM_keyName].T

        def fun_fit(B, n, j):
            # return f_SPM_with_T(B, n, J=J, T=T)
            return f_PM_with_T(B, n, J=j, T=T)



        B = self.struct_of_data[self.SPM_keyName].xx
        M = self.struct_of_data[self.SPM_keyName].yy
        T = self.struct_of_data[self.SPM_keyName].T
        J = 2.5
        eps = 0.001 * abs(abs(B[0]) - abs(B[1]))

        # reduce a noise:
        Mp = M[(B > eps)]
        Mn = M[(B < -eps)]
        M_for_fit = np.copy(M)

        # Find the intersection of two arrays to avoid conflict with numbers of elements.
        # if abs(self.struct_of_data[self.SPM_keyName].Hmin) == self.struct_of_data[self.SPM_keyName].Hmax:
        if len(M[(B > 0)]) != len(M[(B < 0)]):
            # reduce a noise:
            negVal = abs(np.min(B))
            pozVal = np.max(B)

            if pozVal >= negVal:
                limitVal = negVal
            else:
                limitVal = pozVal

            inx_B = np.logical_or((np.abs(B) <= limitVal + eps), (np.abs(B) <= eps))
            B = np.copy(B[inx_B])

            M_for_fit = np.copy(M[inx_B])

            Mp = M[np.where(B > eps)]
            Mn = M[np.where(B < -eps)]

        if len(M[np.where(B > 0)]) == len(M[np.where(B < 0)]):
            inx_B = np.logical_or((np.abs(B) <= eps), (np.abs(B) >= eps))
            B = np.copy(B[inx_B])

            M_for_fit = np.copy(M[np.where(inx_B)])
            # reduce a noise:
            Mp = M[np.where(B > eps)]
            Mn = M[np.where(B < -eps)]

        M_for_fit[np.where(B > eps)] = 0.5 * (Mp + np.abs(Mn[::-1]))
        M_for_fit[np.where(B < -eps)] = 0.5 * (Mn - np.abs(Mp[::-1]))
            # M_for_fit[(B > 0)] = 0.5*(Mp + np.abs(Mn))
            # M_for_fit[(B < 0)] = 0.5*(Mn - np.abs(Mp))

        if self.SPM_field_region_for_fit[0, 0] >= B.max():
            self.SPM_field_region_for_fit[0, 0] = np.fix(10 * B.max()) / 10 - \
                                                      5 * self.struct_of_data[self.SPM_keyName].Hstep

        # sabtruct a PM magnetic phase which is low temperature independent:
        M_for_fit = M_for_fit - f_PM_with_T(B, n=self.n_diffPM, J=J, T=T)

        condlist1 = (self.SPM_field_region_for_fit[0, 0] < B) & (self.SPM_field_region_for_fit[0, 1] > B)
        initial_guess = [0.01, 2, ]
        # fit the data:
        pres, pcov = curve_fit(fun_fit, B[np.logical_or(condlist1, condlist1)],
                            M_for_fit[np.where(np.logical_or(condlist1, condlist1))],
                            p0=initial_guess,
                            bounds=([0, 0, ], [np.inf, np.inf, ]))

        self.n_SPM = pres[0]
        self.J_SPM = pres[1]
        print('->> fit SPM have been done. For m(T={0}K) obtained n = {1:1.3g} *1e27 [1/m^3] or {2:1.3g} % of the n(GaAs)'\
          .format(self.SPM_keyName, self.n_SPM, self.n_SPM/22.139136*100))
        print('->> SPM magnetic phase has total momentum J = {0:1.3g} or {1:1.3g} numbers of J(Mn2+)'\
          .format(self.J_SPM, self.J_SPM/2.5))

        # setup variables for plot:
        self.ax.cla()
        self.suptitle_txt = 'case: {}, '.format(self.selectCase) + '$Ga_{1-x}Mn_xAs$,' \
                            + ' where $x = {0:1.2g}\%$ '.format(self.how_many_Mn_in_percent)
        self.ylabel_txt = '$M(T={0}K) (A/m)$'.format(T)
        self.xlabel_txt = '$B (T)$'
        self.setupAxes()


        self.ax.scatter(B[np.logical_or(condlist1, condlist1)],
                        M_for_fit[np.where(np.logical_or(condlist1, condlist1))],
                        label='region for $fit$',
                        facecolor='none',
                        alpha=.1,
                        s=200, marker='s', edgecolors='k',
                        )
        self.ax.plot(B, fun_fit(B, self.n_SPM, self.J_SPM), 'r-',
                     label='$fit$, $n_{{SPM}}={0:1.3g}\%$, \n$J={1:1.3g}(Mn^{{2+}})$'.format(self.n_SPM/22.139136*100,
                                                                                                  self.J_SPM/2.5)+\
                           ', $n_{{[SPM,\;x={0:1.3g}\%]}}={1:1.3g}\%$'.format(self.how_many_Mn_in_percent,
                            self.n_SPM / 22.139136 * 100/self.how_many_Mn_in_percent*100)
                     )
        self.ax.plot(B, f_PM_with_T(B, n=self.n_diffPM, J=J, T=T), 'b-',
                     label='$PM(T={2:1.2g}K)$, $n_{{PM}}={0:1.3g}\%$, \n$J={1:1.3g}(Mn^{{2+}})$'\
                     .format(self.n_diffPM/22.139136*100, 1, T, )+\
                           ', $n_{{[PM,\;x={0:1.3g}\%]}}={1:1.3g}\%$'.format(self.how_many_Mn_in_percent,
                            self.n_diffPM / 22.139136 * 100/self.how_many_Mn_in_percent*100))
        self.ax.scatter(B, M[np.where(inx_B)],  label='$raw$', alpha=.2, color='g')
        self.ax.scatter(B, M_for_fit, label='set for $fit$, $\\frac{{\left(M_++M_-\\right)}}{{2}} - PM(T={0:1.2g}K)$'.format(T), color='k', alpha=.2)
        self.ax.plot(B, 10*(M_for_fit - fun_fit(B, self.n_SPM, self.J_SPM)), 'k.', label='residuals $10*(raw - fit)$')

        self.ax.legend(shadow=True, fancybox=True, loc='upper left')
        self.ax.grid(True)
        plt.show()
        plt.draw()

    def calculate(self):

        n_m3 = number_density(rho=rho_GaAs, M=self.mass_Molar_kg)
        n_m3_GaAs = number_density(rho=rho_GaAs, M=mass_Molar_kg_GaAs)
        print('Ga(1-x)Mn(x)As molar mass for {0:1.2g}% Mn is: {1:1.6g} [kg/mol]'.format(self.how_many_Mn_in_percent, self.mass_Molar_kg))
        print('GaAs molar mass for {0:1.2g}% Mn is: {1:1.6g} [kg/mol]'.format(0, mass_Molar_kg_GaAs))
        print( 'table concentration (Number density) is: {:1.6g} *10^27 [1/m3] or 10^21 [1/cm3]'.format( (n_m3/1e27) ) )
        print( 'pure GaAs table concentration (Number density) is: {:1.9g} *10^27 [1/m3] or 10^21 [1/cm3]'.format( (n_m3_GaAs/1e27) ) )
        n_m3 = n_m3_GaAs
        print( '-> we use this table concentration (Number density) is: {:1.6g} *10^27 [1/m3] or 10^21 [1/cm3]'.format( (n_m3/1e27) ) )
        print('The area size of the sample surface is: {:1.5g} [mm^2]'.format(self.area*1e6))
        print('if suppose that one side has 5mm that other side will be have: {:1.5g} [mm]'.format(self.area/0.005*1e3))

        # old uncorrected calc from M saturation values:
        # magnetization_SI = from_EMU_cm3_to_A_by_m(moment_emu = self.magnetic_moment_saturation_experiment, V_cm3 = self.volumeOfTheFilm_GaMnAs*1e6)
        # n_PM_phase = concentration_PM(M = magnetization_SI, p_exp = g_J_Mn2_plus)

        print('===>>')
        print('For case {0} :'.format(self.selectCase))
        print('Concentration of FM ions Mn is: {:1.6}%'.format(self.n_FM_phase/n_m3_GaAs/1e27*100))
        print('<<===')

    def setupAxes(self):
        # create figure with axes:

        pylab.ion()  # Force interactive
        plt.close('all')
        ### for 'Qt4Agg' backend maximize figure
        plt.switch_backend('QT5Agg')

        plt.rc('font', family='serif')
        self.fig = plt.figure()
        # gs1 = gridspec.GridSpec(1, 2)
        # fig.show()
        # fig.set_tight_layout(True)
        self.figManager = plt.get_current_fig_manager()
        DPI = self.fig.get_dpi()
        self.fig.set_size_inches(800.0 / DPI, 600.0 / DPI)

        gs = gridspec.GridSpec(1, 1)

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.grid(True)
        self.ax.set_ylabel(self.ylabel_txt, fontsize=20, fontweight='bold')
        self.ax.set_xlabel(self.xlabel_txt, fontsize=20, fontweight='bold')

        self.fig.suptitle(self.suptitle_txt, fontsize=22, fontweight='normal')

        # Change the axes border width
        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[axis].set_linewidth(2)
        # plt.subplots_adjust(top=0.85)
        # gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        self.fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)

        # put window to the second monitor
        # figManager.window.setGeometry(1923, 23, 640, 529)
        # self.figManager.window.setGeometry(780, 20, 800, 600)
        self.figManager.window.setGeometry(780, 20, 1024, 768)
        self.figManager.window.setWindowTitle('Magnetic fitting')
        self.figManager.window.showMinimized()

        self.fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)


        # save to the PNG file:
        # out_file_name = '%s_' % (case) + "%05d.png" % (numOfIter)
        # fig.savefig(os.path.join(out_dir, out_file_name))


if __name__ =='__main__':
    print ('-> you run ',  __file__, ' file in a main mode' )
    from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from feff.libs.dir_and_file_operations import create_out_data_folder

    # open GUI filedialog to select feff_0001 working directory:
    A = StoreAndLoadVars()
    A.fileNameOfStoredVars = 'GaMnAs_SQUID.pckl'
    print('last used: {}'.format(A.getLastUsedDirPath()))
    # openfile dialoge
    root = tk.Tk()
    root.withdraw()

    # load A-model data:
    txt_info = "select the SQUID measurements\noutput data files for B180 sample\ndefault: Origin_Sawicki_measur_(B180)"
    messagebox.showinfo("info", txt_info)
    dir_path = filedialog.askdirectory(initialdir=A.getLastUsedDirPath())
    if os.path.isdir(dir_path):
        A.lastUsedDirPath = dir_path
        A.saveLastUsedDirPath()

    # # check is the 'calc' folder exist:
    # out_dir = os.path.join(dir_path, 'calc')
    # print('out dir path is: {}'.format(out_dir))
    # if not os.path.isdir(out_dir):
    #     os.mkdir(out_dir)

    out_dir = create_out_data_folder(dir_path, 'calc')

    # Change the Case Name for calculation: [B180a, B180b, B180c, B180v]
    a = ConcentrationOfMagneticIons()
    a.dataFolderSorceBase = dir_path
    a.selectCase = 'B180c'

    # check is 'a.selectCase' sub-folder exist:
    out_dir = os.path.join(out_dir, a.selectCase)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)


    a.prepareData()
    a.diff_PM_keyName1 = '2'
    a.diff_PM_keyName2 = '5'
    a.diff_PM_field_region_for_fit = np.array([[-6, -0.1], [0.1, 6]])  # [T]
    a.calc_diff_PM()
    # save to the PNG file:
    out_file_name = '%s' % (a.selectCase) + "_diff.png"
    a.fig.savefig(os.path.join(out_dir, out_file_name))

    a.subtract_PM_after_calc_diff_PM()
    # save to the PNG file:
    out_file_name = '%s' % (a.selectCase) + "_RAW-PM.png"
    a.fig.savefig(os.path.join(out_dir, out_file_name))


    a.SPM_keyName = '50'
    a.SPM_field_region_for_fit = np.array([[4, 6]]) # [T]
    a.calc_SPM_Langevin()
    # save to the PNG file:
    out_file_name = '%s' % (a.selectCase) + "_SPM.png"
    a.fig.savefig(os.path.join(out_dir, out_file_name))

    a.FM_field_region_for_fit = np.array([[-6, -2], [2, 6]])  # [T]
    a.subtract_Line()
    # save to the PNG file:
    out_file_name = '%s' % (a.selectCase) + "_FM.png"
    a.fig.savefig(os.path.join(out_dir, out_file_name))
    # sortKeys = sorted(a.struct_of_data, key=lambda key: a.struct_of_data[key].T)
    # for i in sortKeys:
    #     a.ax.cla()
    #     # struct_of_data[i].plotLogT(self.ax)
    #     a.struct_of_data[i].plot(a.ax)
    #
    #     a.ax.legend(shadow=True, fancybox=True, loc='upper left')
    #     plt.draw()
    #     # print([a.struct_of_data[i].forFit_x, a.struct_of_data[i].forFit_y])
    #     x = np.array(a.struct_of_data[i].forFit_x)
    #     y = np.array(a.struct_of_data[i].forFit_y)
    #
    #
    #     x = np.array(a.struct_of_data[i].forFit_x)[:,0]
    #     y = np.array(a.struct_of_data[i].forFit_y)[:,0]
    #
    #     x1 = np.array(( 5.12259078,  5.29071205,  5.45883331,  5.62695458,  5.79507585,  5.96319712,  6.13131838,  6.29943965,  6.46756092,  6.63568218 , 6.80380345,  6.97192472, 7.14004599,  7.30816725,  7.47628852,  7.64440979,  7.81253105,  7.98065232, 8.14877359,  8.31689486,  8.48501612,  8.65313739,  8.82125866,  8.98937992,  9.15750119 , 9.32562246,  9.49374373 , 9.66186499,  9.82998626 , 9.99810753))
    #     y1 = np.array((36562.255,  36704.364,  36846.472,  36959.829,  37071.074,  37182.319,  37288.874,  37394.099,  37499.323,  37575.449,  37634.214,  37692.98 ,  37769.696,  37866.358,  37963.021,  38026.567,  38020.08 ,  38013.593,  38025.271,  38126.641,  38228.011,  38328.095,  38347.998,  38367.901,  38387.804,  38459.712,  38539.864,  38620.017,  38658.663,  38680.4 ))
    #
    #     # np.set_printoptions(precision=3, suppress=True, linewidth=750,)
    #     # print(repr(y))
    #     n = return_fit_param(x, y)
    #     popt, pcov = curve_fit(func, x, y)
    # a.calculate()
    print('------')
    print('-> file ',  __file__, '  was finished')

