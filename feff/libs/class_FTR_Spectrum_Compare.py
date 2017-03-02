'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-28
'''
from feff.libs.class_Spectrum import Spectrum
import os
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name, get_upper_folder_name
import matplotlib.pyplot as plt
import numpy as np

class FTR_gulp_to_feff_A_model():
    '''
    Class to search optimal snapshot coordinates by compare chi(k) nad FTR(r) spectra between the snapshots and
     average spectrum from all snapshots

    '''
    def __init__(self):
        self.graph_title_txt = 'model name'
        # define average (from all snapshots in directory) spectrum data:
        self.experiment = Spectrum()
        self.experiment.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/result_1mono1SR2VasVga2_6.txt'
        self.experiment.label = 'average model'
        self.experiment.label_latex = 'average model'
        self.experiment.loadSpectrumData()


        self.theory_one = Spectrum()
        self.theory_one.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000002_00001.dat'
        self.theory_one.label = 'snapshot model'
        self.theory_one.label_latex = 'snapshot model'
        self.theory_one.loadSpectrumData()

        self.theory_one.r_vector = self.theory_one.r_vector
        self.theory_one.ftr_vector = self.theory_one.ftr_vector


    def get_name_of_model_from_fileName(self):
        modelName = os.path.split(os.path.split(os.path.dirname(self.theory_one.pathToLoadDataFile))[0])[1]
        name = os.path.split(os.path.basename(self.theory_one.pathToLoadDataFile))[1]
        name = name.split('.')[0]
        snapNumberStr = name.split('chi_'+modelName+'_')[1]
        return modelName, snapNumberStr

    def updateInfo(self):
        modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        self.graph_title_txt = 'model: ' + modelName
        self.theory_one.label_latex = 'snapshot: {0}'.format(snapNumberStr)

    def get_R_factor(self):
        R_chi = self.theory_one.get_chi_R_factor()
        R_ftr = self.theory_one.get_FTR_R_factor()

        R_tot = 0.5*(R_ftr + R_chi)

        return R_tot, R_ftr, R_chi

    def plotSpectra_chi_k(self):
        self.theory_one.plotTwoSpectrum_chi_k()
        # self.experiment.plotOneSpectrum_FTR_r()
        plt.text(3, 0.18, '$R_{{tot}}$ = {0}, $R_{{FT(r)}}$ = {1}, $R_{{\chi(k)}}$ = {2}'.format(round(self.get_R_factor()[0], 4),
                                                                                           round(self.get_R_factor()[1], 4),
                                                                                           round(self.get_R_factor()[2], 4),
                                                                                           ),
                 fontdict={'size': 20})
        plt.title(self.graph_title_txt)
        plt.legend()
        plt.show()

    def plotSpectra_FTR_r(self):
        self.theory_one.plotTwoSpectrum_FTR_r()
        # self.experiment.plotOneSpectrum_FTR_r()
        plt.text(3, 0.18, '$R_{{tot}}$ = {0}, $R_{{FT(r)}}$ = {1}, $R_{{\chi(k)}}$ = {2}'.format(round(self.get_R_factor()[0], 4),
                                                                                           round(self.get_R_factor()[1], 4),
                                                                                           round(self.get_R_factor()[2], 4),
                                                                                           ),
                 fontdict={'size': 20})
        plt.title(self.graph_title_txt)
        plt.legend()
        plt.show()

    def findOptimum(self):
        pass

class FTR_gulp_to_feff_A_B_mix_models():
    def __init__(self):
        self.mix_parameter = 0.1
        self.so2_bonds = [(0.81, 1)]


        # define experiment data:
        self.experiment = Spectrum()
        self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data', '450.chik')
        self.experiment.label = 'experiment T=450 C'
        self.experiment.label_latex = 'experiment $T=450^{\circ}$ '
        self.experiment.loadSpectrumData()


        self.theory_one = Spectrum()
        self.theory_one.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000002_00001.dat'
        self.theory_one.label = 'theory 1mono1SR2VasVga2_6_000002_00001'
        self.theory_one.label_latex = 'theory 1mono1SR2VasVga2_6_000002_00001'
        self.theory_one.loadSpectrumData()


        self.theory_two = Spectrum()
        self.theory_two.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000131_00130.dat'
        self.theory_two.label = 'theory 1mono1SR2VasVga2_6_000131_00130'
        self.theory_two.label_latex = 'theory 1mono1SR2VasVga2_6_000131_00130'
        self.theory_two.loadSpectrumData()


        self.theory_mix_state = Spectrum()
        self.theory_mix_state.label = 'mix phases'
        self.theory_mix_state.label_latex = 'mix phases $A_{{{0:.2f}}}B_{{{1:.2f}}}$'.format(self.mix_parameter, 1-self.mix_parameter)
        self.theory_mix_state.r_vector = self.theory_one.r_vector
        self.theory_mix_state.ftr_vector = self.theory_one.ftr_vector * self.mix_parameter + \
                                           self.theory_two.ftr_vector * (1-self.mix_parameter)
        self.theory_mix_state.ideal_curve_x = self.experiment.r_vector
        self.theory_mix_state.ideal_curve_y = self.experiment.ftr_vector

    def updateInfo(self):
        self.theory_mix_state.label_latex = 'mix phases $A_{{0:1.3f}}B_{{1:1.3f}}$'.format(self.mix_parameter, 1-self.mix_parameter)

    def get_R_factor(self):
        return self.theory_mix_state.get_FTR_R_factor()

    def plotSpectra(self):
        self.theory_one.plotOneSpectrum_FTR_r()
        self.theory_two.plotOneSpectrum_FTR_r()
        self.theory_mix_state.plotTwoSpectrum_FTR_r()
        plt.text(3, 0.18, '$R$-$factor$ = {0}, $x$ = {1}'.format(round(self.get_R_factor(), 4), round(self.mix_parameter, 4)),
                 fontdict={'size': 20})
        plt.legend()
        plt.show()

    def findOptimum(self):
        pass


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    a = FTR_gulp_to_feff_A_model()
    a.updateInfo()
    print(a.get_R_factor())
    a.plotSpectra_chi_k()