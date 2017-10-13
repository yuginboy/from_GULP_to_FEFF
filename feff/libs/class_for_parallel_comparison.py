'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-07-12
'''
from feff.libs.class_Spectrum import Spectrum, GraphElement, TableData, BaseData
import os
import datetime
from collections import OrderedDict as odict
from timeit import default_timer as timer
import copy
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name, get_upper_folder_name, \
    listOfFilesFN_with_selected_ext, create_out_data_folder, create_data_folder
from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars
from feff.libs.class_SpectraSet import SpectraSet
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec
import numpy as np

import progressbar
class Model_for_spectra():
    def __init__(self):
        # How many serial snapshots from different center atoms we have:
        self.numberOfSerialEquivalentAtoms = 2

        # store to the ASCII table on a file:
        self.table = TableData()

        # weights of R-factors (Rtot = (Rchi* w1 + Rrtf*w2)/(w1+w2)):
        # weights for calc tota R-factor in minimization procedure
        self.weight_R_factor_FTR = 1
        self.weight_R_factor_chi = 1

        # coefficient for multiplying theory spectra in FTR space
        self.scale_theory_factor_FTR = 1
        self.scale_experiment_factor_FTR = 1

        # coefficient for multiplying theory spectra in CHI space
        self.scale_theory_factor_CHI = 1

        self.listOfSnapshotFiles = []
        self.projectWorkingFEFFoutDirectory = '/home/yugin/VirtualboxShare/GaMnO/debug/1mono1SR2VasVga2_6/feff__0001'

        # create object for theoreticaly calculated spectra:
        self.theory = Spectrum()
        # create object which will be collected serial snapshot spectra
        self.setOfSnapshotSpectra = SpectraSet()

        # create dict for store all snapshots in path:
        self.dictOfAllSnapshotsInDirectory = odict()

    def get_Model_name(self):
        path = os.path.join(self.projectWorkingFEFFoutDirectory, self.listOfSnapshotFiles[0])
        modelName = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        name = os.path.split(os.path.basename(path))[1]
        name = name.split('.')[0]
        return modelName

    def get_name_of_model_from_fileName(self):
        modelName = os.path.split(os.path.split(os.path.dirname(self.theory.pathToLoadDataFile))[0])[1]
        name = os.path.split(os.path.basename(self.theory.pathToLoadDataFile))[1]
        name = name.split('.')[0]
        # take only the number of snapshot:
        snapNumberStr = name.split('chi_'+modelName+'_')[1]
        snapNumberStr = snapNumberStr.split('_')[0]
        return modelName, snapNumberStr

    def updateInfo(self):
        modelName, snapNumberStr = self.get_name_of_model_from_fileName()

        self.theory.label_latex = 'model: ' + modelName + 'snapshot: {0}'.format(snapNumberStr)


        self.theory.label = snapNumberStr

    def get_R_factor(self):
        R_chi = self.theory.get_chi_R_factor()
        R_ftr = self.theory.get_FTR_R_factor()

        R_tot = (self.weight_R_factor_FTR * R_ftr + self.weight_R_factor_chi * R_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return R_tot, R_ftr, R_chi

class FTR_gulp_to_feff_A_model_base():
    '''
    Class to search optimal snapshot coordinates by compare chi(k) nad FTR(r) spectra between the snapshots and
     average spectrum from all snapshots

    '''

    def __init__(self):
        # store minimum values:
        self.minimum = BaseData()
        self.minimum.fill_initials()
        # store current values of R-factor:
        self.currentValues = BaseData()
        self.currentValues.fill_initials()

        # How many serial snapshots from different center atoms we have:
        self.numberOfSerialEquivalentAtoms = 2

        self.model_A = Model_for_spectra()
        self.model_A.numberOfSerialEquivalentAtoms = 3
        self.model_B = Model_for_spectra()
        self.model_B.numberOfSerialEquivalentAtoms = 1
        self.model_C = Model_for_spectra()
        self.model_C.numberOfSerialEquivalentAtoms = 2

        # user's parameters for xftf preparation ['PK'- Pavel Konstantinov, 'ID' - Iraida Demchenko]:
        self.user = 'PK'

        # sample preparation mode: AG, 250, 350, 450
        self.sample_preparation_mode = 'AG'

        # store to the ASCII table on a file:
        self.table = TableData()

        # weights of R-factors (Rtot = (Rchi* w1 + Rrtf*w2)/(w1+w2)):
        # weights for calc tota R-factor in minimization procedure
        self.weight_R_factor_FTR = 1
        self.weight_R_factor_chi = 1

        # coefficient for multiplying theory spectra in FTR space
        self.scale_theory_factor_FTR = 1
        self.scale_experiment_factor_FTR = 1

        # coefficient for multiplying theory spectra in CHI space
        self.scale_theory_factor_CHI = 1

        self.FTR = GraphElement()
        self.Chi_k = GraphElement()
        self.suptitle_fontsize = 18

        self.projectWorkingFEFFoutDirectory = '/home/yugin/VirtualboxShare/GaMnO/debug/1mono1SR2VasVga2_6/feff__0001'
        self.listOfSnapshotFiles = []
        # out dir for  Rfactor minimum files:
        self.outMinValsDir = ''
        self.outMinValsDir_mask = ''

        self.graph_title_txt = 'model name'
        self.showFigs = True
        # define average (from all snapshots in directory) spectrum data:
        self.experiment = Spectrum()
        self.experiment.user = self.user
        self.experiment.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/result_1mono1SR2VasVga2_6.txt'
        self.experiment.label = 'average model'
        self.experiment.label_latex = 'average model'
        # self.experiment.loadSpectrumData()


        self.theory_one = Spectrum()
        self.theory_one.user = self.user
        self.theory_one.pathToLoadDataFile = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000002_00001.dat'
        self.theory_one.label = 'snapshot model'
        self.theory_one.label_latex = 'snapshot model'

        self.model_A.theory = copy.deepcopy(self.theory_one)
        self.model_B.theory = copy.deepcopy(self.theory_one)
        self.model_C.theory = copy.deepcopy(self.theory_one)

        self.outDirectoryForTowModelsFitResults = '/home/yugin/VirtualboxShare/GaMnO/out_fit_two_models'
        self.outDirectoryFor_3_type_ModelsFitResults = '/home/yugin/VirtualboxShare/GaMnO/out_fit_3_models'

        # self.theory_one.loadSpectrumData()

        # create object which will be collected serial snapshot spectra
        self.setOfSnapshotSpectra = SpectraSet()

        # self.theory_SimpleComposition = Spectrum()
        # self.theory_LinearComposition = Spectrum()

        self.find_min_Rtot_in_single_snapshot = True
        self.do_SimpleSpectraComposition = True
        self.do_LinearSpectraComposition = False
        self.do_FTR_from_linear_Chi_k_SpectraComposition = True

        self.saveDataToDisk = True
        self.parallel_job_numbers = 2

        self.set_ideal_curve_params()

    def getInDirectoryStandardFilePathes(self):
        '''
        return experiment ASCII data filepath if selected the corrected filepath and list of all snapshots ASCII data
        :return:
        '''
        self.listOfSnapshotFiles = listOfFilesFN_with_selected_ext(self.projectWorkingFEFFoutDirectory, ext='dat')
        self.experiment.pathToLoadDataFile = \
        listOfFilesFN_with_selected_ext(self.projectWorkingFEFFoutDirectory, ext='txt')[0]

    def set_ideal_curve_params(self):
        self.theory_one.ideal_curve_ftr = self.experiment.ftr_vector
        self.theory_one.ideal_curve_r = self.experiment.r_vector
        self.theory_one.ideal_curve_k = self.experiment.k_vector
        self.theory_one.ideal_curve_chi = self.experiment.chi_vector
        self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve

        self.theory_one.user = self.user
        self.setOfSnapshotSpectra.user = self.user

        # repeat for two models:
        self.model_A.theory = copy.deepcopy(self.theory_one)
        self.model_A.setOfSnapshotSpectra.user = self.user

        self.model_B.theory = copy.deepcopy(self.theory_one)
        self.model_B.setOfSnapshotSpectra.user = self.user

    def get_name_of_model_from_fileName(self):
        modelName = os.path.split(os.path.split(os.path.dirname(self.theory_one.pathToLoadDataFile))[0])[1]
        name = os.path.split(os.path.basename(self.theory_one.pathToLoadDataFile))[1]
        name = name.split('.')[0]
        # take only the number of snapshot:
        snapNumberStr = name.split('chi_' + modelName + '_')[1]
        snapNumberStr = snapNumberStr.split('_')[0]
        return modelName, snapNumberStr

    def updateInfo(self):
        modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        if self.scale_theory_factor_FTR == 1:
            self.graph_title_txt = 'model: ' + modelName + ', $R_{{tot}}$  = {0}'.format(
                round(self.get_R_factor()[0], 4))
            self.theory_one.label_latex = 'snapshot: {0}'.format(snapNumberStr)
        else:
            self.graph_title_txt = 'model: ' + modelName + ', $R_{{tot}}$  = {0}, $S_0^2$ = {1:1.3f}'.format(
                round(self.get_R_factor()[0], 4), self.scale_theory_factor_FTR)
            self.theory_one.label_latex = 'snapshot [$S_0^2$ = {1:1.3f}]: {0}'.format(snapNumberStr,
                                                                                      self.scale_theory_factor_FTR)

        self.theory_one.label = snapNumberStr

    def get_R_factor(self):
        R_chi = self.theory_one.get_chi_R_factor()
        R_ftr = self.theory_one.get_FTR_R_factor()

        R_tot = (self.weight_R_factor_FTR * R_ftr + self.weight_R_factor_chi * R_chi) / \
                (self.weight_R_factor_FTR + self.weight_R_factor_chi)

        return R_tot, R_ftr, R_chi

    def plotSpectra_chi_k(self):
        self.theory_one.plotTwoSpectrum_chi_k()
        # self.experiment.plotOneSpectrum_FTR_r()
        # plt.text(3, 0.18, '$R_{{tot}}$ = {0}, $R_{{FT(r)}}$ = {1}, $R_{{\chi(k)}}$ = {2}'.format(round(self.get_R_factor()[0], 4),
        #                                                                                    round(self.get_R_factor()[1], 4),
        #                                                                                    round(self.get_R_factor()[2], 4),
        #                                                                                    ),
        #          fontdict={'size': 20})
        plt.title('$R_{{\chi(k)}}$ = {0}'.format(round(self.get_R_factor()[2], 4)))
        plt.legend()
        plt.show()

    def plotSpectra_FTR_r(self):
        self.theory_one.plotTwoSpectrum_FTR_r()
        # self.experiment.plotOneSpectrum_FTR_r()
        # plt.text(3, 0.18, '$R_{{tot}}$ = {0}, $R_{{FT(r)}}$ = {1}, $R_{{\chi(k)}}$ = {2}'.format(round(self.get_R_factor()[0], 4),
        #                                                                                    round(self.get_R_factor()[1], 4),
        #                                                                                    round(self.get_R_factor()[2], 4),
        #                                                                                    ),
        #          fontdict={'size': 20})
        plt.title('$R_{{FT(r)}}$  = {0}'.format(round(self.get_R_factor()[1], 4)))
        plt.legend()
        plt.show()

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
            self.figManager.window.setWindowTitle('CoO fitting')
            self.figManager.window.showMinimized()


            # save to the PNG file:
            # out_file_name = '%s_' % (case) + "%05d.png" % (numOfIter)
            # fig.savefig(os.path.join(out_dir, out_file_name))

    def setAxesLimits_FTR(self):
        plt.axis([1, 5, plt.ylim()[0], plt.ylim()[1]])

    def setAxesLimits_Chi(self):
        # plt.axis([3.5, 13, plt.ylim()[0], plt.ylim()[1]])
        plt.axis([3.5, 13, -0.1, 0.1])

    def updatePlot(self, saveFigs=True):

        if self.showFigs:
            # self.suptitle_txt = '$Fit$ $model$ $for$ $sample$: '+ \
            #     '$Au[{0:1.3f}\AA]/Co[{1:1.3f}\AA]/CoO[{2:1.3f}\AA]/Au[{3:1.3f}\AA]/MgO[{4:1.3f}\AA]/MgCO_3[{5:1.3f}\AA]/Mg(OH)_2[{6:1.3f}\AA]/C[{7:1.3f}\AA]$'.format(
            #     self.thicknessVector[0], self.thicknessVector[1], self.thicknessVector[2], self.thicknessVector[3],
            #     self.thicknessVector[4], self.thicknessVector[5], self.thicknessVector[6], self.thicknessVector[7],
            # )


            self.fig.clf()
            gs = gridspec.GridSpec(1, 2)

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            plt.axes(self.FTR.axes)
            self.plotSpectra_FTR_r()
            self.FTR.axes.grid(True)
            self.setAxesLimits_FTR()

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.plotSpectra_chi_k()
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
            self.figManager.window.showMinimized()

        if saveFigs and self.showFigs:
            # save to the PNG file:
            # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
            modelName, snapNumberStr = self.get_name_of_model_from_fileName()
            if self.scale_theory_factor_FTR == 1:
                out_file_name = snapNumberStr + '_R={0:1.4}.png'.format(self.minimum.Rtot)
            else:
                out_file_name = snapNumberStr + \
                                '_So={1:1.3f}_R={0:1.4}.png'.format(self.minimum.Rtot, self.scale_theory_factor_FTR)
            if self.saveDataToDisk:
                self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))

    def updatePlotOfSnapshotsComposition_Simple(self, saveFigs=True):

        if self.showFigs:
            # self.suptitle_txt = '$Fit$ $model$ $for$ $sample$: '+ \
            #     '$Au[{0:1.3f}\AA]/Co[{1:1.3f}\AA]/CoO[{2:1.3f}\AA]/Au[{3:1.3f}\AA]/MgO[{4:1.3f}\AA]/MgCO_3[{5:1.3f}\AA]/Mg(OH)_2[{6:1.3f}\AA]/C[{7:1.3f}\AA]$'.format(
            #     self.thicknessVector[0], self.thicknessVector[1], self.thicknessVector[2], self.thicknessVector[3],
            #     self.thicknessVector[4], self.thicknessVector[5], self.thicknessVector[6], self.thicknessVector[7],
            # )


            self.fig.clf()
            gs = gridspec.GridSpec(1, 2)

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            plt.axes(self.FTR.axes)
            self.setOfSnapshotSpectra.plotSpectra_FTR_r_SimpleComposition()
            self.FTR.axes.grid(True)
            self.setAxesLimits_FTR()

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.setOfSnapshotSpectra.plotSpectra_chi_k_SimpleComposition()
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
            self.figManager.window.showMinimized()

        if saveFigs and self.showFigs:
            # save to the PNG file:
            # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
            # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
            if self.scale_theory_factor_FTR == 1:
                out_file_name = self.theory_one.label + '_R={0:1.4}.png'.format(self.minimum.Rtot)
            else:
                out_file_name = self.theory_one.label + '_So={1:1.3}_R={0:1.4}.png'.format(self.minimum.Rtot,
                                                                                           self.scale_theory_factor_FTR)
            if self.saveDataToDisk:
                self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))
                self.minimum.pathToImage = os.path.join(self.outMinValsDir, out_file_name)

    def updatePlotOfSnapshotsComposition_Linear(self, saveFigs=True):

        if self.showFigs:
            # self.suptitle_txt = '$Fit$ $model$ $for$ $sample$: '+ \
            #     '$Au[{0:1.3f}\AA]/Co[{1:1.3f}\AA]/CoO[{2:1.3f}\AA]/Au[{3:1.3f}\AA]/MgO[{4:1.3f}\AA]/MgCO_3[{5:1.3f}\AA]/Mg(OH)_2[{6:1.3f}\AA]/C[{7:1.3f}\AA]$'.format(
            #     self.thicknessVector[0], self.thicknessVector[1], self.thicknessVector[2], self.thicknessVector[3],
            #     self.thicknessVector[4], self.thicknessVector[5], self.thicknessVector[6], self.thicknessVector[7],
            # )


            self.fig.clf()
            gs = gridspec.GridSpec(1, 2)

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            plt.axes(self.FTR.axes)
            self.setOfSnapshotSpectra.plotSpectra_FTR_r_LinearComposition()
            self.FTR.axes.grid(True)
            self.setAxesLimits_FTR()

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.setOfSnapshotSpectra.plotSpectra_chi_k_LinearComposition()
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
            self.figManager.window.showMinimized()

        if saveFigs and self.showFigs:
            # save to the PNG file:
            # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
            # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
            if self.scale_theory_factor_FTR == 1:
                out_file_name = self.theory_one.label + '_R={0:1.4}.png'.format(self.minimum.Rtot)
            else:
                out_file_name = self.theory_one.label + '_So={1:1.3}_R={0:1.4}.png'.format(self.minimum.Rtot,
                                                                                           self.scale_theory_factor_FTR)
            if self.saveDataToDisk:
                self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))

    def updatePlotOfSnapshotsComposition_Linear_FTR_from_linear_Chi_k(self, saveFigs=True):

        if self.showFigs:
            # self.suptitle_txt = '$Fit$ $model$ $for$ $sample$: '+ \
            #     '$Au[{0:1.3f}\AA]/Co[{1:1.3f}\AA]/CoO[{2:1.3f}\AA]/Au[{3:1.3f}\AA]/MgO[{4:1.3f}\AA]/MgCO_3[{5:1.3f}\AA]/Mg(OH)_2[{6:1.3f}\AA]/C[{7:1.3f}\AA]$'.format(
            #     self.thicknessVector[0], self.thicknessVector[1], self.thicknessVector[2], self.thicknessVector[3],
            #     self.thicknessVector[4], self.thicknessVector[5], self.thicknessVector[6], self.thicknessVector[7],
            # )


            self.fig.clf()
            gs = gridspec.GridSpec(1, 2)

            self.FTR.axes = self.fig.add_subplot(gs[0, 0])
            plt.axes(self.FTR.axes)
            self.setOfSnapshotSpectra.plotSpectra_FTR_r_LinearComposition_FTR_from_linear_Chi_k()
            self.FTR.axes.grid(True)
            self.setAxesLimits_FTR()

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.setOfSnapshotSpectra.plotSpectra_chi_k_LinearComposition_FTR_from_linear_Chi_k()
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
            self.figManager.window.showMinimized()

        if saveFigs and self.showFigs:
            # save to the PNG file:
            # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
            # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
            if self.scale_theory_factor_FTR == 1:
                out_file_name = self.theory_one.label + '_R={0:1.4}.png'.format(self.minimum.Rtot)
            else:
                out_file_name = self.theory_one.label + '_So={1:1.3}_R={0:1.4}.png'.format(self.minimum.Rtot,
                                                                                           self.scale_theory_factor_FTR)
            if self.saveDataToDisk:
                self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))
                self.minimum.pathToImage = os.path.join(self.outMinValsDir, out_file_name)

    def findBestSnapshotFromList(self):
        '''
        seaching procedure of One model linear fit
        method has 3 different ways to fit the experimental data:
        equal concentrations, linear coefficients, and linear spectra combinations in k-space but fit in R-space
        :return: coefficients of data fit
        '''

        if self.scale_theory_factor_FTR == 1:
            mask_ss = ''
        else:
            mask_ss = '_So={0:1.3f}'.format(self.scale_theory_factor_FTR)

        mask_ss = mask_ss + self.outMinValsDir_mask

        if (self.weight_R_factor_FTR / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rchi' + mask_ss)
        elif (self.weight_R_factor_chi / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rftr' + mask_ss)
        else:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rtot' + mask_ss)
        self.setupAxes()
        number = 0

        self.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.setOfSnapshotSpectra.set_ideal_curve_params()
        currentSerialSnapNumber = 0

        for filePath in self.listOfSnapshotFiles:
            number = number + 1
            # print('==> file is: {0}'.format(filePath))
            # print('==> Number is: {0}'.format(number))

            currentSerialSnapNumber = currentSerialSnapNumber + 1

            self.theory_one.pathToLoadDataFile = filePath
            self.theory_one.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.theory_one.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.theory_one.loadSpectrumData()
            self.updateInfo()

            modelName, snapNumberStr = self.get_name_of_model_from_fileName()

            currentSpectra = copy.deepcopy(self.theory_one)

            self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra)
            self.setOfSnapshotSpectra.weight_R_factor_chi = self.weight_R_factor_chi
            self.setOfSnapshotSpectra.weight_R_factor_FTR = self.weight_R_factor_FTR
            self.setOfSnapshotSpectra.result.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.setOfSnapshotSpectra.result_simple.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.label_latex_ideal_curve = \
                self.experiment.label_latex_ideal_curve

            self.setOfSnapshotSpectra.result.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.setOfSnapshotSpectra.result_simple.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR = self.scale_theory_factor_FTR

            self.setOfSnapshotSpectra.result.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.setOfSnapshotSpectra.result_simple.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_CHI = self.scale_theory_factor_CHI

            R_tot, R_ftr, R_chi = self.get_R_factor()

            self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
            self.currentValues.number = number
            self.currentValues.snapshotName = os.path.basename(filePath)
            self.table.addRecord(self.currentValues)
            if self.find_min_Rtot_in_single_snapshot:
                if R_tot < self.minimum.Rtot:
                    self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                    self.updatePlot()

            if currentSerialSnapNumber == self.numberOfSerialEquivalentAtoms:

                currentSerialSnapNumber = 0

                if self.do_SimpleSpectraComposition:
                    # ----- Simple Composition of Snapshots:
                    self.setOfSnapshotSpectra.calcSimpleSpectraComposition()
                    # print('Simple composition has been calculated')
                    self.setOfSnapshotSpectra.updateInfo_SimpleComposition()
                    number = number + 1
                    R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_SimpleComposition()
                    self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
                    self.currentValues.number = number
                    self.currentValues.snapshotName = self.setOfSnapshotSpectra.result_simple.label
                    self.table.addRecord(self.currentValues)

                    self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result_simple)
                    self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    if R_tot < self.minimum.Rtot:
                        self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                        self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                               modelName + ', simple snapshots composition,  $R_{{tot}}$  = {0}'.format(
                            round(self.minimum.Rtot, 4))
                        self.updatePlotOfSnapshotsComposition_Simple()
                        # save ASCII column data:
                        self.setOfSnapshotSpectra.saveSpectra_SimpleComposition(
                            output_dir=self.outMinValsDir)

                if self.do_FTR_from_linear_Chi_k_SpectraComposition:
                    # ----- Linear Composition _FTR_from_linear_Chi_k of Snapshots:
                    self.setOfSnapshotSpectra.calcLinearSpectraComposition_FTR_from_linear_Chi_k()
                    print('Linear FTR from Chi(k) composition has been calculated')
                    self.setOfSnapshotSpectra.updateInfo_LinearComposition_FTR_from_linear_Chi_k()
                    number = number + 1
                    R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()
                    self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
                    self.currentValues.number = number
                    self.currentValues.snapshotName = self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.label
                    self.table.addRecord(self.currentValues)

                    self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k)
                    self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    if R_tot < self.minimum.Rtot:
                        self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                        self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                               modelName + ', linear $FT(r)\leftarrow\chi(k)$ snapshots composition,  $R_{{tot}}$  = {0}'.format(
                            round(R_tot, 4))
                        self.updatePlotOfSnapshotsComposition_Linear_FTR_from_linear_Chi_k()
                        # save ASCII column data:
                        self.setOfSnapshotSpectra.saveSpectra_LinearComposition_FTR_from_linear_Chi_k(
                            output_dir=self.outMinValsDir)

                if self.do_LinearSpectraComposition:
                    # ----- Linear Composition of Snapshots:
                    self.setOfSnapshotSpectra.calcLinearSpectraComposition()
                    print('Linear composition has been calculated')
                    self.setOfSnapshotSpectra.updateInfo_LinearComposition()
                    number = number + 1
                    R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_LinearComposition()
                    self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
                    self.currentValues.number = number
                    self.currentValues.snapshotName = self.setOfSnapshotSpectra.result.label
                    self.table.addRecord(self.currentValues)

                    self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result)
                    self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    if R_tot < self.minimum.Rtot:
                        self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                        self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                               modelName + ', linear snapshots composition,  $R_{{tot}}$  = {0}'.format(
                            round(self.minimum.Rtot, 4))
                        self.updatePlotOfSnapshotsComposition_Linear()
                        # save ASCII column data:
                        self.setOfSnapshotSpectra.saveSpectra_LinearComposition(
                            output_dir=self.outMinValsDir)

                # flush Dict of Set of Snapshots
                self.setOfSnapshotSpectra.flushDictOfSpectra()

        # store table to ASCII file:
        self.table.outDirPath = self.outMinValsDir
        timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
        # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        self.table.outFileName = modelName + timestamp + '_So={1:1.3f}_R={0:1.4}.txt'.format(self.minimum.Rtot,
                                                                                             self.scale_theory_factor_FTR)
        self.table.writeToASCIIFile()

    def findBestSnapshotsCombinationFrom_2_type_Models(self):
        '''
        searching procedure of Two Models (A - first, B - second) linear model:
        k/n(A1 + A2 + .. + An) + (1-k)/m(B1 + B2 + .. + Bm)
        :return: k - coefficient which corresponds to concentration A phase in A-B compound
        '''
        if self.scale_theory_factor_FTR == 1:
            mask_ss = ''
        else:
            mask_ss = '_So={0:1.3f}'.format(self.scale_theory_factor_FTR)

        mask_ss = mask_ss + self.outMinValsDir_mask

        if (self.weight_R_factor_FTR / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryForTowModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rchi' + mask_ss)
        elif (self.weight_R_factor_chi / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryForTowModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rftr' + mask_ss)
        else:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryForTowModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rtot' + mask_ss)
        self.setupAxes()
        number = 0

        self.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.setOfSnapshotSpectra.set_ideal_curve_params()

        self.model_A.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.model_A.setOfSnapshotSpectra.set_ideal_curve_params()
        self.model_B.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.model_B.setOfSnapshotSpectra.set_ideal_curve_params()

        # load all snapshots to the RAM
        self.listOfSnapshotFiles = self.model_A.listOfSnapshotFiles
        self.numberOfSerialEquivalentAtoms = self.model_A.numberOfSerialEquivalentAtoms
        self.model_A.dictOfAllSnapshotsInDirectory = self.loadAllSpectraToDict()

        self.listOfSnapshotFiles = self.model_B.listOfSnapshotFiles
        self.numberOfSerialEquivalentAtoms = self.model_B.numberOfSerialEquivalentAtoms
        self.model_B.dictOfAllSnapshotsInDirectory = self.loadAllSpectraToDict()

        currentSerialSnapNumber_modelA = 0

        length = len(self.model_A.dictOfAllSnapshotsInDirectory)

        bar = progressbar.ProgressBar(maxval=length, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        i = 0
        # self.minimum.Rtot = 1000
        # self.minimum.Rchi = 1000
        # self.minimum.Rftr = 1000
        start = timer()
        for i in self.model_A.dictOfAllSnapshotsInDirectory:
            # model A
            bar.update(i)

            val = self.model_A.dictOfAllSnapshotsInDirectory[i]
            current_model_A = val['data']

            # store current result of model-A simple composition:
            currentSpectra_model_A_resultSimpleComposition = \
                copy.deepcopy(current_model_A.result_simple)

            # -----------------------------------------------------------------------------------------------------
            # intrinsic loop for Model-B snapshots   --------------------------------------------------------------
            for j in self.model_B.dictOfAllSnapshotsInDirectory:
                # model B
                val = self.model_B.dictOfAllSnapshotsInDirectory[j]
                current_model_B = val['data']

                # store current result of model-B simple composition:
                currentSpectra_model_B_resultSimpleComposition = \
                    copy.deepcopy(current_model_B.result_simple)

                # if model A and B were loaded then calc linear composition of two resulting spectra
                # from these two models:

                #     flush Dict of Set of TWO models results:
                self.setOfSnapshotSpectra.flushDictOfSpectra()
                currentSpectra_model_A_resultSimpleComposition.label = self.model_A.get_Model_name()
                self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra_model_A_resultSimpleComposition)
                currentSpectra_model_B_resultSimpleComposition.label = self.model_B.get_Model_name()
                self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra_model_B_resultSimpleComposition)

                self.setOfSnapshotSpectra.weight_R_factor_chi = self.weight_R_factor_chi
                self.setOfSnapshotSpectra.weight_R_factor_FTR = self.weight_R_factor_FTR
                self.setOfSnapshotSpectra.result.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                self.setOfSnapshotSpectra.result_simple.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.label_latex_ideal_curve = \
                    self.experiment.label_latex_ideal_curve

                # we already did (applied scale factors) it in model_A and model_B:
                self.setOfSnapshotSpectra.result.scale_theory_factor_FTR = 1
                self.setOfSnapshotSpectra.result_simple.scale_theory_factor_FTR = 1
                # was some problem with a scaling, therefore we store
                # result_FTR_from_linear_Chi_k.scale_theory_factor_FTR without changes:
                # self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR = 1

                self.setOfSnapshotSpectra.result.scale_theory_factor_CHI = 1
                self.setOfSnapshotSpectra.result_simple.scale_theory_factor_CHI = 1
                self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_CHI = 1

                if self.do_FTR_from_linear_Chi_k_SpectraComposition:
                    # if i == 4:
                    #     print(i)
                    # ----- Linear Composition _FTR_from_linear_Chi_k of Snapshots:
                    self.setOfSnapshotSpectra.calcLinearSpectraComposition_FTR_from_linear_Chi_k()
                    # print('Linear FTR from Chi(k) composition has been calculated')
                    self.setOfSnapshotSpectra.updateInfo_LinearComposition_FTR_from_linear_Chi_k()
                    R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()

                    self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k)
                    self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    if R_tot < self.minimum.Rtot:
                        self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                        modelNameTxt = self.setOfSnapshotSpectra.getInfo_LinearComposition_FTR_from_linear_Chi_k()
                        self.minimum.snapshotName = modelNameTxt + '\n' + current_model_A.result_simple.label_latex + \
                                                    '\n' + current_model_B.result_simple.label_latex
                        self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                               modelNameTxt + \
                                               ',\nlinear $FT(r)\leftarrow\chi(k)$ snapshots composition,  $R_{{tot}}$  = {0}'.format(
                                                   round(R_tot, 4))
                        self.suptitle_fontsize = 14
                        self.updatePlotOfSnapshotsComposition_Linear_FTR_from_linear_Chi_k()
                        self.suptitle_fontsize = 18
                        self.minimum.indicator_minimum_from_FTRlinear_chi = True
                        self.minimum.model_A = copy.deepcopy(current_model_A)
                        self.minimum.model_B = copy.deepcopy(current_model_B)
                        self.minimum.setOfSnapshotSpectra = copy.deepcopy(self.setOfSnapshotSpectra)
                        # save ASCII column data:
                        if self.saveDataToDisk:
                            self.setOfSnapshotSpectra.saveSpectra_LinearComposition_FTR_from_linear_Chi_k(
                                output_dir=self.outMinValsDir)

                            # store model-A snapshots for this minimum case:
                            current_model_A.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)
                            # store model-B snapshots for this minimum case:
                            current_model_B.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)

                # flush Dict of Set of TWO models results:
                self.setOfSnapshotSpectra.flushDictOfSpectra()
                # #     flush Dict of Set of model-B Snapshots:
                # self.model_B.setOfSnapshotSpectra.flushDictOfSpectra()
                # #     flush Dict of Set of model-A Snapshots:
                # self.model_A.setOfSnapshotSpectra.flushDictOfSpectra()
        bar.update(i)
        bar.finish()
        print()
        print('minimum Rtot = {}'.format(self.minimum.Rtot))
        print('{}'.format(self.minimum.snapshotName))
        runtime = timer() - start
        print("runtime is {0:f} seconds".format(runtime))

        # # store table to ASCII file:
        # self.table.outDirPath = self.outMinValsDir
        # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
        # # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        # modelName = self.model_A.get_Model_name() + '_and_' + self.model_B.get_Model_name()
        # self.table.outFileName = modelName + timestamp + '_So={1:1.3f}_R={0:1.4}.txt'.format(self.minimum.Rtot,
        #                                                                                      self.scale_theory_factor_FTR)
        # self.table.writeToASCIIFile()

    def findBestSnapshotsCombinationFrom_3_type_Models(self):
        '''
        searching procedure of 3 type Models (A - first, B - second, C - third) linear model:
        a/n(A1 + A2 + .. + An) + b/m(B1 + B2 + .. + Bm) + c/l(C1 + C2 + .. + Cl)
        a/n + b/m + c/l = 1
        :return: a/n,  b/m, c/l - coefficient which corresponds to concentration A,B,C phases in A-B-C compound
        '''
        if self.scale_theory_factor_FTR == 1:
            mask_ss = ''
        else:
            mask_ss = '_So={0:1.3f}'.format(self.scale_theory_factor_FTR)

        mask_ss = mask_ss + self.outMinValsDir_mask

        if (self.weight_R_factor_FTR / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryFor_3_type_ModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rchi' + mask_ss)
        elif (self.weight_R_factor_chi / (self.weight_R_factor_FTR + self.weight_R_factor_chi)) < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryFor_3_type_ModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rftr' + mask_ss)
        else:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.outDirectoryFor_3_type_ModelsFitResults,
                                                        first_part_of_folder_name=self.user + '_Rmin=Rtot' + mask_ss)
        self.setupAxes()
        number = 0

        self.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.setOfSnapshotSpectra.set_ideal_curve_params()

        self.model_A.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.model_A.setOfSnapshotSpectra.set_ideal_curve_params()
        self.model_B.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.model_B.setOfSnapshotSpectra.set_ideal_curve_params()
        self.model_C.setOfSnapshotSpectra.target = copy.deepcopy(self.experiment)
        self.model_C.setOfSnapshotSpectra.set_ideal_curve_params()

        # load all snapshots to the RAM
        self.listOfSnapshotFiles = self.model_A.listOfSnapshotFiles
        self.numberOfSerialEquivalentAtoms = self.model_A.numberOfSerialEquivalentAtoms
        self.model_A.dictOfAllSnapshotsInDirectory = self.loadAllSpectraToDict()

        self.listOfSnapshotFiles = self.model_B.listOfSnapshotFiles
        self.numberOfSerialEquivalentAtoms = self.model_B.numberOfSerialEquivalentAtoms
        self.model_B.dictOfAllSnapshotsInDirectory = self.loadAllSpectraToDict()

        self.listOfSnapshotFiles = self.model_C.listOfSnapshotFiles
        self.numberOfSerialEquivalentAtoms = self.model_C.numberOfSerialEquivalentAtoms
        self.model_C.dictOfAllSnapshotsInDirectory = self.loadAllSpectraToDict()

        currentSerialSnapNumber_modelA = 0

        length = len(self.model_A.dictOfAllSnapshotsInDirectory) * len(self.model_B.dictOfAllSnapshotsInDirectory) * \
                 len(self.model_C.dictOfAllSnapshotsInDirectory)

        bar = progressbar.ProgressBar(maxval=length, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        i = 0
        # self.minimum.Rtot = 1000
        # self.minimum.Rchi = 1000
        # self.minimum.Rftr = 1000
        start = timer()
        bar_index = 1
        for i in self.model_A.dictOfAllSnapshotsInDirectory:
            # model A
            bar.update(bar_index)

            val = self.model_A.dictOfAllSnapshotsInDirectory[i]
            current_model_A = val['data']

            # store current result of model-A simple composition:
            currentSpectra_model_A_resultSimpleComposition = \
                copy.deepcopy(current_model_A.result_simple)

            # -----------------------------------------------------------------------------------------------------
            # intrinsic loop for Model-B snapshots   --------------------------------------------------------------
            for j in self.model_B.dictOfAllSnapshotsInDirectory:
                # model B
                val = self.model_B.dictOfAllSnapshotsInDirectory[j]
                current_model_B = val['data']

                # store current result of model-B simple composition:
                currentSpectra_model_B_resultSimpleComposition = \
                    copy.deepcopy(current_model_B.result_simple)

                for k in self.model_C.dictOfAllSnapshotsInDirectory:
                    # model C
                    val = self.model_C.dictOfAllSnapshotsInDirectory[k]
                    current_model_C = val['data']

                    # store current result of model-C simple composition:
                    currentSpectra_model_C_resultSimpleComposition = \
                        copy.deepcopy(current_model_C.result_simple)


                    # if model A-B-C were loaded then calc linear composition of two resulting spectra
                    # from these two models:

                    #     flush Dict of Set of TWO models results:
                    self.setOfSnapshotSpectra.flushDictOfSpectra()
                    currentSpectra_model_A_resultSimpleComposition.label = self.model_A.get_Model_name()
                    self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra_model_A_resultSimpleComposition)
                    currentSpectra_model_B_resultSimpleComposition.label = self.model_B.get_Model_name()
                    self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra_model_B_resultSimpleComposition)
                    currentSpectra_model_C_resultSimpleComposition.label = self.model_C.get_Model_name()
                    self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra_model_C_resultSimpleComposition)

                    self.setOfSnapshotSpectra.weight_R_factor_chi = self.weight_R_factor_chi
                    self.setOfSnapshotSpectra.weight_R_factor_FTR = self.weight_R_factor_FTR
                    self.setOfSnapshotSpectra.result.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    self.setOfSnapshotSpectra.result_simple.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                    self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.label_latex_ideal_curve = \
                        self.experiment.label_latex_ideal_curve

                    # we already did (applied scale factors) it in models A, B, C:
                    self.setOfSnapshotSpectra.result.scale_theory_factor_FTR = 1
                    self.setOfSnapshotSpectra.result_simple.scale_theory_factor_FTR = 1
                    # was some problem with a scaling, therefore we store
                    # result_FTR_from_linear_Chi_k.scale_theory_factor_FTR without changes:
                    # self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR = 1

                    self.setOfSnapshotSpectra.result.scale_theory_factor_CHI = 1
                    self.setOfSnapshotSpectra.result_simple.scale_theory_factor_CHI = 1
                    self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_CHI = 1

                    if self.do_FTR_from_linear_Chi_k_SpectraComposition:
                        # if i == 4:
                        #     print(i)
                        # ----- Linear Composition _FTR_from_linear_Chi_k of Snapshots:
                        self.setOfSnapshotSpectra.calcLinearSpectraComposition_FTR_from_linear_Chi_k()
                        # print('Linear FTR from Chi(k) composition has been calculated')
                        self.setOfSnapshotSpectra.updateInfo_LinearComposition_FTR_from_linear_Chi_k()
                        R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_LinearComposition_FTR_from_linear_Chi_k()

                        self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k)
                        self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                        if R_tot < self.minimum.Rtot:
                            self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                            modelNameTxt = self.setOfSnapshotSpectra.getInfo_LinearComposition_FTR_from_linear_Chi_k()
                            self.minimum.snapshotName = modelNameTxt + '\n' + current_model_A.result_simple.label_latex + \
                                                        '\n' + current_model_B.result_simple.label_latex + \
                                                        '\n' + current_model_C.result_simple.label_latex
                            self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                                   modelNameTxt + \
                                                   ',\nlinear $FT(r)\leftarrow\chi(k)$ snapshots composition,  $R_{{tot}}$  = {0}'.format(
                                                       round(R_tot, 4))
                            self.suptitle_fontsize = 14
                            self.updatePlotOfSnapshotsComposition_Linear_FTR_from_linear_Chi_k()
                            self.suptitle_fontsize = 18
                            self.minimum.indicator_minimum_from_FTRlinear_chi = True
                            self.minimum.model_A = copy.deepcopy(current_model_A)
                            self.minimum.model_B = copy.deepcopy(current_model_B)
                            self.minimum.model_C = copy.deepcopy(current_model_C)
                            self.minimum.setOfSnapshotSpectra = copy.deepcopy(self.setOfSnapshotSpectra)
                            # save ASCII column data:
                            if self.saveDataToDisk:
                                self.setOfSnapshotSpectra.saveSpectra_LinearComposition_FTR_from_linear_Chi_k(
                                    output_dir=self.outMinValsDir)

                                # store model-A snapshots for this minimum case:
                                current_model_A.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)
                                # store model-B snapshots for this minimum case:
                                current_model_B.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)
                                # store model-C snapshots for this minimum case:
                                current_model_C.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)

                    # flush Dict of Set of TWO models results:
                    self.setOfSnapshotSpectra.flushDictOfSpectra()
                    bar_index += 1
                    # #     flush Dict of Set of model-B Snapshots:
                    # self.model_B.setOfSnapshotSpectra.flushDictOfSpectra()
                    # #     flush Dict of Set of model-A Snapshots:
                    # self.model_A.setOfSnapshotSpectra.flushDictOfSpectra()
        print('Number of calculated cases: {}'.format(bar_index-1))
        print('Total Number of cases: {}'.format(length))
        bar.update(bar_index-1)
        bar.finish()
        print()
        print('minimum Rtot = {}'.format(self.minimum.Rtot))
        print('{}'.format(self.minimum.snapshotName))
        runtime = timer() - start
        print("runtime is {0:f} seconds".format(runtime))

        # # store table to ASCII file:
        # self.table.outDirPath = self.outMinValsDir
        # timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
        # # modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        # modelName = self.model_A.get_Model_name() + '_and_' + self.model_B.get_Model_name()
        # self.table.outFileName = modelName + timestamp + '_So={1:1.3f}_R={0:1.4}.txt'.format(self.minimum.Rtot,
        #                                                                                      self.scale_theory_factor_FTR)
        # self.table.writeToASCIIFile()

    def calcSelectedSnapshotFile(self):
        '''
        calculate and plot graphs only for selected snapshot file
        :return:
        '''
        import tkinter as tk
        from tkinter import filedialog
        # open GUI filedialog to select snapshot file:
        a = StoreAndLoadVars()
        print('last used: {}'.format(a.getLastUsedFilePath()))
        # openfile dialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("snapshot files", '*.dat')],
                                               initialdir=a.getLastUsedDirPath())
        if os.path.isfile(file_path):
            a.lastUsedFilePath = file_path
            a.saveLastUsedFilePath()

            # change the working directory path to selected one:
            self.projectWorkingFEFFoutDirectory = os.path.dirname(file_path)
            # search for experiment and theory files:
            self.getInDirectoryStandardFilePathes()
            self.listOfSnapshotFiles = [file_path]

            if self.user == 'PK':
                self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                                  f'{self.sample_preparation_mode}.chik')
            elif self.user == 'ID':
                self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                                  f'SM_{self.sample_preparation_mode}_av.chik')
            # load experiment/ideal curve:
            self.experiment.user = self.user
            self.experiment.loadSpectrumData()
            self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
            self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
            self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
            if self.sample_preparation_mode == 'AG':
                self.experiment.label_latex_ideal_curve = self.user + ': as grown'
                self.outMinValsDir_mask = '_as_grown_'
            # set experiment spectra:
            self.set_ideal_curve_params()
            # start searching procedure:
            self.findBestSnapshotFromList()

    def calcAllSnapshotFiles(self):
        '''
        main method to run searching procedure of minimum R-factor snapshot
        :return:
        '''
        import tkinter as tk
        from tkinter import filedialog
        # open GUI filedialog to select feff_0001 working directory:
        a = StoreAndLoadVars()
        print('last used: {}'.format(a.getLastUsedDirPath()))
        # openfile dialoge
        root = tk.Tk()
        root.withdraw()
        dir_path = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            a.lastUsedDirPath = dir_path
            a.saveLastUsedDirPath()

        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = dir_path
        # search for experiment and theory files:
        self.getInDirectoryStandardFilePathes()

        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotFromList()

    def calcAllSnapshotFiles_temperature(self):
        '''
        main method to run searching procedure of minimum R-factor snapshot
        compare with 350.chik in data folder
        :return:
        '''
        import tkinter as tk
        from tkinter import filedialog, messagebox
        # open GUI filedialog to select feff_0001 working directory:
        a = StoreAndLoadVars()
        a.fileNameOfStoredVars = 'model_single.pckl'
        print('last used: {}'.format(a.getLastUsedDirPath()))
        # openfile dialoge
        root = tk.Tk()
        root.withdraw()
        txt_info = "select the model FEFF-out folder\nN={} Mn atoms".format(
            self.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            a.lastUsedDirPath = dir_path
            a.saveLastUsedDirPath()

        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = dir_path
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.listOfSnapshotFiles = listOfFilesFN_with_selected_ext(self.projectWorkingFEFFoutDirectory, ext='dat')
        if self.user == 'PK':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'{self.sample_preparation_mode}.chik')
        elif self.user == 'ID':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'SM_{self.sample_preparation_mode}_av.chik')
        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
        self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
        if self.sample_preparation_mode == 'AG':
            self.experiment.label_latex_ideal_curve = self.user + ': as grown'
            self.outMinValsDir_mask = '_as_grown_'

        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotFromList()

    def calcAllSnapshotFilesFor_2_type_Models_single(self):
        '''
        main method to run searching procedure of minimum R-factor
        Combine with the linear coefficients of two models A and B
        :return:
        '''
        model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles, \
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles, \
        outDirectoryForTowModelsFitResults = self.loadListOfFilesFor_2_type_Models()

        # change the working directory path to selected one:
        self.model_A.projectWorkingFEFFoutDirectory = model_A_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_A.listOfSnapshotFiles = model_A_listOfSnapshotFiles
        # ==============================================================================================================

        # load B-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        self.model_B.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_B.listOfSnapshotFiles = model_B_listOfSnapshotFiles
        # ==============================================================================================================
        # select outDirectoryForTowModelsFitResults:

        self.outDirectoryForTowModelsFitResults = outDirectoryForTowModelsFitResults

        if self.user == 'PK':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'{self.sample_preparation_mode}.chik')
        elif self.user == 'ID':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'SM_{self.sample_preparation_mode}_av.chik')
        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
        self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
        if self.sample_preparation_mode == 'AG':
            self.experiment.label_latex_ideal_curve = self.user + ': as grown'
            self.outMinValsDir_mask = '_as_grown_'

        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotsCombinationFrom_2_type_Models()

    def calcAllSnapshotFilesFor_3_type_Models_single(self):
        '''
        main method to run searching procedure of minimum R-factor
        Combine with the linear coefficients of two models A and B
        :return:
        '''
        model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles, \
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles, \
        model_C_projectWorkingFEFFoutDirectory, model_C_listOfSnapshotFiles, \
        outDirectoryFor_3_type_ModelsFitResults = self.loadListOfFilesFor_3_type_Models()

        # change the working directory path to selected one:
        self.model_A.projectWorkingFEFFoutDirectory = model_A_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_A.listOfSnapshotFiles = model_A_listOfSnapshotFiles
        # ==============================================================================================================

        # load B-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        self.model_B.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_B.listOfSnapshotFiles = model_B_listOfSnapshotFiles
        # ==============================================================================================================

        # load C-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_C_projectWorkingFEFFoutDirectory
        self.model_C.projectWorkingFEFFoutDirectory = model_C_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_C.listOfSnapshotFiles = model_C_listOfSnapshotFiles
        # ==============================================================================================================


        # select outDirectoryForTowModelsFitResults:
        self.outDirectoryFor_3_type_ModelsFitResults = outDirectoryFor_3_type_ModelsFitResults

        if self.user == 'PK':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'{self.sample_preparation_mode}.chik')
        elif self.user == 'ID':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'SM_{self.sample_preparation_mode}_av.chik')
        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
        self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
        if self.sample_preparation_mode == 'AG':
            self.experiment.label_latex_ideal_curve = self.user + ': as grown'
            self.outMinValsDir_mask = '_as_grown_'

        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotsCombinationFrom_3_type_Models()

    def calcAllSnapshotFilesFor_2_type_Models_parallel(self,
                                                       model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles,
                                                       model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles,
                                                       outDirectoryForTowModelsFitResults):
        '''
        main method to run searching procedure of minimum R-factor
        Combine with the linear coefficients of two models A and B
        method to load sliced lists of files.
        :return:
        '''

        # change the working directory path to selected one:
        self.model_A.projectWorkingFEFFoutDirectory = model_A_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_A.listOfSnapshotFiles = model_A_listOfSnapshotFiles
        # ==============================================================================================================

        # load B-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        self.model_B.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_B.listOfSnapshotFiles = model_B_listOfSnapshotFiles
        # ==============================================================================================================
        # select outDirectoryForTowModelsFitResults:

        self.outDirectoryForTowModelsFitResults = outDirectoryForTowModelsFitResults

        if self.user == 'PK':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'{self.sample_preparation_mode}.chik')
        elif self.user == 'ID':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'SM_{self.sample_preparation_mode}_av.chik')
        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
        self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
        if self.sample_preparation_mode == 'AG':
            self.experiment.label_latex_ideal_curve = self.user + ': as grown'
            self.outMinValsDir_mask = '_as_grown_'

        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotsCombinationFrom_2_type_Models()

    def calcAllSnapshotFilesFor_3_type_Models_parallel(self,
        model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles,
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles,
        model_C_projectWorkingFEFFoutDirectory, model_C_listOfSnapshotFiles,
        outDirectoryFor_3_type_ModelsFitResults):
        '''
        main method to run searching procedure of minimum R-factor
        Combine with the linear coefficients of 3 models A, B and C
        method to load sliced lists of files.
        :return:
        '''

        # change the working directory path to selected one:
        self.model_A.projectWorkingFEFFoutDirectory = model_A_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_A.listOfSnapshotFiles = model_A_listOfSnapshotFiles
        # ==============================================================================================================

        # load B-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        self.model_B.projectWorkingFEFFoutDirectory = model_B_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_B.listOfSnapshotFiles = model_B_listOfSnapshotFiles
        # ==============================================================================================================

        # load C-model data:
        # change the working directory path to selected one:
        self.projectWorkingFEFFoutDirectory = model_C_projectWorkingFEFFoutDirectory
        self.model_C.projectWorkingFEFFoutDirectory = model_C_projectWorkingFEFFoutDirectory
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        self.model_C.listOfSnapshotFiles = model_C_listOfSnapshotFiles
        # ==============================================================================================================

        # select outDirectoryForTowModelsFitResults:
        self.outDirectoryFor_3_type_ModelsFitResults = outDirectoryFor_3_type_ModelsFitResults

        if self.user == 'PK':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'{self.sample_preparation_mode}.chik')
        elif self.user == 'ID':
            self.experiment.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data',
                                                              f'SM_{self.sample_preparation_mode}_av.chik')
        # load experiment/ideal curve:
        self.experiment.user = self.user
        self.experiment.loadSpectrumData()
        self.experiment.ftr_vector = self.experiment.ftr_vector * self.scale_experiment_factor_FTR
        self.experiment.label_latex_ideal_curve = self.user + f': T={self.sample_preparation_mode}' + '$^{\circ}$'
        self.outMinValsDir_mask = f'_T={self.sample_preparation_mode}_'
        if self.sample_preparation_mode == 'AG':
            self.experiment.label_latex_ideal_curve = self.user + ': as grown'
            self.outMinValsDir_mask = '_as_grown_'

        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotsCombinationFrom_3_type_Models()

    def loadListOfFilesFor_2_type_Models(self):
        '''
        load directories and lists of files for two models A-B
        :return:
        '''
        import tkinter as tk
        from tkinter import filedialog, messagebox

        # open GUI filedialog to select feff_0001 working directory:
        a = StoreAndLoadVars()
        a.fileNameOfStoredVars = 'model_2_a_vars.pckl'
        print('last used: {}'.format(a.getLastUsedDirPath()))
        # openfile dialoge
        root = tk.Tk()
        root.withdraw()

        # load A-model data:
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the A-model \n" \
                   "        (top layer model !)\n" \
                   "-=============================-\n" \
                   "    This list of files will be divided\n" \
                   "       by parallel job value\n" \
                   "________________________________\n\n" \
                   "         FEFF-out folder \n" \
                   "         (Ex: feff_0001)\n" \
                   "________________________________\n\n" \
                   "          N={} Mn atoms".format(
            self.model_A.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path_model_A = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
        if os.path.isdir(dir_path_model_A):
            a.lastUsedDirPath = dir_path_model_A
            a.saveLastUsedDirPath()

        # change the working directory path to selected one:
        model_A_projectWorkingFEFFoutDirectory = dir_path_model_A
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        model_A_listOfSnapshotFiles = listOfFilesFN_with_selected_ext(model_A_projectWorkingFEFFoutDirectory,
                                                                           ext='dat')
        # ==============================================================================================================

        # load B-model data:
        b = StoreAndLoadVars()
        b.fileNameOfStoredVars = 'model_2_b_vars.pckl'
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the B-model \n" \
                   "        (down layer model !)\n" \
                   "-=============================-\n" \
                   "________________________________\n\n" \
                   "         FEFF-out folder \n" \
                   "         (Ex: feff_0001)\n" \
                   "________________________________\n\n" \
                   "          N={} Mn atoms".format(
            self.model_B.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=b.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            b.lastUsedDirPath = dir_path
            b.saveLastUsedDirPath()

        # change the working directory path to selected one:
        model_B_projectWorkingFEFFoutDirectory = dir_path
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        model_B_listOfSnapshotFiles = listOfFilesFN_with_selected_ext(model_B_projectWorkingFEFFoutDirectory,
                                                                           ext='dat')
        # ==============================================================================================================
        # select outDirectoryForTowModelsFitResults:
        c = StoreAndLoadVars()
        c.fileNameOfStoredVars = 'outdir_vars.pckl'
        txt_info = "select the directory to store the results\nof two A-B models fitting"
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=c.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            c.lastUsedDirPath = dir_path
            c.saveLastUsedDirPath()

        model_A_modelName = os.path.split(os.path.split((model_A_projectWorkingFEFFoutDirectory))[0])[1]
        print(model_A_modelName)
        model_B_modelName = os.path.split(os.path.split((model_B_projectWorkingFEFFoutDirectory))[0])[1]
        print(model_B_modelName)
        maskTxt = '[{0}]__[{1}]__{2}_({3})'.format(model_A_modelName, model_B_modelName,
                                                 self.user, self.sample_preparation_mode)
        outDirectoryForTowModelsFitResults = create_out_data_folder(dir_path, maskTxt)

        return model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles, \
               model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles, \
               outDirectoryForTowModelsFitResults

    def loadListOfFilesFor_3_type_Models(self):
        '''
        load directories and lists of files for two models A-B-C
        :return:
        '''
        import tkinter as tk
        from tkinter import filedialog, messagebox

        # open GUI filedialog to select feff_0001 working directory:
        a = StoreAndLoadVars()
        a.fileNameOfStoredVars = 'model_3_a_vars.pckl'
        print('last used: {}'.format(a.getLastUsedDirPath()))
        # openfile dialoge
        root = tk.Tk()
        # root.option_add('*Dialog.msg.width', 50)
        root.option_add('*font', 'Helvetica -15')
        root.withdraw()

        # load A-model data:
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the A-model \n" \
                   "        (top layer model !)\n" \
                   "-=============================-\n" \
                   "    This list of files will be divided\n" \
                   "       by parallel job value\n" \
                   "________________________________\n\n" \
                   "         FEFF-out folder \n" \
                   "         (Ex: feff_0001)\n" \
                   "________________________________\n\n" \
                   "          N={} Mn atoms".format(
            self.model_A.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path_model_A = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
        if os.path.isdir(dir_path_model_A):
            a.lastUsedDirPath = dir_path_model_A
            a.saveLastUsedDirPath()

        # change the working directory path to selected one:
        model_A_projectWorkingFEFFoutDirectory = dir_path_model_A
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        model_A_listOfSnapshotFiles = listOfFilesFN_with_selected_ext(model_A_projectWorkingFEFFoutDirectory,
                                                                           ext='dat')
        # ==============================================================================================================

        # load B-model data:
        b = StoreAndLoadVars()
        b.fileNameOfStoredVars = 'model_3_b_vars.pckl'
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the B-model \n" \
                   "        (middle layer model !)\n" \
                   "-=============================-\n" \
                   "________________________________\n\n" \
                   "         FEFF-out folder \n" \
                   "         (Ex: feff_0001)\n" \
                   "________________________________\n\n" \
                   "          N={} Mn atoms".format(
            self.model_B.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=b.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            b.lastUsedDirPath = dir_path
            b.saveLastUsedDirPath()

        # change the working directory path to selected one:
        model_B_projectWorkingFEFFoutDirectory = dir_path
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        model_B_listOfSnapshotFiles = listOfFilesFN_with_selected_ext(model_B_projectWorkingFEFFoutDirectory,
                                                                           ext='dat')
        # ==============================================================================================================

        # load C-model data:
        c = StoreAndLoadVars()
        c.fileNameOfStoredVars = 'model_3_c_vars.pckl'
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the C-model \n" \
                   "        (down layer model !)\n" \
                   "-=============================-\n" \
                   "________________________________\n\n" \
                   "         FEFF-out folder \n" \
                   "         (Ex: feff_0001)\n" \
                   "________________________________\n\n" \
                   "          N={} Mn atoms".format(
            self.model_C.numberOfSerialEquivalentAtoms)
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=c.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            c.lastUsedDirPath = dir_path
            c.saveLastUsedDirPath()

        # change the working directory path to selected one:
        model_C_projectWorkingFEFFoutDirectory = dir_path
        # search for experiment and theory files:
        # self.getInDirectoryStandardFilePathes()
        model_C_listOfSnapshotFiles = listOfFilesFN_with_selected_ext(model_C_projectWorkingFEFFoutDirectory,
                                                                           ext='dat')
        # ==============================================================================================================
        # select outDirectoryForThreeModelsFitResults:
        out = StoreAndLoadVars()
        out.fileNameOfStoredVars = 'outdir_3_vars.pckl'
        txt_info = "select the directory to store the results\nof three A-B-C models fitting"
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=out.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            out.lastUsedDirPath = dir_path
            out.saveLastUsedDirPath()

        model_A_modelName = os.path.split(os.path.split((model_A_projectWorkingFEFFoutDirectory))[0])[1]
        print(model_A_modelName)
        model_B_modelName = os.path.split(os.path.split((model_B_projectWorkingFEFFoutDirectory))[0])[1]
        print(model_B_modelName)
        model_C_modelName = os.path.split(os.path.split((model_C_projectWorkingFEFFoutDirectory))[0])[1]
        print(model_C_modelName)
        maskTxt = '[{0}]__[{1}]__[{2}]__{3}_({4})'.format(model_A_modelName, model_B_modelName, model_C_modelName,
                                                 self.user, self.sample_preparation_mode)
        outDirectoryFor_3_type_ModelsFitResults = create_out_data_folder(dir_path, maskTxt)
        print('outdata directory is:\n', outDirectoryFor_3_type_ModelsFitResults, '\n', '---'*15)

        return model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles, \
               model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles, \
               model_C_projectWorkingFEFFoutDirectory, model_C_listOfSnapshotFiles, \
               outDirectoryFor_3_type_ModelsFitResults

    def loadAllSpectraToDict(self):
        # load all snapshots and prepare data for the next calculations
        # define the next variables:
        # self.listOfSnapshotFiles -list of snapshots
        # self.numberOfSerialEquivalentAtoms - current number of serial equivalent atoms
        # --------------------------------------------------------------
        currentSerialSnapNumber = 0
        number = 0
        result_dict = odict()

        for filePath in self.listOfSnapshotFiles:
            # print('==> file is: {0}'.format(filePath))
            # print('==> Number is: {0}'.format(number))

            currentSerialSnapNumber = currentSerialSnapNumber + 1

            self.theory_one.pathToLoadDataFile = filePath
            self.theory_one.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.theory_one.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.theory_one.loadSpectrumData()
            self.updateInfo()

            modelName, snapNumberStr = self.get_name_of_model_from_fileName()

            currentSpectra = copy.deepcopy(self.theory_one)

            self.setOfSnapshotSpectra.addSpectraToDict(currentSpectra)
            # do not need weights, we only create linear combination of model_B spectra
            self.setOfSnapshotSpectra.weight_R_factor_chi = self.weight_R_factor_chi
            self.setOfSnapshotSpectra.weight_R_factor_FTR = self.weight_R_factor_FTR
            self.setOfSnapshotSpectra.result.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.setOfSnapshotSpectra.result_simple.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.label_latex_ideal_curve = \
                self.experiment.label_latex_ideal_curve

            self.setOfSnapshotSpectra.result.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.setOfSnapshotSpectra.result_simple.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_FTR = self.scale_theory_factor_FTR

            self.setOfSnapshotSpectra.result.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.setOfSnapshotSpectra.result_simple.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            self.setOfSnapshotSpectra.result_FTR_from_linear_Chi_k.scale_theory_factor_CHI = self.scale_theory_factor_CHI

            R_tot, R_ftr, R_chi = self.get_R_factor()

            # self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
            # self.currentValues.number = number
            # self.currentValues.snapshotName = os.path.basename(filePath)
            # # self.table.addRecord(self.currentValues)

            # ==========================================================================
            # do not need spectra from single atom in multiatoms structure:
            # if R_tot < self.minimum.Rtot:
            #     # check single spectrum for best fit:
            #     self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
            #     # model B:
            #     self.theory_one.pathToLoadDataFile = self.theory_one.pathToLoadDataFile
            #     self.theory_one.scale_theory_factor_FTR = self.scale_theory_factor_FTR
            #     self.theory_one.scale_theory_factor_CHI = self.scale_theory_factor_CHI
            #     self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
            #     self.theory_one.loadSpectrumData()
            #     self.updateInfo()
            #     self.updatePlot()

            if currentSerialSnapNumber == self.numberOfSerialEquivalentAtoms:

                currentSerialSnapNumber = 0
                # add next number for dict record
                number = number + 1

                # ----- Simple Composition of Snapshots:
                self.setOfSnapshotSpectra.calcSimpleSpectraComposition()
                # print('Simple composition has been calculated')
                self.setOfSnapshotSpectra.updateInfo_SimpleComposition()

                R_tot, R_ftr, R_chi = self.setOfSnapshotSpectra.get_R_factor_SimpleComposition()
                # self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
                # self.currentValues.number = number
                # self.currentValues.snapshotName = self.setOfSnapshotSpectra.result_simple.label
                # self.table.addRecord(self.currentValues)

                self.theory_one = copy.deepcopy(self.setOfSnapshotSpectra.result_simple)
                self.theory_one.label_latex_ideal_curve = self.experiment.label_latex_ideal_curve
                if R_tot < self.minimum.Rtot:
                    # check single model simple composition for best fit:
                    self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                    self.minimum.snapshotName = modelName + ': ' \
                                        + self.setOfSnapshotSpectra.getInfo_SimpleComposition()\
                                        + ', simple snapshots composition'
                    self.graph_title_txt = 'model [$S_0^2$={0:1.3f}]: '.format(self.scale_theory_factor_FTR) + \
                                           modelName + ', simple snapshots composition,  $R_{{tot}}$  = {0}'.format(
                        round(self.minimum.Rtot, 4))

                    # do not want to change methods
                    # updatePlotOfSnapshotsComposition_Simple() in this class therefor do deepcopy:
                    # self.setOfSnapshotSpectra.flushDictOfSpectra()
                    self.updatePlotOfSnapshotsComposition_Simple()
                    # save ASCII column data:
                    if self.saveDataToDisk:
                        self.setOfSnapshotSpectra.saveSpectra_SimpleComposition(
                            output_dir=self.outMinValsDir)

                # store current result of model-i simple composition:
                currentSetOfSpectra = copy.deepcopy(self.setOfSnapshotSpectra)

                # add object to the dict:
                result_dict[number] = odict({'data': currentSetOfSpectra})

                #     flush Dict of Set of TWO models results:
                self.setOfSnapshotSpectra.flushDictOfSpectra()

        return result_dict

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    #
    # a = FTR_gulp_to_feff_A_model()
    # a.updateInfo()
    # print(a.get_R_factor())
    # # a.plotSpectra_chi_k()
    # # a.plotSpectra_FTR_r()
    # a.setupAxes()
    # a.updatePlot()
    # plt.show()


    # # start global searching procedure:
    # fit the sub-components of one model
    # If Model has X-num of Mn then procedure search the weights of X-num of snapshots.
    a = FTR_gulp_to_feff_A_model_base()
    a.numberOfSerialEquivalentAtoms = 9
    a.do_FTR_from_linear_Chi_k_SpectraComposition = False
    a.find_min_Rtot_in_single_snapshot = False
    a.weight_R_factor_FTR = 1.0
    a.weight_R_factor_chi = 0.0
    a.scale_theory_factor_FTR = 0.81
    a.scale_experiment_factor_FTR = 1.0

    #  change the user name, which parameters for xftf transformation you want to use:
    a.user = 'ID'
    # change tha sample preparation method:
    a.sample_preparation_mode = '450'
    # if you want compare with the theoretical average, do this:
    # a.calcAllSnapshotFiles()
    #  if you want to find the minimum from the all snapshots do this:
    a.calcAllSnapshotFiles_temperature()
    # if you want to check only one snapshot do this:
    # a.calcSelectedSnapshotFile()


    # # start global search of Two-model combination:
    # a = FTR_gulp_to_feff_A_model_base()
    # a.weight_R_factor_FTR = 1.0
    # a.weight_R_factor_chi = 0.0
    # a.scale_theory_factor_FTR = 0.81
    # a.scale_experiment_factor_FTR = 1.0
    #
    # a.model_A.numberOfSerialEquivalentAtoms = 5
    # a.model_B.numberOfSerialEquivalentAtoms = 2
    # a.model_C.numberOfSerialEquivalentAtoms = 3
    #
    # #  change the user name, which parameters for xftf transformation you want to use:
    # a.user = 'ID'
    # # change tha sample preparation method:
    # a.sample_preparation_mode = '450'
    # # if you want compare with the theoretical average, do this:
    # # a.calcAllSnapshotFiles()
    #
    # # for debug and profiling:
    # a.saveDataToDisk = True
    #
    # #  if you want to find the minimum from the all snapshots do this:
    # a.calcAllSnapshotFilesFor_2_type_Models_single()
    # # a.calcAllSnapshotFilesFor_3_type_Models_single()
    # # if you want to check only one snapshot do this:
    # # a.calcSelectedSnapshotFile()


    # # start calculate only snapshot file:
    # a = FTR_gulp_to_feff_A_model_base()
    # a.calcSelectedSnapshotFile()