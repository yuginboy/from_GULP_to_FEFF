"""
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-28
"""
from feff.libs.class_Spectrum import Spectrum, GraphElement, TableData, BaseData
import os
import datetime
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name, get_upper_folder_name, \
    listOfFilesFN_with_selected_ext, create_out_data_folder
from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec
import numpy as np

class FTR_gulp_to_feff_A_model():
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

        # store to the ASCII table on a file:
        self.table = TableData()

        # weights of R-factors (Rtot = (Rchi* w1 + Rrtf*w2)/(w1+w2)):
        self.weights_of_R_factor = np.array([1, 1])

        self.FTR = GraphElement()
        self.Chi_k = GraphElement()
        self.suptitle_fontsize = 18

        self.projectWorkingFEFFoutDirectory = '/home/yugin/VirtualboxShare/GaMnO/debug/1mono1SR2VasVga2_6/feff__0001'
        self.listOfSnapshotFiles = []
        # out dir for  Rfactor minimum files:
        self.outMinValsDir = ''

        self.graph_title_txt = 'model name'
        self.showFigs = True
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

        self.set_ideal_curve_params()

    def getInDirectoryStandardFilePathes(self):
        '''
        return experiment ASCII data filepath if selected the corrected filepath and list of all snapshots ASCII data
        :return:
        '''
        self.listOfSnapshotFiles = listOfFilesFN_with_selected_ext(self.projectWorkingFEFFoutDirectory, ext='dat')
        self.experiment.pathToLoadDataFile = listOfFilesFN_with_selected_ext(self.projectWorkingFEFFoutDirectory, ext='txt')[0]

    def set_ideal_curve_params(self):
        self.theory_one.ideal_curve_ftr = self.experiment.ftr_vector
        self.theory_one.ideal_curve_r =   self.experiment.r_vector
        self.theory_one.ideal_curve_k =   self.experiment.k_vector
        self.theory_one.ideal_curve_chi = self.experiment.chi_vector

    def get_name_of_model_from_fileName(self):
        modelName = os.path.split(os.path.split(os.path.dirname(self.theory_one.pathToLoadDataFile))[0])[1]
        name = os.path.split(os.path.basename(self.theory_one.pathToLoadDataFile))[1]
        name = name.split('.')[0]
        snapNumberStr = name.split('chi_'+modelName+'_')[1]
        return modelName, snapNumberStr

    def updateInfo(self):
        modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        self.graph_title_txt = 'model: ' + modelName + ', $R_{{tot}}$  = {0}'.format(round(self.get_R_factor()[0], 4))
        self.theory_one.label_latex = 'snapshot: {0}'.format(snapNumberStr)

    def get_R_factor(self):
        R_chi = self.theory_one.get_chi_R_factor()
        R_ftr = self.theory_one.get_FTR_R_factor()

        R_tot = (self.weights_of_R_factor[1]*R_ftr + self.weights_of_R_factor[0]*R_chi) / np.sum(self.weights_of_R_factor)

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

            self.Chi_k.axes = self.fig.add_subplot(gs[0, 1])
            plt.axes(self.Chi_k.axes)
            self.plotSpectra_chi_k()
            # self.Chi_k.axes.invert_xaxis()
            self.Chi_k.axes.grid(True)

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
            out_file_name =  snapNumberStr + '_R={0:1.4}.png'.format(self.minimum.Rtot)
            self.fig.savefig(os.path.join(self.outMinValsDir, out_file_name))

    def findBestSnapshotFromList(self):

        if self.weights_of_R_factor[1] < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory,
                                                        first_part_of_folder_name='Rmin=Rchi')
        elif self.weights_of_R_factor[0] < 0.001:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory,
                                                        first_part_of_folder_name='Rmin=Rftr')
        else:
            self.outMinValsDir = create_out_data_folder(main_folder_path=self.projectWorkingFEFFoutDirectory, first_part_of_folder_name='Rmin=Rtot')
        self.setupAxes()
        number = 0
        for filePath in self.listOfSnapshotFiles:
            number = number + 1
            self.theory_one.pathToLoadDataFile = filePath
            self.theory_one.loadSpectrumData()
            self.updateInfo()
            R_tot, R_ftr, R_chi = self.get_R_factor()

            self.currentValues.Rtot, self.currentValues.Rftr, self.currentValues.Rchi = R_tot, R_ftr, R_chi
            self.currentValues.number = number
            self.currentValues.snapshotName = os.path.basename(filePath)
            self.table.addRecord(self.currentValues)
            if R_tot < self.minimum.Rtot:
                self.minimum.Rtot, self.minimum.Rftr, self.minimum.Rchi = R_tot, R_ftr, R_chi
                self.updatePlot()

        # store table to ASCII file:
        self.table.outDirPath = self.outMinValsDir
        timestamp = datetime.datetime.now().strftime("_[%Y-%m-%d_%H_%M_%S]_")
        modelName, snapNumberStr = self.get_name_of_model_from_fileName()
        self.table.outFileName = modelName + timestamp + '_R={0:1.4}.txt'.format(self.minimum.Rtot)
        self.table.writeToASCIIFile()

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
        # set experiment spectra:
        self.set_ideal_curve_params()
        # start searching procedure:
        self.findBestSnapshotFromList()

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
        file_path = filedialog.askopenfilename(filetypes = [("snapshot files",'*.dat')], initialdir=a.getLastUsedDirPath())
        if os.path.isfile(file_path):
            a.lastUsedFilePath = file_path
            a.saveLastUsedFilePath()

            # change the working directory path to selected one:
            self.projectWorkingFEFFoutDirectory = os.path.dirname(file_path)
            # search for experiment and theory files:
            self.getInDirectoryStandardFilePathes()
            self.listOfSnapshotFiles = [file_path]
            # set experiment spectra:
            self.set_ideal_curve_params()
            # start searching procedure:
            self.findBestSnapshotFromList()




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
    #
    # a = FTR_gulp_to_feff_A_model()
    # a.updateInfo()
    # print(a.get_R_factor())
    # # a.plotSpectra_chi_k()
    # # a.plotSpectra_FTR_r()
    # a.setupAxes()
    # a.updatePlot()
    # plt.show()


    # start global searching procedure:
    a = FTR_gulp_to_feff_A_model()
    a.weights_of_R_factor = np.array([1, 0])
    a.calcAllSnapshotFiles()


    # # start calculate only snapshot file:
    # a = FTR_gulp_to_feff_A_model()
    # a.calcSelectedSnapshotFile()

