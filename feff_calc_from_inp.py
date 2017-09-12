'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-09-11
'''
from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name, get_upper_folder_name, \
    listOfFilesFN_with_selected_ext, create_out_data_folder, create_data_folder
import os
import numpy as np
import datetime
from timeit import default_timer as timer
import copy

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from shutil import copyfile
# for run executable file:
import subprocess
from feff.libs.plot_data import plotData

from feff.libs.dir_and_file_operations import create_out_data_folder, listOfFiles, listOfFilesFN, \
    deleteAllFilesInFolder, listOfFilesNameWithoutExt, listOfFilesFN_with_selected_ext

class FEFF_calculation_class():
    def __init__(self):
        # dir name of input FEFF files:
        self.dirNameInp = ''
        self.list_of_inp_files = []

        self.dirNameOut = ''

        # path to exe file of the feff program
        # standart feff exe file:
        # feff_exe = 'wine /home/yugin/PycharmProjects/feff/exe/feff84_nclusx_175.exe' # path to exe file of the feff program

        # for big cases. In src files was changed
        # c      max number of atoms in problem for the pathfinder
        #        parameter (natx =10000) before it was: natx =1000
        # c      max number of unique potentials (potph) (nphx must be ODD to
        # c      avoid compilation warnings about alignment in COMMON blocks)
        #        parameter (nphx = 21) before it was: nphx = 7
        self.path_to_feff_exe = 'wine cmd /C "/home/yugin/PycharmProjects/feff/exe/feff_84_2016-08-02.bat"'

        self.is_RAM_disk_exist = True
        self.path_to_RAM_disk = '/mnt/ramdisk/yugin/tmp'

    def get_working_dir(self):
        # load directory with FEFF input files for calculation in parallel mode

        import tkinter as tk
        from tkinter import filedialog, messagebox

        # open GUI filedialog to select feff_0001 working directory:
        a = StoreAndLoadVars()
        a.fileNameOfStoredVars = 'feff_parallel_calc_vars.pckl'
        print('last used: {}'.format(a.getLastUsedDirPath()))
        # openfile dialoge
        root = tk.Tk()
        # root.option_add('*Dialog.msg.width', 50)
        root.option_add('*font', 'Helvetica -15')
        root.withdraw()

        # load A-model data:
        txt_info = "                                    \n" \
                   "-=============================-\n" \
                   "        select the inp__#### \n" \
                   "        \n" \
                   "-=============================-\n" \
                   "    This list of files will be divided\n" \
                   "       by parallel job value\n" \
                   "________________________________\n\n" \
                   "         FEFF-input folder \n" \
                   "         (Ex: inp_0001)\n" \
                   "________________________________\n\n"
        messagebox.showinfo("info", txt_info)
        dir_path = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
        if os.path.isdir(dir_path):
            a.lastUsedDirPath = dir_path
            a.saveLastUsedDirPath()

        self.dirNameInp = dir_path

    def prepare_vars(self):
        self.list_of_inp_files = listOfFilesFN_with_selected_ext(self.dirNameInp, ext='inp')
        self.dirNameOut = create_out_data_folder(main_folder_path=os.path.split(os.path.normpath(self.dirNameInp))[0],
                               first_part_of_folder_name='feff_')

    def do_calculation_serial(self, list_of_inp_files=[]):
        # do calculation in a serial mode (under parallel constructions)

        # create unique tmp folder
        if self.is_RAM_disk_exist:
            outDirFEFFtmp = create_out_data_folder(main_folder_path=self.path_to_RAM_disk,
                                                   first_part_of_folder_name='tmp_')
        else:
            outDirFEFFtmp = create_out_data_folder(main_folder_path=os.path.split(os.path.normpath(self.dirNameInp))[0],
                               first_part_of_folder_name='tmp_')

        # outDirFEFFchi = create_out_data_folder(main_folder_path=os.path.split(os.path.normpath(self.dirNameInp))[0],
        #                                        first_part_of_folder_name='tmp_chi_')
        # self.feffCalcFun(list_of_inp_files=list_of_inp_files,
        #                 dataPath=self.dirNameInp,
        #                 tmpPath=outDirFEFFtmp,
        #                 outDirPath=outDirFEFFchi,
        #                 plotTheData=True)


        self.feffCalcFun(list_of_inp_files=list_of_inp_files,
                        dataPath=self.dirNameInp,
                        tmpPath=outDirFEFFtmp,
                        outDirPath=self.dirNameOut,
                        plotTheData=True)

    def feffCalcFun(self, list_of_inp_files=[],
                    dataPath='',
                    tmpPath='',
                    outDirPath='', plotTheData=True):
        # Please, change only the load data (input) directory name!
        # directory with the output feff files = dataPath

        folder_path, folder_name = os.path.split(os.path.dirname(dataPath))
        filesFullPathName = list_of_inp_files

        # tmpPath = os.path.join(tmpDirPath, folder_name ) # tmp folder for the temporary calculation files
        # if  not (os.path.isdir(tmpPath)):
        #             os.makedirs(tmpPath, exist_ok=True)


        feff_exe = self.path_to_feff_exe

        # outDirPath = os.path.join( '/home/yugin/VirtualboxShare/FEFF/out', folder_name )
        # if  not (os.path.isdir(outDirPath)):
        #             os.makedirs(outDirPath, exist_ok=True)



        result_dir = create_out_data_folder(outDirPath)
        # go to the tmp directory:
        os.chdir(tmpPath)

        i = 0
        numOfColumns = len(filesFullPathName)

        # started to average from that snupshot number:
        shift = round(numOfColumns / 2)

        k = np.r_[0:20.05:0.05]
        numOfRows = len(k)
        chi = np.zeros((numOfRows, numOfColumns))
        chi_std = np.zeros((numOfRows))
        chi_mean = np.zeros((numOfRows))

        chi_median = np.zeros((numOfRows))
        chi_max = np.zeros((numOfRows))
        chi_min = np.zeros((numOfRows))

        # copy file input to the tmp directory and start calculations:
        for f in filesFullPathName:
            # copy input file to the feff.inp file in tmp directory:
            copyfile(f, os.path.join(tmpPath, 'feff.inp'))
            print('copy the ', f, ' to the -> ', os.path.join(tmpPath, 'feff.inp'))
            print('run the feff calculation')

            currentFileNameBase = os.path.basename(f)
            currentFileName = os.path.splitext(currentFileNameBase)[0]

            # run the feff calculation:
            subprocess.call(feff_exe, shell=True)

            # Check if chi.dat is created:
            if os.path.isfile(os.path.join(tmpPath, 'chi.dat')):
                print('copy the chi.dat to the ->', )
                # create a new name to the chi.dat output file:
                # chiOutName = "chi_%05d.dat" %(i)
                # chiOutName = 'chi_' + currentFileName + "_%05d.dat" % (i + 1)
                chiOutName = 'chi_' + currentFileName + ".dat"
                copyfile(os.path.join(tmpPath, 'chi.dat'), os.path.join(outDirPath, chiOutName))
                print('feff calculation is finished')

                deleteAllFilesInFolder(tmpPath)
                # load txt output files:
                data = np.loadtxt(os.path.join(outDirPath, chiOutName), float)
                # select chi values only for k > 0 because FEFF output files contain the different length of k-vector:
                if len(data[:, 0]) < numOfRows:
                    chi[1:, i] = data[:, 1]
                elif len(data[:, 0]) == numOfRows:
                    chi[:, i] = data[:, 1]
                else:
                    print('you have unexpected numbers of rows in your output files')
                    print('input file name is: ', f)
                    print('number of elements is: ', len(data[:, 0]), ' the first k-element is: ', data[0, 0])

                if ((i % 1000) == 0) and (i > 2):

                    chi_std = np.std(chi[:, 0:i], axis=1)
                    chi_mean = np.mean(chi[:, 0:i], axis=1)

                    chi_median = np.median(chi[:, 0:i], axis=1)
                    chi_max = np.amax(chi[:, 0:i], axis=1)
                    chi_min = np.amin(chi[:, 0:i], axis=1)

                    out_array = np.zeros((numOfRows, 3 + 3))
                    out_array[:, 0] = k
                    out_array[:, 1] = chi_mean
                    out_array[:, 2] = chi_std

                    out_array[:, 3] = chi_median
                    out_array[:, 4] = chi_max
                    out_array[:, 5] = chi_min

                    headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
                    out_file_name = "result_%s" % (folder_name) + "_%05d.txt" % (i)
                    np.savetxt(os.path.join(result_dir, out_file_name), out_array, fmt='%1.6e', delimiter='\t',
                               header=headerTxt)
                    print('==> write iter number {0} to the {1} file'.format(i, out_file_name))
                    if plotTheData:
                        plotData(x=k, y=chi_mean, error=chi_std, numOfIter=i, out_dir=result_dir, case=folder_name,
                                 y_median=chi_median, y_max=chi_max, y_min=chi_min)

                    if (i > shift) and (((i - shift) % 1000) == 0) and ((i - shift) > 2):
                        # calc average with a shift (started not from the first snapshot)

                        chi_std = np.std(chi[:, shift:i], axis=1)
                        chi_mean = np.mean(chi[:, shift:i], axis=1)

                        chi_median = np.median(chi[:, shift:i], axis=1)
                        chi_max = np.amax(chi[:, shift:i], axis=1)
                        chi_min = np.amin(chi[:, shift:i], axis=1)

                        out_array = np.zeros((numOfRows, 3 + 3))
                        out_array[:, 0] = k
                        out_array[:, 1] = chi_mean
                        out_array[:, 2] = chi_std

                        out_array[:, 3] = chi_median
                        out_array[:, 4] = chi_max
                        out_array[:, 5] = chi_min

                        headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
                        out_file_name = f'aver_from_{shift}_to_{i}_' + "result_%s" % (folder_name) + "_%05d.txt" % (i)
                        np.savetxt(os.path.join(result_dir, out_file_name), out_array, fmt='%1.6e', delimiter='\t',
                                   header=headerTxt)
                        print('==> write iter number {0} to the {1} file'.format(i, out_file_name))
                        if plotTheData:
                            plotData(x=k, y=chi_mean, error=chi_std, numOfIter=i, out_dir=result_dir,
                                     case=folder_name + f'_from_{shift}_to_{i}]',
                                     y_median=chi_median, y_max=chi_max, y_min=chi_min)
            i += 1

        chi_std = np.std(chi[:, :], axis=1)
        chi_mean = np.mean(chi[:, :], axis=1)

        chi_median = np.median(chi[:, :], axis=1)
        chi_max = np.amax(chi[:, :], axis=1)
        chi_min = np.amin(chi[:, :], axis=1)

        out_array = np.zeros((numOfRows, 3 + 3))
        out_array[:, 0] = k
        out_array[:, 1] = chi_mean
        out_array[:, 2] = chi_std

        out_array[:, 3] = chi_median
        out_array[:, 4] = chi_max
        out_array[:, 5] = chi_min

        headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
        np.savetxt(os.path.join(result_dir, 'result.txt'), out_array, fmt='%1.6e', delimiter='\t', header=headerTxt)
        # copy result.txt file to the outDirPath folder:
        copyfile(os.path.join(result_dir, 'result.txt'), os.path.join(outDirPath, 'result_' + folder_name + '.txt'))
        if plotTheData:
            plotData(x=k, y=chi_mean, error=chi_std, numOfIter=i, out_dir=result_dir, case=folder_name,
                     y_median=chi_median, y_max=chi_max, y_min=chi_min)

        # ==========================================================================
        #  Calculate average data started from the middle snapshot number.
        # We suppose that structure will be already relaxed to the moment of middle snapshot.

        chi_std = np.std(chi[:, shift:-1], axis=1)
        chi_mean = np.mean(chi[:, shift:-1], axis=1)

        chi_median = np.median(chi[:, shift:-1], axis=1)
        chi_max = np.amax(chi[:, shift:-1], axis=1)
        chi_min = np.amin(chi[:, shift:-1], axis=1)

        out_array = np.zeros((numOfRows, 3 + 3))
        out_array[:, 0] = k
        out_array[:, 1] = chi_mean
        out_array[:, 2] = chi_std

        out_array[:, 3] = chi_median
        out_array[:, 4] = chi_max
        out_array[:, 5] = chi_min

        headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
        np.savetxt(os.path.join(result_dir, f'aver_from_{shift}_to_{numOfColumns}_result.txt'), out_array, fmt='%1.6e',
                   delimiter='\t', header=headerTxt)
        # copy result.txt file to the outDirPath folder:
        if plotTheData:
            plotData(x=k, y=chi_mean, error=chi_std, numOfIter=i, out_dir=result_dir,
                     case=folder_name + f'_shift={shift}',
                     y_median=chi_median, y_max=chi_max, y_min=chi_min)

        print('delete TMP folder: ', tmpPath)
        import shutil
        shutil.rmtree(tmpPath)
        print('program is finished')
        # print('-> create a video file:')
        # create_graphs_and_save_images_from_chi_dat(dataPath = outDirPath, each_elem_to_draw = 10)

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    obj = FEFF_calculation_class()
    obj.get_working_dir()
    obj.prepare_vars()
    obj.do_calculation_serial(list_of_inp_files=obj.list_of_inp_files[0:10])
