'''
 Program to run FEFF 8.4.175 calculation in cycle
 We need only determine the data folder with feff input files
 Program return chi dat files for each input file and calculate mean, median, std, max, min values of chi
 for all chi dat files

 author: Zhenia Syriany (Yevgen Syryanyy)
 e-mail: yuginboy@gmail.com
 License: this code is in the GPL license
 Last modified: 2016-07-27
'''
"Main file for run feff calculation package"
# import pandas as pd
import sys
import os
import time
from io import StringIO
import numpy as np

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.gridspec as gridspec
from matplotlib import pylab
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
import re
from shutil import copyfile
# for run executable file:
import subprocess

# import plotting procedure:
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libs'))
from feff.libs.plot_data import plotData

from feff.libs.dir_and_file_operations import create_out_data_folder, listOfFiles, listOfFilesFN, \
    deleteAllFilesInFolder, listOfFilesNameWithoutExt, listOfFilesFN_with_selected_ext
# from libs.create_images import create_graphs_and_save_images_from_chi_dat

# for parallel calculation:
from joblib import Parallel, delayed
import multiprocessing

# def runInParallel(*fns):
#   proc = []
#   for fn in fns:
#     p = Process(target=fn)
#     p.start()
#     proc.append(p)
#   for p in proc:
#     p.join()

def touch(path):
    '''create empty file'''
    with open(path, 'a'):
        os.utime(path, None)

def feffCalcFun(dataPath = '/home/yugin/VirtualboxShare/FEFF/load/60/', tmpPath = '/home/yugin/VirtualboxShare/FEFF/tmp',
                outDirPath = '/home/yugin/VirtualboxShare/FEFF/out', plotTheData = True):
    # Please, change only the load data (input) directory name!
    # directory with the input feff files = dataPath

    folder_path, folder_name = os.path.split( os.path.dirname(dataPath) )
    files = listOfFiles(dataPath) # only files name
    # only the names without extansion:
    names = listOfFilesNameWithoutExt(dataPath)
    filesFullPathName = listOfFilesFN_with_selected_ext(dataPath, ext = 'inp')



    # tmpPath = os.path.join(tmpDirPath, folder_name ) # tmp folder for the temporary calculation files
    # if  not (os.path.isdir(tmpPath)):
    #             os.makedirs(tmpPath, exist_ok=True)

    # standart feff exe file:
    # feff_exe = 'wine /home/yugin/PycharmProjects/feff/exe/feff84_nclusx_175.exe' # path to exe file of the feff program

    # for big cases. In src files was changed
    # c      max number of atoms in problem for the pathfinder
    #        parameter (natx =10000) before it was: natx =1000
    # c      max number of unique potentials (potph) (nphx must be ODD to
    # c      avoid compilation warnings about alignment in COMMON blocks)
    #        parameter (nphx = 21) before it was: nphx = 7
    feff_exe = 'wine cmd /C "/home/yugin/PycharmProjects/feff/exe/feff_84_2016-08-02.bat"' # path to exe file of the feff program


    # outDirPath = os.path.join( '/home/yugin/VirtualboxShare/FEFF/out', folder_name )
    # if  not (os.path.isdir(outDirPath)):
    #             os.makedirs(outDirPath, exist_ok=True)



    result_dir = create_out_data_folder( outDirPath )
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
        print('copy the ',  f,  ' to the -> ',  os.path.join(tmpPath, 'feff.inp'))
        print('run the feff calculation')

        currentFileNameBase = os.path.basename(f)
        currentFileName     = os.path.splitext(currentFileNameBase)[0]

        # run the feff calculation:
        subprocess.call(feff_exe,  shell=True)

        # Check if chi.dat is created:
        if os.path.isfile( os.path.join(tmpPath, 'chi.dat') ):
            print('copy the chi.dat to the ->', )
            # create a new name to the chi.dat output file:
            # chiOutName = "chi_%05d.dat" %(i)
            chiOutName = 'chi_' + currentFileName + "_%05d.dat" %(i+1)
            copyfile(os.path.join(tmpPath, 'chi.dat'), os.path.join(outDirPath, chiOutName))
            print('feff calculation is finished')


            deleteAllFilesInFolder(tmpPath)
            # load txt output files:
            data = np.loadtxt(os.path.join(outDirPath, chiOutName), float)
            # select chi values only for k > 0 because FEFF output files contain the different length of k-vector:
            if len(data[:, 0]) < numOfRows:
                chi[1:,i] = data[:,1]
            elif len(data[:, 0]) == numOfRows:
                chi[:,i] = data[:,1]
            else:
                print('you have unexpected numbers of rows in your output files')
                print('input file name is: ', f)
                print('number of elements is: ', len(data[:, 0]), ' the first k-element is: ', data[0, 0])

            if ((i % 500) == 0 ) and ( i > 2 ):

                chi_std = np.std(chi[:, 0:i], axis=1)
                chi_mean = np.mean(chi[:, 0:i], axis=1)

                chi_median = np.median(chi[:, 0:i], axis=1)
                chi_max    = np.amax(chi[:, 0:i], axis=1)
                chi_min    = np.amin(chi[:, 0:i], axis=1)


                out_array = np.zeros((numOfRows, 3+3))
                out_array[:, 0] = k
                out_array[:, 1] = chi_mean
                out_array[:, 2] = chi_std

                out_array[:, 3] = chi_median
                out_array[:, 4] = chi_max
                out_array[:, 5] = chi_min

                headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
                out_file_name = "result_%s" %(folder_name) + "_%05d.txt" %(i)
                np.savetxt(os.path.join(result_dir, out_file_name), out_array, fmt='%1.6e', delimiter='\t',header=headerTxt)
                print('==> write iter number {0} to the {1} file'.format(i, out_file_name))
                if plotTheData:
                    plotData(x = k, y = chi_mean, error = chi_std, numOfIter = i, out_dir = result_dir, case = folder_name,
                            y_median= chi_median, y_max=chi_max, y_min=chi_min)

                if (i > shift) and (((i-shift) % 500)==0) and ( (i-shift) > 2 ):
                    # calc average with a shift (started not from the first snapshot)

                    chi_std = np.std(chi[:,  shift:i], axis=1)
                    chi_mean = np.mean(chi[:,  shift:i], axis=1)

                    chi_median = np.median(chi[:,  shift:i], axis=1)
                    chi_max    = np.amax(chi[:,  shift:i], axis=1)
                    chi_min    = np.amin(chi[:,  shift:i], axis=1)


                    out_array = np.zeros((numOfRows, 3+3))
                    out_array[:, 0] = k
                    out_array[:, 1] = chi_mean
                    out_array[:, 2] = chi_std

                    out_array[:, 3] = chi_median
                    out_array[:, 4] = chi_max
                    out_array[:, 5] = chi_min

                    headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
                    out_file_name = f'aver_from_{shift}_to_{i}_' + "result_%s" %(folder_name) + "_%05d.txt" %(i)
                    np.savetxt(os.path.join(result_dir, out_file_name), out_array, fmt='%1.6e', delimiter='\t',header=headerTxt)
                    print('==> write iter number {0} to the {1} file'.format(i, out_file_name))
                    if plotTheData:
                        plotData(x = k, y = chi_mean, error = chi_std, numOfIter = i, out_dir = result_dir, case = folder_name + f'_from_{shift}_to_{i}]',
                                y_median= chi_median, y_max=chi_max, y_min=chi_min)
        else:
            # if chi.dat is absent
            print('create the chi_%05d.error file  ->' % (i + 1))
            # create a new name to the chi.dat output file:
            # chiOutName = "chi_%05d.dat" %(i)
            chiOutName = 'chi_' + currentFileName + "_%05d.error" % (i + 1)
            touch(os.path.join(outDirPath, chiOutName))
            print('feff calculation was crushed')

        i += 1

    chi_std    = np.std(chi[:, :], axis=1)
    chi_mean   = np.mean(chi[:, :], axis=1)

    chi_median = np.median(chi[:, :], axis=1)
    chi_max    = np.amax(chi[:, :], axis=1)
    chi_min    = np.amin(chi[:, :], axis=1)


    out_array = np.zeros((numOfRows, 3+3))
    out_array[:, 0] = k
    out_array[:, 1] = chi_mean
    out_array[:, 2] = chi_std

    out_array[:, 3] = chi_median
    out_array[:, 4] = chi_max
    out_array[:, 5] = chi_min

    headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
    np.savetxt(os.path.join(result_dir, 'result.txt'), out_array, fmt='%1.6e', delimiter='\t',header=headerTxt)
    # copy result.txt file to the outDirPath folder:
    copyfile(os.path.join(result_dir, 'result.txt'), os.path.join(outDirPath, 'result_' + folder_name +'.txt'))
    if plotTheData:
        plotData(x = k, y = chi_mean, error = chi_std, numOfIter = i, out_dir = result_dir, case = folder_name,
                 y_median= chi_median, y_max=chi_max, y_min=chi_min)

    #     ==========================================================================
    #  Calculate average data started from the middle snapshot number.
    # We suppose that structure will be already relaxed to the moment of middle snapshot.

    chi_std = np.std(chi[:, shift:-1],axis=1)
    chi_mean = np.mean(chi[:, shift:-1],axis=1)

    chi_median = np.median(chi[:, shift:-1],axis=1)
    chi_max    = np.amax(chi[:, shift:-1],axis=1)
    chi_min    = np.amin(chi[:, shift:-1],axis=1)


    out_array = np.zeros((numOfRows, 3+3))
    out_array[:, 0] = k
    out_array[:, 1] = chi_mean
    out_array[:, 2] = chi_std

    out_array[:, 3] = chi_median
    out_array[:, 4] = chi_max
    out_array[:, 5] = chi_min

    headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
    np.savetxt(os.path.join(result_dir, f'aver_from_{shift}_to_{numOfColumns}_result.txt'), out_array, fmt='%1.6e', delimiter='\t',header=headerTxt)
    # copy result.txt file to the outDirPath folder:
    if plotTheData:
        plotData(x = k, y = chi_mean, error = chi_std, numOfIter = i, out_dir = result_dir, case = folder_name + f'_shift={shift}',
                 y_median= chi_median, y_max=chi_max, y_min=chi_min)



    print('program is finished')
    # print('-> create a video file:')
    # create_graphs_and_save_images_from_chi_dat(dataPath = outDirPath, each_elem_to_draw = 10)


if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )
    # runInParallel(main(dataPath = '/home/yugin/PycharmProjects/feff/load/53/'), main(dataPath = '/home/yugin/PycharmProjects/feff/load/66/'),
    #               main(dataPath = '/home/yugin/PycharmProjects/feff/load/67/'))


    # run in terminal the next command: python3 mainFEFF.py 60
    # where '60' - the name of case-folder, which you want to calculate

    debugMode = True
    userHomeDirPath = os.path.expanduser('~')
    feffLoadDirLocalPath = 'VirtualboxShare/GaMnO/debug/'
    feffLoadDirAbsPath = os.path.join(userHomeDirPath, feffLoadDirLocalPath)

    if debugMode:
        # for test and debug:
        # feffCalcFun(dataPath=feffLoadDirAbsPath + 'test/', plotTheData=False)
        feffCalcFun(dataPath=feffLoadDirAbsPath + 'feff_debug/', tmpPath=feffLoadDirAbsPath + 'feff_debug/tmp/',
        outDirPath = feffLoadDirAbsPath + 'feff_debug/feff_out/', plotTheData=True)

    else:
        dataPath = []
        if len(sys.argv) > 1:
            for i in range(len(sys.argv[1:])):
                tmpPath = feffLoadDirAbsPath + "%s/" % sys.argv[i+1]

                if  (os.path.isdir(tmpPath)):
                    print (tmpPath)
                    # time.sleep(2)
                    dataPath.append(tmpPath)
                    # main(dataPath = dataPath)
                else:
                    print('-> Selected case: ', sys.argv[i+1], ' dose not correct \n-> Program can not find data folder: ', tmpPath)
            print(len(dataPath))
            num_cores = multiprocessing.cpu_count()
            if (len(dataPath) <= num_cores) and (len(dataPath) > 0):
                print ('Program will be calculating on {} numbers of CPUs'.format(len(dataPath)) )
                time.sleep(1)
                print('Programm will calculate the next cases:\n{:}\n'.format(dataPath))
                Parallel(n_jobs=len(dataPath))(delayed(feffCalcFun)(i) for i in dataPath)
            else:
                print('PC doesn''t have these numbers of needed CPUs for parallel calculation' )


        else:
            print('- > No selected case was found. Please, try to use \'run in terminal the next command: python3 mainFEFF.py 60\' \n  '
                  'where \'60\' - the name of case-folder, which you want to calculate')
    # main(dataPath = '/home/yugin/PycharmProjects/feff/load/53/')
    # main(dataPath = '/home/yugin/PycharmProjects/feff/load/66/')
    # main(dataPath = '/home/yugin/PycharmProjects/feff/load/67/')
    # print (sys.argv[:])


    print('-> finished')
