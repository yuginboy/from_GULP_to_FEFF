import sys
import os
from io import StringIO
import inspect
import numpy as np
import scipy as sp
import re
from shutil import copyfile
# for run executable file:
import subprocess

from libs.dir_and_file_operations import create_out_data_folder, listOfFiles, listOfFilesFN, deleteAllFilesInFolder, listOfFilesFN_with_selected_ext
# import plotting procedure:
from libs.plot_data import plotData

def create_file_with_results(dataPath = '/home/yugin/img/test/'):
    # directory with the chi.dat feff files:dataPath
    folder_path, folder_name = os.path.split( os.path.dirname(dataPath) )
    files = listOfFiles(dataPath)

    # load list of chi_%d.dat files:
    filesFullPathName = listOfFilesFN_with_selected_ext(dataPath, ext = 'dat')

    outDirPath = create_out_data_folder(dataPath, first_part_of_folder_name = 'chi_rslt')
    if  not (os.path.isdir(outDirPath)):
                os.makedirs(outDirPath, exist_ok=True)


    print('-> calculating data folder is: ', dataPath)
    print('-> output data folder is: ', outDirPath)

    i = 0
    numOfColumns = len(filesFullPathName)
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
        # select only every each_elem_to_draw file:
            # load data from the chi.dat file:
        data = np.loadtxt(f, float)
        # select chi values only for k > 0 because FEFF output files contain the different length of k-vector:
        if len(data[:, 0]) < numOfRows:
                chi[1:,i] = data[:,1]
        elif len(data[:, 0]) == numOfRows:
            chi[:,i] = data[:,1]
        else:
            print('you have unexpected numbers of rows in your output files')
            print('input file name is: ', f)
            print('number of elements is: ', len(data[:, 0]), ' the first k-element is: ', data[0, 0])

        i+=1

    chi_std = np.std(chi[:,:],axis=1)
    chi_mean = np.mean(chi[:,:],axis=1)

    chi_median = np.median(chi[:,:],axis=1)
    chi_max    = np.amax(chi[:,:],axis=1)
    chi_min    = np.amin(chi[:,:],axis=1)


    out_array = np.zeros((numOfRows, 3+3))
    out_array[:, 0] = k
    out_array[:, 1] = chi_mean
    out_array[:, 2] = chi_std

    out_array[:, 3] = chi_median
    out_array[:, 4] = chi_max
    out_array[:, 5] = chi_min

    headerTxt = 'k\t<chi>\tstd\tchi_median\tchi_max\tchi_min'
    np.savetxt(os.path.join(outDirPath, 'result_' + folder_name +'.txt'), out_array, fmt='%1.6e', delimiter='\t',header=headerTxt)
    plotData(x = k, y = chi_mean, error = chi_std, numOfIter = i, out_dir = outDirPath, case = folder_name,
             y_median= chi_median, y_max=chi_max, y_min=chi_min)



if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )

    for i in ['81A',  '81B',  '82A',  '82B',  '82C',  '83A',  '83B',  '83C',  '83D',]:
        path = '/home/yugin/VirtualboxShare/FEFF/out/%s/' % i
        print('-> calculating case is: ', path)
        create_file_with_results(dataPath = path)

    print('-> finished')