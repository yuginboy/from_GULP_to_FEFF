'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-27
'''
import numpy as np
import os
from feff.libs.dir_and_file_operations import get_folder_name, runningScriptDir
from feff.libs.feff_processing import xftf

def load_experimental_chi_data(file_path):
    # load experimental chi-data file. In  non-existent points we use linear interp procedure
    data = np.loadtxt(file_path, float)
    k = np.r_[0:20.05:0.05]
    numOfRows = len(k)
    chi = np.zeros((numOfRows, 2))
    chi[:, 0] = k
    k_old = data[:, 0]
    chi_old = data[:, 1]
    chi_interp = np.interp(k, k_old, chi_old)
    chi[:, 1] = chi_interp
    return chi

def load_chi_data(file_path):
    # load theoreticaly calculeted chi-data file:

    k = np.r_[0:20.05:0.05]
    numOfRows = len(k)
    chi = np.zeros((numOfRows, 2))
    data = np.loadtxt(file_path, float)
    chi[:, 0] = k
    # select chi values only for k > 0 because FEFF output files contain the different length of k-vector:
    if len(data[:, 0]) == numOfRows - 1:
        chi[1:, 1] = data[:, 1]
        print('theory data without 0-k point has been loaded')
    elif len(data[:, 0]) == numOfRows:
        chi[:, 1] = data[:, 1]
        print('theory data has been loaded')
    elif len(data[:, 0]) < numOfRows - 1:
            chi = load_experimental_chi_data(file_path)
            print('experimental data has been loaded')
    else:
        print('you have unexpected numbers of rows in your output files')
        print('input file name is: ', file_path)
        print('number of elements is: ', len(data[:, 0]), ' the first k-element is: ', data[0, 0])
    return chi

def load_and_apply_xftf(file_path):
    data = load_chi_data(file_path)
    fr = xftf(data[:, 0], data[:, 1])
    return fr

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    file_path1 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000002_00001.dat'
    file_path2 = r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/chi_1mono1SR2VasVga2_6_000131_00130.dat'
    exp_data_path2 = os.path.join(get_folder_name(runningScriptDir), 'data', '350.chik')
    chi1 = load_chi_data(file_path1)
    chi2 = load_chi_data(file_path2)
    chi3 = load_chi_data(exp_data_path2)
    import matplotlib.pyplot as plt
    chi = chi1
    plt.plot(chi[:, 0], chi[:, 1],)
    chi = chi2
    plt.plot(chi[:, 0], chi[:, 1],)
    chi = chi3
    plt.plot(chi[:, 0], chi[:, 1],)
    plt.show()
    print('finish')