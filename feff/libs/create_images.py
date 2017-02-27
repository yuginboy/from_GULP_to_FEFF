"Main file for run feff calculation package"
# import pandas as pd
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
import re
from shutil import copyfile
# for run executable file:
import subprocess

# for parallel calculation:
import time
from joblib import Parallel, delayed
import multiprocessing

# use this if you want to include modules from a subfolder
# cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"")))
# print('cmd_subfolder: ', cmd_subfolder)
# if cmd_subfolder not in sys.path:
#     print('-1-> sys.path: ', sys.path)
#     sys.path.insert(0, cmd_subfolder)
#     print('-2-> sys.path: ', sys.path)

# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if not path in sys.path:
#     sys.path.insert(1, path)
# del path
# import plotting procedure:
# from libs.plot_data import plotData

from libs.dir_and_file_operations import create_out_data_folder, listOfFiles, listOfFilesFN, deleteAllFilesInFolder, listOfFilesFN_with_selected_ext
from libs.video_record import create_avi_from_png

def create_graphs_and_save_images_from_chi_dat (dataPath = '/home/yugin/img/test/', each_elem_to_draw = 10, ymin = -0.3, ymax= 0.5):
    '''

    :param dataPath: directory with the chi.dat feff files
    :param each_elem_to_draw: select only every each_elem_to_draw file
    :return: images
    '''
    # directory with the chi.dat feff files:dataPath
    folder_path, folder_name = os.path.split( os.path.dirname(dataPath) )
    files = listOfFiles(dataPath)
    filesFullPathName = listOfFilesFN_with_selected_ext(dataPath, ext = 'dat')

    outDirPath = create_out_data_folder(dataPath, first_part_of_folder_name = 'chi')
    if  not (os.path.isdir(outDirPath)):
                os.makedirs(outDirPath, exist_ok=True)



    video_dir = create_out_data_folder( outDirPath, first_part_of_folder_name = 'video')

    print('-> calculating data folder is: ', dataPath)

    i = 0
    numOfColumns = len(filesFullPathName)
    k = np.r_[0:20.05:0.05]
    numOfRows = len(k)
    chi = np.zeros((numOfRows))

    # load result <chi> and std values. In the dataPath must be a result.txt file with a '.txt' extension:
    result_file_FullPathName = listOfFilesFN_with_selected_ext(dataPath, ext = 'txt')
    chi_std = np.zeros((numOfRows))
    chi_mean = np.zeros((numOfRows))

    chi_median = np.zeros((numOfRows))
    chi_max = np.zeros((numOfRows))
    chi_min = np.zeros((numOfRows))

    data = np.loadtxt(result_file_FullPathName[0], float)
    chi_mean = data[:,1]
    chi_std = data[:,2]

    chi_median = data[:,3]
    chi_max    = data[:,4]
    chi_min    = data[:,5]

    #  ===============================================================
    # create plot window:
    pylab.ion()  # Force interactive
    plt.close('all')
    ### for 'Qt4Agg' backend maximize figure
    plt.switch_backend('QT4Agg')

    fig = plt.figure()
    # gs1 = gridspec.GridSpec(1, 2)
    # fig.show()
    # fig.set_tight_layout(True)
    figManager = plt.get_current_fig_manager()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0 / DPI, 1080.0 / DPI, dpi=DPI)

    gs = gridspec.GridSpec(1,1)

    ax = fig.add_subplot(gs[0,0])


    txt =  '$\chi(k)$ when the Number of the treated file is: {0}'.format(i)
    fig.suptitle(txt, fontsize=22, fontweight='normal')

    # Change the axes border width
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(2)
    # plt.subplots_adjust(top=0.85)
    # gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)

    # put window to the second monitor
    # figManager.window.setGeometry(1923, 23, 640, 529)
    figManager.window.setGeometry(1920, 20, 1920, 1180)
    figManager.window.setWindowTitle('chi values')
    figManager.window.showMinimized()

    # plt.show()

    x = k
    y = chi_mean
    error = chi_std

    y_median = chi_median
    y_max = chi_max
    y_min = chi_min
    ax.plot( x, y, label = '<$\chi$>' )
    ax.plot( x, y_median, label = '$chi$ median', color = 'darkcyan' )
    ax.plot( x, y_max, label = '$chi$ max', color = 'skyblue' )
    ax.plot( x, y_min, label = '$chi$ min', color = 'lightblue' )
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95], w_pad=1.1)
    ax.plot(x, y, 'k', color='#1B2ACC')
    ax.fill_between(x, y-error, y+error,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=4, linestyle='dashdot', antialiased=True, label = '$\chi(k)$')
    ax.grid(True)


    graph_line, = ax.plot( x, y, label = 'current $\chi$' , color = 'red')

    ax.set_ylabel('$\chi(k)$', fontsize=20, fontweight='bold')
    ax.set_xlabel('$k$', fontsize=20, fontweight='bold')
    ax.set_ylim(ymin = ymin, ymax= ymax)

    # copy file input to the tmp directory and start calculations:
    for f in filesFullPathName:
        # select only every each_elem_to_draw file:
        if ((i % each_elem_to_draw) == 0 ) :
            # load data from the chi.dat file:
            data = np.loadtxt(f, float)
            # select chi values only for k > 0 because FEFF output files contain the different length of k-vector:
            if len(data[:, 0]) < numOfRows:
                chi[1:] = data[:,1]
            elif len(data[:, 0]) == numOfRows:
                chi[:] = data[:,1]
            else:
                print('you have unexpected numbers of rows in your output files')
                print('input file name is: ', f)
                print('number of elements is: ', len(data[:, 0]), ' the first k-element is: ', data[0, 0])


            # plot data:
            # update the data
            graph_line.set_ydata(chi)
            txt =  "GaMnAs case %s, " % (folder_name) + "$\chi(k)$ when the Number of the treated file is: %05d" %(i)
            fig.suptitle(txt, fontsize=22, fontweight='normal')
            plt.draw()
            # save to the PNG file:
            out_file_name = "chi_%05d.png" %(i)
            fig.savefig( os.path.join(outDirPath, out_file_name) )
        i+=1




    print('-> images was created')

    create_avi_from_png(folder = outDirPath, out_avi_name = 'chi_' + folder_name + '.avi', framerate = 25)
    target_video_file = os.path.join(outDirPath,'chi_' + folder_name + '.avi')
    dst_video_file = os.path.join(video_dir,'chi_' + folder_name + '.avi')

    copyfile(target_video_file, dst_video_file) # copy
    os.remove(target_video_file) # delete

    print('delete all created temporary *.png files in folder: ', outDirPath)
    deleteAllFilesInFolder(outDirPath)

if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )
    dataPath = []
    argv = ('81A',  '81B',  '82A',  '82B',  '82C',  '83A',  '83B',  '83C',  '83D',)
    if len(argv) > 0:
        for i in range(len(argv)):
            tmpPath = "/home/yugin/VirtualboxShare/FEFF/out/%s/" % argv[i]

            if  (os.path.isdir(tmpPath)):
                print (tmpPath)
                # time.sleep(2)
                dataPath.append(tmpPath)
                # main(dataPath = dataPath)
            else:
                print('-> Selected case: ', argv[i], ' dose not correct \n-> Program can not find data folder: ', tmpPath)
        print(len(dataPath))
        num_cores = multiprocessing.cpu_count()
        if (len(dataPath) <= num_cores) and (len(dataPath) > 0):
            print ('Program will be calculating on {} numbers of CPUs'.format(len(dataPath)) )
            time.sleep(1)
            print('Programm will calculate the next cases:\n{:}\n'.format(dataPath))
            Parallel(n_jobs=len(dataPath))(delayed(create_graphs_and_save_images_from_chi_dat)(i) for i in dataPath)
        else:
            print('PC doesn''t have this numbers of needed CPUs for parallel calculation' )


    else:
        print('- > No selected case was found.')

    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/51/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/58/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/60/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/63/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/64/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/65/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/53/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/66/', each_elem_to_draw = 10)
    # create_graphs_and_save_images_from_chi_dat(dataPath = '/home/yugin/VirtualboxShare/FEFF/out/67/', each_elem_to_draw = 10)

    print ('-> finished' )