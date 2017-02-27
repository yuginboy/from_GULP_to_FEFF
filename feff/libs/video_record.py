import sys
import os
from io import StringIO
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
import cv2
from libs.dir_and_file_operations import create_out_data_folder, listOfFiles, listOfFilesFN, listOfFilesFN_with_selected_ext





def create_avi_from_png(folder = '/home/yugin/img', out_avi_name = 'output_n=65.avi', framerate = 10):
    # go to the directory:
    os.chdir(folder)
    files = listOfFilesFN_with_selected_ext(folder, ext = 'png')

    image = cv2.imread(files[0])
    height = np.size(image, 0)
    width = np.size(image, 1)
    frame_size = (width, height)
    # frame_size = (1920, 1080)


    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    video_out = cv2.VideoWriter(out_avi_name, fourcc, framerate, frame_size)

    for frame in files:
        img = cv2.imread(frame)
        # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)
        video_out.write(img)
    #     cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
    print ('-> avi file: ', out_avi_name, ' was created' )


if __name__ == "__main__":
    create_avi_from_png()
    out_avi_name = 'output_n=65.avi'
