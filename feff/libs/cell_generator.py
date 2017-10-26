'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-09-19
'''
import numpy as np
import re
import periodictable as pd
from libs.dir_and_file_operations import createUniqFile, createUniqFile_from_idx
from libs.inputheader import writeHeaderToInpFile
from feff.libs.dir_and_file_operations import create_out_data_folder, get_folder_name, runningScriptDir
import os

class Cell():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.tagName = []
        self.atomIndex = []
        self.distance = []

class CellModel():
    def __init__(self, numOfAtoms=1):
        self.latCons = 1
        self.majorElemTag = 'Mn'
        self.structure = dict()


class Strategy():
    def __init__(self):
        pass

class CellGenerator():
    def __init__(self):
        self.numOfAtoms = 512
        self.path_to_GULP_input_file = os.path.join(get_folder_name(runningScriptDir), 'data', 'src',
                                                              'GaMnAs_ideal.gin')

    def load_cell_from_GULP_input_file(self):
        file = open(self.path_to_GULP_input_file, 'r')
        pattern = 'fractional'
        for line in file:
            i = i + 1
            if i > 29:
                tegTmp = re.findall('[a-zA-Z]+|\d+', line)



if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')