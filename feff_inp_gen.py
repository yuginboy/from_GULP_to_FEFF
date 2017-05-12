#!/usr/bin/env python

import re
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import sys
from libs.coord_extract import loadCoords
from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars

# open GUI filedialog to select snapshot file:
a = StoreAndLoadVars()
print('last used: {}'.format(a.getLastUsedFilePath()))
# openfile dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes = [("history files", '*.history')], initialdir=os.path.dirname(a.getLastUsedFilePath()))
if os.path.isfile(file_path):
    a.lastUsedFilePath = file_path
    a.saveLastUsedFilePath()

    file = open(file_path, 'r')

    folder_name = os.path.split( os.path.split( os.path.normpath(file_path) )[0])[1]

    pattern1 = "\s+\d{1}\.\d{3}\s+\d{1}\.\d{3}\s+\d{2}\.\d{2}"
    pattern2 = "timestep"
    pattern3 = '\s{10+}'



    i = 0
    timestep = []
    HO = np.zeros((3, 3))

    print('RUN with ',folder_name)

    for line in file:
        i = i+1
        if re.search(pattern2, line):
            timestep.append(i)
        if i == 2:
            # get the number of atoms from 2 line of the file in the Unit Cell:
            numOfAtoms = np.fromstring(line, sep='\t', dtype=np.int64)[2]

        # import HO vectors:
        if i == 4:
            # get the cell parameters:
            HO[0, :] = np.fromstring(line, sep='\t',)[:]
        if i == 5:
            # get the cell parameters:
            HO[1, :] = np.fromstring(line, sep='\t',)[:]
        if i == 6:
            # get the cell parameters:
            HO[2, :] = np.fromstring(line, sep='\t',)[:]
    numOfLinesInFile = i
    vectForRDF = np.linspace(0, HO[0, 0], 1000)
    # Номер строки:
    i = 0
    # Номер снапшота:
    j = 0
    # intrinsic index of line in snapshot:
    k = 0
    file = open(file_path, 'r')
    loadCoords(file,timestep, numOfAtoms, vectForRDF, HO, numOfLinesInFile)


    print('DONE with ', folder_name)