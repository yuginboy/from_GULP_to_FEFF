"""
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-03-04
* global project name: from_GULP_to_FEFF
"""
import pickle
import os
from feff.libs.dir_and_file_operations import runningScriptDir

class StoreAndLoadVars():
    def __init__(self):
        self.fileNameOfStoredVars = 'vars.pckl'
        self.dirPath = runningScriptDir

        self.lastUsedDirPath = runningScriptDir

    def loadDataFromPickleFile(self):
        # Getting back the objects:
        if os.path.isfile(os.path.join(self.dirPath, self.fileNameOfStoredVars)):
            pcklFile = os.path.join(self.dirPath, self.fileNameOfStoredVars)
            with open(pcklFile, 'rb') as f:
                obj = pickle.load(f)
            # print('')
            try:
                self.lastUsedDirPath = obj[0].lastUsedDirPath
            except Exception:
                print(self.fileNameOfStoredVars + ' does not have attribute "lastUsedDirPath"')

    def storeDataToPickleFile(self):
        pcklFile = os.path.join(self.dirPath, self.fileNameOfStoredVars)
        with open(pcklFile, 'wb') as f:
            pickle.dump([self], f)

    def getLastUsedDirPath(self):
        self.loadDataFromPickleFile()
        return self.lastUsedDirPath

    def saveLastUsedDirPath(self):
        self.storeDataToPickleFile()


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    import tkinter as tk
    from tkinter import filedialog

    a = StoreAndLoadVars()
    print('last used: {}'.format(a.getLastUsedDirPath()))
    # openfile dialoge
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(initialdir=a.getLastUsedDirPath())
    if os.path.isdir(dir_path):
        a.lastUsedDirPath = dir_path
        a.saveLastUsedDirPath()

    
