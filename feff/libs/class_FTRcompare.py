'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-27
'''
import os
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name
class Spectrum():
    # base spectrum class
    def __init__(self):
        self.x = []
        self.y = []
        self.pathToLoadDataFile = os.path.join(get_folder_name(runningScriptDir), 'data', '350.chik')


class FTR_gulp_to_feff():
    def __init__(self):
        self.ideal = Spectrum()
if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')