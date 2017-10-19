'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-10-19
'''
import sys
import os
from io import StringIO


runningScriptDir = os.path.dirname(os.path.abspath(__file__))
# get root project folder name:
runningScriptDir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]


def create_out_data_folder(main_folder_path, first_part_of_folder_name = ''):
    '''
    create out data directory like 0005 or 0004
    :param main_folder_path: path to the main project folder
    :return: full path to the new directory
    return folder path like: main_folder_path + first_part_of_folder_name + '%04d' % i
    '''
    checkFile = 1
    i = 1

    # check, if first_part_of_folder_name is not absent then add '_' symbol to the end
    if len(first_part_of_folder_name) > 0:
        first_part_of_folder_name += '_'

    while checkFile > 0:

        out_data_folder_path = os.path.join( main_folder_path, first_part_of_folder_name + '%04d' % i )
        if  not (os.path.isdir(out_data_folder_path)):
            checkFile = 0
            os.makedirs(out_data_folder_path, exist_ok=True)
        i+=1
    return  out_data_folder_path

def create_data_folder(main_folder_path, first_part_of_folder_name = ''):
    '''
    create directory
    :param main_folder_path: path to the main project folder
    :return: full path to the new directory
    return folder path like: main_folder_path + first_part_of_folder_name + '%04d' % i
    '''

    out_data_folder_path = os.path.join(main_folder_path, first_part_of_folder_name)
    if  not (os.path.isdir(out_data_folder_path)):
        os.makedirs(out_data_folder_path, exist_ok=True)
    return  out_data_folder_path


def listOfFiles(dirToScreens):
    '''
    return only the names of the files in directory
    :param folder:
    :return:
    '''
    '''
    :param dirToScreens: from which directory you want to take a list of the files
    :return:
    '''
    files = [f for f in os.listdir(dirToScreens) if os.path.isfile(os.path.join(dirToScreens,f))]
    return sorted(files)

def listOfFilesNameWithoutExt(dirToScreens):
    '''
    return only the name of the files in directory without extansion
    :param folder:
    :return:
    '''
    '''
    :param dirToScreens: from which directory you want to take a list of the files
    :return:
    '''
    base = [f for f in os.listdir(dirToScreens) if os.path.isfile(os.path.join(dirToScreens,f))]
    names = [os.path.splitext(f)[0] for f in base]
    return sorted(names)

def listOfFilesFN(folder):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    return sorted([os.path.join(folder,f) for f in os.listdir(folder)])

def listOfFilesFN_with_selected_ext(folder, ext = 'png'):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    out = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(ext)]
    out = sorted(out)
    return out

def deleteAllFilesInFolder(folder):
    # delete all files in the current directory:
    filelist = [ f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    for f in filelist:
        os.remove(f)
    return None

def get_upper_folder_name(file_path):
    # return only a directory name when file is placed
    return os.path.split(os.path.split(os.path.normpath(file_path))[0])[1]

def get_folder_name(file_path):
    # return only a directory name when file is placed
    return os.path.split(os.path.normpath(file_path))[0]

def touch(path):
    '''create empty file'''
    with open(path, 'a'):
        os.utime(path, None)

if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )
    runningScriptDir = os.path.dirname(os.path.abspath(__file__))
    print(get_folder_name(get_folder_name(runningScriptDir)))