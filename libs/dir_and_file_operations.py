import sys
import os
from io import StringIO

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

def createUniqFile(filename, mask='test'):
    outfile = filename
    dirname = os.path.dirname(os.path.abspath(filename))
    fn = os.path.basename(os.path.abspath(filename)).split('.')[0]
    ext = os.path.basename(os.path.abspath(filename)).split('.')[1]
    checkFile = 1
    i = 1
    while checkFile > 0:
        outfile = os.path.join(dirname, mask + '%06d' % i + '.' + ext)
        if not os.path.exists(outfile):
            checkFile =0

        i+=1
    return outfile

def createUniqFile_from_idx(filename, mask='test', idx=0):
    outfile = filename
    dirname = os.path.dirname(os.path.abspath(filename))
    fn = os.path.basename(os.path.abspath(filename)).split('.')[0]
    ext = os.path.basename(os.path.abspath(filename)).split('.')[1]
    # checkFile = 1
    # i = 1
    # while checkFile > 0:
    #     outfile = os.path.join(dirname, mask + '%06d' % i + '.' + ext)
    #     if not os.path.exists(outfile):
    #         checkFile =0
    #
    #     i+=1
    outfile = os.path.join(dirname, mask + '%06d' % idx + '.' + ext)
    return outfile

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
    return files

def listOfFilesFN(folder):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    return [os.path.join(folder,f) for f in os.listdir(folder)]

def listOfFilesFN_with_selected_ext(folder, ext = 'png'):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    return [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(ext)]

def deleteAllFilesInFolder(folder):
    # delete all files in the current directory:
    filelist = [ f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    for f in filelist:
        os.remove(f)
    return None

def deleteAllSelectExtenFilesInFolder(folder, ext = 'dat'):
    # delete all files in the current directory:
    filelist = [ f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))
                 and os.path.basename(f).split('.')[1] == ext]
    for f in filelist:
        os.remove(os.path.join(folder, f))
    return None

if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )