"""
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-28
"""
from feff.libs.class_Spectrum import Spectrum, GraphElement, TableData, BaseData
from feff.libs.class_for_parallel_comparison import Model_for_spectra, FTR_gulp_to_feff_A_model_base
import os
import datetime
from timeit import default_timer as timer
import copy
from shutil import copyfile
from feff.libs.dir_and_file_operations import runningScriptDir, get_folder_name, get_upper_folder_name, \
    listOfFilesFN_with_selected_ext, create_out_data_folder, create_data_folder
from feff.libs.class_StoreAndLoadVars import StoreAndLoadVars
from feff.libs.class_SpectraSet import SpectraSet
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec
import numpy as np

import progressbar
# from joblib import Parallel, delayed
import pathos.multiprocessing as mp

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def slice_list(input, size, whole_num=2):
    # divide list into N equal parts which rounds to whole_num
    # if input = [123456] but size=2, whole_num=2 we expect [1234], [56]
    input_size = len(input)
    slice_size = int(input_size / size)
    while slice_size % whole_num:
        slice_size = slice_size + 1
    result = list(chunks (input, slice_size))
    return result

def one_thread_calculation(model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles,
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles,
        outDirectoryForTowModelsFitResults, weight_R_factor_FTR=1.0, weight_R_factor_chi=0.0,
        scale_theory_factor_FTR=0.81, scale_experiment_factor_FTR=1.0,
        model_A_numberOfSerialEquivalentAtoms=2, model_B_numberOfSerialEquivalentAtoms=2,
        user='ID', sample_preparation_mode='AG', saveDataToDisk=True):
    # create one thread of calculation by creating the object which will get to the input
    # sliced list of files of model_A
    a = FTR_gulp_to_feff_A_model_base()
    a.weight_R_factor_FTR = weight_R_factor_FTR
    a.weight_R_factor_chi = weight_R_factor_chi
    a.scale_theory_factor_FTR = scale_theory_factor_FTR
    a.scale_experiment_factor_FTR = scale_experiment_factor_FTR

    a.model_A.numberOfSerialEquivalentAtoms = model_A_numberOfSerialEquivalentAtoms
    a.model_B.numberOfSerialEquivalentAtoms = model_B_numberOfSerialEquivalentAtoms

    #  change the user name, which parameters for xftf transformation you want to use:
    a.user = user
    # change tha sample preparation method:
    a.sample_preparation_mode = sample_preparation_mode

    # for debug and profiling:
    a.saveDataToDisk = saveDataToDisk

    #  if you want to find the minimum from the all snapshots do this:
    a.calcAllSnapshotFilesForTwoModels_temperature_parallel(
        model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles,
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles,
        outDirectoryForTowModelsFitResults)
    return a.minimum


class FTR_gulp_to_feff_A_model(FTR_gulp_to_feff_A_model_base):
    '''
    Class to search optimal snapshot coordinates by compare chi(k) nad FTR(r) spectra between the snapshots and
     average spectrum from all snapshots

    '''

    def findBestSnapshotsCombinationFromTwoModels_parallel(self):
        '''
        searching procedure of Two Models (A - first, B - second) linear model:
        k/n(A1 + A2 + .. + An) + (1-k)/m(B1 + B2 + .. + Bm)
        :return: k - coefficient which corresponds to concentration A phase in A-B compound
        Parallel realization
        '''
        model_A_projectWorkingFEFFoutDirectory, model_A_listOfSnapshotFiles, \
        model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles, \
        outDirectoryForTowModelsFitResults = self.loadListOfFilesForTwoModels_temperature()


        def func(list_of_files):
            return one_thread_calculation(model_A_projectWorkingFEFFoutDirectory, list_of_files,
                                   model_B_projectWorkingFEFFoutDirectory, model_B_listOfSnapshotFiles,
                                   outDirectoryForTowModelsFitResults,
                                   weight_R_factor_FTR=self.weight_R_factor_FTR,
                                   weight_R_factor_chi=self.weight_R_factor_chi,
                                   scale_theory_factor_FTR=self.scale_theory_factor_FTR,
                                   scale_experiment_factor_FTR=self.scale_experiment_factor_FTR,
                                   model_A_numberOfSerialEquivalentAtoms=self.model_A.numberOfSerialEquivalentAtoms,
                                   model_B_numberOfSerialEquivalentAtoms=self.model_B.numberOfSerialEquivalentAtoms,
                                   user=self.user,
                                   sample_preparation_mode=self.sample_preparation_mode,
                                   saveDataToDisk=self.saveDataToDisk)

        start = timer()
        p = mp.Pool(self.parallel_job_numbers)
        number = self.model_A.numberOfSerialEquivalentAtoms
        listOfIndexes = slice_list(model_A_listOfSnapshotFiles, size=self.parallel_job_numbers, whole_num=number)
        # print(listOfIndexes)
        result = p.map(func, listOfIndexes)
        # bar.update(i)
        # bar.finish()
        vec_Rtot = list((i.Rtot for i in result))
        minIdx, = np.where(vec_Rtot == np.min(vec_Rtot))
        Rtot = result[minIdx[0]].Rtot
        snapshotName = result[minIdx[0]].snapshotName

        # save ASCII column data:
        if self.saveDataToDisk:
            self.outMinValsDir = outDirectoryForTowModelsFitResults
            obj = result[minIdx[0]]
            obj.setOfSnapshotSpectra.saveSpectra_LinearComposition_FTR_from_linear_Chi_k(
                output_dir=self.outMinValsDir)

            # store model-A snapshots for this minimum case:
            obj.model_A.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)
            # store model-B snapshots for this minimum case:
            obj.model_B.saveSpectra_SimpleComposition(output_dir=self.outMinValsDir)
            dst = os.path.join(self.outMinValsDir, os.path.basename(obj.pathToImage))
            copyfile(obj.pathToImage, dst)

        print('======'*10)
        print('======'*10)
        print('======'*10)
        print('global minimum Rtot = {0}'.format(Rtot))
        print('{0}'.format(snapshotName))
        runtime = timer() - start
        print('======'*10)
        print("total runtime is {0:f} seconds".format(runtime))
        print('======'*10)
        print('======'*10)
        print('======'*10)


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')

    # start global search of Two-model combination in Parallel mode:
    a = FTR_gulp_to_feff_A_model()
    a.weight_R_factor_FTR = 1.0
    a.weight_R_factor_chi = 0.0
    a.scale_theory_factor_FTR = 0.81
    a.scale_experiment_factor_FTR = 1.0

    a.model_A.numberOfSerialEquivalentAtoms = 1
    a.model_B.numberOfSerialEquivalentAtoms = 2

    #  change the user name, which parameters for xftf transformation you want to use:
    a.user = 'ID'
    # change tha sample preparation method:
    a.sample_preparation_mode = 'AG'
    # if you want compare with the theoretical average, do this:
    # a.calcAllSnapshotFiles()

    # for debug and profiling:
    a.saveDataToDisk = True

    a.parallel_job_numbers = 5

    #  if you want to find the minimum from the all snapshots do this:
    a.findBestSnapshotsCombinationFromTwoModels_parallel()




