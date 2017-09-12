'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-09-11
'''
from  feff_calc_from_inp import *
import pathos.multiprocessing as mp

# for parallel calculation:
from joblib import Parallel, delayed
import multiprocessing

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

def one_thread_calculation(list_of_inp_files=[], dirNameInp='', dirNameOut=''):
    # create one thread of calculation by creating the object which will get to the input
    # sliced list of files of FEFF input files:
    a = FEFF_calculation_class()

    # reinitialize vars:
    a.list_of_inp_files = list_of_inp_files
    a.dirNameInp = dirNameInp
    a.dirNameOut = dirNameOut

    # start calculations:
    a.do_calculation_serial(list_of_inp_files=list_of_inp_files)


class FEFF_parallel_calculation_class(FEFF_calculation_class):
    def __init__(self):
        super().__init__()
        self.parallel_job_numbers = 1

    def start_parallel_calculations(self):
        # do parallel routine of calculations:
        def func(list_of_inp_files):
            return one_thread_calculation(list_of_inp_files=list_of_inp_files,
                                          dirNameInp=self.dirNameInp,
                                          dirNameOut=self.dirNameOut)

        start = timer()

        number = 1
        listOfIndexes = slice_list(self.list_of_inp_files, size=self.parallel_job_numbers, whole_num=number)
        print('*----'*10)
        print('User calls {} number of threads'.format(self.parallel_job_numbers))
        print('program define {} number of threads'.format(len(listOfIndexes)))
        for idx, elem in enumerate(listOfIndexes):
            print('Thread # {0} will be calculate {1} elements'.format(idx, len(elem)))

        print('*----' * 10)

        # # for debug
        # for lst in listOfIndexes:
        #     result = func(lst)


        # p = mp.Pool(self.parallel_job_numbers)
        p = mp.Pool(len(listOfIndexes))
        p.map(func, listOfIndexes)



        print('======' * 10)
        print('======' * 10)
        print('======' * 10)
        runtime = timer() - start
        print('======' * 10)
        print("total runtime is {0:f} seconds".format(runtime))
        txt = '======' * 10
        txt += '\n'
        txt += '======' * 10
        txt += '\n'
        txt += "total runtime is {0:f} seconds".format(runtime)
        txt += '\n'
        txt_file_name = os.path.join(self.dirNameOut, 'result_info.txt')
        f = open(txt_file_name, 'x')
        f.write(txt)
        print('Resulting information have been saved in: ', txt_file_name)

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')

    MainObj = FEFF_parallel_calculation_class()
    # load inp dir name:
    MainObj.get_working_dir()
    # prepare variables:
    MainObj.prepare_vars()
    # use RAM-disk:
    MainObj.is_RAM_disk_exist = True
    # set parallel jobs number:
    MainObj.parallel_job_numbers = 10
    # do parallel calculation with a pathos multiprocessing tool
    MainObj.start_parallel_calculations()

    #  Do parallel calculation with a standard Parallel tools:
    # start = timer()
    #
    # number = 1
    # listOfIndexes = slice_list(MainObj.list_of_inp_files, size=MainObj.parallel_job_numbers, whole_num=number)
    # for idx, elem in enumerate(listOfIndexes):
    #     print('Thread # {0} will be calculate {1} elements'.format(idx, len(elem)))
    #
    # print('*----' * 10)
    # def func(list_of_inp_files):
    #     one_thread_calculation(list_of_inp_files=list_of_inp_files,
    #                            dirNameInp=MainObj.dirNameInp,
    #                            dirNameOut=MainObj.dirNameOut)
    #
    # Parallel(n_jobs=MainObj.parallel_job_numbers)(delayed(func)(i) for i in listOfIndexes)
    #
    # print('======' * 10)
    # print('======' * 10)
    # print('======' * 10)
    # runtime = timer() - start
    # print('======' * 10)
    # print("total runtime is {0:f} seconds".format(runtime))
    # txt = '======' * 10
    # txt += '\n'
    # txt += '======' * 10
    # txt += '\n'
    # txt += "total runtime is {0:f} seconds".format(runtime)
    # txt += '\n'
    # txt_file_name = os.path.join(MainObj.dirNameOut, 'result_info.txt')
    # f = open(txt_file_name, 'x')
    # f.write(txt)
    # print('Resulting information have been saved in: ', txt_file_name)
#



