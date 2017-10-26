'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-10-26
'''
import os
from feff.libs.class_FTR_Spectrum_Compare_2_type_Models import FTR_gulp_to_feff_A_model as TwoModelsClass
from collections import OrderedDict as odict
from feff.libs.dir_and_file_operations import listOfFolders, create_data_folder
from feff.libs.determine_numbers_of_target_atoms import TargetAtom


class AutomaticCompare:
    '''
    Base class for loading all models in target directory and calculate all possible combinations of the selected models
    with each other (for selected list of parameters: user, sample_preparation_mode and constraints)

    '''
    def __init__(self):
        self.dir_path_of_models = ''
        self.obj = TwoModelsClass()
        self.obj.weight_R_factor_FTR = 1.0
        self.obj.weight_R_factor_chi = 0.0
        self.obj.scale_theory_factor_FTR = 0.81
        self.obj.scale_experiment_factor_FTR = 1.0

        self.obj.model_A.is_GUI = False
        self.obj.model_B.is_GUI = False

        #  change the user name, which parameters for xftf transformation you want to use:
        self.obj.user = 'ID'
        # change tha sample preparation method:
        self.obj.sample_preparation_mode = 'AG'

        # for debug and profiling:
        self.obj.saveDataToDisk = True

        self.obj.parallel_job_numbers = 2

        self.list_of_models = odict()
        self.model_feff_inp_folder_name = 'feff__0001'
        self.dir_path_for_calc_result = ''

        self.list_of_prep_mode_params_for_calc = ('250', '350', '450')

    def load_models_to_dict(self):
        tmp_obj = TargetAtom()

        dir_list = listOfFolders(self.dir_path_of_models)
        for idx, val in enumerate(dir_list):
            print(val)
            tmp_obj.path_to_cfg_file = os.path.join(self.dir_path_of_models, val, 'atoms.cfg')
            tmp_obj.read_cfg_file()
            tmp_obj.print_info()
            self.list_of_models[idx] = {
                'index': idx,
                'name' : val,
                'number of atoms' : tmp_obj.get_number_of_target_atoms(),
                'feff.inp folder' : os.path.join(self.dir_path_of_models, val, self.model_feff_inp_folder_name)
            }
        print(self.list_of_models)

    def create_uniq_result_folder(self):
        '''
        create result folder like a combination of two vars: self.obj.user and self.obj.sample_preparation_mode
        example: ID_(AG)
        :return:
        '''
        name_part = '{}_({})'.format(self.obj.user, self.obj.sample_preparation_mode)
        self.dir_path_for_calc_result = create_data_folder(self.dir_path_for_calc_result,
                                                           first_part_of_folder_name=name_part)

    def start_calculation(self):
        idx = 0
        for current_prep_mode in self.list_of_prep_mode_params_for_calc:
            self.obj.sample_preparation_mode = current_prep_mode
            self.create_uniq_result_folder()
            n = len(self.list_of_models)

            for i in range(n):
                print('i = {}'.format(i))
                if (i > 1) and (i < n-1):
                    for k in range(n-i):
                        idx += 1
                        # path1 = self.list_of_models[i]['feff.inp folder']
                        # path2 = self.list_of_models[i+k]['feff.inp folder']

                        path1 = self.list_of_models[i]['name']
                        path2 = self.list_of_models[i+k]['name']


                        print('{}_{}__{}__{}'.format(idx, path1, path2, current_prep_mode))


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')

    a = AutomaticCompare()
    a.dir_path_of_models = '/mnt/soliddrive/yugin/models/for_testing/'
    a.load_models_to_dict()
    a.dir_path_for_calc_result = '/mnt/soliddrive/yugin/rslt_best_models/'
    a.create_uniq_result_folder()
    print(a.dir_path_for_calc_result)
    a.start_calculation()