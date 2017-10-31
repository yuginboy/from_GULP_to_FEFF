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
import copy


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
        self.obj.is_GUI = False

        #  change the user name, which parameters for xftf transformation you want to use:
        self.obj.user = 'ID'
        # change tha sample preparation method:
        self.obj.sample_preparation_mode = 'AG'

        # for debug and profiling:
        self.obj.saveDataToDisk = True

        self.obj.parallel_job_numbers = 5

        self.list_of_models = odict()
        self.model_feff_inp_folder_name = 'feff__0001'
        self.dir_path_for_calc_result_base = ''
        self.dir_path_for_calc_result_current = ''

        self.list_of_prep_mode_params_for_calc = ['350', '450']
        # self.list_of_prep_mode_params_for_calc = ['250', '350', '450']
        self.dict_of_model_constraints_name_pattern = odict([
            ('AG', ['MnI_caseB', 'Monomer']),
            ('250', ['MnI_caseB', 'Monomer']),
            ('350', ['MnYM', 'Monomer']),
            ('450', ['MnYM', 'Monomer']),
                                                ])
        # self.list_of_prep_mode_params_for_calc = ['AG']

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
        self.dir_path_for_calc_result_current = create_data_folder(self.dir_path_for_calc_result_base,
                                                                   first_part_of_folder_name=name_part)

    def check_name_pattern_constraints(self, prep_mode, name_1, name_2):
        n = len(self.dict_of_model_constraints_name_pattern[prep_mode])
        tmp_list = self.dict_of_model_constraints_name_pattern[prep_mode]
        for ww in tmp_list:
            if ww in name_1:
                for val in tmp_list:
                    if (val in name_1 and val not in name_2) or (val in name_2 and val not in name_1):
                        if val in name_2:
                            return True
            if ww in name_2:
                for val in tmp_list:
                    if (val in name_1 and val not in name_2) or (val in name_2 and val not in name_1):
                        if val in name_1:
                            return True
        return False


    def start_calculation(self):
        idx = 0
        for current_prep_mode in self.list_of_prep_mode_params_for_calc:
            self.obj.sample_preparation_mode = current_prep_mode
            # create unique output folder for results:
            self.create_uniq_result_folder()
            self.obj.projectWorkingFEFFoutDirectory = self.dir_path_for_calc_result_current

            n = len(self.list_of_models)
            print('========================================\n')
            print(f'There are n = {n} models in directory')

            for i in range(n-1):
                name1 = self.list_of_models[i]['name']
                path1 = self.list_of_models[i]['feff.inp folder']
                self.obj.model_A.projectWorkingFEFFoutDirectory = path1

                for k in range(i+1, n):

                    print('i = {}, k = {}, i+k = {}, idx = {}'.format(i, k, i+k+1, idx))
                    name2 = self.list_of_models[k]['name']
                    path2 = self.list_of_models[k]['feff.inp folder']
                    self.obj.model_B.projectWorkingFEFFoutDirectory = path2
                    if self.check_name_pattern_constraints(current_prep_mode, name1, name2):
                        idx += 1
                        print('idx = {}, A = {}, B = {}, prep_mode = {}, user = {}'.format(
                        idx, name1, name2, current_prep_mode, self.obj.user))

                        # calculatting part:
                        current_obj = copy.deepcopy(self.obj)
                        current_obj.findBestSnapshotsCombinationFrom_2_type_Models_parallel()
                        del current_obj


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')

    a = AutomaticCompare()
    # a.dir_path_of_models = '/mnt/soliddrive/yugin/models/for_testing/'
    a.dir_path_of_models = '/mnt/soliddrive/yugin/models/best_models/'
    a.load_models_to_dict()
    a.dir_path_for_calc_result_base = '/mnt/soliddrive/yugin/rslt_best_models/'
    print(a.dir_path_for_calc_result_current)
    a.start_calculation()