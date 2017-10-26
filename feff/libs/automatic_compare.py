'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2017-10-26
'''
import os
from feff.libs.class_FTR_Spectrum_Compare_2_type_Models import FTR_gulp_to_feff_A_model as TwoModelsClass


class AutomaticCompare:
    '''
    Base class for loading all models in target directory and calculate all possible combinations of the selected models
    with each other (for selected list of parameters: user, sample_preparation_mode and constraints)

    '''
    def __init__(self):
        self.models_dir_path = ''
        self.object = TwoModelsClass()
        self.object.weight_R_factor_FTR = 1.0
        self.object.weight_R_factor_chi = 0.0
        self.object.scale_theory_factor_FTR = 0.81
        self.object.scale_experiment_factor_FTR = 1.0

        self.object.model_A.is_GUI = True
        self.object.model_B.is_GUI = True

        #  change the user name, which parameters for xftf transformation you want to use:
        self.object.user = 'ID'
        # change tha sample preparation method:
        self.object.sample_preparation_mode = 'AG'

        # for debug and profiling:
        self.object.saveDataToDisk = True

        self.object.parallel_job_numbers = 2

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')