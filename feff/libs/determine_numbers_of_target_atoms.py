'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-10-24
'''
import re
import os
class TargetAtom():
    def __init__(self):
        self.path_to_cfg_file = ''
        self.atom_type = 'Mn'
        self.number_of_target_atoms = None

    def read_cfg_file(self):
        pattern = self.atom_type + '='
        self.number_of_target_atoms = 0
        if os.path.isfile(self.path_to_cfg_file):
            with open(self.path_to_cfg_file, 'r') as f:
                for line in f:
                    if pattern in line:
                        self.number_of_target_atoms = re.findall('(\d+)', line)[0]
                        self.number_of_target_atoms = int(self.number_of_target_atoms)

    def create_cfg_file(self):
        with open(self.path_to_cfg_file, 'w') as f:
            out_line = '{}={}\n'.format(self.atom_type, self.number_of_target_atoms)
            f.write(out_line)

    def get_number_of_target_atoms(self):
        if self.number_of_target_atoms is None:
            self.read_cfg_file()
            return int(self.number_of_target_atoms)
        else:
            return int(self.number_of_target_atoms)

    def print_info(self):
        txt =  '==========================================\n'
        txt += 'Number of target >> {} << atoms is [ {} ]\n'.format(self.atom_type, self.number_of_target_atoms)
        txt +=  '==========================================\n'
        print(txt)


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    obj = TargetAtom()
    obj.atom_type = 'Mn'
    obj.path_to_cfg_file = '/mnt/soliddrive/yugin/models/1Mn/best_models/MnI_caseB_Mn2[4]/atoms.cfg'
    obj.read_cfg_file()
    obj.print_info()
    # obj.atom_type = 'Mn'
    # obj.number_of_target_atoms = 2
    # obj.path_to_cfg_file = '/mnt/soliddrive/yugin/models/1Mn/best_models/MnI_caseB_Mn2[4]/atoms.cfg'
    # obj.create_cfg_file()
    # obj.print_info()

