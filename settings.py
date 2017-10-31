import os
from feff.exe.sys_path import exe_file_path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

path_to_exe = os.path.join(exe_file_path, r'feff84_nclusx_175.exe')

# set path to RAM manually
ram_disk_path = r'/mnt/ramdisk/yugin/tmp'
