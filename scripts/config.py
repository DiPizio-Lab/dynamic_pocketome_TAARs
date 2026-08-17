"""this is a script for ease of folder path management"""
import os
import platform


def folder_path(super_folder, folder_name):
    path = os.path.join(super_folder, folder_name + sep)
    return path


if platform.system() == 'Darwin':
    sep = '/'
elif platform.system() == 'Windows':
    sep = '\\'
elif platform.system() == 'Linux':
    sep = '/'
else:
    sep = '/'

folder_project = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0].replace('/', sep)
folder_input = folder_path(folder_project, 'input')
folder_output = folder_path(folder_project, 'output')

folder_scripts = folder_path(folder_project, 'scripts')
folder_apo_in = folder_path(folder_input, 'apo_structures')
folder_holo_in = folder_path(folder_input, 'holo_structures')
folder_results = folder_path(folder_output, 'results')
folder_meta_analysis = folder_path(folder_output, 'meta_analysis')
# atclus is not bundled in this repo -- install it separately (see README.md)
# and either drop it at scripts/atclus or point this at your own checkout.
folder_atclus = folder_path(folder_scripts, 'atclus')
