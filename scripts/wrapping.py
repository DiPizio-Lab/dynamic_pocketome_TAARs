""""this is a script for xxxx"""
import pandas as pd
import time
import os, sys
import shutil
import subprocess
import config as conf
# Settings
pd.set_option('expand_frame_repr', False)
# wrapping takes roughly two hours for 3 reps in a system with 92 000 atoms

# I Helper Functions
working_dir = os.getcwd()

INPUT_SETTINGS_EQU = ''
INPUT_SETTINGS_PROD = ''
NO_REPS = '3'


def path_to_input_files():
    """iterates through the directories and produces path to the input files of the PDB folders"""
    list_pdb_folders = []
    for folder in os.listdir(conf.folder_data):
        if (os.path.isdir(folder) and not 'input_file' in folder and not folder.startswith('.')
                and not folder.startswith('_')):
            path_pdb = os.path.abspath(folder)
            list_pdb_folders.append(path_pdb)
        elif 'input_file' in folder:
            global INPUT_PARAMETERS
            INPUT_PARAMETERS = folder
        elif 'equilibration_settings' in folder:
            global INPUT_SETTINGS_EQU
            INPUT_SETTINGS_EQU = folder
        elif 'production_settings' in folder:
            global INPUT_SETTINGS_PROD
            INPUT_SETTINGS_PROD = folder

    # TODO hard-coded-names are SHITTY, but script before mine names them so ok, I guess
    global INPUT_TOPOLOGY
    INPUT_TOPOLOGY = 'NEUTRAL_fis.psf'

    global INPUT_COORDINATES
    INPUT_COORDINATES = 'NEUTRAL_fis.pdb'

    global INPUT_LIG
    INPUT_LIG = 'LIG.par'
    return list_pdb_folders


def create_input_settings_files():
    """iterates through PDB folders and changes the settings input file by adding the folder name to the
    topology, ligand and coordinates, saves the changes in a new file"""
    list_pdb_folders = path_to_input_files()
    for folder_pdb in list_pdb_folders:
        name_folder = os.path.basename(folder_pdb)
        name_folder = str(name_folder).strip('/')
        with open(INPUT_SETTINGS_EQU, 'r') as in_f, open('new_equilib_settings.txt', 'w') as out_f:
            for line in in_f:
                if 'input_topology=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_TOPOLOGY)
                    out_f.write('input_topology=' + new_line + '\n')
                elif 'input_coordinates=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_COORDINATES)
                    out_f.write('input_coordinates=' + new_line + '\n')
                elif 'lig_par=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_LIG)
                    out_f.write('lig_par=' + new_line + '\n')
                else:
                    out_f.write(line)
        try:
            shutil.move(os.path.join(working_dir, 'new_equilib_settings.txt'),
                        os.path.join(working_dir, name_folder))
            os.rmdir(os.path.join(working_dir, 'new_equilib_settings.txt'))
        except:
            shutil.Error()

        with open(INPUT_SETTINGS_PROD, 'r') as in_f, open('new_prod_settings.txt', 'w') as out_f:
            for line in in_f:
                if 'input_topology=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_TOPOLOGY)
                    out_f.write('input_topology=' + new_line + '\n')
                elif 'input_coordinates=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_COORDINATES)
                    out_f.write('input_coordinates=' + new_line + '\n')
                elif 'lig_par=' in line:
                    new_line = os.path.join(working_dir, name_folder, INPUT_LIG)
                    out_f.write('lig_par=' + new_line + '\n')
                else:
                    out_f.write(line)
        try:
            shutil.move(os.path.join(working_dir, 'new_prod_settings.txt'),
                        os.path.join(working_dir, name_folder))
            os.rmdir(os.path.join(working_dir, 'new_prod_settings.txt'))
        except:
            shutil.Error()

global INPUT_TOPOLOGY
INPUT_TOPOLOGY = 'NEUTRAL_fis.psf'


def wrapping_using_vmd():
    """taking the trajectories produced by MD_queue.run() and wraps them using Alessandro's script vmd_wrap.tcl"""
    script_wrap = os.path.join(working_dir, 'vmd_wrap.tcl')

    for folder in os.listdir(conf.folder_data):
        print(folder)
        production_folder = os.path.join(conf.folder_data, folder, 'production')
        topology = os.path.join(conf.folder_data, folder, INPUT_TOPOLOGY)
        os.chdir(production_folder)
        for rep in range(1, int(NO_REPS) + 1):
            # copy topology to rep_folder, because vmd script needs it there
            shutil.copyfile(topology, os.path.join(production_folder, str(rep), 'NEUTRAL_fis.pdb'))
            os.chdir(os.path.join(production_folder, str(rep)))
            if 'output_wrapped.dcd' not in os.listdir(os.path.join(production_folder, str(rep))):
                try:
                    subprocess.run(['vmd', '-dispdev', 'text', '-e', script_wrap], check=True,
                                   stdout=sys.stdout, stderr=sys.stderr, text=True)
                except subprocess.CalledProcessError as e:
                    print(f'Error:  {e}')


def main():
    t0 = time.time()
    wrapping_using_vmd()
    t1 = time.time()
    t_min = (t1 - t0) / 60
    t_hr = t_min / 60
    print('time:  ', t1 - t0, '  sec')
    print('time:  ', t_min, '  min')
    print('time:  ', t_hr, '  hours')


if __name__ == '__main__':
    main()

