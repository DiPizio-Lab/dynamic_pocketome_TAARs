""""this is a script to perform pocket identification and characterisation using mdpocket based onfpocket
https://github.com/Discngine/fpocket """
import re
import shutil
import datetime
import pandas as pd
import numpy as np
import time
from scripts import config as conf
from scripts import extract_iso as iso
from scripts import logging as logger
from sklearn import cluster
from sklearn.metrics import silhouette_score
import sys, os
import subprocess
from biopandas.pdb import PandasPdb
import multiprocessing
import json
import glob

# Settings
pd.set_option('expand_frame_repr', False)
working_dir = os.getcwd()

# TODO: xtc dcd check

class PocketAnalysis:
    """ stores method for pocket search on GPCRs class A;
    The class requires at least a topology, trajectory"""

    def __init__(self, topology, trajectory,  pocket_dir, curr_proj=None, curr_rep=None, verbose=False):
        self.topology = topology
        self.trajectory = trajectory
        self.curr_proj = curr_proj
        self.curr_rep = curr_rep
        self.verbose = verbose
        self.pocket_dir = pocket_dir

    @staticmethod
    def _parse_pdb_to_df(file):
        """ Method to extract the coordinates from a PDB file and store them in a dataframe
        Input:   file: path-like str
        Output:  coords_array: np.array of this file
                 df: pd.DataFrame of the file"""
        atom_coords = []
        full_line = []
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                atom_coords.append(re.split(r'\s+', line)[-5:-2])
                full_line.append(re.split(r'\s+', line))
        df = pd.DataFrame(full_line,
                          columns=['atom', 'atom_no', 'chain', 'molecule_name', 'pocket', 'x', 'y', 'z', 'n1', 'n2'])
        coords_array = np.array(atom_coords, dtype='f')

        return coords_array, df

    def _mdpocket_run_one(self):
        """Method starting the first run of mdpocket using subprocess.run();
        NOTE that the trajectory and topology have to be located in the subfolder results_dbscan_new/curr_proj/curr_rep/pockets
        and have to be named 'aligned_traj.dcd' and 'aligned_top.pdb'. The format for the trajectory has to be DCD.
        If you want to change any of these settings, you have to change the code. BUT the good news is that if you run
        the whole pipeline (as intended) the files are automatically named in a compatibnle way and saved at
        the right place...

        For the documentation of MDpocket please refer to:
        "https://github.com/Discngine/fpocket/blob/master/doc/GETTINGSTARTED.md"
        by   Vincent Le Guilloux, Peter Schmidtke and Pierre Tuffery,
            “Fpocket: An open source platform for ligand pocket detection”, BMC Bioinformatics 2009, 10:168
        as well as
            Peter Schmldtke, Axel Bidon-Chanal, Javier Luque, Xavier Barril,
            “MDpocket: open-source cavity detection and characterization on molecular dynamics trajectories.”,
            Bioinformatics. 2011 Dec 1;27(23):3276-85"""
        curr_pocket_dir = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)
        os.chdir(curr_pocket_dir)
        trajectory = self.trajectory.split('/')[-1]
        topology = self.topology.split('/')[-1]
        # copy trajectory and topology into pocket_dir
        if not os.path.exists( os.path.join(curr_pocket_dir, trajectory)):
            shutil.copy(self.trajectory, os.path.join(curr_pocket_dir, trajectory))
        if not os.path.exists(os.path.join(curr_pocket_dir, topology)):
            shutil.copy(self.topology, os.path.join(curr_pocket_dir, topology))

        if self.trajectory.endswith('.dcd'):
            subprocess.run([f'mdpocket --trajectory_file {trajectory} --trajectory_format dcd -f {topology}'],
                           check=True, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        elif self.trajectory.endswith('.xtc'):
            subprocess.run([f'mdpocket --trajectory_file {trajectory} --trajectory_format xtc -f {topology}'],
                           check=True, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        else:
            logger.log(f'VALUE ERROR: The trajectory file has to be either dcd or xtc format! \n'
                       f'You specified {self.trajectory} as input.')
            raise ValueError(f'The trajectory file has to be either dcd or xtc format!\n'
                             f'You specified {self.trajectory} as input.')
        os.chdir(working_dir)

    def _extract_iso_mdpocket(self, isovalue, density):
        """Using the extractIsoPdb.py included in mdpocket to save a PDB file of specific Isovalues"""
        # default is density grid
        grid_file = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                             'mdpout_dens_grid.dx')
        output_name = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                   f'mdpout_dens_iso_{isovalue}.pdb')
        if not density:
            grid_file = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                     'mdpout_freq_grid.dx')
            output_name = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                       f'mdpout_freq_iso_{isovalue}.pdb')
        elif density:
            grid_file = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                                 'mdpout_dens_grid.dx')
            output_name = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                       f'mdpout_dens_iso_{isovalue}.pdb')

        iso.extract_iso_pdb(grid_file, output_name, isovalue)

    def _optimize_epsilon(self, data, isofile):
        """Using a range of epsilon values and the dataset provided, the function returns the optimal
        eps value (between 0.1 and 20 in steps of 0.1) for DBSCAN according to the silhouette score
        under the constraint that the number of clusters found is higher than 5
        returns: best_eps, rounded to two decimals and report_dict containing info about the silhouette score,
        epsilon value and number of clusters used to produce separate pockets"""
        eps_range = np.arange(0.1, 20.1, 0.1)
        eps_sil_dict = {}
        eps_cluster_dict = {}
        best_eps = None
        best_sil_score = -1
        for eps in eps_range:
            dbscan = cluster.DBSCAN(eps=eps)
            labels = dbscan.fit_predict(data)
            no_cluster = len(np.unique(labels))
            eps_cluster_dict[eps] = no_cluster
            if len(set(labels)) > 1:
                sil_score = silhouette_score(data, labels)
                eps_sil_dict[eps] = sil_score
                if no_cluster > 5 and sil_score > 0.3 and sil_score > best_sil_score:
                    # if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_eps = eps
        print(f'Best eps: {best_eps}')
        print(f'Best silhouette score: {best_sil_score}')
        print(f'Number of clusters: {no_cluster}')
        logger.log(
            f'Best eps: {best_eps} \n Best silhouette score: {best_sil_score} \n Number of clusters: {no_cluster}')
        if best_eps is None:
            best_eps = 0.5
            print('No optimal epsilon found, clustering will be performed using eps=0.5')
            logger.log('No optimal epsilon found, clustering will be performed using eps=0.5')
        if self.verbose:
            print(f'Project {self.curr_proj} replicate {self.curr_rep}: The best silhouette score (according to the '
                  f'search parameters) of {best_sil_score:.3f} '
                  f'is achieved with eps = {best_eps} which is chosen to separate the pockets using DBSCAN. '
                  f'This results_dbscan_new in {no_cluster} clusters.')
        logger.log(f'Project {self.curr_proj} replicate {self.curr_rep}: The best silhouette score (according to the '
                   f'search parameters) of {best_sil_score:.3f} '
                   f'is achieved with eps = {best_eps} which is chosen to separate the pockets using DBSCAN. '
                   f'This results_dbscan_new in {no_cluster} clusters.')
        eps_cluster_ls = [(k, v) for k, v in eps_cluster_dict.items()]
        sil_no_clusters = pd.DataFrame(eps_cluster_ls, columns=['eps', 'no_clusters'])
        sil_no_clusters['sil_score'] = np.NaN
        sil_no_clusters['sil_score'] = sil_no_clusters['eps'].map(eps_sil_dict)

        sil_no_clusters.to_csv(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                            f'silhouette_scores_{isofile.split('.pdb')[0]}.csv'), index=False)
        report_dict = {'best epsilon': str(best_eps), 'silhouette score': str(best_sil_score)}

        return round(best_eps, 2), report_dict

    def _prep_for_second_run(self, isovalues, dbscan, verbose):
        """prepares the second run of mdpocket by reading in the saved pdb files (output of first run),
        separates the information into pockets
        (using DBSCAN clustering or ATClus algorithm depending on DBSCAN flag)
        and saves the coordinates for each pocket as a single PDB file (Input for mdpocket second run;
        Pockets that cannot be separated (DBSCAN label -1) are ignored during this process
        Steps: read in the pdb (output of mdpocket run 1),
               cluster the pockets to separate them,
               save single PDBs for each pocket"""
        path_to_folder = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)
        iso_files = [f'mdpout_dens_iso_{iso}.pdb' for iso in isovalues]

        if dbscan:
            print('INFO: Pockets are separated using DBSCAN clustering')
            logger.log('INFO: Pockets are separated using DBSCAN clustering')
            for isofile in iso_files:
                curr_coords, curr_df = self._parse_pdb_to_df(os.path.join(path_to_folder, isofile))
                eps, report_dict = self._optimize_epsilon(data=curr_coords, isofile=isofile)
                dbscan = cluster.DBSCAN(eps=eps)
                curr_labels = dbscan.fit_predict(curr_coords)
                curr_df['label'] = curr_labels
                report_dict['number of clusters'] = len(curr_df['label'].unique())
                pdb_file = PandasPdb().read_pdb(os.path.join(path_to_folder, isofile))
                pdb_df = pdb_file.df['ATOM']
                map_dict = dict(zip(curr_df['atom_no'].astype(float), curr_df['label']))
                pdb_file.df['ATOM']['pocket_no'] = pdb_file.df['ATOM']['atom_number'].map(map_dict)
                for pocket in pdb_file.df['ATOM']['pocket_no'].unique():
                    path_save = os.path.join(path_to_folder,
                                             f'coords_pocket_{pocket + 1}_{isofile.split('.pdb')[0]}.pdb')
                    if pocket == -1:
                        continue
                    pocket_pdb = PandasPdb()
                    cluster_df = pdb_df[pdb_df['pocket_no'] == pocket]
                    pocket_pdb.df['ATOM'] = cluster_df
                    pocket_pdb.to_pdb(path_save)
                # save the best epsilon, silhouette score and number of clusters as json dict
                json_obj = json.dumps(report_dict)
                with open(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir,
                                       f'values_for_clustering_{isofile.split('.pdb')[0]}.json'), 'w') as out_f:
                    out_f.write(json_obj)
        elif dbscan is False:
            print('INFO: Pockets are separated using ATClus (finite element method based approach)')
            logger.log('INFO: Pockets are separated using ATClus (finite element method based approach)')
            atclus_dir = os.path.join(os.path.join(os.getcwd(), conf.folder_scripts, 'atclus'))
            for prj in os.listdir(conf.folder_results):
                for rep in os.listdir(os.path.join(conf.folder_results, prj)):
                    results_dir = os.path.join(conf.folder_results, prj, rep)
                    if not os.path.exists(os.path.join(results_dir, self.pocket_dir)):
                        os.mkdir(os.path.join(results_dir, self.pocket_dir))
                    # copy the exe into the same directory as the data to be able to run it
                    shutil.copy(os.path.join(atclus_dir, 'atclus.f'), os.path.join(results_dir, self.pocket_dir, 'atclus.f'))
                    shutil.copy(os.path.join(atclus_dir, 'atclus.inc'),
                                os.path.join(results_dir,self.pocket_dir, 'atclus.inc'))
                    shutil.copy(os.path.join(atclus_dir, 'atclus.o'), os.path.join(results_dir, self.pocket_dir, 'atclus.o'))
                    # shutil.copy(os.path.join(atclus_dir, 'go_clean'), os.path.join(results_dir, POCKET_DIR, 'go_clean'))
                    shutil.copy(os.path.join(atclus_dir, 'atclus'), os.path.join(results_dir, self.pocket_dir, 'atclus'))
                    # running fortran77 code via command-line via subprocess
                    for isofile in iso_files:
                        os.chdir(os.path.join(results_dir, self.pocket_dir))

                        if verbose is False:
                            logfile = os.path.join(results_dir, self.pocket_dir, 'atclus_output_log.txt')
                            with open(logfile, 'w') as log:
                                subprocess.run(['./atclus', isofile.split('.pdb')[0], '1.4'],
                                               check=True, stdout=log, stderr=log)
                        elif verbose:
                            subprocess.run(['./atclus', isofile.split('.pdb')[0], '1.4'],
                                           check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
                        os.chdir(conf.folder_results)  # change back to working directory
                    # print number of pockets found --> search for .group files
                    for file in os.listdir(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, 'pockets_dens')):
                        if file.endswith('.group'):
                            with open(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, 'pockets_dens', file),
                                      'r') as f:
                                lines = f.readlines()
                                last_line = lines[-1]
                                parts = last_line.split()
                                if "#:" in parts:
                                    hash_index = parts.index("#:")
                                    number_of_pockets = parts[hash_index - 1]
                                    if self.verbose:
                                        print(f"Number of Pockets found for "
                                              f"{os.path.join(self.curr_proj, self.curr_rep, file, 'pockets_dens')}: "
                                              f"{number_of_pockets}")
                                    else:
                                        logger.log(f"Number of Pockets found for "
                                              f"{os.path.join(self.curr_proj, self.curr_rep, file, 'pockets_dens')}: "
                                              f"{number_of_pockets}")
                                else:
                                    print(f'No number of pockets automatically reported')

    def _md_pocket_two_list(self, pocket_list):
        """let it run without parallelization"""
        for pocket in pocket_list:
            pocket_folder = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)
            os.chdir(pocket_folder)
            pocket_file = pocket.split('/')[-1]
            output_name = pocket_file.split('.pdb')[0]

            trajectory = self.trajectory.split('/')[-1]
            topology = self.topology.split('/')[-1]
            # copy trajectory and topology into pocket_dir
            if not os.path.exists(os.path.join(pocket_folder, trajectory)):
                shutil.copy(self.trajectory, os.path.join(pocket_folder, trajectory))
            if not os.path.exists(os.path.join(pocket_folder, topology)):
               shutil.copy(self.topology, os.path.join(pocket_folder, topology))

            subprocess.run([f'mdpocket -S --trajectory_file {trajectory} '
                            f'--trajectory_format {trajectory.split('.')[-1]} '
                            f'-f {topology} --selected_pocket {pocket_file} -d -o {output_name}'],
                           check=True, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            os.chdir(working_dir)

    def _md_pocket_two(self, pocket_path):
        """starting the second run of mdpocket;
        Input: path to a PDB file with coordinates of a single pocket;
        Output: saves three files:
                1) _descriptors.txt     # useful descriptors, basis for the rest of the analysis
                2) _mdpocket.pdb        # PDB file with single pocket
                3) _mdpocket_atoms.pdb  # PDB file with the residues near the pocket over all frames analysed
        documentation of MDpocket:
             "https://github.com/Discngine/fpocket/blob/master/doc/GETTINGSTARTED.md"
             paper by: Peter Schmldtke, Axel Bidon-Chanal, Javier Luque, Xavier Barril,
            “MDpocket: open-source cavity detection and characterization on molecular dynamics trajectories.”,
            Bioinformatics. 2011 Dec 1;27(23):3276-85
        function to be called in multiprocessing due to the extreme computational time
        """
        # If the subprocess command is not shell=True it will crash.
        pocket_dir = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)
        os.chdir(pocket_dir)
        pocket_file = pocket_path.split('/')[-1]
        output_name = pocket_file.split('.pdb')[0]

        trajectory = self.trajectory.split('/')[-1]
        topology = self.topology.split('/')[-1]
        # copy trajectory and topology into pocket_dir
        if not os.path.exists(os.path.join(pocket_dir, trajectory)):
            shutil.copy(self.trajectory, os.path.join(pocket_dir, trajectory))
        if not os.path.exists(os.path.join(pocket_dir, topology)):
            shutil.copy(self.topology, os.path.join(pocket_dir, topology))
        if self.trajectory.endswith('.dcd'):
            subprocess.run([f'mdpocket -S --trajectory_file {trajectory} --trajectory_format dcd -f '
                            f'{topology} --selected_pocket {pocket_file} -d -o {output_name}'],
                           check=True, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        elif self.trajectory.endswith('.xtc'):
            subprocess.run([f'mdpocket -S --trajectory_file {trajectory} --trajectory_format xtc -f '
                            f'{topology} --selected_pocket {pocket_file} -d -o {output_name}'],
                           check=True, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        else:
            logger.log(f'VALUE ERROR: The trajectory file has to be either dcd or xtc format! '
                       f'\n You specified {self.trajectory} as input.')
            raise ValueError(f'The trajectory file has to be either dcd or xtc format!\n '
                             f'You specified {self.trajectory} as input.')
        os.chdir(working_dir)

    def _parallel_mdpocket_run_two(self, dbscan=False, parallel=True):
        """starting the second run in parallel;
        Since the second run is very time-consuming, the process is parallelized
        parameters for parallelization: 80% of available cores are used (WS7 has 16 --> 12 used for pocket search),
        documentation of multiprocessing: https://docs.python.org/3/library/multiprocessing.html
        Cave: no druggability score is given but could be reproduced using the paper:
        https://pubs.acs.org/doi/full/10.1021/jm100574m"""
        # 0: search for all pocket PDB files
        pockets_dir = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)
        pockets = []
        cpus_used = None
        if dbscan:
            pockets = [os.path.join(pockets_dir, file) for file in os.listdir(pockets_dir)
                       if file.startswith('coords_pocket_') and not file.endswith('descriptors.txt')
                       and 'atoms' not in file and not file.endswith('mdpocket.pdb')]
        elif dbscan is False:
            pockets = [os.path.join(pockets_dir, file) for file in os.listdir(pockets_dir)
                       if file.startswith('mdpout_') and '-out-' in file and file.endswith('.pdb')
                       and not file.endswith('descriptors.txt') and 'atoms' not in file
                       and not file.endswith('mdpocket.pdb')]
        if not pockets:
            logger.log('VALUE ERROR: No pockets were found in the first run or searching for files failed')
            raise ValueError(f'No pockets were found in the first run or searching for files failed')

        if parallel is False:
            self._md_pocket_two_list(pockets)
        # I: run mdpocket using multiprocessing
        # I.1: identify the number of cores to be used --> lets say 80%
        elif parallel is True:
            cpus_avail = multiprocessing.cpu_count()
            if cpus_avail < 6:
                cpus_used = int(cpus_avail * 0.8)
            elif cpus_avail > 6:
                cpus_used = int(cpus_avail - 4)
            if self.verbose:
                print(f'Your workstation has {cpus_avail} cores available of which {cpus_used} '
                      f'cores are used for pocket description. ')
            logger.log(f'Your workstation has {cpus_avail} cores available of which {cpus_used} '
                       f'cores are used for pocket description. ')
            # the Pool object which offers a convenient means of parallelizing the execution of a function across multiple
            # input values, distributing the input data across processes (data parallelism).
            # I.2: run mdpocket for the second time in multiprocessing
            pool_obj = multiprocessing.Pool(processes=cpus_used)
            pool_obj.map(self._md_pocket_two, pockets)
            # maps all entries in iterable to the available cores and executes the function divided onto the cores
            pool_obj.close()
            pool_obj.join()  # makes sure that all the processes are done running
        return cpus_used

    def pocket_search(self, density=True, dbscan=False, isovalues=[3],
                      parallel=True):
        """This method is based on MDpocket:
               [paper by Peter Schmldtke, Axel Bidon-Chanal, Javier Luque, Xavier Barril,
                “MDpocket: open-source cavity detection and characterization on molecular dynamics trajectories.”,
                Bioinformatics. 2011 Dec 1;27(23):3276-85]
            Unfortunately, the method can only search for pockets of trajectories/topologies created by this pipeline or
            saved in results_dbscan_new/curr_proj/curr_rep/pockets called aligned_traj.dcd and aligned_top.pdb. This is due to the
            behaviour of MDpocket which can only be executed from the subdirectory of the traj/top files
            and will not accept absolute paths / formatted strings. Also, there is no druggability score given...
            The Input is therefore none"""
        logger.log(f'Pocket Analysis Starting. The directory used is {self.pocket_dir}. \n')
        # I: creates the subdir "pockets" if it doesn't exist
        if not os.path.exists(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir)):
            os.makedirs(os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir))
        current_datetime = datetime.datetime.now()
        logger.log(f'\n Pocket Identification, Separation and Description \n Data: {self.curr_proj}_{self.curr_rep} \n')
        logger.log(f'Start Pocket Hunting at {current_datetime} . ')
        print(f'Start Pocket Hunting at {current_datetime} .')

        # I.1: copy files to new folder pockets --> so that MDpocket can behave like a diva ...
        # parent_dir = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep)
        trajectory = self.trajectory.split('/')[-1]
        topology = self.topology.split('/')[-1]
        trj_path = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir, trajectory)
        top_path = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep, self.pocket_dir, topology)
        if not os.path.exists(trj_path):
            shutil.copy(self.trajectory, trj_path)
        if not os.path.exists(top_path):
            shutil.copy(self.topology, top_path)
        # II: check that the User understands what is used as input
        if not os.path.exists(trj_path) and not os.path.exists(top_path):
            logger.log(f'FILE NOT FOUND ERROR: There is no trajectory file named "{self.trajectory}" and no file named '
                       f'"{self.topology}". '
                       f'This method can only be used with files saved in the following directory format: '
                       f'{os.path.join(conf.folder_results, self.curr_proj, self.pocket_dir)} .'
                       f'Please read the documentation of the Pocket Search Tool.')
            raise FileNotFoundError(f'There is no trajectory file named "{self.trajectory}" and no file named '
                       f'"{self.topology}". '
                       f'This method can only be used with files saved in the following directory format: '
                       f'{os.path.join(conf.folder_results, self.curr_proj, self.pocket_dir)} .'
                       f'Please read the documentation of the Pocket Search Tool.')
        elif not os.path.exists(trj_path) and os.path.exists(top_path):
            logger.log(f'FILE NOT FOUND ERROR:There is no trajectory file named "{self.trajectory}" in the directory '
                       f'{self.curr_proj} / {self.curr_rep}'
                       f'This method can only be used with files saved in the following directory format: '
                       f'{trj_path} .'
                       f'Please read the documentation of the Pocket Search Tool.')
            raise FileNotFoundError(f'FILE NOT FOUND ERROR:There is no trajectory file named "{self.trajectory}" in the directory '
                       f'{self.curr_proj} / {self.curr_rep}'
                       f'This method can only be used with files saved in the following directory format: '
                       f'{trj_path} .'
                       f'Please read the documentation of the Pocket Search Tool.')
        elif os.path.exists(trj_path) and not os.path.exists(top_path):
            logger.log(f'FILE NOT FOUND ERROR: There is no topology file named "{self.topology}" in the directory '
                       f'{self.curr_proj} / {self.curr_rep}'
                       f'This method can only be used with files saved in the following directory format: '
                       f'{top_path} .'
                       f'Please read the documentation of the Pocket Search Tool.')
            raise FileNotFoundError(f'FILE NOT FOUND ERROR: There is no topology file named "{self.topology}" in the directory '
                       f'{self.curr_proj} / {self.curr_rep}'
                       f'This method can only be used with files saved in the following directory format: '
                       f'{top_path} .'
                       f'Please read the documentation of the Pocket Search Tool.')
        # III: Call the actual runs of MDpocket and the intermediate data preparation (clustering, PDB file management)
        start_run1 = time.time()
        self._mdpocket_run_one()
        end_run1 = time.time()
        print(f'Finding Pockets took {(end_run1 - start_run1) / 60} minutes')
        print('Starting IsoValue Extraction')
        logger.log(f'Finding Pockets took {(end_run1 - start_run1) / 60} minutes \n Starting IsoValue Extraction')
        if not density:
            for isovalue in isovalues:
                self._extract_iso_mdpocket(isovalue=isovalue, density=density)
            start_prep = time.time()
            self._prep_for_second_run(isovalues=isovalues, dbscan=dbscan, verbose=self.verbose)
            end_prep = time.time()
        elif density:
            for isovalue in isovalues:
                self._extract_iso_mdpocket(isovalue=isovalue, density=density)
            start_prep = time.time()
            self._prep_for_second_run(isovalues=isovalues, dbscan=dbscan, verbose=self.verbose)
            end_prep = time.time()
        print(f'Separating Pockets took {(end_prep - start_prep) / 60} minutes')
        logger.log(f'Separating Pockets took {(end_prep - start_prep) / 60} minutes')
        start_run2 = time.time()
        cpus_used = self._parallel_mdpocket_run_two(parallel=parallel, dbscan=dbscan)
        end_run2 = time.time()
        print(f'Describing Pockets took {(end_run2 - start_run2) / 60} minutes using {cpus_used} cores')
        print(f'End Pocket Hunting at {current_datetime} . '
              f'The results are saved in the subdirectory {self.pocket_dir}.')
        logger.log(f'Describing Pockets took {(end_run2 - start_run2) / 60} minutes using {cpus_used} cores \n'
                   f'End Pocket Hunting at {current_datetime} . '
                   f'The results are saved in the subdirectory {self.pocket_dir}.')
