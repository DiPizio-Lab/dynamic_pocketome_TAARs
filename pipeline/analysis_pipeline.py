""""Step 1: turns wrapped MD trajectories into per-experiment analysis results.

For every replicate under input/apo_structures/ and input/holo_structures/ (PDB+rep
layout: <state><PDBID>/<rep>/{structure.pdb, structure.psf, traj_wrapped.xtc}), this:
  1. Pre-processes the trajectory (dry protein, align to the reference topology) --
     scripts/preprocessing.py
  2. Calculates RMSD/RMSF -- scripts/basic_analysis.py
  3. Runs the MDpocket-based pocket search -- scripts/pocket_analysis.py

Results are written to output/apo_structures/ and output/holo_structures/, mirroring
the input PDB+rep tree (see config.py). This is the first stage of the pipeline;
Step 2 (comparative_study.py / complete_pocket_analysis.py) reads its pockets/
output.

Originally developed for Clarissa Rienäcker's Master's Thesis, analysing TAAR (and
OR) MD trajectories with MDAnalysis and fpocket/MDpocket.
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf
from scripts import preprocessing as prepro
from scripts import basic_analysis
from scripts import pocket_analysis as pocket
from scripts import gene_selections as gene_sel
from scripts import logging as logger

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

VERBOSE = True
SHOW_PLOTS = False  # True: plots are displayed and saved; False: saved only

# Simulation timing: frames are written every traj_period integration steps of
# time_step fs, then strided by `stride` in make_selection_n_save() below --
# 25000 * 4fs = 0.1 ns/raw frame, * stride 10 = 1 ns/frame downstream (matches
# consecutive_zeros_transiency.NS_PER_FRAME).
TRJ_PERIOD_STEP_STRIDE = {'traj_period': 25000, 'time_step': 4, 'stride': 10}
STRIDE = TRJ_PERIOD_STEP_STRIDE['stride']

# Binding site & ICL3 residue numbers per gene (NOT the orthosteric/is_binding_site pocket
# classification, which lives in orthosteric_filter_ligand_based.py and is computed from the
# ligand centroid per PDB) are read from reference_data/binding_site_residues.txt and
# reference_data/ICL3_definition.txt by scripts/gene_selections.py.


def get_project_list():
    """[<APO_STRUCTURES_DIR or HOLO_STRUCTURES_DIR>/<state><PDBID>/<rep>, ...] for every
    replicate under both input structure trees."""
    project_ls = []
    for structures_dir in (conf.APO_STRUCTURES_DIR, conf.HOLO_STRUCTURES_DIR):
        if not os.path.isdir(structures_dir):
            continue
        for folder in sorted(os.listdir(structures_dir)):
            folder_path = os.path.join(structures_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for rep in sorted(os.listdir(folder_path)):
                rep_path = os.path.join(folder_path, rep)
                if os.path.isdir(rep_path):
                    project_ls.append(rep_path)
    return project_ls


def setup_results_folder():
    """Mirrors the input PDB+rep tree into the matching output results tree
    (output/apo_structures/ or output/holo_structures/, see config.results_dir_for)."""
    for structures_dir in (conf.APO_STRUCTURES_DIR, conf.HOLO_STRUCTURES_DIR):
        if not os.path.isdir(structures_dir):
            continue
        for folder in sorted(os.listdir(structures_dir)):
            folder_path = os.path.join(structures_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            results_dir = conf.results_dir_for(folder)
            for rep in sorted(os.listdir(folder_path)):
                if os.path.isdir(os.path.join(folder_path, rep)):
                    os.makedirs(os.path.join(results_dir, folder, rep), exist_ok=True)


def run_analyses():
    """Wrapper calling all per-replicate analysis steps; files are saved and then
    not overwritten if they already exist."""
    project_ls = get_project_list()

    for proj_rep in project_ls:
        curr_rep = os.path.basename(proj_rep)
        folder_name = os.path.basename(os.path.dirname(proj_rep))
        curr_traj = os.path.join(proj_rep, 'traj_wrapped.xtc')
        curr_topol = os.path.join(proj_rep, 'structure.pdb')
        results_dir = conf.results_dir_for(folder_name)

        # binding site & ICL3 residues differ per gene, so the group selection is built per project
        gene = gene_sel.gene_for_project(folder_name)
        group_sel = gene_sel.group_selection_for(gene)

        # I: pre-processing -- dry protein, then align to the reference topology
        preprocess_obj = prepro.PreProcess(curr_topol, curr_traj, folder_name, curr_rep, verbose=VERBOSE)
        preprocess_obj.make_selection_n_save(stride=STRIDE)  # if selection None selects 'protein'
        # Note: dry protein is saved, align to ref takes path to saving loc
        #  (otherwise issues with the number of frames: only 1st frame is returned as AtomGroup)
        universe_path = os.path.join(results_dir, folder_name, curr_rep, 'dry_prot.xtc')
        dry_top_path = os.path.join(results_dir, folder_name, curr_rep, 'dry_prot.pdb')
        preprocess_obj.align_to_reference(reference=curr_topol, universe=(dry_top_path, universe_path))

        # II: RMSD / RMSF
        basic_analysis_obj = basic_analysis.BasicAnalysis(curr_topol, curr_traj, curr_proj=folder_name,
                                                          curr_rep=curr_rep, verbose=VERBOSE, show_plots=SHOW_PLOTS)
        basic_analysis_obj.calc_rmsd(trj_period_step_stride=TRJ_PERIOD_STEP_STRIDE,
                                     selection='protein and name CA', group_selection=group_sel)
        basic_analysis_obj.calc_rmsf()

        # III: pocket search (MDpocket) -- runs on the aligned, dry-protein output from step I,
        # not the raw input (which still has solvent/membrane)
        aligned_topol, aligned_traj = conf.aligned_paths_for(folder_name, curr_rep, 'xtc')
        pocket_obj = pocket.PocketAnalysis(aligned_topol, aligned_traj, curr_proj=folder_name, curr_rep=curr_rep,
                                           verbose=VERBOSE, pocket_dir='pockets')
        pocket_obj.pocket_search()


def main():
    t0 = time.time()
    logger.log("Welcome to MMLab's custom MD Analysis Pipeline! Here is what happened while you had a coffee: \n \n ")
    setup_results_folder()
    run_analyses()
    t1 = time.time()
    t_min = (t1 - t0) / 60
    t_hr = t_min / 60
    print('time:  ', t1 - t0, '  sec')
    print('time:  ', t_min, '  min')
    print('time:  ', t_hr, '  hours')
    logger.log(f'\n \n The pipeline took {round(t_min, 3)} minutes ({round(t_hr, 3)} hours) '
               f'to run completely. \n \n  Thank you for using our code, have a nice day!')


if __name__ == '__main__':
    main()
