""""Step 1: turns wrapped MD trajectories into per-experiment analysis results.

For every replicate under input/apo_structures/ and input/holo_structures/ (PDB+rep
layout: <state><PDBID>/<rep>/{structure.pdb, structure.psf, traj_wrapped.xtc}), this:
  1. Pre-processes the trajectory (dry protein, align to the reference topology) --
     scripts/preprocessing.py
  2. Calculates RMSD/RMSF -- scripts/basic_analysis.py
  3. Runs the MDpocket-based pocket search -- scripts/pocket_analysis.py

Results are written to output/apo_structures/ and output/holo_structures/, mirroring
the input PDB+rep tree (see config.py). This is the first stage of the pipeline;

Originally developed by Clarissa Rienaecker analysing TAAR (and
OR) MD trajectories with MDAnalysis and fpocket/MDpocket.
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as conf
from scripts import preprocessing as prepro
from scripts import basic_analysis
from scripts import pocket_analysis as pocket
from scripts import gene_selections as gene_sel
from scripts import logging as logger
from scripts import run_summary

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

VERBOSE = True
SHOW_PLOTS = False

# Simulation timing: input/ is expected to already be at 1 frame = 1 ns `stride` is kept here as a knob for anyone whose
# input isn't pre-normalized, but stays 1 for this dataset
TRJ_PERIOD_STEP_STRIDE = {'traj_period': 25000, 'time_step': 4, 'stride': 1}
STRIDE = TRJ_PERIOD_STEP_STRIDE['stride']


def get_project_list(pdb_ids=None, exclude_pdb_ids=None):
    """[<APO_STRUCTURES_DIR or HOLO_STRUCTURES_DIR>/<state><PDBID>/<rep>, ...] for every
    replicate under both input structure trees, covering both states for whichever PDB IDs are
    included (bare IDs, e.g. '8ITF', not 'apo8ITF').

    pdb_ids: if given, only these PDB IDs (both apo and holo) - e.g. a small reproducibility-
    check subset that should be prioritized ahead of the rest.
    exclude_pdb_ids: if given, every PDB ID EXCEPT these - the complement of a pdb_ids run, for
    submitting the "everything else" work as a separate, non-overlapping job so the two never
    race on the same replicate's output.
    Mutually exclusive - pass at most one."""
    if pdb_ids is not None and exclude_pdb_ids is not None:
        raise ValueError("pass at most one of pdb_ids / exclude_pdb_ids, not both")
    project_ls = []
    for structures_dir in (conf.APO_STRUCTURES_DIR, conf.HOLO_STRUCTURES_DIR):
        if not os.path.isdir(structures_dir):
            continue
        for folder in sorted(os.listdir(structures_dir)):
            folder_path = os.path.join(structures_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            _, folder_pdb_id = run_summary.parse_experiment(folder)
            if pdb_ids is not None and folder_pdb_id not in pdb_ids:
                continue
            if exclude_pdb_ids is not None and folder_pdb_id in exclude_pdb_ids:
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


def run_analyses(shard_index=0, num_shards=1, pdb_ids=None, exclude_pdb_ids=None):
    """Wrapper calling all per-replicate analysis steps; files are saved and then
    not overwritten if they already exist.

    shard_index/num_shards split project_ls (project_ls[shard_index::num_shards])
    so this can be run as several concurrent processes (e.g. one per SLURM node) each
    covering a disjoint set of replicates with no coordination needed between them (every
    replicate writes to its own output subtree, see setup_results_folder). Defaults (0, 1)
    run every replicate, unchanged from before sharding existed.

    pdb_ids/exclude_pdb_ids: see get_project_list() - scope this run to a PDB ID subset (or its
    complement) instead of everything."""
    project_ls = get_project_list(pdb_ids=pdb_ids, exclude_pdb_ids=exclude_pdb_ids)[shard_index::num_shards]
    print(f'Shard {shard_index}/{num_shards}: {len(project_ls)} replicate(s) to process')

    for proj_rep in project_ls:
        curr_rep = os.path.basename(proj_rep)
        folder_name = os.path.basename(os.path.dirname(proj_rep))
        curr_traj = os.path.join(proj_rep, 'traj_wrapped.xtc')
        curr_topol = os.path.join(proj_rep, 'structure.pdb')
        results_dir = conf.results_dir_for(folder_name)
        state, pdb_id = run_summary.parse_experiment(folder_name)

        try:
            # binding site & ICL3 residues differ per gene, so the group selection is built per project
            gene = gene_sel.gene_for_project(folder_name)
            group_sel = gene_sel.group_selection_for(gene)

            # I: pre-processing - dry protein, then align to the reference topology
            with run_summary.stage(state, pdb_id, curr_rep, 'preprocessing'):
                preprocess_obj = prepro.PreProcess(curr_topol, curr_traj, folder_name, curr_rep, verbose=VERBOSE)
                preprocess_obj.make_selection_n_save(stride=STRIDE)  # if None selects 'protein'
                # Note: dry protein is saved, align to ref takes path to saving loc
                universe_path = os.path.join(results_dir, folder_name, curr_rep, 'dry_prot.xtc')
                dry_top_path = os.path.join(results_dir, folder_name, curr_rep, 'dry_prot.pdb')
                preprocess_obj.align_to_reference(reference=curr_topol, universe=(dry_top_path, universe_path))

            # II: RMSD / RMSF
            with run_summary.stage(state, pdb_id, curr_rep, 'rmsd_calculation'):
                basic_analysis_obj = basic_analysis.BasicAnalysis(curr_topol, curr_traj, curr_proj=folder_name,
                                                                  curr_rep=curr_rep, verbose=VERBOSE, show_plots=SHOW_PLOTS)
                basic_analysis_obj.calc_rmsd(trj_period_step_stride=TRJ_PERIOD_STEP_STRIDE,
                                             selection='protein and name CA', group_selection=group_sel,
                                             icl3_free_selection=gene_sel.ca_without_icl3_selection_for(gene))
                basic_analysis_obj.calc_rmsf()

            # III: pocket search (MDpocket) - runs on the aligned, dry-protein output from step I,
            # not the raw input (which still has solvent/membrane). Records its own
            # pocket_detection/pocket_separation/pocket_characterisation checkpoints internally
            # (see PocketAnalysis.pocket_search), since those are three distinct sub-stages.
            aligned_topol, aligned_traj = conf.aligned_paths_for(folder_name, curr_rep, 'xtc')
            pocket_obj = pocket.PocketAnalysis(aligned_topol, aligned_traj, curr_proj=folder_name, curr_rep=curr_rep,
                                               verbose=VERBOSE, pocket_dir=conf.POCKETS_DIRNAME)
            pocket_obj.pocket_search()
        except Exception as exc:
            # One experiment's failure must not take down the rest of this shard's replicate
            # list so here we just log and move on to the next replicate.
            message = run_summary.format_exception_message(exc)
            print(f'ERROR: {folder_name}/{curr_rep} failed, skipping to next replicate: {message}')
            logger.log(f'ERROR: {folder_name}/{curr_rep} failed, skipping to next replicate: {message}')
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: per-replicate preprocessing, RMSD/RMSF "
                                                  "and pocket search.")
    parser.add_argument('--num-shards', type=int, default=1,
                         help="Split the replicate list into this many shards, e.g. one per "
                              "SLURM node (default: 1, i.e. no sharding -- run everything)")
    parser.add_argument('--shard-index', type=int, default=0,
                         help="0-indexed shard this process should run (default: 0)")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--pdb-ids', nargs='+', default=None,
                        help="Only these PDB IDs (both apo and holo), e.g. --pdb-ids 8ITF 8JLJ")
    scope.add_argument('--exclude-pdb-ids', nargs='+', default=None,
                        help="Every PDB ID except these -- the complement of a --pdb-ids run, "
                             "for a non-overlapping companion job covering the rest")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    return args


def main():
    args = parse_args()
    t0 = time.time()
    logger.log("Welcome to CPM's custom MD Analysis Pipeline! Here is what happened while you had a coffee: \n \n ")
    setup_results_folder()
    run_analyses(shard_index=args.shard_index, num_shards=args.num_shards,
                 pdb_ids=args.pdb_ids, exclude_pdb_ids=args.exclude_pdb_ids)
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
