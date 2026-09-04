"""Single place all scripts resolve project paths from.

PROJECT_ROOT defaults to the current working directory, so run scripts from the
repo root (or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient, e.g. when submitting jobs from a different working directory).
"""
import os

PROJECT_ROOT = os.environ.get("TAAR_ROOT", os.getcwd())
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REFERENCE_DATA_DIR = os.path.join(PROJECT_ROOT, "reference_data")

# Raw MD input -- published separately, never written to by the pipeline.
# Layout: {..._STRUCTURES_DIR}/<state><PDBID>/<replicate>/
#           {structure.pdb, structure.psf, traj_wrapped.xtc}
APO_STRUCTURES_DIR = os.path.join(INPUT_DIR, "apo_structures")
HOLO_STRUCTURES_DIR = os.path.join(INPUT_DIR, "holo_structures")

# Ligand reference for the orthosteric filter: the raw holo input itself, since
# every holo structure carries its co-crystallized ligand.
HOLO_BASE_DIR = HOLO_STRUCTURES_DIR

# Stage 1/2 per-structure results (dry_prot, aligned_traj, pockets/, ...).
# Mirrors the input PDB+rep tree one-to-one, one output subtree per input
# subtree, populated by the pipeline as it runs.
APO_RESULTS_DIR = os.path.join(OUTPUT_DIR, "apo_structures")
HOLO_RESULTS_DIR = os.path.join(OUTPUT_DIR, "holo_structures")


def results_dir_for(curr_proj):
    """APO_RESULTS_DIR or HOLO_RESULTS_DIR, picked from a project id's apo/holo prefix
    (e.g. 'apo8ITF' -> APO_RESULTS_DIR)."""
    return APO_RESULTS_DIR if curr_proj.startswith("apo") else HOLO_RESULTS_DIR


def aligned_paths_for(curr_proj, curr_rep, trajectory_ext):
    """(aligned_top.pdb, aligned_traj.<trajectory_ext>) paths written by
    PreProcess.align_to_reference for a given project + replicate -- the single place this
    naming convention is resolved, so preprocessing.py / basic_analysis.py / pocket_analysis.py
    can't drift out of sync with each other about where these files live."""
    base = os.path.join(results_dir_for(curr_proj), curr_proj, curr_rep)
    return os.path.join(base, 'aligned_top.pdb'), os.path.join(base, f'aligned_traj.{trajectory_ext}')


# Everything that aggregates across PDB+rep: comparative_study driver runs
# (within_structure/, apo_vs_holo/, within_gene/, across_genes/), Stage 3
# global-ID clustering/matching/replicate-concordance, and paper figures.
META_ANALYSIS_ROOT = os.path.join(OUTPUT_DIR, "meta_analysis")
META_ANALYSIS_DIR = os.path.join(META_ANALYSIS_ROOT, "across_genes")
