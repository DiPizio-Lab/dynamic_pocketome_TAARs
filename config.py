"""Single place all scripts resolve project paths from.

PROJECT_ROOT defaults to the current working directory, so run scripts from the
repo root (or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient, e.g. when submitting jobs from a different working directory).
"""
import glob
import os


def find_external_root(relative_suffix):
    """First existing /mnt/<share>/<relative_suffix> or /mnt/<share>/<subshare>/<relative_suffix>."""
    candidates = sorted(glob.glob(f"/mnt/*/{relative_suffix}")) + \
        sorted(glob.glob(f"/mnt/*/*/{relative_suffix}"))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"No /mnt/*/{relative_suffix} (or /mnt/*/*/{relative_suffix}) found on "
        f"{os.uname().nodename} -- is the share mounted on this machine?")


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
    PreProcess.align_to_reference for a given project + replicate."""
    base = os.path.join(results_dir_for(curr_proj), curr_proj, curr_rep)
    return os.path.join(base, 'aligned_top.pdb'), os.path.join(base, f'aligned_traj.{trajectory_ext}')


# Everything that aggregates across PDB+rep: comparative_study driver runs
# (within_structure/, apo_vs_holo/, within_gene/, across_genes/), Stage 3
# global-ID clustering/matching/replicate-concordance, and paper figures.
META_ANALYSIS_ROOT = os.path.join(OUTPUT_DIR, "meta_analysis")
META_ANALYSIS_DIR = os.path.join(META_ANALYSIS_ROOT, "across_genes")

# One PDB ID per gene (see reference_data/hard_coded_gene_dict.txt): 8ITF -> mTAAR9,
# 8JLJ -> mTAAR1, 8JLR -> hTAAR1, 8PM2 -> mTAAR7f.
# Default `pdb_ids=` for global_id_and_comparison.run_global_id_states() (Step 2.2 only).
REPRESENTATIVE_PDB_IDS = ['8ITF', '8JLJ', '8JLR', '8PM2']

# Step 1's per-replicate pocket-search output subdirectory name.
POCKETS_DIRNAME = "pockets"

# Isovalues the whole pipeline runs pocket detection at.
ISOVALUES = [3.0]


def require_float_isovalues(isovalues):
    """Raises TypeError unless every isovalue is a float (3.0, not 3)."""
    bad = [v for v in isovalues if not isinstance(v, float)]
    if bad:
        raise TypeError(f"isovalues must all be floats (e.g. 3.0, not 3): got {bad!r}")
    return list(isovalues)


def _isovalue_dirname(isovalue):
    return f"isovalue_{float(isovalue)}"


def isovalue_subpath(base_dir, isovalue, isovalues=ISOVALUES):
    """base_dir, or base_dir/isovalue_<X.X> when isovalues has more than one entry."""
    return base_dir if len(isovalues) <= 1 else os.path.join(base_dir, _isovalue_dirname(isovalue))


def isovalue_meta_analysis_root(isovalue, isovalues=ISOVALUES):
    return isovalue_subpath(META_ANALYSIS_ROOT, isovalue, isovalues)


def isovalue_meta_analysis_dir(isovalue, isovalues=ISOVALUES):
    return os.path.join(isovalue_meta_analysis_root(isovalue, isovalues), "across_genes")
