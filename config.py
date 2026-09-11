"""Single place all scripts resolve project paths from.

PROJECT_ROOT defaults to the current working directory, so run scripts from the
repo root (or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient, e.g. when submitting jobs from a different working directory).
"""
import glob
import os


def find_external_root(relative_suffix):
    """First existing /mnt/<share>/<relative_suffix> (or /mnt/<share>/<subshare>/<relative_suffix>,
    one directory level deeper) -- for raw external source trees (apo_holo_analysis's own
    simulation output, the K_rienaecker holo source) that live on a shared drive mounted under a
    DIFFERENT top-level path on different machines: this interactive workstation sees
    /mnt/cpm_crienaecker/Z/..., the SLURM compute nodes see /mnt/mmlab_shared/... instead -- same
    underlying share, different local mount name. Searching every top-level /mnt/* entry (and one
    level deeper) means the code adapts to whichever machine runs it instead of hardcoding one
    machine's mount name, which breaks as soon as a job lands on a different node.
    Raises FileNotFoundError (loudly, immediately, with the hostname) rather than returning None
    if nothing matches -- a raw source path silently resolving to nothing is exactly the kind of
    bug that should never fail quietly this far upstream of the actual computation."""
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

# One PDB ID per gene (see reference_data/hard_coded_gene_dict.txt): 8ITF -> mTAAR9,
# 8JLJ -> mTAAR1, 8JLR -> hTAAR1, 8PM2 -> mTAAR7f.
#
# Global ID clustering (Step 2.2) is meant to answer a specific, scoped research question --
# e.g. comparing apo vs holo states of one receptor, comparing its triplicate MD replicates
# against each other, or (this subset's purpose) showcasing Global ID conservation/divergence
# across a biologically diverse set of genes. Clustering pockets across every solved structure
# of the *same* receptor indiscriminately isn't itself a meaningful comparison -- it mixes
# redundant near-duplicate structures into the same run rather than the deliberate, one-receptor-
# per-gene comparison this subset represents. (It also happens to keep Global ID clustering's
# in-memory working set tractable -- clustering the full 26-PDB-ID dataset needs 100+GB and has
# OOM-killed a node with anon-rss over 230GB -- but that's a consequence of choosing a scoped
# subset, not the reason to choose one.)
#
# Used as the default `pdb_ids=` passed to global_id_and_comparison.run_global_id_states() (Step
# 2.2 only) -- NOT to pipeline.pocket_dataframes.pocket_dirs_for()/build_pocket_dataframes()
# (Step 2.1), which always parses every PDB ID so all_pockets/pocket_summary stay the complete
# dataset. Filter those tables by this same list (the `pdb_id` column) to see exactly what a
# Global ID run scoped to it was clustering.
REPRESENTATIVE_PDB_IDS = ['8ITF', '8JLJ', '8JLR', '8PM2']

# Step 1's per-replicate pocket-search output subdirectory name.
POCKETS_DIRNAME = "pockets"

# Isovalues the whole pipeline runs pocket detection at. mdpocket/ATClus pocket numbering
# (p01, p02, ...) restarts independently per isovalue, so a shared pocket number between two
# isovalues is not reliably the same physical pocket -- Global ID clustering (Step 2.2) must
# never compare pockets across isovalues. To guarantee that, every stage gets a fully separate
# output tree per isovalue: a single isovalue keeps the flat legacy layout everywhere; more than
# one adds an isovalue_<X.X> subdirectory to every stage's output (see isovalue_subpath()).
ISOVALUES = [3.0]


def require_float_isovalues(isovalues):
    """Raises TypeError unless every isovalue is a float (3.0, not 3) -- an int silently breaks
    pock_file_parser()'s on-disk filename matching (round-trips through numpy.float64 and picks
    up a spurious trailing '_0', so 'iso_3-out-*.pdb' is never found)."""
    bad = [v for v in isovalues if not isinstance(v, float)]
    if bad:
        raise TypeError(f"isovalues must all be floats (e.g. 3.0, not 3): got {bad!r}")
    return list(isovalues)


def _isovalue_dirname(isovalue):
    return f"isovalue_{float(isovalue)}"


def isovalue_subpath(base_dir, isovalue, isovalues=ISOVALUES):
    """base_dir, or base_dir/isovalue_<X.X> when isovalues has more than one entry -- the
    flat/split rule shared by every stage's output layout."""
    return base_dir if len(isovalues) <= 1 else os.path.join(base_dir, _isovalue_dirname(isovalue))


def isovalue_meta_analysis_root(isovalue, isovalues=ISOVALUES):
    return isovalue_subpath(META_ANALYSIS_ROOT, isovalue, isovalues)


def isovalue_meta_analysis_dir(isovalue, isovalues=ISOVALUES):
    return os.path.join(isovalue_meta_analysis_root(isovalue, isovalues), "across_genes")
