# taar-pocketome

Pipeline for comparative pocket (pocketome) analysis of TAAR GPCR structures in
apo and holo states, extended with a **global pocket ID** scheme that makes
pockets comparable across MD replicates and across the apo/holo split.

This repo is a self-contained continuation of
[`comparative_study_apo_holo_TAARs`](https://github.com/CRienaecker/comparative_study_apo_holo_TAARs),
which covers the original apo/holo pocketome comparison. It ships its own
(updated) copy of that pipeline rather than depending on the other repo, so it
can be run and reviewed on its own. **It does not share git history with that
repo** — the two were developed as separate lines of work and are published as
separate repos, one per paper/chapter.

Data (MD trajectories, raw pocket-search output, generated figures) is
published separately: **TODO — link to data repository / DOI**. This repo is
code only.

---

## Setup

```bash
conda env create -f environment.yml
conda activate taar-pocketome
```

Three external tools are used by parts of the pipeline but are not
pip/conda-installable and are not bundled here:

| Tool | Used by | Install |
|---|---|---|
| **atclus** | `scripts/pocket_analysis.py` (pocket clustering) | external GitHub repo — TODO: link |
| **PyMOL** | opening the `.pml` scenes written by `global_id/pymol_ambiguous_pockets.py` | https://pymol.org/ |
| **Blender + Molecular Nodes** | rendering the pocket figures from the PDBs written by `global_id/blender_global_id_prep.py` | https://molecularnodes.org/ |

### Directory layout

```
taar-pocketome/
├── config.py            # PROJECT_ROOT / INPUT_DIR / OUTPUT_DIR — single source of truth for paths
├── input/                # put your data here (apo_structures/, holo_structures/, ...) — empty in this repo
├── output/               # pipeline results and figures get written here — empty in this repo
├── notebooks/            # documented walkthrough
├── pipeline/             # Stage 1 + Stage 2 — parsing, clustering, orthosteric classification, plots
├── global_id/            # Stage 3 — global pocket IDs, apo/holo matching, replicate concordance, paper figures
├── scripts/              # shared library (config, preprocessing, RMSD/RMSF, pocket search, QC, BW numbering)
├── reference_data/        # small lookup tables the pipeline needs to run (GPCRdb numbering, gene-name map)
└── archive/scripts/       # superseded/one-off scripts, kept for reference only
```

`config.py` resolves `PROJECT_ROOT` from the `TAAR_ROOT` environment variable,
falling back to the current working directory. **Run scripts from the repo
root**, or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient (e.g. submitting a job from elsewhere). Drop your own data into
`input/` following the layout referenced by `scripts/config.py`
(`apo_structures/`, `holo_structures/`) before running Stage 1.

---

## Stage 1 — parsing & global-ID clustering → `pipeline/`

| Script | Purpose |
|---|---|
| `comparative_study.py` | Main driver: parsing, within-PDB clustering, orthosteric classification |
| `meta_analysis_class.py` | Core `MetaAnalysis` class (parsing, voxel-IoU clustering) |
| `comp_with_sel.py` | CLI wrapper (optional; SLURM-friendly) |

```bash
python pipeline/comp_with_sel.py --across-genes
```

## Stage 2 — post-processing & visualizations → `pipeline/`

| Script | Purpose |
|---|---|
| `complete_pocket_analysis.py` | Main analysis entry point + calls all visualizations |
| `orthosteric_filter_ligand_based.py` | Ligand-based orthosteric detection (5 Å from ligand centroid) — single source of truth for `is_orthosteric` |
| `consecutive_zeros_transiency.py` | `max_consecutive_zero_frames` per pocket, to separate real transiency from noise |
| `visualizations.py` | Plotting module (~30 functions) |

```python
from complete_pocket_analysis import run_complete_pocket_analysis

results = run_complete_pocket_analysis(
    saving_loc='output/meta_analysis/across_genes',
    holo_base='input/holo_structures/input',
    distance_threshold=5.0,
    export_pdfs=True,
    run_3d_heatmap=True,
    run_apo_holo_comparison=True,
    run_violin_plots=True,
)
```

## Stage 3 — global pocket IDs → `global_id/`

A **global pocket ID** is a dataset-scoped identifier that lets the same
pocket be tracked across MD replicates and across the apo/holo split, where
each run otherwise produces its own independent local numbering.

| Script | Purpose |
|---|---|
| `global_id_wrapper_paper.py` | Runs the Stage-1 driver body per state (apo/holo/both/combined), producing a non-colliding result set per run |
| `gid_state_mapping.py` | Matches apo global IDs onto holo global IDs by bidirectional nearest-centroid matching (both states are already superposed into one coordinate frame) |
| `gid_replicate_analysis_wrapper.py` | Replicate concordance: descriptive stats (counts, occupancy spectrum, Jaccard vs. permutation null, rarefaction), a silhouette check that global IDs are spatially real, and a threshold-sweep robustness check |
| `replicate_figures.py` | Renders the replicate-concordance figure (count collapse, occupancy, presence matrix, Jaccard heatmap, rarefaction) |
| `global_id_heatmap.py` | Cross-experiment heatmap: rows = global ID, columns = gene → state → replicate, cell = median pocket volume |
| `summary_global_ids_paper.py` | One row per global ID over the combined apo+holo run |
| `pymol_ambiguous_pockets.py` | Writes a PyMOL session per pocket whose occupancy class is unstable across the threshold sweep, for visual inspection |
| `blender_global_id_prep.py` | Picks, per global pocket, the MD frame closest to its median volume and exports it as a Molecular-Nodes-ready PDB |
| `calibrate_pocket_radii.py` | Solves the sphere radius that reproduces each pocket's reported volume, for the Blender render (raw alpha-sphere radii overstate pocket size ~3-5x) |
| `taar_style.py` | Shared palette, gene order, and style helpers reused across the Stage-3 figures |

```bash
python global_id/global_id_wrapper_paper.py --state apo
python global_id/global_id_wrapper_paper.py --state holo
python global_id/gid_state_mapping.py
python global_id/gid_replicate_analysis_wrapper.py
python global_id/replicate_figures.py
```

---

## `scripts/` — shared library

Config/path helpers, MD preprocessing (RMSD/RMSF/wrapping), pocket search
(mdpocket/ATClus) drivers, QC, and Ballesteros-Weinstein residue numbering.
Actively imported by both Stage 1/2 and Stage 3 — not archived, not optional.

## `reference_data/`

- `TAARs_numbered/` — GPCRdb residue-numbering tables per receptor, used by `scripts/bw_mapping.py` / `scripts/TAARs_BW.py`.
- `hard_coded_gene_dict.txt` — PDB-ID → gene-name lookup used by `global_id/taar_style.py`.

Both are small, static reference tables required for the code to run — not
experimental output, so they're checked in rather than published with the
data.

## `archive/scripts/`

One-off, superseded, or poster/QA utilities, kept for reference only. Nothing
in `pipeline/` or `global_id/` imports anything from here, and these scripts
still contain the original hardcoded cluster paths (not rewritten, since
they're not meant to run as-is).

---

## Notebook

`notebooks/pipeline_walkthrough.ipynb` is a documented, end-to-end tutorial
through Stages 1-3, meant as the entry point for anyone new to the codebase.
