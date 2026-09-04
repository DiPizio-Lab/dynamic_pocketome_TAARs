# taar-pocketome

Pipeline for comparative pocket (pocketome) analysis of TAAR GPCR structures in
apo and holo states, extended with a **global pocket ID** scheme that makes
pockets comparable across MD replicates and across the apo/holo split.

Data (MD trajectories, raw pocket-search output, generated figures) is
published separately: **TODO — link to data repository / DOI**. This repo is
code only.

The pipeline has two required steps, plus an optional one-time migration and
an optional figures step:

0. **Step 0** *(optional, dataset-specific)* — normalize raw simulation
   output from wherever it was written into the uniform `input/` layout the
   pipeline assumes → `step0_prepare_clean_input.py`.
1. **Step 1** — turn each wrapped MD trajectory into per-experiment analysis
   results (RMSD/RMSF, pocket search) → `pipeline/analysis_pipeline.py`.
2. **Step 2** — parse and characterise every pocket (2.1), then assign
   **global pocket IDs** so the same pocket is comparable across MD
   replicates, genes and the apo/holo split, and compare apo vs. holo (2.2) →
   `pipeline/pocket_dataframes.py` + `pipeline/global_id_and_comparison.py`.
3. **Step 3** *(optional)* — publication figures and supporting analyses
   (replicate concordance, global-ID heatmap, Blender renders) →
   `taar_paper_figures/`.

`step1_preprocessing_pocket_detection.ipynb` and `step2_pocket_analysis.ipynb`
are documented, end-to-end walkthroughs of Steps 1 and 2 — the entry point for
anyone new to the codebase.

---

## Setup

```bash
conda env create -f environment.yml
conda activate taar-pocketome
```

Tools used by parts of the pipeline but not pip/conda-installable, not bundled
here:

| Tool | Used by | Install |
|---|---|--|
| **atclus** | `scripts/pocket_analysis.py` (pocket clustering); Fortran source + build artifacts vendored at `scripts/atclus/` | https://github.com/aachen1995/atclus-4 |
| **PyMOL** | opening the `.pml` scenes written by `taar_paper_figures/pymol_ambiguous_pockets.py` | https://pymol.org/ |
| **Blender + Molecular Nodes** | rendering the pocket figures from the PDBs written by `taar_paper_figures/blender_global_id_prep.py`, and the panel-D renders in `taar_paper_figures/blender/` | https://molecularnodes.org/ |

### Directory layout

```
taar-pocketome/
├── config.py               # PROJECT_ROOT / INPUT_DIR / OUTPUT_DIR / ... — single source of truth for paths
├── step0_prepare_clean_input.py   # Step 0 — one-time raw-data migration (see below)
├── step1_preprocessing_pocket_detection.ipynb   # Step 1 walkthrough
├── step2_pocket_analysis.ipynb                  # Step 2 walkthrough
├── input/                   # raw MD data — read-only, never written to by the pipeline
│   ├── apo_structures/<state><PDBID>/<replicate>/   # structure.pdb, structure.psf, traj_wrapped.xtc
│   └── holo_structures/<state><PDBID>/<replicate>/
├── output/                  # everything the pipeline writes
│   ├── apo_structures/<state><PDBID>/<replicate>/   # Step 1 per-experiment results, mirrors input/ 1:1
│   │                                                 #   (dry_prot.*, aligned_traj.*, RMSD/RMSF csvs, pockets/, ...)
│   ├── holo_structures/<state><PDBID>/<replicate>/
│   └── meta_analysis/       # everything that aggregates across PDB+rep: Step 2.1's across_genes/
│                             #   dataframes, Step 2.2's global_ID_*/ clustering runs and
│                             #   apo_holo_global_id_mapping/, Step 3's paper_figures/
├── pipeline/                 # Step 1 + Step 2 drivers and library — RMSD/RMSF & pocket-search
│                             #   orchestration, pocket-table parsing, global-ID clustering,
│                             #   binding-site & transiency classification, plotting
├── taar_paper_figures/       # Step 3 — publication figures, replicate concordance, global-ID
│                             #   heatmap/summary, PyMOL/Blender helpers (own README below)
├── scripts/                  # shared library of per-trajectory primitives (preprocessing,
│                             #   RMSD/RMSF, pocket search) used by Step 1, plus the vendored
│                             #   atclus/ pocket-clustering tool
└── reference_data/            # small lookup tables the pipeline needs to run (GPCRdb numbering,
                                #   gene-name map, binding-site/ICL3 residues)
```

`input/` is exclusively raw data — nothing under it is ever created or
modified by the pipeline. `output/apo_structures/` and
`output/holo_structures/` mirror `input/`'s PDB+rep tree exactly, one output
subtree per input subtree, and are populated by Step 1 as it runs. Anything
that aggregates across more than one PDB+rep pair — Step 2's pocket tables and
global-ID runs, Step 3's figures — lives under `output/meta_analysis/`
instead.

`config.py` resolves `PROJECT_ROOT` from the `TAAR_ROOT` environment variable,
falling back to the current working directory. **Run scripts from the repo
root**, or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient (e.g. submitting a job from elsewhere). Drop your own data into
`input/apo_structures/` and `input/holo_structures/` following the layout
above before running Step 1 (or normalize it there with Step 0 first).

---

## Step 0 — one-time raw-data migration → `step0_prepare_clean_input.py`

Dataset-specific, **not** part of the general pipeline: it is the step that
gets raw simulation output (written by two different upstream sources, at two
different raw frame frequencies, in three different file layouts) into the
uniform `input/apo_structures/`, `input/holo_structures/` shape every later
step assumes, so nothing downstream ever has to guess about simulation
settings again. Detection is pattern-based and reports every replicate's
outcome (`OK`/`SKIPPED`/`REVIEW`/`MISSING`/`ERROR`) to
`reference_data/input_data_settings_summary.csv`, flushed after each
replicate so an interrupted run leaves an accurate partial record. Re-running
it leaves already-migrated replicates untouched.

Only relevant if you're reproducing this dataset's migration from its
original sources; if you already have data in the `input/` layout, skip
straight to Step 1.

---

## Step 1 — per-experiment RMSD/RMSF & pocket search → `pipeline/analysis_pipeline.py`

For every replicate under `input/apo_structures/` and `input/holo_structures/`:

1. Pre-process the trajectory: dry the protein, align to the reference
   topology — `scripts/preprocessing.py`.
2. Calculate RMSD/RMSF — `scripts/basic_analysis.py`. RMSD is computed for the
   protein backbone (Cα), for the binding-site residues and their side
   chains, and for Cα excluding the flexible ICL3 loop — group selections are
   resolved per gene (hTAAR1, mTAAR1, mTAAR7f, mTAAR9) by
   `scripts/gene_selections.py` from `reference_data/binding_site_residues.txt`
   and `reference_data/ICL3_definition.txt`. As the implementation uses
   MDAnalysis, the user can decide what RMSD values should be calculated,
   following their documentation.
3. Run the MDpocket-based pocket search — `scripts/pocket_analysis.py`
   (dummy-atom clustering via the vendored `scripts/atclus/` tool).

Results are written to `output/apo_structures/` and `output/holo_structures/`,
mirroring the input PDB+rep tree (dry protein, aligned trajectory, RMSD/RMSF
CSVs, `pockets/`). This is the only stage that touches raw trajectories;
everything downstream reads its output.

---

## Step 2 — pocket dataframes, global pocket IDs & apo/holo comparison

### 2.1. Pocket dataframes → `pipeline/pocket_dataframes.py`

Turns Step 1's `pockets/` output (mdpocket/ATClus dummy-atom PDBs, descriptor
files, residue files) into per-pocket dataframes:

| Module | Purpose |
|---|---|
| `pocket_dataframes.py` | Main driver (`PocketFileParser` + `build_pocket_dataframes`): parses pocket files, interpolates volume across short closures, flags orthosteric/largest/transient pockets, derives `volume_category` |
| `orthosteric_filter_ligand_based.py` | Single source of truth for `is_binding_site` (see *Binding-site (orthosteric) definition* below) |
| `consecutive_zeros_transiency.py` | Single source of truth for the `transient` column (see *Transiency definition* below) |
| `pocket_io.py` | Shared CSV/Parquet I/O — every big table is written as both, read back through `load_table()` (parquet preferred) |
| `visualizations.py` | Plotting module (~30 functions), incl. `categorize_volume` (the single source of truth for size-category boundaries) |

```python
from pipeline.pocket_dataframes import build_pocket_dataframes, pocket_dirs_for

build_pocket_dataframes(pocket_dirs_for(), saving_loc='output/meta_analysis/across_genes')
```

Writes `all_pockets` (one row per pocket, frame, alpha sphere) and
`pocket_summary` (one row per pocket, first frame) as both `.csv` and
`.parquet`, plus `orthosteric_perframe_volumes.csv`, to
`output/meta_analysis/across_genes/` by default.

#### Binding-site (orthosteric) definition

Computed once by `orthosteric_filter_ligand_based.classify_binding_site`, this
is the only definition of "orthosteric" in the pipeline:

1. Per holo PDB, take the co-crystallized ligand's centroid
   (`load_ligand_centroids`). Apo structures have no ligand of their own (it's
   stripped out before the MD system is built), so an apo pocket is scored
   against its holo counterpart's centroid instead.
2. A pocket is binding-site if its own centroid (the mean x/y/z of all its
   dummy-atom/alpha-sphere coordinates, across every frame) is within 5 Å
   (`DEFAULT_DISTANCE_THRESHOLD`) of that ligand centroid.

Stored as `is_binding_site` in Step 2.2's `pocket_comparison_table.csv`, and
merged into `pocket_dataframes.py`'s own output (`all_pockets` /
`pocket_summary`) as `is_orthosteric` — the name every `taar_paper_figures/`
reader expects. Both names refer to the same underlying classification,
computed exactly once.

#### Transiency definition

Computed once by `consecutive_zeros_transiency.classify_transient`, the only
definition of "transient" in the pipeline. A pocket is `transient = True` if
`interpolated_pock_volume == 0`:

- for **at least 10% of its frames in total** (`n_zero_frames`,
  `MIN_ZERO_FRACTION`), **and**
- for **at least 50 of them consecutively** (`max_consecutive_zero_frames`,
  `MIN_CONSECUTIVE_ZERO_FRAMES`) — at 1 ns/frame, a run of ≥50 ns.

Both conditions must hold — this tells real, sustained pocket closures apart
from a handful of scattered zero-volume frames (noise). Consecutive-only
over-flags (a single long closure anywhere counts, however short the rest of
the trajectory); fraction-only under-flags a pocket that's scattered shut for
many individual frames but never closed for long.

### 2.2. Global pocket IDs & apo/holo comparison → `pipeline/global_id_and_comparison.py`

A **global pocket ID** is a dataset-scoped identifier that lets the same
pocket be tracked across MD replicates and across the apo/holo split, where
each run otherwise produces its own independent local numbering. A pocket's
global ID is only meaningful **within the run that produced it** — two
pockets can only share one if they were voxel-clustered together in the same
call, so IDs from two different runs are not directly comparable (unless
reconciled via `match_states()`, see below).

`run_global_id_states()` runs one or more clustering subsets in a single call
(`states=['apo', 'holo', 'both']`, any combination):

| Run | Subset clustered | Output |
|---|---|---|
| `'both'` (default) | apo + holo together, one shared numbering | `global_ID_combined/`; also runs the apo/holo comparison automatically |
| `'apo'` | apo only | `global_ID_apo/` |
| `'holo'` | holo only | `global_ID_holo/` |

```python
from pipeline.global_id_and_comparison import run_global_id_states
from pipeline import pocket_io
import config as conf

all_pockets = pocket_io.load_table(conf.META_ANALYSIS_DIR, 'all_pockets')
pocket_summary = pocket_io.load_table(conf.META_ANALYSIS_DIR, 'pocket_summary')
results = run_global_id_states(all_pockets, pocket_summary, states=['apo', 'holo', 'both'],
                                source_loc=conf.META_ANALYSIS_DIR, gid_root=conf.META_ANALYSIS_ROOT,
                                reconcile_apo_holo=True)
```

Clustering method: voxelize every pocket's alpha-sphere point cloud, dilate to
absorb small shifts, compute pairwise Intersection-over-Union, and take
connected components (IoU ≥ threshold) as global IDs. Writes, per run,
`pocket_comparison_table` (one row per local pocket, with its global ID,
gene/state/PDB uniqueness annotations, and the `is_orthosteric` / `transient`
/ `volume_category` columns inherited from Step 2.1), plus diagnostic/QC plots
(`pocket_clusters_qc.html`, per-PDB 3D scatter, upset/bar-chart gene-overlap
plots).

If `'apo'` and `'holo'` are both requested with `reconcile_apo_holo=True`,
`match_states()` additionally reconciles their independent numberings by
bidirectional nearest-centroid matching, writing
`apo_holo_global_id_mapping/global_id_state_mapping.csv`.

The `'both'` run also calls `run_apo_holo_comparison()`, writing
`apo_holo_pocketome_summary.csv`: per-PDB allosteric pocketome metrics (Δ
pocket count, Δ median volume, Jensen–Shannon distance of the size-class
profile, Δ size-class fractions, from `taar_paper_figures/pocketome_metrics.py`)
plus the orthosteric binding-site volume shift (from
`taar_paper_figures/fig2_binding_site.py`).

---

## Step 3 (optional) — publication figures → `taar_paper_figures/`

A self-contained figure package (RMSD heatmaps, binding-site, allosteric
pocketome, replicate concordance, global-ID heatmap, Blender panel-D renders)
with its own README (`taar_paper_figures/paper_figures_README.md`) covering
inputs/outputs, the colour system, and the Blender workflow in detail.

```bash
python taar_paper_figures/paper_plots.py
```

| Module | Purpose |
|---|---|
| `paper_plots.py` | Entry point for the three publication figures — one colour system, one row order, one entry point |
| `taar_style.py` | Shared palette, gene map loader, row order, panel-letter helpers — every figure module imports this |
| `pocketome_metrics.py` | Apo-vs-holo comparison maths (no plotting); also reused directly by `pipeline/global_id_and_comparison.py` |
| `fig1_rmsd_heatmaps.py` | Figure 1 — RMSD heatmaps + differential |
| `fig2_binding_site.py` | Figure 2 — orthosteric binding site |
| `fig2_panel_d_pockets.py` | Figure 2 panel D — renders + silhouette trace |
| `fig3_pocketome.py` | Figure 3 — allosteric pocketome |
| `global_id_heatmap.py` | Cross-experiment heatmap: rows = global ID, columns = gene → state → replicate, cell = median pocket volume |
| `summary_global_ids_paper.py` | One row per global ID over the combined apo+holo run |
| `gid_replicate_analysis_wrapper.py` | Replicate concordance: descriptive stats (counts, occupancy spectrum, Jaccard vs. permutation null, rarefaction), a silhouette check that global IDs are spatially real, and a threshold-sweep robustness check |
| `replicate_figures.py` | Renders the replicate-concordance figure (count collapse, occupancy, presence matrix, Jaccard heatmap, rarefaction) |
| `pymol_ambiguous_pockets.py` | Writes a PyMOL session per pocket whose occupancy class is unstable across the threshold sweep, for visual inspection |
| `blender_global_id_prep.py` | Picks, per global pocket, the MD frame closest to its median volume and exports it as a Molecular-Nodes-ready PDB |
| `calibrate_pocket_radii.py` | Solves the sphere radius that reproduces each pocket's reported volume, for the Blender render (raw alpha-sphere radii overstate pocket size ~3-5x) |
| `blender/export_pocket_coords.py`, `blender/set_pocket_colours.py`, `blender/set_render_background.py` | Run inside Blender: camera-projected pocket vertices for panel D, pocket material colours, transparent-film render settings |

Note the two-way coupling with `pipeline/`: `global_id_and_comparison.py`
imports `taar_style`, `pocketome_metrics` and `fig2_binding_site` from this
package (as `taar_paper_figures.*`), while several modules here import
`pocket_io` and `global_id_and_comparison` back from `pipeline/` via a
`sys.path` insert — see `paper_figures_README.md` for exact inputs/outputs of
each figure.

---

## `scripts/` — shared per-trajectory library

MD preprocessing (drying/aligning), RMSD/RMSF, and pocket search
(mdpocket/ATClus) — the classes Step 1 orchestrates. Actively imported by
Step 1 and by `taar_paper_figures/global_id_heatmap.py` — not archived, not
optional.

| Module | Purpose |
|---|---|
| `preprocessing.py` | `PreProcess` — dry the protein, align to reference topology |
| `basic_analysis.py` | `BasicAnalysis` — RMSD/RMSF calculation and plots |
| `pocket_analysis.py` | `PocketAnalysis` — MDpocket-based pocket search, ATClus dummy-atom clustering |
| `extract_iso.py` | Isosurface PDB extraction from mdpocket's density grid, used by `pocket_analysis.py` |
| `gene_selections.py` | PDB ID → gene lookup and per-gene RMSD group selections (binding site, ICL3) |
| `logging.py` | Trivial print-to-file logger used across the pipeline |
| `atclus/` | Vendored Fortran source + build artifacts for the ATClus pocket-clustering tool (see Setup) |

## `reference_data/`

- `TAARs_numbered/` — GPCRdb residue-numbering tables per receptor, used by `pipeline/visualizations.py` (`plot_largest_pockets`, via its `bw_csv_file` argument) to draw Ballesteros-Weinstein helix boundaries.
- `binding_site_residues.txt` / `ICL3_definition.txt` — per-gene residue numbers (+ BW numbering) used for Step 1's RMSD group selections (`scripts/gene_selections.py`).
- `hard_coded_gene_dict.txt` — PDB-ID → gene-name lookup, the single source of truth used by `scripts/gene_selections.py` and both `taar_style.py` helpers.
- `input_data_settings_summary.csv` — Step 0's per-replicate migration report (generated, not hand-maintained).

The first three are small, static reference tables required for the code to
run — not experimental output, so they're checked in rather than published
with the data.

---

## Known gaps

- `reference_data/binding_site_residues.txt` / `ICL3_definition.txt` don't
  have an entry for mTAAR9's TM5 BW-5.46 position — it's a one-residue
  deletion relative to the other three genes, so no equivalent resid exists.
- `taar_paper_figures/fig2_panel_d_pockets.py`'s `BLENDER_DIR` points at an
  absolute path outside this repo — configure it to your own Blender project
  location, same as the other external tools above.
- `pipeline/meta_analysis.sh` is a stale SLURM launcher for a `comp_with_sel.py`
  CLI wrapper that no longer exists in the pipeline (superseded by the
  `pipeline/pocket_dataframes.py` / `global_id_and_comparison.py` split
  above) — left as reference for the SLURM invocation shape, not runnable
  as-is.
