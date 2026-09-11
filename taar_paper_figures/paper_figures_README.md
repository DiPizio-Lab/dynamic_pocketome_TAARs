# TAAR paper figures

Publication figures for the TAAR pocketome paper. One colour system, one row order,
one entry point.

Run everything from `paper_plots.py` — comment in the figure you want to redraw.

```bash
python paper_plots.py
```

---

## What each figure says

| Figure | Scope | Message |
|---|---|---|
| **1** | Whole receptor | How much does the fold move? RMSD, apo vs holo |
| **2** | Orthosteric site | How big is the binding site, and how does the ligand change it? |
| **3** | Allosteric pocketome | Does ligand binding change the *number* and *size* of distal pockets? |

Figures 2 and 3 differ by exactly one filter: `is_orthosteric` True vs False. 

---

## Files in this package

### Entry point
| File | Role |
|---|---|
| `paper_plots.py` | The only file you run. Calls the three figure builders. |

### Shared modules
| File | Role |
|---|---|
| `taar_style.py` | Palette, gene map loader, row order, gene labels, diverging Δ bars, panel letters. Every figure imports this. |
| `pocketome_metrics.py` | Apo-vs-holo comparison maths. No plotting. |

### Figure builders
| File | Produces |
|---|---|
| `fig1_rmsd_heatmaps.py` | Figure 1 — RMSD heatmaps + differential |
| `fig2_binding_site.py` | Figure 2 — orthosteric binding site |
| `fig2_panel_d_pockets.py` | Figure 2 panel D — renders + silhouette trace (also standalone) |
| `fig3_pocketome.py` | Figure 3 — allosteric pocketome |

The Blender scripts that produce panel D's render and camera export are not included in this repo. 
The process on how to get to the exact visualisations of the paper is not described or shipped here.

---

## Inputs

**Put this folder inside `apo_holo_analysis/`.** `taar_style.ROOT` is derived from the
package location (`Path(__file__).parent.parent`), so no path editing is needed. To keep
the package elsewhere, set the `TAAR_ROOT` environment variable or hardcode `ROOT` in
`taar_style.py`.

Remaining paths are constants at the top of each module.

| Input | Used by | Notes |
|---|---|---|
| `hard_coded_gene_dict.txt` | `taar_style.py` | PDB ID → gene. Hand-curated |
| `meta_analysis/across_genes/pocket_summary.csv`/`.parquet` | `fig3_pocketome.py` | One row per pocket. Needs `median_pock_volume_open`, `volume_category`, `transient`, `is_orthosteric`. |
| `meta_analysis/across_genes/orthosteric_perframe_volumes.csv` | `fig2_binding_site.py` | Per-frame orthosteric volumes. ID format `apo8ITF_2_p19_i3`. |
| RMSD median CSV | `fig1_rmsd_heatmaps.py` | As configured inside that script. |
| `binding_site_8JLR_hTAAR1_{side,top}view.png` | `fig2_binding_site.py` | Panel C. Autocropped on load. |
| `pocket_camera_coords.npz` | `fig2_panel_d_pockets.py` | Camera-projected pocket vertices, written by the (excluded) Blender export script -- see Panel D workflow below. Keys are the blender object names; styles are derived from them, so reselecting pockets needs no code edits. |
| `blender_apo_holo_smallest_largest_pockets_nobg.png` | `fig2_panel_d_pockets.py` | Panel D render. The RGBA (no-background) export is preferred and composited onto white; an opaque render is autocropped instead. Optional -- `load_render()` degrades quietly (panel D just omits the render half) if missing. |

**Upstream, not in this package:** `pipeline/pocket_dataframes.py` builds `pocket_summary`.
It derives `volume_category` from the per-pocket **trajectory median** volume, not the
first frame. If figure 3 fails on a missing `median_pock_volume_open` column, that
pipeline needs rerunning.

---

## Outputs

Written to `analysis_results/paper_figures/` at 600 dpi:

```
figure1_rmsd_combined.png
figure2_binding_site.png
figure3_allosteric_pocketome.png
panel_d_pocket_silhouettes.png
```



---

## The colour system

One channel per variable. Never reuse a channel.

| Variable | Channel | Values                                    |
|---|---|-------------------------------------------|
| Gene | **Position + label**, black text | No colour spent                           |
| State | **Colour** | apo `#D9B98B` sand · holo `#9B5560` mauve |
| Volume (continuous) | **viridis** | RMSD and pocket volume                    |
| Size category | **viridis samples** | 0.12 / 0.40 / 0.68 / 0.92                 |
| Transiency | **Texture** | stable solid · transient hatched          |

The Blender materials use the same state colours, so the renders and the plots agree.

---

## Metrics (`pocketome_metrics.py`)

| Function | Answers |
|---|---|
| `delta_pocket_count` | Does the **number** of pockets change? → figure 3 panel C |
| `delta_category_fraction` | **Which size classes** gained or lost? → figure 3 panel D |
| `delta_median_volume` | Does the **typical pocket size** change? (median of per-pocket medians — not a sum, so it stays independent of pocket count) |
| `js_distance` / `js_per_structure` | How far did the whole size profile shift, as one number? **Not plotted** — the values span only ~0.07–0.35 and carry no direction, so they read as flat. Use in the results text. |

---

## Conventions worth keeping

Things that silently break these layouts, learned the hard way:

- **Axes-level legends reserve horizontal space** and blow open the gap between panels.
  Use `fig.legend(..., loc="outside lower center")`.
- **`sharey` propagates tick labels to every panel**, and `labelleft=False` only affects
  *major* ticks. Use `which="both"`.
- **Panel letters drift** unless offset in *points* from each axes corner
  (`ts.panel_letter`), not in axes fractions.
- **One `TITLE_PAD` per figure** so titles align by construction.
- Row height below ~0.21 in makes the `a`/`h` state labels collide.


---

## Turning panels off

Each figure module has a `PANELS` dict at the top. Set any entry to `False` and the panel
is dropped — widths, row labels, gene labels, panel letters and the legend all
re-flow automatically, and the leftmost surviving panel picks up the PDB rows.

```python
# fig3_pocketome.py
PANELS = {"size": True, "stability": False, "delta_count": True, "delta_class": True}
```

| Module | Keys |
|---|---|
| `fig1_rmsd_heatmaps.py` | `heat_ca`, `heat_bs`, `colourbar`, `diff_ca`, `diff_bs` |
| `fig2_binding_site.py` | `volume`, `delta`, `site`, `pockets` |
| `fig3_pocketome.py` | `size`, `stability`, `delta_count`, `delta_class` |

---

## Transiency

`true_transient` as stored in `pocket_analysis_summary.csv` is
`max_consecutive_zero_frames >= 50` **alone** — the 10%-of-frames condition was never
combined in, which is why it flags ~74% of pockets. `fig3_pocketome.py` therefore derives
transiency itself:

```
transient = (>= 10% of frames at zero) AND (max_consecutive_zero_frames >= 50)
```

Both columns are already in the summary table, so no pipeline rerun is needed. Set
`DERIVE_TRANSIENCY = False` to fall back to the stored column.


---

## Panel D workflow

The render and the silhouettes must come from the **same camera**. Steps 2 and 4 need the
Blender scripts, which are not included in this repo:

1. Frame the camera in Blender.
2. Set transparent film, RGBA output, `Standard` view transform.
3. Render → save as `*_nobg.png`.
4. Export pocket surface vertices through the camera → `pocket_camera_coords.npz`
   (+ `camera_pose.json`, to reproduce the framing later).
5. Place both files under `fig2_panel_d_pockets.BLENDER_DIR` (default:
   `blender_renders/` next to that script), then
   `python -c "import fig2_panel_d_pockets as d; d.panel_d()"` to check panel D alone,
   or `fig2.figure2()` for the whole figure.

Steps 3 and 4 must use the same camera — if the camera moves, redo both. Material or
colour changes do **not** require a re-export; only camera moves and geometry changes do.
