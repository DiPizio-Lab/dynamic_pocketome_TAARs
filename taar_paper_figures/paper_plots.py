"""TAAR paper figures - single entry point.

Comment in the figure(s) you want to redraw, then run this file.
Everything shares one colour system and one row order via taar_style.

    python paper_plots.py

Upstream of this package: pipeline/pocket_dataframes.py writes pocket_summary
(csv+parquet) including the trajectory-median volume columns that figure 3 reads.
If those columns are missing, rerun that pipeline first."""
import time

import taar_style as ts                # palette, gene map, row order, Delta bars
import pocketome_metrics as pm         # JS distance, Delta count / volume / size class
import fig1_rmsd_heatmaps as fig1      # figure 1: RMSD
import fig2_binding_site as fig2       # figure 2: orthosteric binding site
import fig2_panel_d_pockets as fig2d   # figure 2 panel D: renders + silhouettes
import fig3_pocketome as fig3          # figure 3: allosteric pocketome


def main():
    # ---- figure 1: RMSD ---------------------------------------------------
    fig1.combined_plot(value_annot=False, replicates=True)
    # fig1.heatmap_plots(value_annot=False)     # the two standalone heatmaps
    # fig1.differential_plot()                  # the standalone differential
    # fig1.csv_summary_rmsd()                   # re-aggregate the RMSD medians

    # ---- figure 2: orthosteric binding site --------------------------------
    fig2.figure2()
    # fig2d.panel_d()                           # panel D alone, for checking

    # ---- figure 3: allosteric pocketome ------------------------------------
    fig3.figure3()


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Figures done in {time.time() - started:.1f} s")