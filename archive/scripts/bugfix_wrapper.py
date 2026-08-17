"""
MASTER PIPELINE WRAPPER - Complete Pocket Analysis Chain

Runs the following analysis steps in sequence:
1. Complete pocket analysis (all visualizations)
2. Max consecutive zero frames analysis (60GB file chunked reading)
3. Mapping consecutive zeros to summary with true_transient flag
4. Poster-quality visualizations (pastel pie charts)

Author: Clarissa Rienaecker
Date: 2025-01-16
"""

import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path


class PocketAnalysisPipeline:
    """Orchestrate complete pocket analysis workflow."""

    def __init__(self, saving_loc: str, holo_base: str, verbose: bool = True):
        """
        Initialize pipeline.

        Parameters
        ----------
        saving_loc : str
            Main results directory (e.g., /path/to/across_genes)
        holo_base : str
            Path to holo structures directory
        verbose : bool
            Print progress information
        """
        self.saving_loc = saving_loc
        self.holo_base = holo_base
        self.verbose = verbose
        self.results = {}

    def log(self, message: str, level: str = "INFO"):
        """Print timestamped log message."""
        if self.verbose:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def step_1_complete_pocket_analysis(self):
        """Run complete pocket analysis with all visualizations."""
        self.log("=" * 80)
        self.log("STEP 1: COMPLETE POCKET ANALYSIS WITH VISUALIZATIONS")
        self.log("=" * 80)

        try:
            from complete_pocket_analysis import run_complete_pocket_analysis

            t0 = time.time()
            self.log("Starting complete pocket analysis...")

            results = run_complete_pocket_analysis(
                saving_loc=self.saving_loc,
                data_path=os.path.join(self.saving_loc, 'summary_df_3d_coords.csv'),
                holo_base=self.holo_base,
                distance_threshold=5.0,
                volume_col='interpolated_pock_volume',
                bw_file_loc=os.path.join(os.getcwd(), 'TAARs_numbered'),
                pdb_file_loc=None,
                export_pdfs=False,
                n_cols=4,
                greyscale=False,
                run_3d_heatmap=False,
                run_apo_holo_comparison=False,
                run_violin_plots=False,
                run_consecutive_zeros=True)

            t1 = time.time()
            self.log(f"✓ Complete pocket analysis finished in {(t1 - t0) / 60:.2f} minutes")
            self.results['step_1'] = results
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def step_2_consecutive_zeros_analysis(self):
        """Calculate max consecutive zero frames on 60GB file (chunked)."""
        self.log("=" * 80)
        self.log("STEP 2: MAX CONSECUTIVE ZERO FRAMES ANALYSIS (CHUNKED 60GB FILE)")
        self.log("=" * 80)

        try:
            from consecutive_zeros_transiency import add_max_consecutive_zeros_chunked

            t0 = time.time()
            data_path = os.path.join(self.saving_loc, 'summary_df_3d_coords.csv')
            output_path = os.path.join(self.saving_loc, 'summary_df_3d_coords_cont_zero.csv')

            self.log(f"Input: {data_path}")
            self.log(f"Output: {output_path}")
            self.log("Starting chunked analysis (this may take 15-30 minutes)...")

            add_max_consecutive_zeros_chunked(
                data_path=data_path,
                output_path=output_path,
                volume_col='interpolated_pock_volume',
                chunksize=100000)

            t1 = time.time()
            self.log(f"✓ Consecutive zeros analysis finished in {(t1 - t0) / 60:.2f} minutes")
            self.results['step_2'] = {'output_path': output_path}
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def step_3_map_consecutive_zeros_to_summary(self):
        """Map consecutive zeros from 60GB file to pocket_analysis_summary.csv."""
        self.log("=" * 80)
        self.log("STEP 3: MAPPING CONSECUTIVE ZEROS TO SUMMARY & CREATING TRUE_TRANSIENT")
        self.log("=" * 80)

        try:
            from helper_new_transient import (
                create_max_zeros_dict_from_large_file,
                add_transient_columns_to_summary)

            t0 = time.time()
            large_file_path = os.path.join(self.saving_loc, 'summary_df_3d_coords_cont_zero.csv')
            summary_path = os.path.join(self.saving_loc, 'pocket_analysis_summary.csv')

            self.log(f"Large file: {large_file_path}")
            self.log(f"Summary file: {summary_path}")
            self.log("Creating dictionary from large file...")

            max_zero_dict = create_max_zeros_dict_from_large_file(large_file_path, chunksize=100000)

            self.log(f"Mapping {len(max_zero_dict):,} pockets to summary...")
            summary_df = add_transient_columns_to_summary(
                summary_path=summary_path,
                max_zero_dict=max_zero_dict,
                output_path=summary_path,  # Overwrite original
                id_column='ID')

            t1 = time.time()
            n_true_transient = (summary_df['true_transient'] == True).sum()
            self.log(f"✓ Mapping complete in {(t1 - t0) / 60:.2f} minutes")
            self.log(f"  Total pockets: {len(summary_df):,}")
            self.log(f"  True transient (≥50 frames): {n_true_transient:,}")
            self.results['step_3'] = {'summary_df': summary_df, 'true_transient_count': n_true_transient}
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def step_4_poster_visualizations(self):
        """Generate poster-quality charts (pastel pie charts)."""
        self.log("=" * 80)
        self.log("STEP 4: POSTER VISUALIZATIONS (PASTEL CHARTS)")
        self.log("=" * 80)

        try:
            from pastel_pie_chart import pocket_analysis_summary, plot_stability_bar_chart

            t0 = time.time()
            summary_path = os.path.join(self.saving_loc, 'pocket_analysis_summary.csv')
            df = pd.read_csv(summary_path)

            self.log(f"Loaded {len(df):,} pockets from {summary_path}")

            # Define color palette
            colors = {
                'Small (<250)': '#8FA8C9',
                'Medium (250-500)': '#8FBFA6',
                'Large (500-750)': '#9A8FBF',
                'Very Large (>750)': '#D49BA8'
            }

            # 1. Pocket volume distribution pie chart
            self.log("Generating pocket volume distribution pie chart...")
            pie_output = os.path.join(self.saving_loc, 'pocket_volume_distribution_pastel.png')
            pocket_analysis_summary(
                df=df,
                title="Overall Pocket Volume Distribution",
                output_path=pie_output,
                figsize=(12, 9),
                dpi=300)
            self.log(f"  Saved: {pie_output}")

            # 2. Stability by volume category bar chart
            self.log("Generating stability by volume category bar chart...")
            bar_output = os.path.join(self.saving_loc, 'stability_volcategories_pastel.png')
            plot_stability_bar_chart(df, colors, output_path=bar_output)
            self.log(f"  Saved: {bar_output}")

            t1 = time.time()
            self.log(f"✓ Poster visualizations finished in {(t1 - t0) / 60:.2f} minutes")
            self.results['step_4'] = {'pie_chart': pie_output, 'bar_chart': bar_output}
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def step_5_pocket_volume_timeseries(self):
        """Generate pocket volume over time plots."""
        self.log("=" * 80)
        self.log("STEP 5: POCKET VOLUME TIME SERIES PLOTS")
        self.log("=" * 80)

        try:
            from pocket_volume_time_series import plot_pocket_volumes_over_time

            t0 = time.time()
            input_csv = os.path.join(self.saving_loc, 'summary_df_3d_coords_cont_zero.csv')

            self.log(f"Input: {input_csv}")
            self.log("Generating individual pocket volume trajectories...")

            plot_count = plot_pocket_volumes_over_time(
                input_csv=input_csv,
                chunk_size=50000,
                verbose=self.verbose,
                dpi_output=300)

            t1 = time.time()
            self.log(f"✓ Time series plots finished in {(t1 - t0) / 60:.2f} minutes")
            self.log(f"  Generated {plot_count} plots")
            self.results['step_5'] = {'plot_count': plot_count}
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def step_6_quick_all_pocket_plots(self):
        """Generate quick all-pockets summary plots."""
        self.log("=" * 80)
        self.log("STEP 6: ALL-POCKETS SUMMARY PLOTS")
        self.log("=" * 80)

        try:
            from quick_all_pocket_plots import replot_summary_for_all_pockets

            t0 = time.time()
            self.log(f"Regenerating all-pockets plots from summary...")

            replot_summary_for_all_pockets(self.saving_loc)

            t1 = time.time()
            self.log(f"✓ All-pockets plots finished in {(t1 - t0) / 60:.2f} minutes")
            self.results['step_6'] = {'status': 'complete'}
            return True

        except Exception as e:
            self.log(f"✗ FAILED: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def run_full_pipeline(self):
        """Execute all analysis steps in sequence."""
        pipeline_start = time.time()

        self.log("=" * 80)
        self.log("STARTING COMPLETE POCKET ANALYSIS PIPELINE")
        self.log("=" * 80)
        self.log(f"Results directory: {self.saving_loc}")
        self.log(f"Holo base: {self.holo_base}")

        steps = [
            ("Step 1: Complete Pocket Analysis", self.step_1_complete_pocket_analysis),
            ("Step 2: Consecutive Zeros Analysis", self.step_2_consecutive_zeros_analysis),
            ("Step 3: Map Consecutive Zeros", self.step_3_map_consecutive_zeros_to_summary),
            ("Step 4: Poster Visualizations", self.step_4_poster_visualizations),
            ("Step 5: Volume Time Series", self.step_5_pocket_volume_timeseries),
            ("Step 6: All-Pockets Plots", self.step_6_quick_all_pocket_plots),
        ]

        results_summary = {}

        for step_name, step_func in steps:
            self.log("")
            success = step_func()
            results_summary[step_name] = "✓ SUCCESS" if success else "✗ FAILED"

            if not success:
                self.log(f"Pipeline halted at {step_name}", level="WARNING")
                break

        pipeline_end = time.time()

        # Print summary
        self.log("=" * 80)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 80)
        for step, result in results_summary.items():
            self.log(f"{result} - {step}")

        total_time = (pipeline_end - pipeline_start) / 60
        self.log(f"\nTotal pipeline time: {total_time:.2f} minutes ({total_time / 60:.2f} hours)")
        self.log("=" * 80)

        return all("✓ SUCCESS" in v for v in results_summary.values())


# =============================================================================

if __name__ == '__main__':
    # Configuration
    saving_loc = '/mnt/mmlab_shared/RienaeckerC/example_case_global_ID/meta_analysis/across_genes'
    holo_base = '/mnt/mmlab_shared/RienaeckerC/example_case_global_ID/holo_structures/input'

    # Initialize and run pipeline
    pipeline = PocketAnalysisPipeline(saving_loc=saving_loc, holo_base=holo_base, verbose=True)
    success = pipeline.run_full_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)