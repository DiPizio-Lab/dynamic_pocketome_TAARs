"""
Pocket volume-over-time plotting function
Generates line plots for each pocket showing volume across all frames.
Uses global maximum volume for all Y-axes (same scale across all plots),
if the local max volume is below 30% of the global max volume the y axis is broken
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys

angstrom_sign = "r’$\AA$’"
cubic_angstrom_sign = "r'($\AA^3$)'"

def create_broken_axis_plot(frames, volumes, pocket_num, state, pdb_id, rep,
                            global_max_volume, broken_axis_threshold=0.30,
                            dpi_output=300):
    """
    Create a plot with broken Y-axis for small pockets.

    If pocket_max_volume < broken_axis_threshold * global_max_volume:
    - Bottom subplot: zoomed view of 0 to ~40% of global_max
    - Top subplot: compressed view of 40% to global_max
    - Diagonal break lines indicate discontinuity

    Parameters
    ----------
    frames : list
        Frame numbers
    volumes : list
        Volume values
    pocket_num : int
        Pocket number identifier
    state : str
        apo or holo
    pdb_id : str
        PDB ID
    rep : int
        Replicate number
    global_max_volume : float
        Global maximum volume across all pockets
    broken_axis_threshold : float
        Threshold fraction (default 0.30 = 30%)
    dpi_output : int
        Output DPI resolution

    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """

    pocket_max_volume = max(volumes)
    PLOT_COLOR = '#00008B'  # midnightblue
    GRID_COLOR = '#808080'  # grey
    BACKGROUND_COLOR = 'white'
    BREAK_COLOR = 'black'

    # Create figure with gridspec for two subplots
    fig = plt.figure(figsize=(10, 8), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)

    # Top subplot = upper portion
    ax_top = fig.add_subplot(gs[0])
    # Bottom subplot = zoomed, lower portion
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    # Break point: show detail up to ~40% of global max, compress rest
    break_threshold = global_max_volume * 0.40

    # Plot data on both axes
    ax_bottom.plot(frames, volumes, color=PLOT_COLOR, linewidth=2, zorder=3)
    ax_top.plot(frames, volumes, color=PLOT_COLOR, linewidth=2, zorder=3)

    ax_bottom.set_ylim(0, break_threshold)
    ax_top.set_ylim(break_threshold, global_max_volume)
    ax_bottom.set_xlim(0, 1000)

    for ax in [ax_top, ax_bottom]:
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.patch.set_facecolor(BACKGROUND_COLOR)

    ax_bottom.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax_bottom.set_ylabel(f'Volume {cubic_angstrom_sign}', fontsize=11, fontweight='bold')
    ax_top.set_ylabel(f'Volume {cubic_angstrom_sign}', fontsize=11, fontweight='bold')

    ax_top.tick_params(labelbottom=False)

    # Add break indicators (diagonal lines) - get figure coordinates for the break line
    fig.canvas.draw()

    line_width = 0.015
    line_color = BREAK_COLOR

    rect_left = Rectangle((0.02, 0.475), line_width, line_width * 2,
                          transform=fig.transFigure, color=line_color, zorder=10)
    fig.add_artist(rect_left)

    rect_right = Rectangle((0.96, 0.475), line_width, line_width * 2,
                           transform=fig.transFigure, color=line_color, zorder=10)
    fig.add_artist(rect_right)

    # Add "break" symbol
    ax_top.text(0.02, break_threshold * 0.5, '⁝⁝', fontsize=16,
                color=line_color, fontweight='bold', transform=ax_top.transData,
                ha='left', va='center')

    # Title
    fig.suptitle(
        f'Pocket {pocket_num} Volume Trajectory \n{state.upper()} {pdb_id} Rep{rep}\nMax volume: {pocket_max_volume:.2f}',
        fontsize=12, y=0.98)

    return fig


def create_standard_plot(frames, volumes, pocket_num, state, pdb_id, rep,
                         global_max_volume, dpi_output=300):
    """
    Create a standard single-axis plot for larger pockets.

    Parameters
    ----------
    frames : list
        Frame numbers
    volumes : list
        Volume values
    pocket_num : int
        Pocket number identifier
    state : str
        apo or holo
    pdb_id : str
        PDB ID
    rep : int
        Replicate number
    global_max_volume : float
        Global maximum volume across all pockets
    dpi_output : int
        Output DPI resolution

    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """

    PLOT_COLOR = '#00008B'
    GRID_COLOR = '#808080'
    BACKGROUND_COLOR = 'white'

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(frames, volumes, color=PLOT_COLOR, linewidth=2, zorder=3)

    ax.set_facecolor(BACKGROUND_COLOR)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, global_max_volume)

    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Volume {cubic_angstrom_sign}', fontsize=11, fontweight='bold')
    ax.set_title(f'Pocket {pocket_num} Volume Trajectory\n{state.upper()} {pdb_id} Rep{rep}',
                 fontsize=12)

    return fig


def plot_pocket_volumes_over_time(
        input_csv: str = "/mnt/mmlab_shared/RienaeckerC/apo_holo_analysis/meta_analysis/across_genes/summary_df_3d_coords.csv",
        chunk_size: int = 50000,
        verbose: bool = True,
        dpi_output: int = 300):
    """
    Generate volume-over-time plots for each individual pocket.

    Processes large CSV in chunks, groups by state+pdb_id+replicate, and creates
    individual plots for each pocket with global Y-axis scaling (all plots use
    the same maximum volume across all pockets).

    Parameters
    ----------
    input_csv : str
        Path to the 60GB summary CSV file containing all pocket coordinates
    chunk_size : int
        Rows per chunk for memory-efficient reading (default 50000)
    verbose : bool
        Print progress information
    dpi_output : int
        DPI for output PNG files (default 300 for publication quality)

    Output
    ------
    Creates /volume subdirectories in:
    /mnt/mmlab_shared/RienaeckerC/apo_holo_analysis/{state}_structures/results/{pdb_id}/{rep}/plots/volume/

    One PNG file per pocket named: pocket_{pocket_number}.png
    All plots share same Y-axis: 0 to global_max_volume
    """

    # Dictionary to store data for each group (state, pdb_id, rep)
    # Structure: {(state, pdb_id, rep): {pocket_number: {frame: volume}}}
    group_pocket_data = defaultdict(lambda: defaultdict(dict))

    # First pass: collect pocket volume data grouped by state+pdb_id+rep
    if verbose:
        print(f"Reading and processing {input_csv}...")
        print(f"Using chunk size: {chunk_size} rows")
        print("(Pass 1: Collecting all pocket volume data...)\n")

    try:
        chunk_count = 0
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
            chunk_count += 1
            if verbose and chunk_count % 10 == 0:
                print(f"Processed {chunk_count * chunk_size} rows...")

            # Ensure required columns exist
            required_cols = {'prj', 'rep', 'snapshot', 'pocket_number', 'interpolated_pock_volume'}
            available_cols = set(chunk.columns)

            # Parse state and pdb_id from prj column (e.g., "apo8ITF" -> state="apo", pdb_id="8ITF")
            chunk['state'] = chunk['prj'].apply(lambda x: 'apo' if str(x).startswith('apo') else 'holo')
            chunk['pdb_id'] = chunk['prj'].apply(lambda x: str(x).replace('apo', '').replace('holo', ''))

            if not required_cols.issubset(available_cols):
                missing = required_cols - available_cols
                raise ValueError(f"Missing columns: {missing}\nAvailable: {sorted(available_cols)}")

            # Group by state, pdb_id, rep and collect volumes
            for (state, pdb_id, rep), group in chunk.groupby(['state', 'pdb_id', 'rep']):
                key = (state, pdb_id, rep)

                # Collect volume data by pocket and frame
                for _, row in group.iterrows():
                    pocket_num = row['pocket_number']
                    frame = int(row['snapshot'])
                    volume = float(row['interpolated_pock_volume'])
                    group_pocket_data[key][pocket_num][frame] = volume

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing column {e}")
        print(f"Available columns: {chunk.columns.tolist()}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)

    if verbose:
        num_groups = len(group_pocket_data)
        total_pockets = sum(len(pockets) for pockets in group_pocket_data.values())
        print(f"\nData collection complete:")
        print(f"  {num_groups} state+pdb_id+replicate groups")
        print(f"  {total_pockets} unique pockets total")

    # Calculate GLOBAL maximum volume across all pockets
    if verbose:
        print("\n(Pass 2: Finding global maximum volume...)")

    all_volumes = []
    for pocket_dict in group_pocket_data.values():
        for frame_dict in pocket_dict.values():
            all_volumes.extend(frame_dict.values())

    global_max_volume = max(all_volumes) if all_volumes else 1.0

    if verbose:
        print(f"Global maximum volume: {global_max_volume:.2f}\n")
        print("(Pass 3: Generating plots...)\n")

    # Generate plots for each group
    total_plots = sum(len(pockets) for pockets in group_pocket_data.values())
    plot_count = 0

    for group_idx, (key, pocket_dict) in enumerate(group_pocket_data.items(), 1):
        state, pdb_id, rep = key

        # Set up output directory
        output_dir = Path(
            f"/mnt/mmlab_shared/RienaeckerC/apo_holo_analysis/"
            f"{state}_structures/results/{state}{pdb_id}/{rep}/plots/volume")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR creating directory {output_dir}: {e}")
            continue

        if verbose:
            print(f"[{group_idx}/{len(group_pocket_data)}] {state}/{pdb_id}/{rep}: {len(pocket_dict)} pockets")

        # Generate plot for each pocket
        for pocket_num, frame_dict in pocket_dict.items():
            frames = sorted(frame_dict.keys())
            volumes = [frame_dict[f] for f in frames]
            pocket_max_volume = max(volumes)

            # Decide whether to use broken axis (threshold: 30% of global max)
            use_broken_axis = pocket_max_volume < (0.30 * global_max_volume)

            if use_broken_axis:
                fig = create_broken_axis_plot(frames, volumes, pocket_num, state, pdb_id, rep,
                                              global_max_volume, broken_axis_threshold=0.30,
                                              dpi_output=dpi_output)
            else:
                fig = create_standard_plot(frames, volumes, pocket_num, state, pdb_id, rep,
                                           global_max_volume, dpi_output=dpi_output)

            # Save figure
            output_file = output_dir / f"pocket_{pocket_num}.png"
            fig.savefig(output_file, dpi=dpi_output, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            plot_count += 1
            if verbose and plot_count % 100 == 0:
                print(f"  Generated {plot_count}/{total_plots} plots...")

    if verbose:
        print(f"COMPLETE!")
        print(f"Generated {plot_count} pocket volume plots")
        print(f"Global Y-axis scale: 0 - {global_max_volume:.2f}")
        print(f"\nOutput structure:")
        print(f"  /mnt/mmlab_shared/RienaeckerC/apo_holo_analysis/")
        print(f"  └── {{state}}_structures/results/{{state}}{{pdb_id}}/{{rep}}/plots/volume/")


    return plot_count


if __name__ == "__main__":
    plot_pocket_volumes_over_time(verbose=True)