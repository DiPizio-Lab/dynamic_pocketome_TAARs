import os
import time
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# =============================================================================
# GLOBAL COLOR CONFIGURATION
# =============================================================================
# Viridis palette samples (green -> blue -> violet range: 0.2 to 0.9)
# Adjust these values to change colors throughout all plots

# Colorscale range for multi-experiment plots (start, end values for Viridis)
# Range 0.2-0.9 gives green -> teal -> blue -> violet (avoiding yellow)
VIRIDIS_RANGE = (0.2, 0.9)

# Greyscale mode - set to True for simpler color scheme
GREYSCALE_MODE = False

# For volume categories (4 colors: Small, Medium, Large, Very Large)
VOLUME_CATEGORY_COLORS = {
    'Small (<250)': px.colors.sample_colorscale("Viridis", 0.25)[0],  # Green
    'Medium (250-500)': px.colors.sample_colorscale("Viridis", 0.45)[0],  # Teal
    'Large (500-750)': px.colors.sample_colorscale("Viridis", 0.65)[0],  # Blue
    'Very Large (>750)': px.colors.sample_colorscale("Viridis", 0.85)[0]  # Violet
}

# Greyscale versions
VOLUME_CATEGORY_COLORS_GREY = {
    'Small (<250)': '#d9d9d9',  # Light grey
    'Medium (250-500)': '#969696',  # Medium grey
    'Large (500-750)': '#525252',  # Dark grey
    'Very Large (>750)': '#252525'  # Very dark grey
}

# For stability categories (2 colors: Stable, Transient)
# Using darker green (#1a5e1a) for transient to make it more visible
STABILITY_COLORS = {
    'Stable': 'midnightblue',  # Blue
    'Transient': '#1a5e1a'  # Dark forest green
}

# Greyscale versions
STABILITY_COLORS_GREY = {
    'Stable': '#525252',  # Dark grey
    'Transient': '#bdbdbd'  # Light grey
}

# For state categories (2 colors: apo, holo) -- sand / dusty burgundy, matching
# taar_paper_figures/taar_style.py's STATE_COLORS (single source of truth for this
# palette). Deliberately not sampled from Viridis: these plots also use Viridis for
# volume categories, so state needs to read as visually distinct from that scale.
STATE_COLORS = {
    'apo': '#D9B98B',   # sand
    'holo': '#9B5560'   # mauve / dusty burgundy
}

STATE_COLORS_GREY = {
    'apo': '#737373',  # Medium grey
    'holo': '#252525'  # Dark grey
}

# Default single color for individual plots
DEFAULT_PLOT_COLOR = px.colors.sample_colorscale("Viridis", 0.5)[0]  # Middle blue
DEFAULT_PLOT_COLOR_GREY = '#525252'  # Dark grey

# Category order (for consistent plotting)
VOLUME_CATEGORY_ORDER = ['Small (<250)', 'Medium (250-500)', 'Large (500-750)', 'Very Large (>750)']
STABILITY_ORDER = ['Stable', 'Transient']

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================
# Default settings for high-resolution exports
DEFAULT_DPI = 300
DEFAULT_SCALE = 2  # For Plotly exports (2x resolution)
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900


def save_plotly_figure(fig, filepath_base, width=None, height=None, scale=None):
    """
    Save a Plotly figure in HTML, PNG, and PDF formats.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to save
    filepath_base : str
        Base filepath without extension (e.g., '/path/to/plot')
        Will create: plot.html, plot.png, plot.pdf
    width : int, optional
        Width in pixels (default: DEFAULT_WIDTH)
    height : int, optional
        Height in pixels (default: DEFAULT_HEIGHT)
    scale : int, optional
        Scale factor for image export (default: DEFAULT_SCALE)

    Returns:
    --------
    dict : Paths to saved files {'html': path, 'png': path, 'pdf': path}
    """
    width = width or DEFAULT_WIDTH
    height = height or DEFAULT_HEIGHT
    scale = scale or DEFAULT_SCALE

    saved_files = {}

    # Save HTML (always works)
    html_path = f"{filepath_base}.html"
    fig.write_html(html_path, include_plotlyjs='cdn')
    saved_files['html'] = html_path

    # Save PNG and PDF (requires kaleido)
    try:
        png_path = f"{filepath_base}.png"
        fig.write_image(png_path, format='png', width=width, height=height, scale=scale)
        saved_files['png'] = png_path

        pdf_path = f"{filepath_base}.pdf"
        fig.write_image(pdf_path, format='pdf', width=width, height=height, scale=scale)
        saved_files['pdf'] = pdf_path

    except Exception as e:
        print(f"  WARNING: Could not export PNG/PDF ({type(e).__name__}). Install kaleido: pip install -U kaleido")

    print(f"Saved {os.path.basename(filepath_base)} ({', '.join(saved_files)})")
    return saved_files


def save_matplotlib_figure(fig, filepath_base, dpi=None):
    """
    Save a Matplotlib figure in PNG and PDF formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure to save
    filepath_base : str
        Base filepath without extension (e.g., '/path/to/plot')
        Will create: plot.png, plot.pdf
    dpi : int, optional
        Resolution in dots per inch (default: DEFAULT_DPI)

    Returns:
    --------
    dict : Paths to saved files {'png': path, 'pdf': path}
    """
    dpi = dpi or DEFAULT_DPI
    saved_files = {}

    png_path = f"{filepath_base}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    saved_files['png'] = png_path

    pdf_path = f"{filepath_base}.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    saved_files['pdf'] = pdf_path

    print(f"Saved {os.path.basename(filepath_base)} ({', '.join(saved_files)})")
    return saved_files


def get_color_dict(color_type, greyscale=None):
    """
    Get the appropriate color dictionary based on greyscale mode.

    Parameters:
    -----------
    color_type : str
        One of 'volume', 'stability', 'state', 'default'
    greyscale : bool, optional
        Override global GREYSCALE_MODE setting

    Returns:
    --------
    dict or str with colors
    """
    use_grey = greyscale if greyscale is not None else GREYSCALE_MODE

    if color_type == 'volume':
        return VOLUME_CATEGORY_COLORS_GREY if use_grey else VOLUME_CATEGORY_COLORS
    elif color_type == 'stability':
        return STABILITY_COLORS_GREY if use_grey else STABILITY_COLORS
    elif color_type == 'state':
        return STATE_COLORS_GREY if use_grey else STATE_COLORS
    elif color_type == 'default':
        return DEFAULT_PLOT_COLOR_GREY if use_grey else DEFAULT_PLOT_COLOR
    else:
        raise ValueError(f"Unknown color_type: {color_type}")


def set_greyscale_mode(enabled=True):
    """
    Enable or disable greyscale mode globally.

    Parameters:
    -----------
    enabled : bool
        If True, use greyscale colors throughout all plots
    """
    global GREYSCALE_MODE
    GREYSCALE_MODE = enabled
    print(f"Greyscale mode: {'enabled' if enabled else 'disabled'}")


def get_viridis_colors(n, start=None, end=None, greyscale=None):
    """
    Generate n colors from Viridis palette within specified range.

    Parameters:
    -----------
    n : int
        Number of colors to generate
    start : float, optional
        Start position in colorscale (0-1), defaults to VIRIDIS_RANGE[0]
    end : float, optional
        End position in colorscale (0-1), defaults to VIRIDIS_RANGE[1]
    greyscale : bool, optional
        Override global GREYSCALE_MODE setting

    Returns:
    --------
    list of color strings
    """
    use_grey = greyscale if greyscale is not None else GREYSCALE_MODE

    if use_grey:
        # Generate greyscale gradient
        if n == 1:
            return ['#525252']
        grey_vals = np.linspace(0.3, 0.8, n)  # Range from dark to light grey
        return [f'rgb({int(v * 255)},{int(v * 255)},{int(v * 255)})' for v in grey_vals]

    if start is None:
        start = VIRIDIS_RANGE[0]
    if end is None:
        end = VIRIDIS_RANGE[1]

    if n == 1:
        return [px.colors.sample_colorscale("Viridis", (start + end) / 2)[0]]

    vals = np.linspace(start, end, n)
    return [px.colors.sample_colorscale("Viridis", v)[0] for v in vals]


def update_color_scheme(volume_colors=None, stability_colors=None, state_colors=None,
                        default_color=None, viridis_range=None, greyscale=None):
    """
    Update global color configuration.

    Parameters:
    -----------
    volume_colors : dict, optional
        Dictionary mapping volume categories to colors
    stability_colors : dict, optional
        Dictionary mapping stability categories to colors
    state_colors : dict, optional
        Dictionary mapping states to colors
    default_color : str, optional
        Default color for single-color plots
    viridis_range : tuple, optional
        (start, end) range for Viridis colorscale (0-1)
    greyscale : bool, optional
        Enable/disable greyscale mode

    Example:
    --------
    # Use different Viridis range (more towards violet)
    update_color_scheme(viridis_range=(0.4, 0.95))

    # Enable greyscale mode
    update_color_scheme(greyscale=True)

    # Custom stability colors
    update_color_scheme(stability_colors={
        'Stable': '#31688e',
        'Transient': '#1a5e1a'
    })
    """
    global VOLUME_CATEGORY_COLORS, STABILITY_COLORS, STATE_COLORS
    global DEFAULT_PLOT_COLOR, VIRIDIS_RANGE, GREYSCALE_MODE

    if greyscale is not None:
        GREYSCALE_MODE = greyscale

    if viridis_range is not None:
        VIRIDIS_RANGE = viridis_range
        # Regenerate default colors based on new range
        positions = np.linspace(viridis_range[0], viridis_range[1], 4)
        VOLUME_CATEGORY_COLORS = {
            'Small (<250)': px.colors.sample_colorscale("Viridis", positions[0])[0],
            'Medium (250-500)': px.colors.sample_colorscale("Viridis", positions[1])[0],
            'Large (500-750)': px.colors.sample_colorscale("Viridis", positions[2])[0],
            'Very Large (>750)': px.colors.sample_colorscale("Viridis", positions[3])[0]
        }
        stab_positions = np.linspace(viridis_range[0], viridis_range[1], 2)
        STABILITY_COLORS = {
            'Stable': px.colors.sample_colorscale("Viridis", stab_positions[1])[0],
            'Transient': '#1a5e1a'  # Keep dark green for transient
        }
        # STATE_COLORS (apo/holo) stays fixed to sand/mauve regardless of viridis_range --
        # it's not part of the Viridis continuum, see the module-level comment above.
        DEFAULT_PLOT_COLOR = px.colors.sample_colorscale("Viridis",
                                                         (viridis_range[0] + viridis_range[1]) / 2)[0]

    if volume_colors is not None:
        VOLUME_CATEGORY_COLORS.update(volume_colors)
    if stability_colors is not None:
        STABILITY_COLORS.update(stability_colors)
    if state_colors is not None:
        STATE_COLORS.update(state_colors)
    if default_color is not None:
        DEFAULT_PLOT_COLOR = default_color


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_bw_numbering(bw_csv_file):
    """Read BW numbering to identify TM helices."""
    df = pd.read_csv(bw_csv_file)
    helices = {}

    for helix_type in ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7', 'H8']:
        helix_residues = df[df['CHAIN'] == helix_type]
        if not helix_residues.empty:
            helices[helix_type] = helix_residues['RES_ID'].tolist()

    return helices


def extract_helix_coordinates(pdb_file, bw_helices, pdb_coords):
    """Extract 3D coordinates for each helix from PDB structure."""
    helix_coords = {}

    for helix_name, res_ids in bw_helices.items():
        coords = []
        for res_id in res_ids:
            if res_id in pdb_coords:
                coords.append(pdb_coords[res_id])

        if coords:
            helix_coords[helix_name] = np.array(coords)

    return helix_coords


def extract_pdbid_info(pocket_id):
    """Extract state, PDB ID, and replicate from pocket ID.
    e.g., 'apo8ITF_1_100_4_2' -> ('apo8ITF', 'apo', '8ITF', '1') """
    m = re.match(r'^([a-z]+)([A-Za-z0-9]+)_(\d+)', pocket_id)
    if m:
        state = m.group(1)
        pdb_id = m.group(2)
        rep = m.group(3)
        state_pdbid = f"{state}{pdb_id}"
        return state_pdbid, state, pdb_id, rep
    return None, None, None, None


def sort_experiments(experiment_list, group_apo_holo=True):
    """
    Sort experiment labels, optionally grouping apo/holo pairs together.

    Parameters:
    -----------
    experiment_list : list or Index
        List of experiment labels like ['apo8ITF rep 1', 'holo8ITF rep 1', ...]
    group_apo_holo : bool
        If True, sort so apo and holo of same PDB ID are adjacent

    Returns:
    --------
    sorted list
    """
    experiments = list(experiment_list)

    if not group_apo_holo:
        return sorted(experiments)

    # Parse each experiment into (pdb_id, state, rep) for sorting
    def sort_key(exp):
        # Extract components: "apo8ITF rep 1" -> pdb_id=8ITF, state=apo, rep=1
        match = re.match(r'^(apo|holo)([A-Za-z0-9]+)\s*rep\s*(\d+)', exp, re.IGNORECASE)
        if match:
            state = match.group(1).lower()
            pdb_id = match.group(2)
            rep = int(match.group(3))
            # Sort by: PDB ID, then replicate, then state (apo before holo)
            state_order = 0 if state == 'apo' else 1
            return (pdb_id, rep, state_order)
        # For non-matching entries, sort alphabetically at the end
        return ('zzz', 999, exp)

    return sorted(experiments, key=sort_key)


def sort_dataframe_index(df, group_apo_holo=True):
    """
    Sort DataFrame index (experiment labels), grouping apo/holo pairs.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with experiment labels as index
    group_apo_holo : bool
        If True, sort so apo and holo of same PDB ID are adjacent

    Returns:
    --------
    DataFrame with sorted index
    """
    sorted_idx = sort_experiments(df.index, group_apo_holo=group_apo_holo)
    return df.reindex(sorted_idx)


def prepare_time_series_data(pocket_df, volume_col='interpolated_pock_volume', hydro_col='hydrophobicity_score'):
    """Aggregate time series data per snapshot."""
    if 'snapshot' not in pocket_df.columns:
        raise ValueError('For time series data you need a column called "snapshot" representing the frames or '
                         'nanoseconds of a simulation')

    agg_dict = {}
    if volume_col in pocket_df.columns:
        agg_dict[volume_col] = 'first'
    if hydro_col in pocket_df.columns:
        agg_dict[hydro_col] = 'first'

    if not agg_dict:
        return pocket_df[['snapshot']].drop_duplicates().sort_values('snapshot')

    time_series = pocket_df.groupby('snapshot').agg(agg_dict).reset_index()
    return time_series.sort_values('snapshot')


def categorize_volume(volume):
    """Categorize pocket volume into size categories."""
    if volume < 250:
        return 'Small (<250)'
    elif volume < 500:
        return 'Medium (250-500)'
    elif volume < 750:
        return 'Large (500-750)'
    else:
        return 'Very Large (>750)'


def get_first_frame_data(df, snapshot_col='snapshot'):
    """Extract only the first frame for 3D visualization."""
    if snapshot_col not in df.columns:
        return df
    first_snapshot = df[snapshot_col].min()
    return df[df[snapshot_col] == first_snapshot].copy()


# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_largest_pockets(saving_loc, pocket_df, pdb_file=None, bw_csv_file=None, out_html='largest_pocket.html',
                         marker_size=6, backbone=False, helix_opacity=0.2, title='Largest Pocket Analysis',
                         export_pdfs=True, pdf_output_dir=None, hydro_col='hydrophobicity_score',
                         volume_col='interpolated_pock_volume',
                         use_cdn=True,
                         downsample_timeseries=None):
    """
    Plot largest pocket with interactive subplots for volume and hydrophobicity over time.
    - 3D scatter plot uses ONLY FIRST FRAME to reduce file size
    - Time series plots use all frames (or downsampled if specified)
    - Uses global VIRIDIS_RANGE for color generation
    """
    pocket_df = pocket_df.copy()

    # Convert coordinates to numeric
    pocket_df['x'] = pd.to_numeric(pocket_df['x'], errors='coerce')
    pocket_df['y'] = pd.to_numeric(pocket_df['y'], errors='coerce')
    pocket_df['z'] = pd.to_numeric(pocket_df['z'], errors='coerce')

    pocket_df['x'] = pocket_df['x'].round(2)
    pocket_df['y'] = pocket_df['y'].round(2)
    pocket_df['z'] = pocket_df['z'].round(2)

    pocket_df = pocket_df.dropna(subset=['x', 'y', 'z', volume_col])

    pocket_df['state_pdbid'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[0])
    pocket_df['rep'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[3])

    unique_combinations = pocket_df[['state_pdbid', 'rep']].drop_duplicates()
    unique_combinations = unique_combinations.dropna()
    unique_combinations['label'] = unique_combinations['state_pdbid'] + ' rep ' + unique_combinations['rep']
    combination_labels = ['All'] + unique_combinations['label'].tolist()

    print(f"Found {len(combination_labels) - 1} unique PDB ID / replicate combinations")

    total_points = len(pocket_df)
    first_frame_points = len(get_first_frame_data(pocket_df))
    print(f"Total data points: {total_points:,}")
    print(f"First frame points (for 3D): {first_frame_points:,}")
    print(f"Reduction: {100 * (1 - first_frame_points / total_points):.1f}%")

    if pdf_output_dir is None:
        pdf_output_dir = os.path.join(saving_loc, 'pdfs')
    os.makedirs(pdf_output_dir, exist_ok=True)

    # Load backbone data if requested
    helix_coords = None
    if backbone and pdb_file and bw_csv_file:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        pdb_coords = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        res_id = residue.get_id()[1]
                        coord = residue['CA'].get_coord()
                        pdb_coords[res_id] = coord

        bw_helices = read_bw_numbering(bw_csv_file)
        helix_coords = extract_helix_coordinates(pdb_file, bw_helices, pdb_coords)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}], [None, {'type': 'xy'}]],
        subplot_titles=('Pocket Location (First Frame)', 'Volume Over Time', 'Hydrophobicity Over Time'),
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.08,
        vertical_spacing=0.12)

    # Generate colors using global Viridis range
    num_combos = len(unique_combinations)
    viridis_colors = get_viridis_colors(num_combos) if num_combos > 0 else [DEFAULT_PLOT_COLOR]

    # Track traces for visibility toggling
    trace_info = []

    # Add traces for each combination
    for idx, (_, row) in enumerate(unique_combinations.iterrows()):
        state_pdbid = row['state_pdbid']
        rep = row['rep']
        label = row['label']
        color = viridis_colors[idx] if idx < len(viridis_colors) else DEFAULT_PLOT_COLOR

        mask = (pocket_df['state_pdbid'] == state_pdbid) & (pocket_df['rep'] == rep)
        combo_data = pocket_df[mask]
        if len(combo_data) == 0:
            continue

        pocket_volume = combo_data[volume_col].iloc[0]

        # === 3D scatter plot: USE ONLY FIRST FRAME ===
        combo_data_3d = get_first_frame_data(combo_data)

        fig.add_trace(go.Scatter3d(
            x=combo_data_3d['x'], y=combo_data_3d['y'], z=combo_data_3d['z'],
            mode='markers', name=label,
            text=[f"<b>{label}</b><br>Volume: {pocket_volume:.2f}" for _ in range(len(combo_data_3d))],
            hoverinfo='text',
            marker=dict(size=marker_size, color=color, opacity=0.8, line=dict(width=0.5, color='white')),
            showlegend=True, legendgroup=label))
        trace_info.append((label, '3d'))

        # === Time series: USE ALL FRAMES ===
        time_series = prepare_time_series_data(combo_data, volume_col, hydro_col)

        if downsample_timeseries and len(time_series) > downsample_timeseries:
            time_series = time_series.iloc[::downsample_timeseries]

        fig.add_trace(go.Scatter(
            x=time_series['snapshot'], y=time_series[volume_col],
            mode='lines', name=f'{label} Volume',
            line=dict(color=color, width=2),
            showlegend=False, legendgroup=label), row=1, col=2)
        trace_info.append((label, 'volume'))

        fig.add_trace(go.Scatter(
            x=time_series['snapshot'], y=time_series[hydro_col],
            mode='lines', name=f'{label} Hydro',
            line=dict(color=color, width=2),
            showlegend=False, legendgroup=label), row=2, col=2)
        trace_info.append((label, 'hydro'))

    # Add backbone traces
    if helix_coords:
        for helix_name, coords in helix_coords.items():
            if len(coords) > 1:
                fig.add_trace(go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='lines', name=helix_name,
                    line=dict(color='rgba(80, 80, 80, 0.5)', width=4),
                    hoverinfo='name', showlegend=False, visible=True))
                trace_info.append(('_backbone', 'backbone'))

    # Create dropdown buttons
    buttons = []
    for combo_label in combination_labels:
        visibility = []
        for i, (trace_label, trace_type) in enumerate(trace_info):
            if trace_type == 'backbone':
                visibility.append(True)
            elif combo_label == 'All':
                visibility.append(trace_type == '3d')
            else:
                visibility.append(trace_label == combo_label)
        buttons.append(dict(label=combo_label, method='update', args=[{'visible': visibility}]))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            domain=dict(x=[0, 0.55], y=[0, 1]),
            xaxis=dict(title='X (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)', showbackground=True),
            yaxis=dict(title='Y (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)', showbackground=True),
            zaxis=dict(title='Z (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)', showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        updatemenus=[dict(active=0, buttons=buttons, direction='down', showactive=True,
                          x=0.0, xanchor='left', y=1.15, yanchor='top', bgcolor='white', bordercolor='#cccccc')],
        annotations=[dict(text='Select PDB ID:', x=0.0, xref='paper', y=1.18, yref='paper', showarrow=False)],
        template='plotly_white',
        paper_bgcolor='white',
        legend=dict(x=1.02, y=0.95, bgcolor='rgba(255,255,255,0.95)', bordercolor='#cccccc', borderwidth=1),
        height=900, width=1400,
        margin=dict(l=60, r=100, t=150, b=60))

    fig.update_xaxes(title_text='Snapshot', row=1, col=2)
    fig.update_yaxes(title_text='Volume (Å³)', row=1, col=2)
    fig.update_xaxes(title_text='Snapshot', row=2, col=2)
    fig.update_yaxes(title_text='Hydrophobicity', row=2, col=2)

    # Save interactive HTML + PNG + PDF using helper
    output_base = os.path.join(saving_loc, out_html.replace('.html', ''))
    save_plotly_figure(fig, output_base, width=1400, height=900)

    # Export individual PDFs (and PNGs)
    if export_pdfs:
        print(f"\nExporting individual PDFs to: {pdf_output_dir}")
        pdf_failed = False

        for idx, (_, row) in enumerate(unique_combinations.iterrows()):
            state_pdbid = row['state_pdbid']
            rep = row['rep']
            color = viridis_colors[idx] if idx < len(viridis_colors) else DEFAULT_PLOT_COLOR

            pdf_fig = _create_individual_figure(
                pocket_df, state_pdbid, rep, volume_col, hydro_col,
                helix_coords, helix_opacity, marker_size, color)

            pdf_basename = f"{state_pdbid}_rep{rep}"
            pdf_path = os.path.join(pdf_output_dir, f"{pdf_basename}.pdf")
            png_path = os.path.join(pdf_output_dir, f"{pdf_basename}.png")

            if not pdf_failed:
                try:
                    pdf_fig.write_image(pdf_path, format='pdf', width=1400, height=900, scale=2)
                    pdf_fig.write_image(png_path, format='png', width=1400, height=900, scale=2)
                    print(f"  Saved: {pdf_basename}.pdf, {pdf_basename}.png")
                except Exception as e:
                    pdf_failed = True
                    print(f"\n  PDF/PNG export failed ({type(e).__name__}). Install kaleido for image support.")
                    html_path = os.path.join(pdf_output_dir, f"{pdf_basename}.html")
                    pdf_fig.write_html(html_path, include_plotlyjs='cdn')
                    print(f"  Saved: {pdf_basename}.html")
            else:
                html_path = os.path.join(pdf_output_dir, f"{pdf_basename}.html")
                pdf_fig.write_html(html_path, include_plotlyjs='cdn')
                print(f"  Saved: {pdf_basename}.html")

    return fig


def _create_individual_figure(pocket_df, state_pdbid, rep, volume_col, hydro_col,
                              helix_coords, helix_opacity, marker_size, color):
    """Create a static figure for a single PDB ID / replicate."""
    mask = (pocket_df['state_pdbid'] == state_pdbid) & (pocket_df['rep'] == rep)
    combo_data = pocket_df[mask]

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}], [None, {'type': 'xy'}]],
                        subplot_titles=(f'Pocket Location - {state_pdbid} rep {rep}', 'Volume Over Time',
                                        'Hydrophobicity Over Time'),
                        column_widths=[0.6, 0.4], row_heights=[0.5, 0.5],
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    pocket_volume = combo_data[volume_col].iloc[0]

    # 3D scatter - FIRST FRAME ONLY
    combo_data_3d = get_first_frame_data(combo_data)

    fig.add_trace(go.Scatter3d(
        x=combo_data_3d['x'], y=combo_data_3d['y'], z=combo_data_3d['z'],
        mode='markers', name=f'{state_pdbid} rep {rep}',
        marker=dict(size=marker_size, color=color, opacity=0.8, line=dict(width=0.5, color='white'))))

    # Backbone
    if helix_coords:
        for helix_name, coords in helix_coords.items():
            if len(coords) > 1:
                fig.add_trace(go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='lines', name=helix_name,
                    line=dict(color='rgba(80, 80, 80, 0.5)', width=4),
                    showlegend=False, hoverinfo='name'))

    # Time series - ALL FRAMES
    time_series = prepare_time_series_data(combo_data, volume_col, hydro_col)

    fig.add_trace(go.Scatter(
        x=time_series['snapshot'], y=time_series[volume_col],
        mode='lines+markers', line=dict(color=color, width=2),
        marker=dict(size=4), showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=time_series['snapshot'], y=time_series[hydro_col],
        mode='lines+markers', line=dict(color=color, width=2),
        marker=dict(size=4), showlegend=False), row=2, col=2)

    fig.update_layout(
        title=dict(text=f'{state_pdbid} rep {rep} - Volume: {pocket_volume:.2f} Å³', x=0.5),
        scene=dict(
            xaxis=dict(title='X (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)'),
            yaxis=dict(title='Y (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)'),
            zaxis=dict(title='Z (Å)', backgroundcolor='white', gridcolor='rgb(240,240,240)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        template='plotly_white', height=900, width=1400,
        margin=dict(l=60, r=100, t=100, b=60))

    fig.update_xaxes(title_text='Snapshot', row=1, col=2)
    fig.update_yaxes(title_text='Volume (Å³)', row=1, col=2)
    fig.update_xaxes(title_text='Snapshot', row=2, col=2)
    fig.update_yaxes(title_text='Hydrophobicity', row=2, col=2)

    return fig


# =============================================================================
# DISTRIBUTION PLOT FUNCTIONS
# =============================================================================

def plot_volume_category_distribution(pocket_df, saving_loc, volume_col='interpolated_pock_volume',
                                      out_prefix='volume_distribution', group_apo_holo=True,
                                      aggregation='per_pocket', greyscale=None):
    """
    Create volume category distribution plots using global VOLUME_CATEGORY_COLORS.

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data with ID and volume columns
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume data
    out_prefix : str
        Prefix for output filenames
    group_apo_holo : bool
        If True, sort x-axis so apo/holo of same PDB ID are adjacent
    aggregation : str
        How to aggregate data:
        - 'per_pocket': Each unique pocket ID counted once (mean volume across frames)
        - 'per_structure': Aggregate replicates per state+PDB (median across replicates)
        - 'per_frame': Count each frame separately (not recommended, inflates counts)
    greyscale : bool, optional
        Override global GREYSCALE_MODE

    Data explanation:
    -----------------
    For 'per_pocket' (default):
        1. Group by unique pocket ID
        2. Calculate mean volume for that pocket across all frames
        3. Categorize based on mean volume
        4. Count pockets per category per experiment

    This means: if experiment apo8ITF_rep1 has 20 unique pockets, you get 20 data points.
    """
    pocket_df = pocket_df.copy()
    colors = get_color_dict('volume', greyscale)
    default_color = get_color_dict('default', greyscale)

    # Extract experiment info
    pocket_df['state_pdbid'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[0])
    pocket_df['state'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[1])
    pocket_df['pdb_id'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[2])
    pocket_df['rep'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[3])
    pocket_df['experiment'] = pocket_df['state_pdbid'] + ' rep ' + pocket_df['rep']

    # Aggregate based on method
    if aggregation == 'per_pocket':
        # Mean volume per unique pocket ID across frames
        pocket_summary = pocket_df.groupby(['experiment', 'state', 'pdb_id', 'rep', 'ID']).agg({
            volume_col: 'mean'
        }).reset_index()
        agg_label = "per unique pocket (mean across frames)"

    elif aggregation == 'per_structure':
        # First get per-pocket means, then median across replicates
        per_pocket = pocket_df.groupby(['state', 'pdb_id', 'rep', 'ID']).agg({
            volume_col: 'mean'
        }).reset_index()
        # Then median per structure (across replicates)
        pocket_summary = per_pocket.groupby(['state', 'pdb_id', 'ID']).agg({
            volume_col: 'median'
        }).reset_index()
        pocket_summary['experiment'] = pocket_summary['state'] + pocket_summary['pdb_id']
        agg_label = "per pocket (median across replicates)"

    else:  # per_frame
        pocket_summary = pocket_df[['experiment', 'state', 'pdb_id', 'rep', 'ID', volume_col]].copy()
        agg_label = "per frame (inflated counts!)"

    # Categorize volumes
    pocket_summary['size_category'] = pocket_summary[volume_col].apply(categorize_volume)

    print(f"Volume distribution aggregation: {agg_label}")
    print(f"Total data points for categorization: {len(pocket_summary)}")

    # === PLOT 1: Stacked Bar Chart per Experiment ===
    experiment_counts = pocket_summary.groupby(['experiment', 'size_category']).size().unstack(fill_value=0)
    experiment_counts = experiment_counts.reindex(columns=VOLUME_CATEGORY_ORDER, fill_value=0)

    # Sort experiments
    experiment_counts = sort_dataframe_index(experiment_counts, group_apo_holo=group_apo_holo)

    fig1 = go.Figure()
    for category in VOLUME_CATEGORY_ORDER:
        if category in experiment_counts.columns:
            fig1.add_trace(go.Bar(
                name=category,
                x=experiment_counts.index,
                y=experiment_counts[category],
                marker_color=colors[category]))

    fig1.update_layout(
        barmode='stack',
        title=dict(text=f'Pocket Size Distribution per Experiment<br><sub>({agg_label})</sub>',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Experiment (State + PDB ID + Replicate)',
        yaxis_title='Number of Pockets',
        legend_title='Size Category',
        template='plotly_white',
        height=600, width=1400,
        xaxis_tickangle=-45)

    save_plotly_figure(fig1, os.path.join(saving_loc, f'{out_prefix}_per_experiment'), width=1400, height=600)

    # === PLOT 2: Overall Distribution (Bar Chart) ===
    overall_counts = pocket_summary['size_category'].value_counts().reindex(VOLUME_CATEGORY_ORDER, fill_value=0)

    fig2 = go.Figure(data=[
        go.Bar(
            x=overall_counts.index,
            y=overall_counts.values,
            marker_color=[colors[c] for c in overall_counts.index],
            text=overall_counts.values,
            textposition='auto')])

    fig2.update_layout(
        title=dict(text=f'Overall Pocket Size Distribution<br><sub>({agg_label})</sub>',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Size Category',
        yaxis_title='Number of Pockets',
        template='plotly_white',
        height=500, width=700)

    save_plotly_figure(fig2, os.path.join(saving_loc, f'{out_prefix}_overall_bar'), width=700, height=500)

    # === PLOT 3: Pie Chart ===
    fig3 = go.Figure(data=[
        go.Pie(
            labels=overall_counts.index,
            values=overall_counts.values,
            marker_colors=[colors[c] for c in overall_counts.index],
            textinfo='label+percent+value',
            hole=0.3)])

    fig3.update_layout(
        title=dict(text='Pocket Size Distribution', x=0.5, font=dict(size=18)),
        template='plotly_white',
        height=500, width=600)

    save_plotly_figure(fig3, os.path.join(saving_loc, f'{out_prefix}_pie'), width=600, height=500)

    # === PLOT 4: Histogram with counts on top ===
    # Calculate histogram bins and counts manually to add text labels
    hist_data = pocket_summary[volume_col].dropna()
    counts, bin_edges = np.histogram(hist_data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig4 = go.Figure(data=[
        go.Bar(
            x=bin_centers,
            y=counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            marker_color=default_color,
            opacity=0.7,
            text=[str(c) if c > 0 else '' for c in counts],
            textposition='outside',
            textfont=dict(size=8))])

    # Add vertical lines for category boundaries
    boundary_color = px.colors.sample_colorscale("Viridis", 0.8)[0]
    for boundary, label in [(250, '250'), (500, '500'), (750, '750')]:
        fig4.add_vline(x=boundary, line_dash="dash", line_color=boundary_color,
                       annotation_text=f"{label} Å³", annotation_position="top")

    fig4.update_layout(
        title=dict(text=f'Pocket Volume Distribution<br><sub>({agg_label})</sub>',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Volume (Å³)',
        yaxis_title='Count',
        template='plotly_white',
        height=500, width=900,
        bargap=0.05)

    save_plotly_figure(fig4, os.path.join(saving_loc, f'{out_prefix}_histogram'), width=900, height=500)

    # Save summary statistics
    summary_stats = pocket_summary.groupby(['state', 'size_category']).size().unstack(fill_value=0)
    summary_stats.to_csv(os.path.join(saving_loc, f'{out_prefix}_summary.csv'))
    print(f"Saved: {out_prefix}_summary.csv")

    return fig1, fig2, fig3, fig4


def plot_transient_stable_distribution(pocket_df, transient_dict, saving_loc,
                                       out_prefix='transient_stable', group_apo_holo=True,
                                       greyscale=None):
    """
    Create transient vs stable pocket distribution plots using global STABILITY_COLORS.

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status (boolean)
    saving_loc : str
        Output directory
    out_prefix : str
        Prefix for output filenames
    group_apo_holo : bool
        If True, sort x-axis so apo/holo of same PDB ID are adjacent
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    pocket_df = pocket_df.copy()
    colors = get_color_dict('stability', greyscale)

    # Extract experiment info
    pocket_df['state_pdbid'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[0])
    pocket_df['state'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[1])
    pocket_df['rep'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[3])
    pocket_df['experiment'] = pocket_df['state_pdbid'] + ' rep ' + pocket_df['rep']

    # Get unique pockets
    unique_pockets = pocket_df[['ID', 'experiment', 'state', 'rep']].drop_duplicates()
    unique_pockets['is_transient'] = unique_pockets['ID'].map(transient_dict)
    unique_pockets['stability'] = unique_pockets['is_transient'].map({True: 'Transient', False: 'Stable'})

    # === PLOT 1: Stacked Bar per Experiment ===
    experiment_counts = unique_pockets.groupby(['experiment', 'stability']).size().unstack(fill_value=0)
    experiment_counts = sort_dataframe_index(experiment_counts, group_apo_holo=group_apo_holo)

    fig1 = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in experiment_counts.columns:
            fig1.add_trace(go.Bar(
                name=stability,
                x=experiment_counts.index,
                y=experiment_counts[stability],
                marker_color=colors[stability]))

    fig1.update_layout(
        barmode='stack',
        title=dict(text='Transient vs Stable Pockets per Experiment', x=0.5, font=dict(size=18)),
        xaxis_title='Experiment',
        yaxis_title='Number of Pockets',
        legend_title='Stability',
        template='plotly_white',
        height=600, width=1400,
        xaxis_tickangle=-45)

    save_plotly_figure(fig1, os.path.join(saving_loc, f'{out_prefix}_per_experiment'), width=1400, height=600)

    # === PLOT 2: Percentage Bar Chart ===
    experiment_pct = experiment_counts.div(experiment_counts.sum(axis=1), axis=0) * 100

    fig2 = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in experiment_pct.columns:
            fig2.add_trace(go.Bar(
                name=stability,
                x=experiment_pct.index,
                y=experiment_pct[stability],
                marker_color=colors[stability],
                text=[f'{v:.1f}%' for v in experiment_pct[stability]],
                textposition='inside'))

    fig2.update_layout(
        barmode='stack',
        title=dict(text='Percentage of Transient vs Stable Pockets per Experiment', x=0.5, font=dict(size=18)),
        xaxis_title='Experiment',
        yaxis_title='Percentage (%)',
        legend_title='Stability',
        template='plotly_white',
        height=600, width=1400,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 100]))

    save_plotly_figure(fig2, os.path.join(saving_loc, f'{out_prefix}_percentage'), width=1400, height=600)

    # === PLOT 3: Pie Chart ===
    overall_counts = unique_pockets['stability'].value_counts().reindex(STABILITY_ORDER, fill_value=0)

    fig3 = go.Figure(data=[
        go.Pie(
            labels=overall_counts.index,
            values=overall_counts.values,
            marker_colors=[colors.get(s, get_color_dict('default', greyscale)) for s in overall_counts.index],
            textinfo='label+percent+value',
            hole=0.3)])

    fig3.update_layout(
        title=dict(text='Overall Transient vs Stable Distribution', x=0.5, font=dict(size=18)),
        template='plotly_white',
        height=500, width=600)

    save_plotly_figure(fig3, os.path.join(saving_loc, f'{out_prefix}_pie'), width=600, height=500)

    # === PLOT 4: By State ===
    state_counts = unique_pockets.groupby(['state', 'stability']).size().unstack(fill_value=0)

    fig4 = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in state_counts.columns:
            fig4.add_trace(go.Bar(
                name=stability,
                x=state_counts.index,
                y=state_counts[stability],
                marker_color=colors[stability],
                text=state_counts[stability],
                textposition='auto'))

    fig4.update_layout(
        barmode='group',
        title=dict(text='Transient vs Stable by State (Apo/Holo)', x=0.5, font=dict(size=18)),
        xaxis_title='State',
        yaxis_title='Number of Pockets',
        legend_title='Stability',
        template='plotly_white',
        height=500, width=600)

    save_plotly_figure(fig4, os.path.join(saving_loc, f'{out_prefix}_by_state'), width=600, height=500)

    # Save summary
    summary = unique_pockets.groupby(['state', 'stability']).size().unstack(fill_value=0)
    summary.to_csv(os.path.join(saving_loc, f'{out_prefix}_summary.csv'))
    print(f"Saved: {out_prefix}_summary.csv")

    return fig1, fig2, fig3, fig4


def plot_combined_volume_stability(pocket_df, transient_dict, saving_loc,
                                   volume_col='interpolated_pock_volume',
                                   out_prefix='combined_volume_stability',
                                   broken_yaxis=True, break_threshold_ratio=0.15,
                                   greyscale=None):
    """
    Create combined visualization showing both volume categories and stability.

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume data
    out_prefix : str
        Prefix for output filenames
    broken_yaxis : bool
        If True, create a broken y-axis plot to show small values
        This creates TWO separate plots: one regular, one with broken axis
    break_threshold_ratio : float
        Ratio of max value below which to place the break (default 0.15 = 15%)
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    pocket_df = pocket_df.copy()
    colors = get_color_dict('stability', greyscale)

    # Extract info
    pocket_df['state_pdbid'] = pocket_df['ID'].apply(lambda x: extract_pdbid_info(x)[0])

    # Get unique pocket summary with mean volume
    pocket_summary = pocket_df.groupby(['state_pdbid', 'ID']).agg({
        volume_col: 'mean'
    }).reset_index()

    pocket_summary['size_category'] = pocket_summary[volume_col].apply(categorize_volume)
    pocket_summary['is_transient'] = pocket_summary['ID'].map(transient_dict)
    pocket_summary['stability'] = pocket_summary['is_transient'].map({True: 'Transient', False: 'Stable'})

    # Cross-tabulation
    cross_tab = pd.crosstab(pocket_summary['size_category'], pocket_summary['stability'])
    cross_tab = cross_tab.reindex(VOLUME_CATEGORY_ORDER, fill_value=0)

    # === PLOT 1: Regular grouped bar chart ===
    fig = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in cross_tab.columns:
            fig.add_trace(go.Bar(
                name=stability,
                x=cross_tab.index,
                y=cross_tab[stability],
                marker_color=colors[stability],
                text=cross_tab[stability],
                textposition='auto'))

    fig.update_layout(
        barmode='group',
        title=dict(text='Pocket Stability by Volume Category', x=0.5, font=dict(size=18)),
        xaxis_title='Volume Category',
        yaxis_title='Number of Pockets',
        legend_title='Stability',
        template='plotly_white',
        height=500, width=800)

    save_plotly_figure(fig, os.path.join(saving_loc, f'{out_prefix}'), width=800, height=500)

    # === PLOT 2: Broken y-axis version using subplots ===
    if broken_yaxis:
        max_val = cross_tab.values.max()
        min_nonzero = cross_tab.values[cross_tab.values > 0].min() if (cross_tab.values > 0).any() else 1

        # Define break point
        break_upper = max_val * break_threshold_ratio
        break_lower = min_nonzero * 2  # Show small values clearly

        # Only create broken axis if there's a significant range difference
        if max_val > 10 * min_nonzero and min_nonzero < break_upper:
            from plotly.subplots import make_subplots

            fig_broken = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]  # Top panel larger
            )

            # Add traces to both subplots
            for stability in STABILITY_ORDER:
                if stability in cross_tab.columns:
                    # Top panel (high values)
                    fig_broken.add_trace(go.Bar(
                        name=stability,
                        x=cross_tab.index,
                        y=cross_tab[stability],
                        marker_color=colors[stability],
                        text=cross_tab[stability],
                        textposition='outside',
                        showlegend=True), row=1, col=1)

                    # Bottom panel (low values) - same data
                    fig_broken.add_trace(go.Bar(
                        name=stability,
                        x=cross_tab.index,
                        y=cross_tab[stability],
                        marker_color=colors[stability],
                        text=[str(v) if v <= break_lower * 2 else '' for v in cross_tab[stability]],
                        textposition='outside',
                        showlegend=False), row=2, col=1)

            # Set y-axis ranges
            fig_broken.update_yaxes(range=[break_upper, max_val * 1.1], row=1, col=1, title_text='Count')
            fig_broken.update_yaxes(range=[0, break_lower * 2], row=2, col=1, title_text='Count (zoom)')

            fig_broken.update_layout(
                barmode='group',
                title=dict(text='Pocket Stability by Volume Category (Broken Y-Axis)', x=0.5, font=dict(size=18)),
                template='plotly_white',
                height=700, width=900,
                legend=dict(x=1.02, y=0.95))

            fig_broken.update_xaxes(title_text='Volume Category', row=2, col=1)

            # Add break indicator lines
            fig_broken.add_shape(type="line", x0=-0.5, x1=len(VOLUME_CATEGORY_ORDER) - 0.5,
                                 y0=break_upper, y1=break_upper, line=dict(color="gray", width=2, dash="dot"),
                                 row=1, col=1)

            save_plotly_figure(fig_broken, os.path.join(saving_loc, f'{out_prefix}_broken_yaxis'), width=900,
                               height=700)
        else:
            print("Broken y-axis not needed (values are similar in magnitude)")

    cross_tab.to_csv(os.path.join(saving_loc, f'{out_prefix}.csv'))
    print(f"Saved: {out_prefix}.csv")

    return fig


# =============================================================================
# HIERARCHICAL GROUPED BAR PLOTS (PDB ID > State > Replicate)
# =============================================================================

def prepare_hierarchical_data(pocket_df, transient_dict=None, volume_col='interpolated_pock_volume'):
    """
    Prepare data for hierarchical plotting with PDB ID > State > Replicate structure.

    Returns DataFrame with columns:
    - pdb_id: e.g., '8ITF'
    - state: 'apo' or 'holo'
    - rep: '1', '2', '3'
    - ID: original pocket ID
    - Various count/category columns
    """
    df = pocket_df.copy()

    # Extract hierarchical info
    df['state_pdbid'] = df['ID'].apply(lambda x: extract_pdbid_info(x)[0])
    df['state'] = df['ID'].apply(lambda x: extract_pdbid_info(x)[1])
    df['pdb_id'] = df['ID'].apply(lambda x: extract_pdbid_info(x)[2])
    df['rep'] = df['ID'].apply(lambda x: extract_pdbid_info(x)[3])

    # Get unique pockets with mean volume
    pocket_summary = df.groupby(['pdb_id', 'state', 'rep', 'ID']).agg({
        volume_col: 'mean'
    }).reset_index()

    # Add volume category
    pocket_summary['size_category'] = pocket_summary[volume_col].apply(categorize_volume)

    # Add stability if transient_dict provided
    if transient_dict is not None:
        pocket_summary['is_transient'] = pocket_summary['ID'].map(transient_dict)
        pocket_summary['stability'] = pocket_summary['is_transient'].map({True: 'Transient', False: 'Stable'})

    return pocket_summary


def plot_hierarchical_stability(pocket_df, transient_dict, saving_loc,
                                out_prefix='stability_hierarchical',
                                n_cols=4, greyscale=None):
    """
    Create hierarchical bar plot showing stability per PDB ID > State > Replicate.

    Layout: Subplots arranged in grid, each subplot shows one PDB ID with one bar per
    (state, replicate) actually present in the data (e.g. apo rep1, apo rep2, holo
    rep1, holo rep2 for a 2-replicate dataset)

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status
    saving_loc : str
        Output directory
    out_prefix : str
        Prefix for output filenames
    n_cols : int
        Number of columns in subplot grid
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    # Ensure output directory exists
    os.makedirs(saving_loc, exist_ok=True)

    colors = get_color_dict('stability', greyscale)
    pocket_summary = prepare_hierarchical_data(pocket_df, transient_dict)

    # Get unique PDB IDs sorted
    pdb_ids = sorted(pocket_summary['pdb_id'].unique())
    n_pdbs = len(pdb_ids)
    n_rows = int(np.ceil(n_pdbs / n_cols))

    print(f"Creating hierarchical stability plot: {n_pdbs} PDB IDs in {n_rows}x{n_cols} grid")

    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'PDB {pdb}' for pdb in pdb_ids],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # Define x-axis structure for each subplot -- reps are whatever replicates are
    # actually present in the data, not a fixed count, so e.g. a 2-replicate dataset
    # doesn't get a phantom, always-empty "rep 3" bar
    states = ['apo', 'holo']
    reps = sorted(pocket_summary['rep'].dropna().unique(), key=str)
    x_labels = [f'{s} r{r}' for s in states for r in reps]  # ['apo r1', 'apo r2', ..., 'holo r1', ...]

    for idx, pdb_id in enumerate(pdb_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        pdb_data = pocket_summary[pocket_summary['pdb_id'] == pdb_id]

        # Count stable/transient per state+rep
        stable_counts = []
        transient_counts = []

        for state in states:
            for rep in reps:
                mask = (pdb_data['state'] == state) & (pdb_data['rep'] == rep)
                rep_data = pdb_data[mask]

                if len(rep_data) > 0:
                    stable_counts.append((rep_data['stability'] == 'Stable').sum())
                    transient_counts.append((rep_data['stability'] == 'Transient').sum())
                else:
                    stable_counts.append(0)
                    transient_counts.append(0)

        # Add traces (only show legend for first subplot)
        show_legend = (idx == 0)

        fig.add_trace(go.Bar(
            name='Stable',
            x=x_labels,
            y=stable_counts,
            marker_color=colors['Stable'],
            showlegend=show_legend,
            legendgroup='Stable'
        ), row=row, col=col)

        fig.add_trace(go.Bar(
            name='Transient',
            x=x_labels,
            y=transient_counts,
            marker_color=colors['Transient'],
            showlegend=show_legend,
            legendgroup='Transient'
        ), row=row, col=col)

    fig.update_layout(
        barmode='stack',
        title=dict(text='Pocket Stability: PDB ID > State > Replicate', x=0.5, font=dict(size=20)),
        template='plotly_white',
        height=250 * n_rows + 100,
        width=300 * n_cols,
        legend=dict(x=1.02, y=0.98),
        showlegend=True
    )

    # Update all y-axes
    fig.update_yaxes(title_text='Count')

    save_plotly_figure(fig, os.path.join(saving_loc, f'{out_prefix}'), width=300 * n_cols, height=250 * n_rows + 100)

    return fig


def plot_hierarchical_volume_categories(pocket_df, saving_loc,
                                        volume_col='interpolated_pock_volume',
                                        out_prefix='volume_hierarchical',
                                        n_cols=4, greyscale=None):
    """
    Create hierarchical bar plot showing volume categories per PDB ID > State > Replicate.

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume
    out_prefix : str
        Prefix for output filenames
    n_cols : int
        Number of columns in subplot grid
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    # Ensure output directory exists
    os.makedirs(saving_loc, exist_ok=True)

    colors = get_color_dict('volume', greyscale)
    pocket_summary = prepare_hierarchical_data(pocket_df, volume_col=volume_col)

    # Get unique PDB IDs sorted
    pdb_ids = sorted(pocket_summary['pdb_id'].unique())
    n_pdbs = len(pdb_ids)
    n_rows = int(np.ceil(n_pdbs / n_cols))

    print(f"Creating hierarchical volume plot: {n_pdbs} PDB IDs in {n_rows}x{n_cols} grid")

    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'PDB {pdb}' for pdb in pdb_ids],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    states = ['apo', 'holo']
    reps = sorted(pocket_summary['rep'].dropna().unique(), key=str)
    x_labels = [f'{s} r{r}' for s in states for r in reps]

    for idx, pdb_id in enumerate(pdb_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        pdb_data = pocket_summary[pocket_summary['pdb_id'] == pdb_id]

        # Count per category per state+rep
        category_counts = {cat: [] for cat in VOLUME_CATEGORY_ORDER}

        for state in states:
            for rep in reps:
                mask = (pdb_data['state'] == state) & (pdb_data['rep'] == rep)
                rep_data = pdb_data[mask]

                for cat in VOLUME_CATEGORY_ORDER:
                    if len(rep_data) > 0:
                        category_counts[cat].append((rep_data['size_category'] == cat).sum())
                    else:
                        category_counts[cat].append(0)

        show_legend = (idx == 0)

        for cat in VOLUME_CATEGORY_ORDER:
            fig.add_trace(go.Bar(
                name=cat,
                x=x_labels,
                y=category_counts[cat],
                marker_color=colors[cat],
                showlegend=show_legend,
                legendgroup=cat
            ), row=row, col=col)

    fig.update_layout(
        barmode='stack',
        title=dict(text='Pocket Volume Categories: PDB ID > State > Replicate', x=0.5, font=dict(size=20)),
        template='plotly_white',
        height=250 * n_rows + 100,
        width=300 * n_cols,
        legend=dict(x=1.02, y=0.98),
        showlegend=True
    )

    fig.update_yaxes(title_text='Count')

    save_plotly_figure(fig, os.path.join(saving_loc, f'{out_prefix}'), width=300 * n_cols, height=250 * n_rows + 100)

    return fig


def plot_aggregated_by_structure(pocket_df, transient_dict, saving_loc,
                                 volume_col='interpolated_pock_volume',
                                 out_prefix='aggregated_by_structure',
                                 aggregation='median',
                                 group_apo_holo=True,
                                 greyscale=None):
    """
    Create bar plots aggregated across replicates (median/mean) per state+PDB ID.

    For stability: median count of stable/transient pockets across replicates
    For volume categories: categorize first, count, then take median across replicates

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume
    out_prefix : str
        Prefix for output filenames
    aggregation : str
        'median' or 'mean' for aggregating across replicates
    group_apo_holo : bool
        If True, sort x-axis so apo/holo of same PDB are adjacent
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    # Ensure output directory exists
    os.makedirs(saving_loc, exist_ok=True)

    stability_colors = get_color_dict('stability', greyscale)
    volume_colors = get_color_dict('volume', greyscale)

    pocket_summary = prepare_hierarchical_data(pocket_df, transient_dict, volume_col)

    agg_func = np.median if aggregation == 'median' else np.mean

    # === Aggregate stability counts ===
    stability_per_rep = pocket_summary.groupby(['pdb_id', 'state', 'rep', 'stability']).size().unstack(fill_value=0)

    # Aggregate across replicates
    stability_agg = stability_per_rep.groupby(['pdb_id', 'state']).agg(agg_func).round().astype(int)
    stability_agg['structure'] = stability_agg.index.get_level_values('state') + stability_agg.index.get_level_values(
        'pdb_id')
    stability_agg = stability_agg.reset_index()

    # Sort
    if group_apo_holo:
        stability_agg['sort_key'] = stability_agg['pdb_id'] + '_' + stability_agg['state'].map(
            {'apo': '0', 'holo': '1'})
        stability_agg = stability_agg.sort_values('sort_key')
    else:
        stability_agg = stability_agg.sort_values('structure')

    # === Aggregate volume category counts ===
    volume_per_rep = pocket_summary.groupby(['pdb_id', 'state', 'rep', 'size_category']).size().unstack(fill_value=0)
    volume_per_rep = volume_per_rep.reindex(columns=VOLUME_CATEGORY_ORDER, fill_value=0)

    volume_agg = volume_per_rep.groupby(['pdb_id', 'state']).agg(agg_func).round().astype(int)
    volume_agg['structure'] = volume_agg.index.get_level_values('state') + volume_agg.index.get_level_values('pdb_id')
    volume_agg = volume_agg.reset_index()

    if group_apo_holo:
        volume_agg['sort_key'] = volume_agg['pdb_id'] + '_' + volume_agg['state'].map({'apo': '0', 'holo': '1'})
        volume_agg = volume_agg.sort_values('sort_key')
    else:
        volume_agg = volume_agg.sort_values('structure')

    # === PLOT 1: Stability (aggregated) ===
    fig1 = go.Figure()

    for stability in STABILITY_ORDER:
        if stability in stability_agg.columns:
            fig1.add_trace(go.Bar(
                name=stability,
                x=stability_agg['structure'],
                y=stability_agg[stability],
                marker_color=stability_colors[stability],
                text=stability_agg[stability],
                textposition='auto'
            ))

    fig1.update_layout(
        barmode='stack',
        title=dict(text=f'Pocket Stability per Structure ({aggregation} across replicates)',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Structure (State + PDB ID)',
        yaxis_title=f'{aggregation.capitalize()} Number of Pockets',
        legend_title='Stability',
        template='plotly_white',
        height=600, width=1400,
        xaxis_tickangle=-45
    )

    save_plotly_figure(fig1, os.path.join(saving_loc, f'{out_prefix}_stability'), width=1400, height=600)

    # === PLOT 2: Volume categories (aggregated) ===
    fig2 = go.Figure()

    for cat in VOLUME_CATEGORY_ORDER:
        if cat in volume_agg.columns:
            fig2.add_trace(go.Bar(
                name=cat,
                x=volume_agg['structure'],
                y=volume_agg[cat],
                marker_color=volume_colors[cat],
                text=volume_agg[cat],
                textposition='auto'
            ))

    fig2.update_layout(
        barmode='stack',
        title=dict(text=f'Pocket Volume Categories per Structure ({aggregation} across replicates)',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Structure (State + PDB ID)',
        yaxis_title=f'{aggregation.capitalize()} Number of Pockets',
        legend_title='Size Category',
        template='plotly_white',
        height=600, width=1400,
        xaxis_tickangle=-45
    )

    save_plotly_figure(fig2, os.path.join(saving_loc, f'{out_prefix}_volume'), width=1400, height=600)

    # Save summary tables
    stability_agg.to_csv(os.path.join(saving_loc, f'{out_prefix}_stability_summary.csv'), index=False)
    volume_agg.to_csv(os.path.join(saving_loc, f'{out_prefix}_volume_summary.csv'), index=False)
    print(f"Saved: {out_prefix}_stability_summary.csv")
    print(f"Saved: {out_prefix}_volume_summary.csv")

    return fig1, fig2


def plot_overall_distributions(pocket_df, transient_dict, saving_loc,
                               volume_col='interpolated_pock_volume',
                               out_prefix='overall_distribution',
                               greyscale=None):
    """
    Create overall distribution plots considering ALL pockets (no aggregation).

    This shows the true distribution across all pockets in the dataset.

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume
    out_prefix : str
        Prefix for output filenames
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    # Ensure output directory exists
    os.makedirs(saving_loc, exist_ok=True)

    stability_colors = get_color_dict('stability', greyscale)
    volume_colors = get_color_dict('volume', greyscale)

    pocket_summary = prepare_hierarchical_data(pocket_df, transient_dict, volume_col)

    total_pockets = len(pocket_summary)
    print(f"Overall distribution based on {total_pockets} unique pockets")

    # === PLOT 1: Stability pie chart ===
    stability_counts = pocket_summary['stability'].value_counts().reindex(STABILITY_ORDER, fill_value=0)

    fig1 = go.Figure(data=[
        go.Pie(
            labels=stability_counts.index,
            values=stability_counts.values,
            marker_colors=[stability_colors[s] for s in stability_counts.index],
            textinfo='label+percent+value',
            hole=0.3
        )
    ])

    fig1.update_layout(
        title=dict(text=f'Overall Pocket Stability Distribution<br><sub>(n={total_pockets} unique pockets)</sub>',
                   x=0.5, font=dict(size=18)),
        template='plotly_white',
        height=500, width=600
    )

    save_plotly_figure(fig1, os.path.join(saving_loc, f'{out_prefix}_stability_pie'), width=600, height=500)

    # === PLOT 2: Volume category pie chart ===
    volume_counts = pocket_summary['size_category'].value_counts().reindex(VOLUME_CATEGORY_ORDER, fill_value=0)

    fig2 = go.Figure(data=[
        go.Pie(
            labels=volume_counts.index,
            values=volume_counts.values,
            marker_colors=[volume_colors[c] for c in volume_counts.index],
            textinfo='label+percent+value',
            hole=0.3
        )
    ])

    fig2.update_layout(
        title=dict(text=f'Overall Pocket Volume Distribution<br><sub>(n={total_pockets} unique pockets)</sub>',
                   x=0.5, font=dict(size=18)),
        template='plotly_white',
        height=500, width=600
    )

    save_plotly_figure(fig2, os.path.join(saving_loc, f'{out_prefix}_volume_pie'), width=600, height=500)

    # === PLOT 3: Stability by state (apo vs holo) ===
    stability_by_state = pocket_summary.groupby(['state', 'stability']).size().unstack(fill_value=0)
    stability_by_state_pct = stability_by_state.div(stability_by_state.sum(axis=1), axis=0) * 100

    fig3 = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in stability_by_state_pct.columns:
            fig3.add_trace(go.Bar(
                name=stability,
                x=stability_by_state_pct.index,
                y=stability_by_state_pct[stability],
                marker_color=stability_colors[stability],
                text=[f'{v:.1f}%' for v in stability_by_state_pct[stability]],
                textposition='auto'
            ))

    fig3.update_layout(
        barmode='stack',
        title=dict(text='Pocket Stability by State (% of pockets)', x=0.5, font=dict(size=18)),
        xaxis_title='State',
        yaxis_title='Percentage (%)',
        legend_title='Stability',
        template='plotly_white',
        height=500, width=500,
        yaxis=dict(range=[0, 100])
    )

    save_plotly_figure(fig3, os.path.join(saving_loc, f'{out_prefix}_stability_by_state'), width=500, height=500)

    # === PLOT 4: Combined stability + volume (all pockets) ===
    cross_tab = pd.crosstab(pocket_summary['size_category'], pocket_summary['stability'])
    cross_tab = cross_tab.reindex(VOLUME_CATEGORY_ORDER, fill_value=0)

    fig4 = go.Figure()
    for stability in STABILITY_ORDER:
        if stability in cross_tab.columns:
            fig4.add_trace(go.Bar(
                name=stability,
                x=cross_tab.index,
                y=cross_tab[stability],
                marker_color=stability_colors[stability],
                text=cross_tab[stability],
                textposition='auto'
            ))

    fig4.update_layout(
        barmode='group',
        title=dict(text=f'Pocket Stability by Volume Category<br><sub>(n={total_pockets} unique pockets)</sub>',
                   x=0.5, font=dict(size=18)),
        xaxis_title='Volume Category',
        yaxis_title='Number of Pockets',
        legend_title='Stability',
        template='plotly_white',
        height=500, width=800
    )

    save_plotly_figure(fig4, os.path.join(saving_loc, f'{out_prefix}_stability_by_volume'), width=800, height=500)

    # Save summary
    summary = pd.DataFrame({
        'Category': ['Total Pockets', 'Stable', 'Transient', 'Small', 'Medium', 'Large', 'Very Large'],
        'Count': [total_pockets,
                  stability_counts.get('Stable', 0),
                  stability_counts.get('Transient', 0),
                  volume_counts.get('Small (<250)', 0),
                  volume_counts.get('Medium (250-500)', 0),
                  volume_counts.get('Large (500-750)', 0),
                  volume_counts.get('Very Large (>750)', 0)],
        'Percentage': [100.0,
                       100 * stability_counts.get('Stable', 0) / total_pockets,
                       100 * stability_counts.get('Transient', 0) / total_pockets,
                       100 * volume_counts.get('Small (<250)', 0) / total_pockets,
                       100 * volume_counts.get('Medium (250-500)', 0) / total_pockets,
                       100 * volume_counts.get('Large (500-750)', 0) / total_pockets,
                       100 * volume_counts.get('Very Large (>750)', 0) / total_pockets]
    })
    summary.to_csv(os.path.join(saving_loc, f'{out_prefix}_summary.csv'), index=False)
    print(f"Saved: {out_prefix}_summary.csv")

    return fig1, fig2, fig3, fig4


def create_all_distribution_plots(pocket_df, transient_dict, saving_loc,
                                  volume_col='interpolated_pock_volume',
                                  n_cols=4, greyscale=None):
    """
    Convenience function to create ALL distribution plots at once.

    Creates:
    1. Hierarchical plots (PDB ID > State > Replicate) - all replicates shown
    2. Aggregated plots (median across replicates per structure)
    3. Overall distribution plots (all pockets considered)

    Parameters:
    -----------
    pocket_df : DataFrame
        Pocket data
    transient_dict : dict
        Dictionary mapping pocket ID to transient status
    saving_loc : str
        Output directory
    volume_col : str
        Column name for volume
    n_cols : int
        Number of columns for hierarchical subplot grids
    greyscale : bool, optional
        Override global GREYSCALE_MODE
    """
    print("Creating all distribution plots")

    # Create output directory if it doesn't exist
    os.makedirs(saving_loc, exist_ok=True)

    print("1. Hierarchical plots (all replicates)")
    plot_hierarchical_stability(pocket_df, transient_dict, saving_loc,
                                out_prefix='hierarchical_stability',
                                n_cols=n_cols, greyscale=greyscale)

    plot_hierarchical_volume_categories(pocket_df, saving_loc, volume_col,
                                        out_prefix='hierarchical_volume',
                                        n_cols=n_cols, greyscale=greyscale)

    print("2. Aggregated plots (median across replicates)")
    plot_aggregated_by_structure(pocket_df, transient_dict, saving_loc, volume_col,
                                 out_prefix='aggregated_median',
                                 aggregation='median',
                                 group_apo_holo=True, greyscale=greyscale)

    print("3. Overall distribution plots (all pockets)")
    plot_overall_distributions(pocket_df, transient_dict, saving_loc, volume_col,
                               out_prefix='overall',
                               greyscale=greyscale)

    print("All distribution plots created successfully.")


# =============================================================================
# ADDITIONAL VISUALIZATIONS (from original visualizations.py)
# =============================================================================
# These functions include:
# - plot_3d_heatmap: Interactive 3D heatmap with slider
# - plot_3d_apo_holo_comparison: Side-by-side 3D comparison with camera sync
# - vis_apo_holo_number_pockets: Bar charts for pocket counts
# - plot_binding_site_volume_violin: Violin plots with chunked CSV loading

def plot_3d_heatmap(saving_loc, df='first_frame_summary_df.csv',
                    out_html='slider_3D_coordinates_frequency_colour_coded.html',
                    cluster_col='Global ID', genes_of_interest=None, colorscale='Viridis', marker_size=10,
                    title='3D pockets colored by n_structures (first frame)', highlighted_pocket=None):
    if isinstance(df, str):
        df = pd.read_csv(os.path.join(saving_loc, df))

    # Convert numeric columns
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df[cluster_col] = df[cluster_col].astype(str)

    # Extract IDs
    df[['PDB ID', 'replicate', 'pocket']] = df['ID'].str.extract(
        r'^((?:apo|holo)[A-Z0-9]+)_(\d)_([a-z]\d+)(?:_.*)?$'
    )
    df['primary_genes'] = df['gene'].fillna('').astype(str)
    df['state'] = df['PDB ID'].str.extract(r'^(apo|holo)')

    symbol_map = {"apo": "circle", "holo": "square"}

    # Build hover text
    pdb_per_global = (df.groupby('Global ID')['PDB ID'].unique()
                      .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['pdb_list'] = df['Global ID'].map(pdb_per_global)
    gene_per_global = (df.groupby('Global ID')['gene'].unique()
                       .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['gene_list'] = df['Global ID'].map(gene_per_global)

    df['hover_text'] = ("ID: " + df['ID'].astype(str)
                        + "<br>Global: " + df[cluster_col].astype(str)
                        + "<br>PDB: " + df['PDB ID'].astype(str)
                        + "<br>n_structures: " + df['n_structures'].astype(str) if 'n_structures' in df.columns else ""
                                                                                                                     + "<br>in genes: " +
                                                                                                                     df[
                                                                                                                         'gene_list'].astype(
                                                                                                                         str)
                                                                                                                     + "<br>in PDB structures: " +
                                                                                                                     df[
                                                                                                                         'pdb_list'].astype(
                                                                                                                         str))

    # Compute n_structures
    counts = (df.dropna(subset=[cluster_col, 'PDB ID'])
              .groupby(cluster_col)['PDB ID'].nunique()
              .rename('n_structures').reset_index())
    df = df.merge(counts, on=cluster_col, how='left')
    df['n_structures'] = pd.to_numeric(df['n_structures'].fillna(0), downcast='integer')

    # Update hover text with n_structures
    df['hover_text'] = ("ID: " + df['ID'].astype(str)
                        + "<br>Global: " + df[cluster_col].astype(str)
                        + "<br>PDB: " + df['PDB ID'].astype(str)
                        + "<br>n_structures: " + df['n_structures'].astype(str)
                        + "<br>in genes: " + df['gene_list'].astype(str)
                        + "<br>in PDB structures: " + df['pdb_list'].astype(str))

    unique_pdbs = sorted(df['PDB ID'].dropna().unique())
    if genes_of_interest is None:
        genes_of_interest = ['hTAAR1', 'mTAAR1', 'mTAAR7f', 'mTAAR9']

    data_traces = []
    pdb_to_traceidx = {}
    gene_to_traceidxs = {g: [] for g in genes_of_interest}
    idx = 0

    cmin = int(df['n_structures'].min())
    cmax = int(df['n_structures'].max())

    # Build traces - store original sizes for each marker
    for pdb in unique_pdbs:
        sub = df[df['PDB ID'] == pdb].copy()
        if sub.empty:
            continue

        # Create marker sizes array based on threshold
        # We'll update these via the slider
        trace = go.Scatter3d(
            x=sub['x'], y=sub['y'], z=sub['z'],
            mode='markers',
            name=f'PDB {pdb}',
            text=sub['hover_text'],
            hoverinfo='text',
            marker=dict(
                size=marker_size,  # Will be updated by slider
                color=sub['n_structures'],
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                colorbar=dict(title='Structures') if idx == 0 else None,
                symbol=[symbol_map.get(s, "circle") for s in sub['state']]
            ),
            customdata=sub[['ID', 'primary_genes', 'n_structures']].values,
            showlegend=True,
            visible=True
        )
        data_traces.append(trace)
        pdb_to_traceidx[pdb] = [idx]

        present_genes = {g for s in sub['primary_genes'].dropna().astype(str)
                         for g in s.split(';') if g}
        for g in genes_of_interest:
            if g in present_genes:
                gene_to_traceidxs[g].append(idx)
        idx += 1

    highlight_idx = None
    if highlighted_pocket is not None:
        highlight_df = df[df['ID'].isin(highlighted_pocket)].copy()
        if not highlight_df.empty:
            highlight_trace = go.Scatter3d(
                x=highlight_df['x'],
                y=highlight_df['y'],
                z=highlight_df['z'],
                mode='markers',
                name='Highlighted Pockets',
                text=highlight_df['hover_text'],
                hoverinfo='text',
                marker=dict(size=marker_size + 2, color='darkred', symbol='diamond'),
                customdata=highlight_df[['ID', 'primary_genes', 'n_structures']].values,
                showlegend=True,
                visible=True
            )
            data_traces.append(highlight_trace)
            highlight_idx = idx
            idx += 1

    total_traces = len(data_traces)
    vis_all = [True] * total_traces

    # Build dropdowns with legend preservation
    pdb_buttons = [dict(
        label='All PDBs',
        method='restyle',
        args=[
            {'visible': vis_all}
        ]
    )]

    for pdb in unique_pdbs:
        vis = [False] * total_traces
        for i in pdb_to_traceidx[pdb]:
            vis[i] = True
        if highlight_idx is not None:
            vis[highlight_idx] = True
        pdb_buttons.append(dict(
            label=str(pdb),
            method='restyle',
            args=[
                {'visible': vis}]))

    gene_buttons = [dict(
        label='All Genes',
        method='restyle',
        args=[
            {'visible': vis_all}])]

    for g in genes_of_interest:
        vis = [False] * total_traces
        for i in gene_to_traceidxs[g]:
            vis[i] = True
        if highlight_idx is not None:
            vis[highlight_idx] = True
        gene_buttons.append(dict(
            label=g,
            method='restyle',
            args=[
                {'visible': vis}]))

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        template='simple_white',
        showlegend=True,
        legend=dict(
            x=1.12,
            y=0.95,
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=0.5),
        updatemenus=[
            dict(
                buttons=pdb_buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.18,
                yanchor='top',
                bgcolor='white'),
            dict(
                buttons=gene_buttons,
                direction='down',
                showactive=True,
                x=0.28,
                xanchor='left',
                y=1.18,
                yanchor='top',
                bgcolor='white')],
        margin=dict(l=60, r=220, t=120, b=60))

    # Build slider - use marker.size to hide/show individual points
    slider_steps = []
    for threshold in range(cmin, cmax + 1):
        # For each trace, create size array based on threshold
        size_updates = []
        for i, trace in enumerate(data_traces):
            if i == highlight_idx:
                # Keep highlighted pockets always visible
                size_updates.append([marker_size + 2] * len(trace.x))
            else:
                # Set size to 0 for markers above threshold, normal size otherwise
                n_struct_values = trace.customdata[:, 2]
                sizes = [marker_size if val <= threshold else 0 for val in n_struct_values]
                size_updates.append(sizes)

        step = dict(
            method='restyle',
            args=[
                {'marker.size': size_updates},
                list(range(total_traces))],
            label=str(threshold))
        slider_steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=cmax - cmin,  # Start with all pockets visible
            currentvalue={"prefix": "Max structures: "},
            pad={"t": 50},
            steps=slider_steps)])

    # Save outputs - use helper for all formats
    out_base = out_html.replace('.html', '')
    save_plotly_figure(fig, os.path.join(saving_loc, out_base), width=1400, height=900)


def plot_3d_apo_holo_comparison(saving_loc, df='first_frame_summary_df.csv',
                                out_html='apo_holo_comparison_3D.html',
                                cluster_col='Global ID', colorscale='Viridis', marker_size=4,
                                title='Apo vs Holo Pocket Comparison'):
    """
    Creates side-by-side 3D scatter plots comparing apo and holo protein structures.

    Parameters:
    -----------
    saving_loc : str
        Directory path for input CSV and output HTML
    df : str or DataFrame
        Either a CSV filename or a pandas DataFrame
    out_html : str
        Output HTML filename
    cluster_col : str
        Column name for cluster/global IDs
    colorscale : str
        Plotly colorscale name
    marker_size : int
        Size of scatter plot markers
    title : str
        Plot title
    """
    if isinstance(df, str):
        df = pd.read_csv(os.path.join(saving_loc, df))

    # Convert numeric columns
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df[cluster_col] = df[cluster_col].astype(str)

    # Extract IDs - example: apoITF_1_p01_i01
    df[['PDB ID', 'replicate', 'pocket', 'isovalue']] = df['ID'].str.extract(
        r'^((?:apo|holo)[A-Z0-9]+)_(\d+)_([a-z]\d+)(?:_([i]\d+))?$' )
    df['primary_genes'] = df['gene'].fillna('').astype(str)
    df['state'] = df['PDB ID'].str.extract(r'^(apo|holo)')
    df['base_pdb'] = df['PDB ID'].str.replace(r'^(apo|holo)', '', regex=True)

    # Use ID directly as the pocket identifier!
    # Strip apo/holo prefix so matching pockets get same color
    df['pocket_identifier'] = df['ID'].str.replace(r'^(apo|holo)', '', regex=True)

    # Store unique_ids before it's used at the end
    unique_ids = df['ID'].unique()

    # Keep Global ID mapping for reference
    unique_global_ids = sorted(df[cluster_col].unique())
    global_id_to_num = {gid: i for i, gid in enumerate(unique_global_ids)}
    df['global_id_num'] = df[cluster_col].map(global_id_to_num)
    pdb_per_global = (df.groupby('Global ID')['PDB ID'].unique()
                      .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['pdb_list'] = df['Global ID'].map(pdb_per_global)
    gene_per_global = (df.groupby('Global ID')['gene'].unique()
                       .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['gene_list'] = df['Global ID'].map(gene_per_global)

    # Compute n_structures
    counts = (df.dropna(subset=[cluster_col, 'PDB ID'])
              .groupby(cluster_col)['PDB ID'].nunique()
              .rename('n_structures').reset_index())
    df = df.merge(counts, on=cluster_col, how='left')
    df['n_structures'] = pd.to_numeric(df['n_structures'].fillna(0), downcast='integer')

    df['hover_text'] = ("ID: " + df['ID'].astype(str)
                        + "<br>Global: " + df[cluster_col].astype(str)
                        + "<br>PDB: " + df['PDB ID'].astype(str)
                        + "<br>State: " + df['state'].astype(str)
                        + "<br>n_structures: " + df['n_structures'].astype(str)
                        + "<br>Gene: " + df['gene_list'].astype(str))

    # Find PDBs that have both apo and holo versions
    pdb_states = df.groupby('base_pdb')['state'].unique()
    paired_pdbs = sorted([base for base, states in pdb_states.items()
                          if len(states) == 2 and 'apo' in states and 'holo' in states])

    if not paired_pdbs:
        print("No PDB IDs found with both apo and holo states!")
        return

    # FIX 1: Create numeric mapping for local IDs (pocket IDs) to use with colorscale
    # Each unique pocket ID gets a different color
    unique_pocket_ids = sorted(df['ID'].unique())
    id_to_num = {pid: i for i, pid in enumerate(unique_pocket_ids)}
    df['id_num'] = df['ID'].map(id_to_num)

    # Normalize to 0-1 range for colorscale
    if len(unique_pocket_ids) > 1:
        df['color_value'] = df['id_num'] / (len(unique_pocket_ids) - 1)
    else:
        df['color_value'] = 0.5  # Single pocket case

    # FIX 2: Store unique_ids before it's used at the end
    unique_ids = df['ID'].unique()

    cmin = 0
    cmax = len(unique_pocket_ids) - 1

    # Keep Global ID mapping for reference
    unique_global_ids = sorted(df[cluster_col].unique())
    global_id_to_num = {gid: i for i, gid in enumerate(unique_global_ids)}
    df['global_id_num'] = df[cluster_col].map(global_id_to_num)

    # Create subplots with three 3D scenes side by side
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Holo - Each Pocket Colored', 'Apo - Each Pocket Colored', 'Overlay (Yellow=Holo, Purple=Apo)'],
        horizontal_spacing=0.03)

    # Create traces for each paired PDB and replicate combination
    all_traces = []
    for base_pdb in paired_pdbs:
        # Get all replicates for this PDB
        holo_pdb = f'holo{base_pdb}'
        apo_pdb = f'apo{base_pdb}'

        pdb_data = df[(df['base_pdb'] == base_pdb)]
        available_reps = sorted(pdb_data['replicate'].dropna().unique())

        # Create traces for "All replicates" option
        for rep_option in ['All'] + list(available_reps):
            if rep_option == 'All':
                holo_data = df[df['PDB ID'] == holo_pdb].copy()
                apo_data = df[df['PDB ID'] == apo_pdb].copy()
                trace_suffix = 'all_reps'
            else:
                holo_data = df[(df['PDB ID'] == holo_pdb) & (df['replicate'] == rep_option)].copy()
                apo_data = df[(df['PDB ID'] == apo_pdb) & (df['replicate'] == rep_option)].copy()
                trace_suffix = f'rep_{rep_option}'

            # Create LOCAL color mapping for this specific combination
            pair_data = pd.concat([holo_data, apo_data])
            unique_pockets_this_combo = sorted(pair_data['ID'].dropna().unique())
            local_id_to_num = {pid: i for i, pid in enumerate(unique_pockets_this_combo)}

            holo_data['local_id_num'] = holo_data['ID'].map(local_id_to_num)
            apo_data['local_id_num'] = apo_data['ID'].map(local_id_to_num)

            local_cmin = 0
            local_cmax = max(len(unique_pockets_this_combo) - 1, 1)

            # Holo trace
            if not holo_data.empty:
                holo_trace = go.Scatter3d(
                    x=holo_data['x'],
                    y=holo_data['y'],
                    z=holo_data['z'],
                    mode='markers',
                    name=f'{holo_pdb}_rep{rep_option}',
                    text=holo_data['hover_text'],
                    hoverinfo='text',
                    marker=dict(
                        size=marker_size,
                        color=holo_data['local_id_num'],
                        colorscale=colorscale,
                        cmin=local_cmin,
                        cmax=local_cmax,
                        colorbar=dict(
                            title='Pocket',
                            x=1.01,
                            len=0.5,
                            tickmode='linear',
                            tick0=0,
                            dtick=max(1, local_cmax // 10)),
                        symbol='square',
                        line=dict(width=0.5, color='white')),
                    customdata=holo_data[['ID', 'primary_genes', 'n_structures', cluster_col]].values,
                    showlegend=True,
                    visible=False,
                    legendgroup=f'{base_pdb}_{trace_suffix}')
            else:
                holo_trace = go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='markers',
                    name=f'{holo_pdb}_rep{rep_option} (no data)',
                    showlegend=True,
                    visible=False,
                    legendgroup=f'{base_pdb}_{trace_suffix}')

            # Apo trace
            if not apo_data.empty:
                apo_trace = go.Scatter3d(
                    x=apo_data['x'],
                    y=apo_data['y'],
                    z=apo_data['z'],
                    mode='markers',
                    name=f'{apo_pdb}_rep{rep_option}',
                    text=apo_data['hover_text'],
                    hoverinfo='text',
                    marker=dict(
                        size=marker_size,
                        color=apo_data['local_id_num'],
                        colorscale=colorscale,
                        cmin=local_cmin,
                        cmax=local_cmax,
                        showscale=False,
                        symbol='circle',
                        line=dict(width=0.5, color='white')),
                    customdata=apo_data[['ID', 'primary_genes', 'n_structures', cluster_col]].values,
                    showlegend=True,
                    visible=False,
                    legendgroup=f'{base_pdb}_{trace_suffix}')
            else:
                apo_trace = go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='markers',
                    name=f'{apo_pdb}_rep{rep_option} (no data)',
                    showlegend=True,
                    visible=False,
                    legendgroup=f'{base_pdb}_{trace_suffix}' )

            # Overlay - Holo (yellow-green from Viridis - distinct from apo blues/purples)
            overlay_holo_trace = go.Scatter3d(
                x=holo_data['x'] if not holo_data.empty else [],
                y=holo_data['y'] if not holo_data.empty else [],
                z=holo_data['z'] if not holo_data.empty else [],
                mode='markers',
                name=f'{holo_pdb}_rep{rep_option} (Holo)',
                text=holo_data['hover_text'] if not holo_data.empty else [],
                hoverinfo='text',
                marker=dict(
                    size=marker_size,
                    color='#FDE725',  # Bright yellow from Viridis
                    symbol='square',
                    line=dict(width=0.5, color='white')),
                customdata=holo_data[
                    ['ID', 'primary_genes', 'n_structures', cluster_col]].values if not holo_data.empty else [],
                showlegend=True,
                visible=False,
                legendgroup=f'{base_pdb}_{trace_suffix}_overlay')

            # Overlay - Apo (purple to match Viridis purple/blue range)
            overlay_apo_trace = go.Scatter3d(
                x=apo_data['x'] if not apo_data.empty else [],
                y=apo_data['y'] if not apo_data.empty else [],
                z=apo_data['z'] if not apo_data.empty else [],
                mode='markers',
                name=f'{apo_pdb}_rep{rep_option} (Apo)',
                text=apo_data['hover_text'] if not apo_data.empty else [],
                hoverinfo='text',
                marker=dict(
                    size=marker_size,
                    color='#440154',  # Dark purple
                    symbol='circle',
                    line=dict(width=0.5, color='white')),
                customdata=apo_data[['ID', 'primary_genes', 'n_structures', cluster_col]].values if not apo_data.empty else [],
                showlegend=True,
                visible=False,
                legendgroup=f'{base_pdb}_{trace_suffix}_overlay')

            all_traces.append((holo_trace, apo_trace, overlay_holo_trace, overlay_apo_trace, base_pdb, rep_option))

    # Add all traces to figure
    for holo_trace, apo_trace, overlay_holo_trace, overlay_apo_trace, _, _ in all_traces:
        fig.add_trace(holo_trace, row=1, col=1)
        fig.add_trace(apo_trace, row=1, col=2)
        fig.add_trace(overlay_holo_trace, row=1, col=3)
        fig.add_trace(overlay_apo_trace, row=1, col=3)

    # Make first combination visible by default
    if all_traces:
        fig.data[0].visible = True  # holo
        fig.data[1].visible = True  # apo
        fig.data[2].visible = True  # overlay holo
        fig.data[3].visible = True  # overlay apo

    # Create dropdown buttons for combined PDB and Replicate selection
    # Build a proper index mapping
    trace_index_map = {}  # (pdb, rep) -> trace_start_index
    for idx, (_, _, _, _, base_pdb, rep_option) in enumerate(all_traces):
        trace_index_map[(base_pdb, rep_option)] = idx * 4

    # Get all unique replicates across all PDBs
    all_reps_set = set()
    for base_pdb in paired_pdbs:
        pdb_data = df[df['base_pdb'] == base_pdb]
        reps = pdb_data['replicate'].dropna().unique()
        all_reps_set.update(reps)
    all_reps = ['All'] + sorted(list(all_reps_set))

    # Create combined buttons for PDB selection
    pdb_buttons = []
    for base_pdb in paired_pdbs:
        button = dict(
            label=base_pdb,
            method='skip',  # handled this with JavaScript
            args=[])
        pdb_buttons.append(button)

    # Create buttons for replicate selection
    rep_buttons = []
    for rep in all_reps:
        button = dict( label=f'Rep {rep}',
            method='skip',  # handled this with JavaScript
            args=[])
        rep_buttons.append(button)

    # Update layout with two dropdown menus
    fig.update_layout(
        title=f'{title} - {paired_pdbs[0] if paired_pdbs else ""} - All Replicates',
        template='simple_white',
        showlegend=True,
        legend=dict(
            x=1.01,
            y=0.95,
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=0.5),
        updatemenus=[
            # PDB selector (left dropdown)
            dict(
                buttons=pdb_buttons,
                direction='down',
                showactive=True,
                x=0.35,
                xanchor='center',
                y=1.12,
                yanchor='top',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1),
            # Replicate selector (right dropdown)
            dict(
                buttons=rep_buttons,
                direction='down',
                showactive=True,
                x=0.65,
                xanchor='center',
                y=1.12,
                yanchor='top',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1)],
        margin=dict(l=60, r=250, t=120, b=60),height=900, width=2700)

    # Make subplot titles larger and more visible
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, color='black', family='Arial Black')

    # Update 3D scene properties with synchronized cameras
    scene_props = dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    fig.update_layout(scene=scene_props, scene2=scene_props, scene3=scene_props)

    # Add JavaScript to synchronize cameras and handle combined dropdown selections
    # Build the trace index map for JavaScript
    trace_map_js = {}
    for base_pdb in paired_pdbs:
        trace_map_js[base_pdb] = {}
        for rep_option in all_reps:
            key = (base_pdb, rep_option)
            if key in trace_index_map:
                trace_map_js[base_pdb][str(rep_option)] = trace_index_map[key]
    # Java written by Claude AI
    sync_js = """   
    <script>
    (function() {
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        var isUpdating = false;

        // Store current selections
        var currentPDB = '""" + (paired_pdbs[0] if paired_pdbs else "") + """';
        var currentRep = 'All';

        // Trace index mapping: traceMap[pdb][rep] = starting_trace_index
        var traceMap = """ + str(trace_map_js).replace("'", '"') + """;

        // Function to update visibility based on current selections
        function updateVisibility() {
            var vis = new Array(gd.data.length).fill(false);

            if (traceMap[currentPDB]) {
                if (currentRep === 'All') {
                    // Show all replicates for this PDB
                    Object.keys(traceMap[currentPDB]).forEach(function(rep) {
                        if (rep !== 'All') {  // Don't include 'All' key if it exists
                            var idx = traceMap[currentPDB][rep];
                            if (idx !== undefined && idx !== null) {
                                vis[idx] = true;     // holo
                                vis[idx + 1] = true; // apo
                                vis[idx + 2] = true; // overlay holo
                                vis[idx + 3] = true; // overlay apo
                            }
                        }
                    });
                    // Also check if there's an 'All' entry
                    if (traceMap[currentPDB]['All'] !== undefined) {
                        var idx = traceMap[currentPDB]['All'];
                        vis[idx] = true;
                        vis[idx + 1] = true;
                        vis[idx + 2] = true;
                        vis[idx + 3] = true;
                    }
                } else {
                    // Show specific replicate only
                    var idx = traceMap[currentPDB][currentRep];
                    if (idx !== undefined && idx !== null) {
                        vis[idx] = true;     // holo
                        vis[idx + 1] = true; // apo
                        vis[idx + 2] = true; // overlay holo
                        vis[idx + 3] = true; // overlay apo
                    }
                }
            }

            Plotly.restyle(gd, 'visible', vis.map(function(v) { return v; }));
            var repText = currentRep === 'All' ? 'All Replicates' : 'Replicate ' + currentRep;
            Plotly.relayout(gd, {'title': '""" + title + """ - ' + currentPDB + ' - ' + repText});
        }

        // Camera synchronization
        gd.on('plotly_relayout', function(eventData) {
            if (isUpdating) return;

            var scene1Camera = eventData['scene.camera'];
            var scene2Camera = eventData['scene2.camera'];
            var scene3Camera = eventData['scene3.camera'];

            if (scene1Camera) {
                isUpdating = true;
                Plotly.relayout(gd, {
                    'scene2.camera': scene1Camera,
                    'scene3.camera': scene1Camera
                }).then(function() {
                    isUpdating = false;
                });
            } else if (scene2Camera) {
                isUpdating = true;
                Plotly.relayout(gd, {
                    'scene.camera': scene2Camera,
                    'scene3.camera': scene2Camera
                }).then(function() {
                    isUpdating = false;
                });
            } else if (scene3Camera) {
                isUpdating = true;
                Plotly.relayout(gd, {
                    'scene.camera': scene3Camera,
                    'scene2.camera': scene3Camera
                }).then(function() {
                    isUpdating = false;
                });
            }
        });

        // Handle dropdown button clicks
        gd.on('plotly_buttonclicked', function(eventData) {
            if (eventData.menu.x === 0.35) {
                // PDB dropdown clicked (left dropdown)
                currentPDB = eventData.button.label;
            } else if (eventData.menu.x === 0.65) {
                // Replicate dropdown clicked (right dropdown)
                currentRep = eventData.button.label.replace('Rep ', '');
            }

            // Update visibility after selection change
            updateVisibility();

            return false; // Prevent default Plotly behavior
        });

        // Initialize with first PDB and All replicates
        updateVisibility();
    })();
    </script>
    """

    # Save output with synchronized camera script
    html_string = fig.to_html()
    # Insert the sync script before the closing body tag
    html_string = html_string.replace('</body>', sync_js + '</body>')

    output_path = os.path.join(saving_loc, out_html)
    with open(output_path, 'w') as f:
        f.write(html_string)

    # Also save PNG and PDF (static versions)
    out_base = out_html.replace('.html', '')
    saved_formats = ['html']
    try:
        png_path = os.path.join(saving_loc, f"{out_base}.png")
        fig.write_image(png_path, format='png', width=2700, height=900, scale=DEFAULT_SCALE)
        saved_formats.append('png')

        pdf_path = os.path.join(saving_loc, f"{out_base}.pdf")
        fig.write_image(pdf_path, format='pdf', width=2700, height=900, scale=DEFAULT_SCALE)
        saved_formats.append('pdf')
    except Exception as e:
        print(f"  WARNING: Could not export PNG/PDF ({type(e).__name__}). Install kaleido: pip install -U kaleido")

    print(f"Saved {out_base} ({', '.join(saved_formats)})")

    print(f"Found {len(paired_pdbs)} PDB IDs with both apo and holo states")
    print(f"Total unique pocket IDs: {len(unique_ids)}")


def vis_apo_holo_number_pockets(df, figsize=(14, 6), save_path=None, version='median'):
    """
    Visualize the number of pockets found per PDB ID, separated by apo and holo states.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pocketome data with 'ID' column in format:
        {state}{PDBID}_{replicate}_p{pocket_num}_i{isovalue}
        Example: apo8ITF_1_p01_i3
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 6)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    version : str, optional
        'median' - Plot median pocket counts across replicates (default)
        'replicates' - Plot one subplot per replicate actually present in the data

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Parse the ID column to extract all components
    def parse_id(id_string):
        """Extract state, PDB ID, replicate, and pocket number from ID string"""
        match = re.match(r'(apo|holo)(\w+)_(\d+)_p(\d+)_i(\d+)', id_string)
        if match:
            state = match.group(1)
            pdb_id = match.group(2)
            replicate = match.group(3)
            pocket_local = match.group(4)
            isovalue = match.group(5)
            return state, pdb_id, replicate, pocket_local, isovalue
        return None, None, None, None, None

    # Apply parsing to create new columns
    df_parsed = df.copy()
    df_parsed[['state', 'pdb_id', 'replicate', 'pocket_local', 'isovalue']] = df_parsed['ID'].apply(
        lambda x: pd.Series(parse_id(x))
    )

    # Count unique pockets per state/PDB/replicate combination
    # Each unique combination of pocket_local represents one pocket
    pocket_counts = df_parsed.groupby(['state', 'pdb_id', 'replicate', 'pocket_local']).size().reset_index(
        name='voxel_count')
    pocket_summary = pocket_counts.groupby(['state', 'pdb_id', 'replicate']).size().reset_index(name='num_pockets')

    # Define colors -- sand / mauve, matching STATE_COLORS
    apo_color = STATE_COLORS['apo']
    holo_color = STATE_COLORS['holo']

    if version == 'median':
        return _plot_median(pocket_summary, figsize, save_path, apo_color, holo_color)
    elif version == 'replicates':
        return _plot_replicates(pocket_summary, figsize, save_path, apo_color, holo_color)
    else:
        raise ValueError("version must be 'median' or 'replicates'")


def _plot_median(pocket_summary, figsize, save_path, apo_color, holo_color):
    """Create a single plot with median pocket counts across replicates"""

    # Calculate median across replicates for each state/PDB combination
    median_counts = pocket_summary.groupby(['state', 'pdb_id'])['num_pockets'].median().reset_index()

    # Pivot to get apo and holo counts side by side
    pivot_data = median_counts.pivot(index='pdb_id', columns='state', values='num_pockets').fillna(0)

    # Ensure both apo and holo columns exist
    if 'apo' not in pivot_data.columns:
        pivot_data['apo'] = 0
    if 'holo' not in pivot_data.columns:
        pivot_data['holo'] = 0

    # Sort by total number of pockets (descending)
    pivot_data['total'] = pivot_data['apo'] + pivot_data['holo']
    pivot_data = pivot_data.sort_values('total', ascending=False)
    pivot_data = pivot_data.drop('total', axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    x = np.arange(len(pivot_data))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width / 2, pivot_data['apo'], width, label='Apo',
                   color=apo_color, alpha=0.9, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width / 2, pivot_data['holo'], width, label='Holo',
                   color=holo_color, alpha=0.9, edgecolor='black', linewidth=0.7)

    # Customize the plot
    ax.set_xlabel('PDB ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median Number of Pockets', fontsize=13, fontweight='bold')
    n_reps = pocket_summary['replicate'].nunique()
    ax.set_title(f'Median Number of Pockets per PDB ID: Apo vs Holo States\n(across {n_reps} replicates)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(pivot_data[['apo', 'holo']].max()) * 1.15)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided - save both PNG and PDF
    if save_path:
        # Remove extension if present
        save_base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        save_matplotlib_figure(fig, save_base)

    return fig, ax


def _plot_replicates(pocket_summary, figsize, save_path, apo_color, holo_color):
    """Create one subplot per replicate actually present in the data, stacked vertically"""

    replicates = sorted(pocket_summary['replicate'].dropna().unique(), key=str)
    n_reps = len(replicates)

    fig, axes = plt.subplots(n_reps, 1, figsize=(figsize[0], figsize[1] * 2.5 * n_reps / 3), sharex=True)
    if n_reps == 1:
        axes = [axes]

    # Process each replicate
    for rep_idx, replicate in enumerate(replicates):
        ax = axes[rep_idx]

        rep_data = pocket_summary[pocket_summary['replicate'] == replicate].copy()

        # Pivot to get apo and holo counts side by side
        pivot_data = rep_data.pivot(index='pdb_id', columns='state', values='num_pockets').fillna(0)

        # Ensure both apo and holo columns exist
        if 'apo' not in pivot_data.columns:
            pivot_data['apo'] = 0
        if 'holo' not in pivot_data.columns:
            pivot_data['holo'] = 0

        # Sort by total number of pockets (descending)
        pivot_data['total'] = pivot_data['apo'] + pivot_data['holo']
        pivot_data = pivot_data.sort_values('total', ascending=False)
        pivot_data = pivot_data.drop('total', axis=1)

        # Set up bar positions
        x = np.arange(len(pivot_data))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width / 2, pivot_data['apo'], width, label='Apo',
                       color=apo_color, alpha=0.9, edgecolor='black', linewidth=0.7)
        bars2 = ax.bar(x + width / 2, pivot_data['holo'], width, label='Holo',
                       color=holo_color, alpha=0.9, edgecolor='black', linewidth=0.7)

        # Customize the subplot
        ax.set_ylabel('Number of Pockets', fontsize=12, fontweight='bold')
        ax.set_title(f'Replicate {replicate}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 30)  # Fixed scale for comparison

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        add_value_labels(bars1)
        add_value_labels(bars2)

    # Set common x-axis label
    axes[-1].set_xlabel('PDB ID', fontsize=13, fontweight='bold')

    # Overall title
    fig.suptitle('Number of Pockets per PDB ID: Apo vs Holo States\n(Individual Replicates)',
                 fontsize=15, fontweight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided - save both PNG and PDF
    if save_path:
        # Remove extension if present
        save_base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        save_matplotlib_figure(fig, save_base)

    return fig, axes


# Volume of binding site

def _load_summary_data_for_binding_site(summary_csv_path):
    """Load the (prj, rep, pocket_number, snapshot, interpolated_pock_volume, ID) columns
    needed by plot_binding_site_volume_violin, in chunks (the source CSV can be too large to
    read at once)."""
    if not os.path.exists(summary_csv_path):
        raise FileNotFoundError(f"File not found: {summary_csv_path}")

    needed_cols = ['prj', 'rep', 'pocket_number', 'snapshot', 'interpolated_pock_volume', 'ID']
    chunks = [chunk for chunk in pd.read_csv(summary_csv_path, usecols=needed_cols, chunksize=200000)]
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {summary_csv_path}")

    return df


def plot_binding_site_volume_violin(pocket_comparison_df, summary_df=None,
                                    saving_loc='.', summary_csv_path=None,
                                    create_individual_plots=True):
    """
    Create violin plots of BINDING SITE pocket volume distribution.
    Shows apo vs holo for each PDB ID.

    Creates:
    1. One combined plot with ALL PDB IDs (apo vs holo for each)
    2. Individual plots per PDB ID (if create_individual_plots=True)

    Logic:
    1. Filter binding site pockets from pocket_comparison_df
    2. Get Local Pocket IDs for binding sites
    3. Filter summary_df for these pockets
    4. Group by: PDB ID, state (apo/holo), frame
    5. Calculate median volume per frame ACROSS REPLICATES
    6. Create violin plots

    Parameters:
    -----------
    pocket_comparison_df : pandas.DataFrame
        Pocket comparison table with 'Local Pocket ID' and 'is_orthosteric' columns
    summary_df : pandas.DataFrame, optional
        Full summary with volume data. If None, loads from summary_csv_path
    saving_loc : str
        Directory to save plot
    summary_csv_path : str, optional
        Path to summary_df_3d_coords.csv
    create_individual_plots : bool
        If True, creates one plot per PDB ID in addition to the combined plot

    Returns:
    --------
    dict : Paths to generated plot files
        {'combined': path, 'individual': [list of paths]}
    """
    violin_dir = os.path.join(saving_loc, 'violin_plots_volume')
    os.makedirs(violin_dir, exist_ok=True)

    ortho_col = 'is_orthosteric'
    if ortho_col not in pocket_comparison_df.columns:
        print(f"ERROR: No '{ortho_col}' column found in pocket_comparison_df "
              f"(available: {list(pocket_comparison_df.columns)})")
        return None

    # Check for ID column (support both 'Local Pocket ID' and 'ID')
    if 'Local Pocket ID' in pocket_comparison_df.columns:
        id_col = 'Local Pocket ID'
    elif 'ID' in pocket_comparison_df.columns:
        id_col = 'ID'
    else:
        print(f"ERROR: No ID column found in pocket_comparison_df. Expected 'Local Pocket ID' or 'ID' "
              f"(available: {list(pocket_comparison_df.columns)})")
        return None

    binding_pockets = pocket_comparison_df[
        pocket_comparison_df[ortho_col] == True
        ].copy()

    if len(binding_pockets) == 0:
        print(f"ERROR: No orthosteric pockets found ({ortho_col} == True)")
        return None

    binding_local_ids = binding_pockets[id_col].unique()

    # Load summary data
    if summary_df is None:
        if summary_csv_path is None:
            summary_csv_path = os.path.join(saving_loc, 'summary_df_3d_coords.csv')
        summary_df = _load_summary_data_for_binding_site(summary_csv_path)

    if 'snapshot' not in summary_df.columns:
        print("ERROR: summary_df has no 'snapshot' column -- can't build a per-frame volume series")
        return None

    # Extract state and clean PDB ID
    summary_df = summary_df.copy()
    summary_df['state'] = summary_df['prj'].str.extract(r'^(apo|holo)', expand=False)
    summary_df['clean_id'] = summary_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)

    # A pocket has multiple rows per real MD frame -- one per alpha sphere/dummy atom, all
    # sharing the same volume (see consecutive_zeros_transiency._per_frame_volumes) -- so this
    # must dedupe to one row per (pocket, snapshot) before using snapshot as the frame axis.
    # Using a raw per-group row counter here instead would silently number alpha spheres, not
    # frames, and make adjacent "frames" jump between unrelated volumes.
    summary_df = summary_df.drop_duplicates(subset=['ID', 'snapshot'])
    summary_df = summary_df.dropna(subset=['state', 'clean_id', 'interpolated_pock_volume', 'ID'])

    # Filter for binding site pockets. binding_local_ids may be full IDs (with the isovalue
    # suffix, e.g. 'apo8ITF_1_p10_i3.0') or local IDs (without it, e.g. 'apo8ITF_1_p10').
    sample_id = str(binding_local_ids[0]) if len(binding_local_ids) > 0 else ""
    is_full_id = '_i' in sample_id

    if is_full_id:
        binding_data = summary_df[summary_df['ID'].isin(binding_local_ids)].copy()
    else:
        binding_data_list = [summary_df[summary_df['ID'].str.startswith(str(local_id) + '_i')]
                             for local_id in binding_local_ids]
        binding_data_list = [d for d in binding_data_list if len(d) > 0]
        if not binding_data_list:
            print("ERROR: No orthosteric data found in summary")
            return None
        binding_data = pd.concat(binding_data_list, ignore_index=True)

    if len(binding_data) == 0:
        print("ERROR: No orthosteric data found in summary")
        return None

    print(f"Orthosteric binding-site volumes: {len(binding_local_ids)} pockets, "
         f"{binding_data['clean_id'].nunique()} PDB ID(s), {len(binding_data):,} (pocket, frame) rows")

    # Median binding-site volume at each real MD frame (snapshot), across every replicate/
    # pocket that contributes to that (PDB ID, state) at that snapshot
    median_volumes = binding_data.groupby(
        ['clean_id', 'state', 'snapshot']
    )['interpolated_pock_volume'].median().reset_index()
    median_volumes.columns = ['clean_id', 'state', 'snapshot', 'median_volume']

    print(f"Median binding-site volume per (PDB ID, state, snapshot) -- {len(median_volumes):,} rows, "
         f"first 10 shown:")
    print(median_volumes.head(10))

    print("Median binding-site volume per PDB ID and state, summarized across all snapshots:")
    summary_stats = median_volumes.groupby(['clean_id', 'state'])['median_volume'].agg(
        ['count', 'mean', 'std', 'min', 'max']
    ).round(2)
    print(summary_stats)

    # Create COMBINED violin plot (all PDB IDs)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    # Color scheme -- sand / mauve, matching STATE_COLORS
    APO_COLOR = STATE_COLORS['apo']
    HOLO_COLOR = STATE_COLORS['holo']

    pdb_ids = sorted(median_volumes['clean_id'].unique())
    n_pdbs = len(pdb_ids)

    # Create figure - wide enough for all PDB IDs
    fig, ax = plt.subplots(figsize=(max(16, n_pdbs * 1.5), 8))

    plot_data = []
    positions = []
    colors = []

    for i, pdb_id in enumerate(pdb_ids):
        # Apo state
        apo_data = median_volumes[
            (median_volumes['clean_id'] == pdb_id) &
            (median_volumes['state'] == 'apo')
            ]['median_volume'].values

        if len(apo_data) > 0:
            plot_data.append(apo_data)
            positions.append(i * 3)
            colors.append(APO_COLOR)

        # Holo state
        holo_data = median_volumes[
            (median_volumes['clean_id'] == pdb_id) &
            (median_volumes['state'] == 'holo')
            ]['median_volume'].values

        if len(holo_data) > 0:
            plot_data.append(holo_data)
            positions.append(i * 3 + 1)
            colors.append(HOLO_COLOR)

    # Create violin plot
    parts = ax.violinplot(plot_data, positions=positions, widths=0.8,
                          showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Customize other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(2)

    # Labels and title
    ax.set_xlabel('PDB ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Binding Site Volume (Å³)', fontsize=14, fontweight='bold')
    ax.set_title('Binding Site Volume Distribution: Apo vs Holo (All PDB IDs)',
                 fontsize=16, fontweight='bold', pad=20)

    # X-axis labels
    pdb_positions = [i * 3 + 0.5 for i in range(n_pdbs)]
    ax.set_xticks(pdb_positions)
    ax.set_xticklabels(pdb_ids, rotation=45, ha='right', fontsize=10)

    # Legend
    legend_elements = [
        Patch(facecolor=APO_COLOR, edgecolor='black', label='Apo', alpha=0.8),
        Patch(facecolor=HOLO_COLOR, edgecolor='black', label='Holo', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fontsize=12, title='State')

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save combined plot
    combined_path = os.path.join(violin_dir, "violin_plot_binding_site_ALL_PDBs.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')

    combined_pdf = os.path.join(violin_dir, "violin_plot_binding_site_ALL_PDBs.pdf")
    plt.savefig(combined_pdf, bbox_inches='tight')
    print(f"Saved combined plot: {os.path.basename(combined_path)} (png, pdf)")

    plt.close()

    # Create INDIVIDUAL plots per PDB ID
    individual_paths = []

    if create_individual_plots:
        for pdb_id in pdb_ids:
            # Filter data for this PDB ID
            pdb_data = median_volumes[median_volumes['clean_id'] == pdb_id]

            # Get apo and holo data
            apo_data = pdb_data[pdb_data['state'] == 'apo']['median_volume'].values
            holo_data = pdb_data[pdb_data['state'] == 'holo']['median_volume'].values

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))

            plot_data = []
            labels = []
            colors = []
            positions = []

            if len(apo_data) > 0:
                plot_data.append(apo_data)
                labels.append('Apo')
                colors.append(APO_COLOR)
                positions.append(0)

            if len(holo_data) > 0:
                plot_data.append(holo_data)
                labels.append('Holo')
                colors.append(HOLO_COLOR)
                positions.append(1)

            if len(plot_data) == 0:
                plt.close()
                continue

            # Create violin plot
            parts = ax.violinplot(plot_data, positions=positions, widths=0.6,
                                  showmeans=True, showmedians=True)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.8)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

            # Customize other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(2)

            # Labels and title
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('Median Binding Site Volume (Å³)', fontsize=12, fontweight='bold')
            ax.set_title(f'{pdb_id} Binding Site: Apo vs Holo',
                         fontsize=14, fontweight='bold', pad=15)

            # Grid
            ax.yaxis.grid(True, alpha=0.3, linewidth=0.8)
            ax.set_axisbelow(True)

            plt.tight_layout()

            # Save individual plot - both PNG and PDF
            individual_base = os.path.join(violin_dir, f"violin_plot_binding_site_{pdb_id}")
            save_matplotlib_figure(fig, individual_base)
            individual_paths.append(f"{individual_base}.png")

            plt.close()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"Generated {1 + len(individual_paths)} plots (PNG + PDF each):")
    print(f"  - 1 combined plot (all PDB IDs)")
    print(f"  - {len(individual_paths)} individual plots")

    return {
        'combined': combined_path,
        'combined_pdf': combined_pdf,
        'individual': individual_paths
    }