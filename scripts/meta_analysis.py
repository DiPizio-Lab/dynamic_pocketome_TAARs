""""this is a script for xxxx"""
import pandas as pd
from matplotlib import cm
from scripts import config as conf
import time
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from ipywidgets import Play, IntSlider, HBox, jslink
import MDAnalysis as mda
import re

# Settings
pd.set_option('expand_frame_repr', False)

# POCKET_DIR = 'pockets_dens'
# SAVING_PATH = os.path.join(conf.folder_meta_analysis, POCKET_DIR)
# if not os.path.exists(SAVING_PATH):
#     os.mkdir(SAVING_PATH)
# ISO_VALUES = [1, 2, 3]
# DBSCAN = False

# TODO: check if play button doesn't work because of large amount of data to be displayed
#  --> comment out fig.add_trace3d to check if line of lineplot is moving;
#  rather because trajectory cannot build universe


###################### RMSD, RMSF PLOTS ####################################################
def plot_rmsd_rmsf_per_proj(df, col_selection_y, col_selection_x, max_y, max_x, title, saving_path):
    """plots the RMSD/RMSF over all frames for each replicate and the mean of the three replicates
    df: DataFrame including the columns 'Time in nanoseconds' and the column selection column
    col_selection: column to plot the RMSD from, e.g., 'RMSD of protein and name CA in Angström'
    max_y: maximum y value
    max_x: maximum x value
    saving_path: path to save the figure"""
    angstrom_sign = r'$\AA$'
    y_title = col_selection_y.replace('in Angström', f' [ {angstrom_sign} ]')
    replicates = df['replicate'].unique()
    plt.figure(figsize=(14, 6))
    colours = plt.cm.Blues(np.linspace(0.3, 0.9, len(replicates)))

    for i, rep in enumerate(replicates):
        rep_df = df[df['replicate'] == rep]
        plt.plot(rep_df[col_selection_x], rep_df[col_selection_y], label=rep, color=colours[i])

    if 'RMSD' in col_selection_y:
        mean_rmsd = df.groupby(col_selection_x)[col_selection_y].mean()
        plt.plot(mean_rmsd.index, mean_rmsd, label='mean of the replicates', color='lightgrey')
    plt.xlabel = col_selection_x
    plt.ylabel = col_selection_y
    ax = plt.gca()
    ax.set_xlim([0, max_x])
    ax.set_ylim([0, max_y])
    ax.set_title(title)
    ax.set_xlabel(col_selection_x)
    ax.set_ylabel(y_title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()
    # print(os.path.join(saving_path, 'all_replicates_' + col_selection_y.replace(' ', '_') + '.tiff'))
    plt.savefig(os.path.join(saving_path, 'all_replicates_' + col_selection_y.replace(' ', '_') + '.tiff'), dpi=300)
    plt.savefig(os.path.join(saving_path, 'all_replicates_' + col_selection_y.replace(' ', '_') + '.pdf'), dpi=300)


def plot_mean_rmsf_indicated_bind_site(df, binding_site, col_sel_y, col_sel_x, title, saving_path):
    """Plots the mean RMSF of several projects (included in the df) into a single plot,
    where the residue ids contained in binding_site are highlighted by lightgrey bars
    df: should include the mean values of different projects within the columns of [project, resid_idx, RMSF]
    binding_site: list of residue ids that are to be highlighted by lightgrey bars
    col_sel__y and col_sel_x: name of the columns used to plot
    title: str
    saving_path: str (path-like)"""
    angstrom_sign = r'$\AA$'
    y_title = col_sel_y + f' [ {angstrom_sign} ]'
    projects = df['project'].unique()
    colours = plt.cm.Blues(np.linspace(0.4, 0.9, len(projects)))
    plt.xlabel = col_sel_x
    plt.ylabel = col_sel_y

    fig, ax = plt.subplots()
    for i, proj in enumerate(projects):
        data = df[df['project'] == proj]
        ax.plot(data[col_sel_x], data[col_sel_y], label=proj, color=colours[i])

    for bind_res in binding_site:
        ax.axvspan(bind_res - 0.5, bind_res + 0.5, alpha=0.4, color='lightgrey')

    ax.set_title(title)
    ax.set_xlabel(col_sel_x)
    ax.set_ylabel(y_title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='lightgrey', alpha=0.8))
    labels.append('Binding Site \n Residues')
    ax.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.grid(False)
    # plt.show()
    plt.savefig(os.path.join(saving_path, col_sel_y.replace(' ', '_') + '_mean_highlighted_binding_site' +
                             '.tiff'), dpi=300)
    plt.savefig(os.path.join(saving_path, col_sel_y.replace(' ', '_') + '_mean_highlighted_binding_site'
                             + '.pdf'), dpi=300)



###################### POCKET PLOTS ####################################################
def plot_descriptor_static(pocket_df, descriptor_col, saving_path, prj, rep, x_lim=1000):
    """IN:
    pocket_df = pd.DataFrame including the column pock_volume for the number of frames to analyse of only an
    individual pocket
    descriptor_col = name of the columns to be plotted, e.g., 'pock_volume'
    OUT:
    static matplotlib plot of pocket volume per frame"""
    angstrom_sign = r'$\AA$'
    plt.figure(figsize=(14, 6))
    plt.plot(pocket_df['snapshot'], pocket_df[descriptor_col], color='darkblue')
    plt.xlabel('Frame')
    plt.ylabel(f'{descriptor_col} {angstrom_sign}³')
    plt.title(f'{descriptor_col} per Frame for {pocket_df['ID'].unique()[0]}')
    ax = plt.gca()
    ax.set_xlim([0, x_lim])
    plt.tight_layout()
    saving_name = os.path.join(prj, rep)
    # plt.show()
    plt.savefig(os.path.join(saving_path, saving_name, f'{descriptor_col}_{pocket_df['ID'].unique()[0]}.pdf'),
                dpi=300)


def plot_pock_vol_dynamic(pock_df, desc_df, descriptor_cols, trajectory, topology, saving_path):
    """and interactive plotly plot (html) of the frame
    INPUT: pock_df = dataframe of the descriptors of the pocket to be plotted
            trajectory = path to trajectory of the pocket to be plotted
            topology = path to topology
            pock_coords_df = list of dataframes that contain pocket coordinates of each pocket of this structure
            saving path """
    # colours = plt.cm.Blues(np.linspace(0.3, 0.9, len(replicates)))
    # TODO: structure on the left, volume plot on the right w/ moving vertical line
    # TODO: updating frames doesnt work, show pocket as darkblue dots
    univ = mda.Universe(topology, trajectory)
    coords = [univ.atoms.positions.copy() for ts in univ.trajectory]
    n_atoms = len(univ.atoms)
    n_frames = len(univ.trajectory)
    saving_path = os.path.join(saving_path, pock_df['prj'].unique()[0], pock_df['rep'].unique()[0])
    saving_name = os.path.join(saving_path, f'position_and_descriptors_of_{pock_df['ID'].unique()[0]}')

    # plot frame 0
    x, y, z = coords[0][:, 0], coords[0][:, 1], coords[0][:, 2]
    # TODO: maybe add the other descriptors also
    #  --> 4 subplots with the 3D image filling half and the three descriptors underneath each other
    rows = len(descriptor_cols)
    cols = 2
    subplot_titles = ['Position of the Pocket']
    for descr in descriptor_cols:
        subplot_titles.append(f'Descriptor: {descr}')
        subplot_titles.append(None)
    # if subplot_titles[-1] is None:
    #     subplot_titles = subplot_titles[:-1]
    specs = [[{'rowspan': len(descriptor_cols), 'type': 'scatter3d'}, {'type': 'scatter'}]]
    # First subplot spans all 3 rows in the first column], the second column is filled within range of rows
    for i in range(rows - 1):
        specs.append([{'type': 'scatter'}, {'type': 'scatter'}])
    fig = make_subplots(rows=rows, cols=cols, column_widths=[0.4, 0.6], subplot_titles=subplot_titles,
                        specs=specs,
                        )
    fig_3d = go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Frame 0',
                          marker=dict(size=5, opacity=0.5, color='lightgrey'))
    fig.add_trace(fig_3d, row=1, col=1)
    # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Frame 0', marker=dict(size=5, opacity=0.4,
    #                                                                                       color='lightblue')),
    #               row=1, col=1)
    for row, descriptor in zip(range(len(descriptor_cols)), descriptor_cols):
        fig_desc = go.Scatter(x=desc_df['snapshot'], y=desc_df[descriptor], mode='lines',
                              name=f'{descriptor} per Frame',
                              line=dict(color='darkblue', width=1))

        fig.add_trace(fig_desc, row=row + 1, col=2)
        fig.update_xaxes(title='Frame')
        fig.update_yaxes(title=descriptor)  # TODO!! angström missing
    frame_marker = go.Scatter(x=[0, 0], y=[pock_df['pock_volume'].min(), pock_df['pock_volume'].max()], mode='lines',
                              name='current frame', line=dict(color='grey', dash='dash'))
    # fig.add_trace(frame_marker, row=1, col=2)  # Play doesn't work anyway and frame marker is only displayed
    # in 1st plot

    fig.update_layout(plot_bgcolor='white', showlegend=True)
    fig.update_xaxes(showgrid=True, gridcolor='black')
    fig.update_yaxes(showgrid=True, gridcolor='black')

    pocket_rep = go.Scatter3d(x=pock_df['x'], y=pock_df['y'], z=pock_df['z'], mode='markers',
                              name='pocket', marker=dict(size=5, color='darkblue'))
    fig.add_trace(pocket_rep, row=1, col=1)
    # link line with play button
    frame_slider = IntSlider(min=0, max=n_frames, step=1, value=0, description='Frame')
    play_bttn = Play(value=0, min=0, max=n_frames, step=1, intervals=100)
    jslink((play_bttn, 'value'), (frame_slider, 'value'))

    # TODO: play button doesnt update the frames

    def update_frame(frame):
        # for frame in range(n_frames):
        x_fr = coords[frame][:, 0]
        y_fr = coords[frame][:, 1]
        z_fr = coords[frame][:, 2]
        fig.update_traces(x=x_fr, y=y_fr, z=z_fr, selector=dict(type='scatter3d'))
        frame_marker.x = [frame, frame]
        fig.update_traces(x=[frame, frame], selector=dict(type='scatter', mode='lines'))
        # fig_3d.x = x_fr
        # fig_3d.y = y_fr
        # fig_3d.z = z_fr

    frame_slider.observe(lambda change: update_frame(change['new']), names='value')
    fig.update_layout(font=dict(family="Times New Roman", size=20),
                      legend=dict(font=dict(family="Times New Roman", size=20)),
                      scene=dict(
                          xaxis=dict(title='x', backgroundcolor='white', gridcolor='black'),
                          yaxis=dict(title='y', backgroundcolor='white', gridcolor='black'),
                          zaxis=dict(title='z', backgroundcolor='white', gridcolor='black')),
                      title=f'MD Simulation of {pock_df['ID'].unique()[0]}',
                      updatemenus=[{'buttons': [{'args': [None, {'frame': {'redraw': True, 'mode': 'immediate',
                                                                           'fromcurrent': True}
                                                                 }],
                                                 'label': 'Play', 'method': 'animate'},
                                                {'args': [None, {'frame': {'redraw': True,
                                                                           'mode': 'immediate',
                                                                           'fromcurrent': True}}],
                                                 'label': 'Pause', 'method': 'animate'}],
                                    'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False,
                                    'type': 'buttons',
                                    'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}])
    fig.update_yaxes(title_font=dict(size=16))
    fig.update_xaxes(title_font=dict(size=16))
    fig.update_annotations(font=dict(size=20, family="Times New Roman"))
    HBox([play_bttn, frame_slider])
    # pio.show(fig)
    fig.write_html(saving_name + '.html')
    fig.write_image(saving_name + '.png', width=2000, height=1500)


def plot_descriptor_heatmap(pock_df, descriptor_col, trajectory, topology, saving_path):
    """Plots the pockets given in pock_df (all pockets for one structure/rep/isovalue)
    in a colourscale based on the descriptor_col
    CAVE: descriptor_col has to be the mean descriptor of each pocket, as shown in the example
    Plots are saved in the pocket_dir/summary_plots"""
    univ = mda.Universe(topology, trajectory)
    coords = [univ.atoms.positions.copy() for ts in univ.trajectory]
    if not os.path.exists(os.path.join(saving_path, 'summary_plots')):
        os.mkdir(os.path.join(saving_path, 'summary_plots'))
    saving_path = os.path.join(saving_path, 'summary_plots')
    saving_name = os.path.join(saving_path, f'{descriptor_col}_summary_of_all_pockets_'
                                            f'{pock_df['prj'].unique()[0]}_{pock_df['rep'].unique()[0]}_{pock_df['isovalue'].unique()[0]}')

    # plot frame 0
    x, y, z = coords[0][:, 0], coords[0][:, 1], coords[0][:, 2]
    fig_heat = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Frame 0',
                                      marker=dict(size=5, opacity=0.5, color='lightgrey')))

    min_descriptor = pock_df[descriptor_col].min()
    max_descriptor = pock_df[descriptor_col].max()
    cmap_viridis = cm.get_cmap('viridis')

    def get_colour(value):
        norm_value = (value - min_descriptor) / (max_descriptor - min_descriptor)
        colour = cmap_viridis(norm_value)
        return f'rgba({colour[0] * 255}, {colour[1] * 255}, {colour[2] * 255}, {colour[3] * 255})'

    for id_p, pocket in pock_df.groupby('pocket_number'):
        pocket_colour = get_colour(pocket[descriptor_col].mean())
        pocket_trace = go.Scatter3d(x=pocket['x'], y=pocket['y'], z=pocket['z'], mode='markers',
                                    name=f'pocket {pocket['pocket_number'].unique()[0]}',
                                    text=[f'{descriptor_col}: {pocket[descriptor_col].mean()} <br> '
                                          f'Pocket Number {pocket['pocket_number'].unique()[0]}'] * len(pocket),
                                    hoverinfo='text',
                                    marker=dict(size=5, color=pocket_colour))

        fig_heat.add_trace(pocket_trace)
    # add dummy colourbar since the colourbar reverts to unlinked default bar
    fig_heat.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                    marker=dict(size=0, color=[min_descriptor, max_descriptor],
                                                colorscale='Viridis', cmin=min_descriptor, cmax=max_descriptor,
                                                colorbar=dict(title=descriptor_col, orientation='h', x=0.5, y=-0.3,
                                                              xanchor='center', yanchor='bottom',
                                                              tickvals=np.linspace(min_descriptor,
                                                                                   max_descriptor, 5),
                                                              ticktext=np.round(np.linspace(min_descriptor,
                                                                                            max_descriptor, 5), 2)
                                                              )), showlegend=False))

    fig_heat.update_layout(template='none', plot_bgcolor='white', showlegend=True,
                           title=f'Pockets of Project {pock_df['prj'].unique()[0]}_{pock_df['rep'].unique()[0]}'
                                 f'<br> Isovalue {pock_df['isovalue'].unique()[0]}'
                                 f'<br> Number of Pockets: {len(pock_df['pocket_number'].unique())}')

    fig_heat.update_xaxes(showgrid=True, gridcolor='black')
    fig_heat.update_yaxes(showgrid=True, gridcolor='black')
    # print(saving_name)
    # fig_heat.show()
    fig_heat.write_html(saving_name + '.html')
    fig_heat.write_image(saving_name + '.pdf')


def summary_plot_pock_sep_atclus(df, saving_path):
    """Input: dataframe representing one structure and replicate for one isovalue but with all pockets"""
    top = df['topology loc'].unique()[0]
    traj = df['trajectory loc'].unique()[0]
    # TODO: trajectory needed?! --> XDT results in AttributeError: 'XTCReader' object has no attribute '_xdr'
    # for file in os.listdir(df['trajectory loc'].unique()[0].rstrip('aligned_traj')):
    #     if 'aligned_traj' in file and file.endswith('.dcd') or file.endswith('.xtc'):
    #         traj = file
    #
    # print('file', traj)
    univ = mda.Universe(topology=top, trajectory=traj)
    coords = [univ.atoms.positions.copy() for ts in univ.trajectory]

    colours = list(itertools.chain.from_iterable([px.colors.qualitative.Dark24, px.colors.qualitative.Light24,
                                                  px.colors.qualitative.Alphabet]))
    x, y, z = coords[0][:, 0], coords[0][:, 1], coords[0][:, 2]
    fig = go.Figure()
    # Plot static protein
    fig_3d = go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Frame 0', marker=dict(size=5, opacity=0.5,
                                                                                     color='lightblue'))
    fig.add_trace(fig_3d)
    # plot the pockets
    for id, pocket in df.groupby('pocket_number'):
        pockets = go.Scatter3d(x=pocket['x'], y=pocket['y'], z=pocket['z'], mode='markers',
                               name=f'pocket {pocket['pocket_number'].unique()[0]}',
                               marker=dict(size=5, color=colours[int(id)]))

        fig.add_trace(pockets)

    fig.update_layout(template='none', plot_bgcolor='white', showlegend=True,
                      title=f'Pockets of Project {df['prj'].unique()[0]}_{df['rep'].unique()[0]}'
                            f'<br> Isovalue {df['isovalue'].unique()[0]}'
                            f'<br> Number of Pockets: {len(df['pocket_number'].unique())}')

    fig.update_xaxes(showgrid=True, gridcolor='black')
    fig.update_yaxes(showgrid=True, gridcolor='black')
    # fig.show()
    if not os.path.exists(os.path.join(saving_path, 'summary_plots')):
        os.mkdir(os.path.join(saving_path, 'summary_plots'))
    saving_name = os.path.join(saving_path, 'summary_plots',
                               str(df['prj'].unique()[0] + '_' + str(df['rep'].unique()[0])
                                   + '_iso_' + str(df['isovalue'].unique()[0])))

    fig.write_html(saving_name + '_all_pockets.html')


def pocket_persistence_heatmap():
    """"""
    # idea: heatmap colouring according to the percentage of in how many structures one pocket is found (darkred if
    # pocket is found in each of the structures, lightblue if the pocket is unique to one of the structures/reps)
