""""this is a script for xxxx"""
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import time
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf
from scripts import logging as log

# Settings
pd.set_option('expand_frame_repr', False)


class BasicAnalysis:
    """should store methods for RMSD, RMSF and pairwise distances of given nested_list, mandatrory method: alignment?
    (alternatively, alignment iun main wrapper / as helper function), """

    def __init__(self, topology=None, trajectory=None, curr_proj='', curr_rep='', verbose=False, show_plots=False):
        self.topology = topology
        self.trajectory = trajectory
        self.curr_proj = curr_proj
        self.curr_rep = curr_rep
        self.verbose = verbose
        self.show_plots = show_plots

    def _line_plot_rmsd(self, df, saving_loc=None):
        """creates line plot using matplotlib for RMSD, RMSF style plots; hidden because called by other methods"""
        angstrom_sign = r'$\AA$'
        alpha = 1
        lw = 1.2  # float - linewidth of line plot
        fontsize = 15  # float - fontsize x and y label
        pad = 30  # float - pad of title, and (x,y) labels
        if saving_loc is None:
            saving_path = os.path.join(conf.results_dir_for(self.curr_proj), self.curr_proj, self.curr_rep, 'plots')
        else:
            saving_path = os.path.join(saving_loc, 'plots')

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        rmsd_cols = [str(col) for col in df.columns if 'RMSD ' in str(col)]

        for rmsd in rmsd_cols:
            fig, ax1 = plt.subplots()
            ax1.plot(df['Time in nanoseconds'], df[rmsd], color='darkblue', ls="-", lw=lw, alpha=alpha)

            plt.tick_params(axis='both', labelsize=12)  # size param
            ax1.set_title(f'{rmsd} \n of project {self.curr_proj} rep {self.curr_rep}',
                          # font='Times New Roman',
                          fontsize=16, pad=pad)
            ax1.set_xlabel('Time in nanoseconds', fontsize=fontsize, labelpad=pad)
            ax1.set_ylabel(f'RMSD {angstrom_sign}', fontsize=fontsize, labelpad=pad)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(saving_path, f'{rmsd.replace(' ', '_')}.tiff'), dpi=300,
                        format='tiff')
            plt.savefig(os.path.join(saving_path, f'{rmsd.replace(' ', '_')}.pdf'), dpi=300,
                        format='pdf')
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _line_plot_rmsf(self, df, saving_loc=None):
        """"""
        if saving_loc is None:
            saving_path = os.path.join(conf.results_dir_for(self.curr_proj), self.curr_proj, self.curr_rep, 'plots')
        else:
            saving_path = os.path.join(saving_loc, 'plots')

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        fig, ax = plt.subplots()
        ax.plot(df['resid_idx'], df['RMSF'], color="darkblue", ls="-", lw=1.5, alpha=1)
        # ax.set_xlim([min(df['Residue Number renum']), max(df['Residue Number renum'])])
        # ax.set_ylim([0, max(df['RMSF']) + 1])
        ax.set_xlabel('Residue Index', fontsize=15, labelpad=20)
        ax.set_ylabel('RMSF', fontsize=15, labelpad=20)
        ax.set_title(f'RMSF \n of project {self.curr_proj} rep {self.curr_rep}',
                     # font='Times New Roman',
                     fontsize=18, pad=30)
        plt.tight_layout()
        plt.savefig(os.path.join(saving_path, f'RMSF_{self.curr_proj}_rep{self.curr_rep}.tiff'), dpi=300,
                    format='tiff')
        plt.savefig(os.path.join(saving_path, f'RMSF_{self.curr_proj}_rep{self.curr_rep}.pdf'), dpi=300,
                    format='pdf')
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def calc_rmsd(self, trj_period_step_stride, topology=None, trajectory=None, selection='protein and name CA',
                  group_selection=None, icl3_free_selection=None, weights=None, verbose=False,
                  reference=(None, None), saving_ext='', saving_loc=None):
        """Calculate the Root Mean Square Deviation (RMSD) of a molecular dynamics (MD) simulation trajectory.

        This function computes the RMSD of a universe over all frames provided.
        The function supports customizable selection of atoms, weighting of the RMSD calculation and
        verbose output for detailed logging.

        Parameters:
        -----------
        trj_period_step_stride : dictionary
            A dictionary containing the trajectory period, timestep, and stride of the trajectory to be analyzed.
            This is used to determine the time in nanoseconds for human-readable plotting.

        topology : str or None, optional
            The file path to the topology file (e.g., a PDB file) for the molecular system. If None, the saved aligned
            topology will be read in from the results_dbscan_new folder. Default is None.

        trajectory : str or None, optional
            The file path to the trajectory file (e.g., a DCD or XTC file) for the molecular system. If None, he saved aligned
            trajectory will be read in from the results_dbscan_new folder. Default is None.

        selection : str, optional
            An atom selection string used to specify which atoms to include in the RMSD calculation. The default
            is 'protein and name CA', which selects the carbon atoms of a protein.
        group_selection: dict, optional
            keys should be the description of the selection, values is a list of CHARMM-like selection language.
            Each is reported off the SAME superposition fit as `selection` (MDAnalysis groupselections
            semantics) -- not independently fit.
        icl3_free_selection : str, optional
            An atom selection (e.g. CA without ICL3) computed as its OWN standalone, self-referenced RMSD --
            fit (superposed) and reported using only this selection, against frame 0 of the same
            (already pre-aligned) universe -- rather than as a groupselection off `selection`'s fit. Use this
            for metrics where including a flexible region (ICL3) in the fit would distort the reported
            deviation of the rest of the selection. Saved as its own 'RMSD of CA without ICL3 in
            Angström' column alongside `selection` and `group_selection`'s columns.
        for weights please refer to the documentation of MD analysis

        verbose : bool, optional
            If True, the function will print detailed logging information during the RMSD calculation. Default is False.

        reference : tuple, optional
            A tuple of two elements where the first element is a reference topology file (str) and the second
            element is a reference trajectory file. The RMSD will be calculated relative to this reference. If
            both elements are None, the RMSD is calculated using the previously saved reference universe topology in
            the resultsfolder. Default is (None, None).
    """

        if saving_loc is None:
            saving_loc = os.path.join(conf.results_dir_for(self.curr_proj), self.curr_proj, self.curr_rep)

        log.log(f'\n Calculating RMSD \n Data: {self.curr_proj}_{self.curr_rep} \n')
        if ('traj_period' not in trj_period_step_stride.keys() and
                'time_step' not in trj_period_step_stride.keys()
                and 'stride' not in trj_period_step_stride.keys()):
            raise ValueError('keys of trj_period_step_stride must contain [traj_period, time_step, stride]')
        print(self.curr_proj, self.curr_rep)

        t0 = time.time()
        if topology is None and trajectory is None:
            if self.verbose:
                print('\n')
                print('INFO: you have not specified a topology or trajectory - '
                      'the method will use the previously generated and saved aligned trajectory and its reference '
                      'topology. \n [saved as "aligned_traj<.dcd/.xtc>", "aligned_top.pdb"  and "reference_univ_prot.pdb" '
                      'in the results folder]')
            log.log('INFO: you have not specified a topology or trajectory - '
                    'the method will use the previously generated and saved aligned trajectory and its reference '
                    'topology. \n [saved as "aligned_traj<.dcd/.xtc>", "aligned_top.pdb"  and "reference_univ_prot.pdb" '
                    'in the results folder]')
            ext = 'dcd' if self.trajectory.endswith('.dcd') else 'xtc'
            topology, trajectory = conf.aligned_paths_for(self.curr_proj, self.curr_rep, ext)

        # Build universe either with aligned_top & traj or with user provided input
        univ = mda.Universe(topology, trajectory)
        if self.verbose:
            print(f'INFO: Universe is built using {topology} and {trajectory}.')
        log.log(f'INFO: Universe is built using {topology} and {trajectory}.')

        # check if reference is provided or not, if not: topology is used as reference
        if reference != (None, None):
            if reference[1] is not None:
                ref_univ = mda.Universe(reference[0], reference[1])
            else:
                ref_univ = mda.Universe(reference[0])
        else:
            ref_univ = mda.Universe(topology)
            if self.verbose:
                print(f'INFO: Using topology {topology} as reference (no explicit reference provided).')
            log.log(f'INFO: Using topology {topology} as reference (no explicit reference provided).')

        if group_selection is None:
            rmsd = rms.RMSD(univ,  # universe to align
                            ref_univ,  # reference universe or atomgroup
                            select=selection,
                            groupselections=None,
                            weights=weights,
                            verbose=self.verbose,
                            superposition=True).run()
            col_names = ['Frame', 'Frame', f'RMSD of {selection} in Angström']
        else:
            appended_sel = []
            RMSD_cols = [f'RMSD of {key} in Angström' for key in group_selection.keys()]
            col_names = ['Frame', 'Frame', f'RMSD of {selection} in Angström', *RMSD_cols]
            for sel in group_selection.values():
                appended_sel.append(*sel)
            rmsd = rms.RMSD(univ,  # universe to align
                            ref_univ,  # reference universe or atomgroup
                            select=selection,
                            groupselections=appended_sel,
                            weights=weights,
                            verbose=self.verbose,
                            superposition=True).run()

        df_rmsd = pd.DataFrame(rmsd.rmsd, columns=col_names)

        if icl3_free_selection is not None:
            # Standalone self-fit: superpose and report using ONLY the ICL3-free selection,
            # against frame 0 of the SAME (already pre-aligned) universe -- deliberately NOT a
            # groupselection off the `selection` fit above, so ICL3's flexibility never
            # contaminates this metric's alignment. See BasicAnalysis.calc_rmsd docstring.
            icl3_rmsd = rms.RMSD(univ, univ, select=icl3_free_selection, ref_frame=0,
                                 verbose=self.verbose).run()
            df_rmsd['RMSD of CA without ICL3 in Angström'] = icl3_rmsd.rmsd[:, 2]

        number_of_frames = len(univ.trajectory)
        frame_idx = np.arange(0, number_of_frames)
        time_ns = ((frame_idx + 1) * trj_period_step_stride['traj_period'] * trj_period_step_stride['time_step']
                   * trj_period_step_stride['stride']) / 1000000
        df_rmsd['Time in nanoseconds'] = time_ns
        df_rmsd = df_rmsd.round(3)
        df_rmsd.to_csv(os.path.join(saving_loc,
                                    f'RMSD_{selection.replace(' ', '_')}{saving_ext}.csv'), index=False)
        self._line_plot_rmsd(df=df_rmsd, saving_loc=saving_loc)

        t1 = time.time()
        if self.verbose:
            print(f'INFO: Calculating the RMSD took {round((t1 - t0), 3)} seconds // '
                  f'{round(((t1 - t0) / 60), 3)} minutes \n')
        log.log(f'INFO: Calculating the RMSD took {round((t1 - t0), 3)} seconds // '
                f'{round(((t1 - t0) / 60), 3)} minutes \n')

    def calc_rmsf(self, topology=None, trajectory=None, selection='protein and name CA', num_shift_rec=None,
                  saving_loc=None):
        """Calculates the RMSF for the given trajectory and topology and plots it across the residue ids"""
        t0 = time.time()
        log.log(f'\n Calculating RMSF \n Data: {self.curr_proj}_{self.curr_rep} \n')
        if saving_loc is None:
            saving_loc = os.path.join(conf.results_dir_for(self.curr_proj), self.curr_proj, self.curr_rep)
        else:
            saving_loc = saving_loc

        using_aligned_default = topology is None and trajectory is None
        if using_aligned_default:
            if self.verbose:
                print('\n')
                print('INFO: you have not specified a topology or trajectory - '
                      'the method will use the previously generated and saved aligned trajectory and its reference '
                      'topology. \n [saved as "aligned_traj<.dcd/.xtc>", "aligned_top.pdb"  and "reference_univ_prot.pdb" '
                      'in the results_dbscan_new folder]')
            log.log('INFO: you have not specified a topology or trajectory - '
                    'the method will use the previously generated and saved aligned trajectory and its reference '
                    'topology. \n [saved as "aligned_traj<.dcd/.xtc>", "aligned_top.pdb"  and "reference_univ_prot.pdb" '
                    'in the results_dbscan_new folder]')
            ext = 'dcd' if self.trajectory.endswith('.dcd') else 'xtc'
            topology, trajectory = conf.aligned_paths_for(self.curr_proj, self.curr_rep, ext)
        else:
            if self.verbose:
                print(f'INFO: calculating the RMSF for {trajectory} with the selection {selection}')
            log.log(f'INFO: calculating the RMSF for {trajectory} with the selection {selection}')

        univ = mda.Universe(topology, trajectory)
        selection_atgr = univ.select_atoms(selection)
        rmsf_rec = rms.RMSF(selection_atgr).run()
        resnames = []
        for x in selection_atgr.resnames:
            resnames.append(x)
        df_rmsf = pd.DataFrame({'resid_idx': selection_atgr.resids, 'RMSF': rmsf_rec.results.rmsf})
        df_rmsf['Residue Name'] = resnames
        if num_shift_rec is not None:
            df_rmsf['Residue Number Renum'] = df_rmsf['resid_index'] + num_shift_rec
        df_rmsf = df_rmsf.round(3)

        self._line_plot_rmsf(df=df_rmsf, saving_loc=saving_loc)
        if using_aligned_default:
            df_rmsf.to_csv(os.path.join(saving_loc,
                                        f'RMSF_{selection.replace(' ', '_')}_aligned_trj.csv'), index=False)
        else:
            # retrieves the filename without extension, i.e., splitting '.pdb' or '.mmcif' etc
            top_filename = Path(topology.split(os.sep)[-1]).stem
            df_rmsf.to_csv(os.path.join(saving_loc,
                                        f'RMSF_{top_filename}_{selection.replace(' ', '_')}.csv'),
                           index=False)
        t1 = time.time()
        if self.verbose:
            print(f'INFO: Calculating the RMSF took {round((t1 - t0), 3)} seconds // '
                  f'{round(((t1 - t0) / 60), 3)} minutes \n')
        log.log(f'INFO: Calculating the RMSF took {round((t1 - t0), 3)} seconds // '
                f'{round(((t1 - t0) / 60), 3)} minutes \n')

    def calc_pairwise_distances(self):
        """use Alessandros backbone for pairwise distances calculations"""
        # TODO: implement pairwise distance calculations
        print('work to do')

    def check_equilibration_temp_pressure(self):
        """checking temperature and pressure of equilibration phase"""
        # TODO: implement equilibration checks
        print('work to do')
