""""this is a script for preprocessing your MD analysis project. It includes only one class called PreProcess
which contains methods usually used as first steps of analyses, such as making a selection and saving it
(e.g., to dry the protein) or aligning to reference structure.
If you want to add basic functionalities this is the right script."""
import os
import time
import warnings
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scripts import config as conf
from scripts import logging as log

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Settings
# warnings.filterwarnings('ignore')


class PreProcess:
    """stores methods dry protein and align protein"""

    def __init__(self, topology=None, trajectory=None, curr_proj=None, curr_rep=None, verbose=False):
        self.topology = topology
        self.trajectory = trajectory
        self.curr_proj = curr_proj
        self.curr_rep = curr_rep
        self.verbose = verbose

    def make_selection_n_save(self, selection=None, stride=1):
        """gets to the dry protein stage if you use None --> protein is selected; use Alessandros backbone
        selection should be a str, default: protein
        stride is int, default: 1, used to stride the trajectory, i.e., only every x-th frame is considered and saved,
        e.g., stride = 10 means that every tenth frame is considered"""
        # dry_prot.write(os.path.join(saving_path, 'dry_prot.dcd'), frames='all')# just wont work (saves only 1st frame)
        log.log(f'\n Preprocessing: Make Selection and Save \n Data: {self.curr_proj}_{self.curr_rep} \n')
        saving_path = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep)
        univ = mda.Universe(self.topology, self.trajectory)

        last_frame = len(univ.trajectory)
        if selection is None:
            dry_prot = univ.select_atoms("protein")
        else:
            dry_prot = univ.select_atoms(selection)

        # file saving
        if not os.path.exists(os.path.join(saving_path, 'dry_prot.pdb')):
            dry_prot.atoms.write(os.path.join(saving_path, 'dry_prot.pdb'))
        else:
            if self.verbose:
                print(
                    f'INFO: The file {os.path.join(saving_path, 'dry_prot.pdb')} already exists and is not overwritten')
            log.log(f'INFO: The file {os.path.join(saving_path, 'dry_prot.pdb')} already exists and is not overwritten')
        if not os.path.exists(os.path.join(saving_path, 'dry_prot.dcd')):
            with mda.Writer(os.path.join(saving_path, 'dry_prot.dcd'), dry_prot.n_atoms, step=stride) as w:
                for ts in univ.trajectory[0:last_frame:stride]:
                    w.write(dry_prot)
        else:
            if self.verbose:
                print(
                    f'INFO: The file {os.path.join(saving_path, 'dry_prot.dcd')} already exists and is not overwritten')
            log.log(f'INFO: The file {os.path.join(saving_path, 'dry_prot.dcd')} already exists and is not overwritten')
        if not os.path.exists(os.path.join(saving_path, 'dry_prot.xtc')):
            with mda.Writer(os.path.join(saving_path, 'dry_prot.xtc'), dry_prot.n_atoms, step=stride) as w:
                for ts in univ.trajectory[0:last_frame:stride]:
                    w.write(dry_prot)
        else:
            if self.verbose:
                print(
                    f'INFO: The file {os.path.join(saving_path, 'dry_prot<.xtc/.dcd>')} already exists and is '
                    f'not overwritten')
            log.log(f'INFO: The file {os.path.join(saving_path, 'dry_prot<.xtc/.dcd>')} already exists and is '
                    f'not overwritten')
        return dry_prot

    def align_to_reference(self, reference, universe=(None, None), selection='protein'):
        """ reference = topology to align to (path-like str)
            universe = tuple (path_to_topology, path_to_trajectory)

        If no universe is given, it will align to the universe made from the class attributes
        reference: PDB file of the reference you want to align to,
        e.g., the PDB file used as input for the simulation or an average of the frames produced by the simulation"""
        # Use either the attributes topology & trajectory or the User Inout, e.g., the universe of the dry protein
        t0 = time.time()
        log.log(f'\n Preprocessing: Alignment to Reference \n Data: {self.curr_proj}_{self.curr_rep} \n')
        saving_path = os.path.join(conf.folder_results, self.curr_proj, self.curr_rep)
        if universe is None:
            univ = mda.Universe(self.topology, self.trajectory)
        else:
            # create universe from the protein selection, e.g., after calling dry protein
            univ = mda.Universe(universe[0], universe[1])

        ref_univ = mda.Universe(reference)
        ref_univ_prot = mda.Merge(ref_univ.select_atoms(selection))
        # for some fucking reasons ypu have to select the atoms of a Universe first to be able to save it
        ref_univ_prot.atoms.write(os.path.join(saving_path, 'aligned_top.pdb'))
        filename_dcd = os.path.join(saving_path, 'aligned_traj.dcd')
        filename_xtc = os.path.join(saving_path, 'aligned_traj.xtc')
        if not os.path.exists(filename_dcd) and not os.path.exists(filename_xtc):
            if self.trajectory.endswith('.dcd'):
                if self.verbose:
                    print(f'INFO: Aligning universe built with {universe[1].split(conf.folder_results)[1]} '
                          f'(with {len(univ.trajectory)} frames) to reference {reference}')
                log.log(f'INFO: Aligning universe built with {universe[1].split(conf.folder_results)[1]} '
                        f'(with {len(univ.trajectory)} frames) to reference {reference}')
                align.AlignTraj(univ, ref_univ_prot,
                                select='protein and name CA',
                                filename=filename_dcd,
                                match_atoms=True,
                                verbose=True,
                                in_memory=False
                                ).run()
                t1 = time.time()
            elif self.trajectory.endswith('.xtc'):
                if self.verbose:
                    print(f'INFO: Aligning universe built with {universe[1].split(conf.folder_results)[1]} '
                          f'(with {len(univ.trajectory)} frames) to reference {reference}')
                log.log(f'INFO: Aligning universe built with {universe[1].split(conf.folder_results)[1]} '
                        f'(with {len(univ.trajectory)} frames) to reference {reference}')
                align.AlignTraj(univ, ref_univ_prot,
                                select='protein and name CA',
                                filename=filename_xtc,
                                match_atoms=True,
                                verbose=True,
                                in_memory=False
                                ).run()
                t1 = time.time()
            if self.verbose:
                print(f'INFO: Aligning the trajectories took {round((t1 - t0), 3)} seconds // '
                      f'{round(((t1 - t0) / 60), 3)} minutes \n')
            log.log(f'INFO: Aligning the trajectories took {round((t1 - t0), 3)} seconds // '
                    f'{round(((t1 - t0) / 60), 3)} minutes \n')
        else:
            if self.verbose:
                print(f'INFO: The file {filename_dcd.split('.')[0]} <.dcd/.xtc> already exists and is not overwritten')
            log.log(f'INFO: The file {filename_dcd.split('.')[0]} <.dcd/.xtc> already exists and is not overwritten')
