import os
import pandas as pd
import MDAnalysis as mda


class QualityChecker:
    def __init__(self, result_folder, output_folder, name_top='aligned_top.pdb', name_trj='aligned_traj.xtc',
                 replicates=3, rmsd_threshold=4.5, expected_frames=1000):
        self.result_folder = result_folder
        self.output_folder = output_folder
        self.name_topo = name_top
        self.name_trj = name_trj
        self.replicate_no = replicates
        self.rmsd_threshold = rmsd_threshold
        self.expected_frames = expected_frames

    def quality_check(self):
        """Public method to run all quality control checks and save the final DataFrame.
        Assumes that the method is used after the RMSD and RMSF calculation using the pipeline
        --> searches for files automatically generated with the pipeline"""
        data_dict, qc_meta_df = self._data_prep()
        qc_meta_df = self._check_rmsd_range(qc_meta_df)
        qc_meta_df = self._check_no_frames(qc_meta_df)
        qc_meta_df = self._check_max_rmsf(qc_meta_df)

        print(qc_meta_df)
        output_path = os.path.join(self.output_folder, 'quality_control_meta_df.csv')
        qc_meta_df.to_csv(output_path)

    def _data_prep(self):
        """Reads in all the needed data and builds a metadata DataFrame."""
        data_dict = {}
        replicates = [str(i) for i in range(1, self.replicate_no + 1)]
        print(self.name_topo)

        for prj in os.listdir(self.result_folder):
            for rep in os.listdir(os.path.join(self.result_folder, prj)):
                if rep in replicates:
                    key = f'{prj}_{rep}'
                    curr_rmsd = curr_rmsf = curr_top = curr_traj = ''

                    out_dir = os.path.join(self.output_folder, prj, rep)
                    os.makedirs(out_dir, exist_ok=True)

                    for file in os.listdir(os.path.join(self.result_folder, prj, rep)):
                        full_path = os.path.join(self.result_folder, prj, rep, file)
                        if file.startswith('RMSD_') and file.endswith('.csv'):
                            curr_rmsd = full_path
                        elif file.startswith('RMSF_') and file.endswith('.csv'):
                            curr_rmsf = full_path
                    curr_top = os.path.join(self.result_folder, prj, rep, self.name_topo)
                    curr_traj = os.path.join(self.result_folder, prj, rep, self.name_trj)

                    data_dict[key] = {'rmsd': curr_rmsd, 'rmsf': curr_rmsf, 'traj': curr_traj, 'top': curr_top}

        qc_meta_df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
        qc_meta_df[['prj', 'rep']] = qc_meta_df['index'].str.split('_', expand=True)

        return data_dict, qc_meta_df

    def _check_rmsd_range(self, meta_df):
        """Flags entries with high RMSD."""
        meta_df['rmsd flag'] = None

        for index in range(len(meta_df)):
            path = meta_df.at[index, 'rmsd']
            curr_rmsd = pd.read_csv(path)
            max_rmsd = curr_rmsd['RMSD of protein and name CA in Angström'].max()
            meta_df.at[index, 'max rmsd'] = max_rmsd
            meta_df.at[index, 'rmsd flag'] = 'high RMSD' if max_rmsd > self.rmsd_threshold else 'normal RMSD'

        return meta_df

    def _check_no_frames(self, meta_df):
        """Flags entries with incorrect frame count."""
        meta_df['frames flag'] = None

        for index, (top, traj) in enumerate(zip(meta_df['top'], meta_df['traj'])):
            univ = mda.Universe(top, traj)
            no_frames = len(univ.trajectory)
            meta_df.at[index, 'number of frames'] = no_frames
            meta_df.at[index, 'frames flag'] = ('incorrect number of frames' if no_frames != self.expected_frames
                                                else 'correct number of frames')

        return meta_df

    def _check_max_rmsf(self, meta_df):
        """Adds max RMSF and corresponding residue ID."""
        meta_df['max RMSF'] = None
        meta_df['residue ID of max RMSF'] = None

        for index in range(len(meta_df)):
            path = meta_df.at[index, 'rmsf']
            curr_rmsf = pd.read_csv(path)
            max_rmsf = curr_rmsf['RMSF'].max()
            max_df = curr_rmsf.loc[curr_rmsf['RMSF'].idxmax()]
            res_id = max_df['resid_idx']
            residue = max_df['Residue Name']
            meta_df.at[index, 'max RMSF'] = max_rmsf
            meta_df.at[index, 'residue ID of max RMSF'] = f'{res_id} ({residue})'

        return meta_df

