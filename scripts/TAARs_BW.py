"""this is a script modified from Francesco's OR pipeline to get BW numbering of a multiple sequence alignment,
input: csv file of MSA, e.g., downloaded from GPCRdb
the script creates a folder for the csv files for each of the aligned protein BW numberings"""

import pandas as pd
import os
import time
from Bio import AlignIO
import config as conf

pd.set_option('expand_frame_repr', False)
# working_dir = os.path.join(os.getcwd(), 'TAAR_numbering')
# os.chdir(working_dir)
working_dir = conf.folder_bw


def alignment_sequence_and_structure_based():
    """"""
    if 'TAARs_numbered' not in os.listdir(conf.folder_bw):
        align_df = pd.read_csv(os.path.join(working_dir, 'BW_corrected_final.csv'), header=None, index_col=0, sep=';')
        chains = []
        for elem in align_df.iloc[0, :].to_list():
            if elem == elem:
                chains.append(elem)
                name = elem
            else:
                chains.append(name)
        align_df.iloc[0] = chains
        for ndx, elem in enumerate(align_df.iloc[1, :].to_list()):
            if elem == elem:
                if '-' in elem:
                    ch = elem.split('.')[0]
                    nm = elem.split('-')[1].split('x')[0]
                    t = ch + '.' + nm
                    align_df.iloc[1, ndx] = t
                else:
                    align_df.iloc[1, ndx] = elem.split('x')[0]

        idx_list = align_df.index.to_list()
        idx_list[0] = 'CHAIN'
        idx_list[1] = 'BW'
        idx_list[2] = 'GPCRdb'
        for ndx, prot in enumerate(idx_list):
            if prot.startswith('[Human]'):
                idx_list[ndx] = prot.split('[Human] ')[1].upper()
            elif prot.startswith('['):
                idx_list[ndx] = prot.split(' ')[1]

        align_df.index = idx_list
        align_df.drop(index='CONSENSUS', inplace=True)
        align_df.replace({'C-term': 'C-Ter', 'N-term': 'N-Ter'}, inplace=True)

        align_df.to_csv(os.path.join(working_dir, 'BW_corrected_final_refined_Nov24.csv'))
        os.mkdir(os.path.join(working_dir, 'TAARs_numbered'))
        for idx, row in align_df.iloc[2:, :].iterrows():
            prot = pd.DataFrame(row, index=row.index)
            prot['BW'] = align_df.iloc[1, :]
            prot['CHAIN'] = align_df.iloc[0, :]
            prot.rename(columns={idx: 'RES_NAME'}, inplace=True)
            prot['RES_NAME'] = prot['RES_NAME'].replace('_', '-')
            prot = prot[prot['RES_NAME'] != '-']
            prot.reset_index(drop=True, inplace=True)
            prot.reset_index(drop=False, inplace=True)
            prot.rename(columns={'index': 'RES_ID'}, inplace=True)
            prot['RES_ID'] = prot['RES_ID'] + 1
            prot.to_csv(os.path.join(working_dir, f'TAARs_numbered/{idx.split('|')[-1]}.csv'), sep=',', index=False)
            del prot


def maestro_structural_alignment():
    """"""
    path = '/home/mmlab/Desktop/Masters_Thesis_CR/Masters_Thesis/TAAR_numbering/structure_based_alignment/'
    file_name = 'alignment_structural_crystals_AF_multistate.fasta'

    input_path = os.path.join(path, file_name)
    with open('crystal_AF_alignment_structural.csv', 'w') as f:
        for seq in AlignIO.read(input_path, 'fasta'):
            seq_csl = []
            for res in seq.seq:
                seq_csl.append(res)
            seq_csl = ','.join(seq_csl)
            f.write(str(seq.id) + ',' + seq_csl + '\n')
            del seq_csl


def main():
    t0 = time.time()
    alignment_sequence_and_structure_based()
    # maestro_structural_alignment()
    t1 = time.time()
    print('time:  ', t1 - t0)


if __name__ == '__main__':
    main()
