""""this is a script for xxxx"""
import pandas as pd
import numpy as np
import time
import warnings
# Settings
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')


# I Helper Functions
def parse_bw_alignment(path_to_alignment_csv):
    """Given the alignment csv produced by Francescos TAARs_BW.py script,
    the function returns a nested dictionary, where initial keys correspond with the identifier of the sequence
    (e.g., TAAR1_HUMAN) and the values are dictionaries mapping the residue amino acid and id to the
    according BW numbering"""

    align_df = pd.read_csv(path_to_alignment_csv)

    bw_mapping = {}
    for row_idx in range(2, len(align_df)):  # Start from row 2 (index 2)
        inner_dict = {}  # Initialize an inner dictionary for each row

        # 2. Loop through each column (excluding 'Unnamed: 0')
        for col in align_df.columns[1:]:
            # Get the modified residue name (e.g., Lys1, Lys2, etc.)
            residue_name = align_df.loc[row_idx, col] + col

            # Only add to inner dictionary if it's not an invalid value (e.g., '_', '_1', '_2')
            if not align_df.loc[row_idx, col].startswith("_"):
                inner_dict[residue_name] = align_df.loc[1, col]  # Use the BW value (row 1) as the value

        # Add the inner dictionary to the nested dictionary with the row identifier as the key
        bw_mapping[align_df.loc[row_idx, 'Unnamed: 0']] = inner_dict
    return align_df, bw_mapping


def caller():
    """"""
    path = '/media/mmlab/DATA/Masters_Thesis_CR/Masters_Thesis/TAAR_numbering/TAARs_alignment_refined.csv'
    _, bw_mapping = parse_bw_alignment(path)
    print(bw_mapping['TAAR1_HUMAN'])  # L366': nan, 'S367': nan, 'S368': nan}


# IV Main 
def main():
    t0 = time.time()
    caller()
    t1 = time.time()
    t_min = (t1 - t0) / 60
    t_hr = t_min / 60
    print('time:  ', t1 - t0, '  sec')
    print('time:  ', t_min, '  min')
    print('time:  ', t_hr, '  hours')


if __name__ == '__main__':
    main()
