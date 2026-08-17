import os
import time
import pandas as pd

# Settings
pd.set_option('expand_frame_repr', False)
pd.options.display.max_colwidth = 500  # long values in columns fully displayed


# Helper Functions
def find_pocket_abnormalities(df: pd.DataFrame):
    """
    Detect key pocket abnormalities:
    - Multiple orthosteric pockets per experiment
    - Large transient pockets
    - State-specific orthosteric pockets
    """
    # 1. multiple orthosteric pockets in same state+PDBID+replicate
    print("\n1. MULTIPLE ORTHOSTERIC POCKETS PER EXPERIMENT:")
    exp_groups = df.groupby(['state', 'pdb_id', 'rep'])

    for (state, pdb_id, rep), group in exp_groups:
        ortho_count = group[group['is_orthosteric']].shape[0]
        if ortho_count > 1:
            ortho_pockets = group[group['is_orthosteric']]
            print(f"  {state}_{pdb_id}_{rep}: {ortho_count} orthosteric pockets")
            print(f"    IDs: {ortho_pockets['ID'].tolist()}")
            print(f"    Volumes in first frame: {ortho_pockets['pock_volume'].tolist()}")

    # 2. Large transient pockets
    print("\n2. LARGE TRANSIENT POCKETS (Medium-Very Large):")
    large_categories = ['Large (500-1000)', 'Very Large (>1000)', 'Medium (250-500)']
    large_transient = df[(df['transient']) & (df['volume_category'].isin(large_categories))]

    if not large_transient.empty:
        for idx, row in large_transient.iterrows():
            print(f"  {row['ID']} ({row['state']}_{row['pdb_id']}_{row['rep']})")
            print(f"    Volume in first frame: {row['pock_volume']:.1f} ({row['volume_category']})")
    else:
        print("  None found")


# Main Functions 
def pockets_data_analysis():
    """"""
    # filter if an experiment has more than one binding site or no binding site reported and check what's going on there
    base_dir = '/mnt/cpm_crienaecker/Z/RienaeckerC/apo_holo_analysis/meta_analysis/across_genes'
    df = pd.read_csv(os.path.join(base_dir, 'pocket_analysis_summary.csv'))
    print(df)
    find_pocket_abnormalities(df)



if __name__ == '__main__':
    t0 = time.time()
    pockets_data_analysis()
    t1 = time.time()
    print(f'Your pipeline took {t1 - t0} seconds')
