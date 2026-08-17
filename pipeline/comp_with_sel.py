import os
import re
import argparse
from comparative_study import run_analysis
from scripts import config as conf

# Example hard-coded mapping (replace with your real IDs/genes)
PDB_TO_GENE = {
    # mTAAR1
    "apo8JLJ": "mTAAR1",
    "holo8JLJ": "mTAAR1",
    "apo8WC4": "mTAAR1",
    "holo8WC4": "mTAAR1",
    "apo8WC5": "mTAAR1",
    "holo8WC5": "mTAAR1",
    "apo8WC3": "mTAAR1",
    "holo8WC3": "mTAAR1",
    "apo8WC6": "mTAAR1",
    "holo8WC6": "mTAAR1",
    "apo8WC7": "mTAAR1",
    "holo8WC7": "mTAAR1",
    "apo8WC9": "mTAAR1",
    "holo8WC9": "mTAAR1",
    "apo8WCB": "mTAAR1",
    "holo8WCB": "mTAAR1",
    "apo8WCC": "mTAAR1",
    "holo8WCC": "mTAAR1",
    "apo8JLK": "mTAAR1",
    "holo8JLK": "mTAAR1",
    # hTAAR1
    "apo8JLR": "hTAAR1",
    "holo8JLR": "hTAAR1",
    "apo8JSO": "hTAAR1",
    "holo8JSO": "hTAAR1",
    "apo8W87": "hTAAR1",
    "holo8W87": "hTAAR1",
    "apo8W88": "hTAAR1",
    "holo8W88": "hTAAR1",
    "apo8W89": "hTAAR1",
    "holo8W89": "hTAAR1",
    "apo8W8A": "hTAAR1",
    "holo8W8A": "hTAAR1",
    "apo8WC8": "hTAAR1",
    "holo8WC8": "hTAAR1",
    "apo8JLN": "hTAAR1",
    "holo8JLN": "hTAAR1",
    "apo8JLO": "hTAAR1",
    "holo8JLO": "hTAAR1",
    "apo8JLP": "hTAAR1",
    "holo8JLP": "hTAAR1",
    "apo8JLQ": "hTAAR1",
    "holo8JLQ": "hTAAR1",
    # mTAAR9
    "apo8ITF": "mTAAR9",
    "holo8ITF": "mTAAR9",
    "apo8IW4": "mTAAR9",
    "holo8IW4": "mTAAR9",
    "apo8IW7": "mTAAR9",
    "holo8IW7": "mTAAR9",
    # mTAAR7f
    "apo8PM2": "mTAAR7f",
    "holo8PM2": "mTAAR7f"}


# TODO: check if the separation works...; adjust run_analysis
# Helpers to collect replicate folders
def find_replicates(base_dir="apo_structures"):
    """Return dict: { 'apo8ITF': [rep1_path, rep2_path, rep3_path], ... }
    Only include replicate folders (numeric names)."""
    groups = {}
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "pockets_dens":
            # root looks like .../apo8ITF/1/pockets_dens
            parts = root.split(os.sep)
            if len(parts) < 3:
                continue
            replicate_dir = parts[-2]
            pdb_id = parts[-3]

            if replicate_dir.isdigit():
                groups.setdefault(pdb_id, []).append(root)
    return groups



# Project list builders
def get_project_groups(mode):
    """
    Return dict of {group_label: [list of pockets_dens dirs]} based on mode.
    Modes:
        - within_rep: each PDB replicate group separately
        - apo_vs_holo: combine apo+holo replicates for same PDB
        - within_gene: group all replicates (apo+holo) by gene, only if present
        - across_genes: group all replicates (apo+holo) by gene, regardless of presence
    """
    apo_groups = find_replicates("apo_structures/results")
    holo_groups = find_replicates("holo_structures/results")
    # {'apo8ITF': ['apo_structures/results/apo8ITF/1/pockets_dens', 'apo_structures/results/apo8ITF/2/pockets_dens', ...
    if mode == "within_rep":
        return {pdb: reps for pdb, reps in {**apo_groups, **holo_groups}.items()}

    elif mode == "apo_vs_holo":
        groups = {}
        for pdb in set([re.sub(r"^(apo|holo)", "", k) for k in PDB_TO_GENE.keys()]):
            apo_key = f"apo{pdb}"
            holo_key = f"holo{pdb}"
            if apo_key in apo_groups and holo_key in holo_groups:
                groups[pdb] = apo_groups[apo_key] + holo_groups[holo_key]
        return groups

    elif mode == "within_gene":
        groups = {}
        for pdb_id, gene in PDB_TO_GENE.items():
            if pdb_id in apo_groups or pdb_id in holo_groups:
                reps = apo_groups.get(pdb_id, []) + holo_groups.get(pdb_id, [])
                groups.setdefault(gene, []).extend(reps)
        return groups

    elif mode == "across_genes":
        # one big group per gene, all apo+holo
        groups = apo_groups | holo_groups
        return groups

    else:
        raise ValueError(f"Unknown mode: {mode}")


# Controller
def run_meta_analysis(within_rep=False, apo_vs_holo=False, within_gene=False, across_genes=False):
    if within_rep:
        for pdb_id, prj_ls in get_project_groups("within_rep").items():
            outdir = os.path.join(conf.folder_meta_analysis, 'within_structure', pdb_id)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            run_analysis(prj_ls, saving_loc=outdir)

    if apo_vs_holo:
        for pdb_id, prj_ls in get_project_groups("apo_vs_holo").items():
            outdir = os.path.join(conf.folder_meta_analysis, 'apo_vs_holo', pdb_id)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            run_analysis(prj_ls, saving_loc=outdir)

    if within_gene:
        for gene, prj_ls in get_project_groups("within_gene").items():
            outdir = os.path.join(conf.folder_meta_analysis, 'within_gene', gene)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            run_analysis(prj_ls, saving_loc=outdir)

    if across_genes:
        outdir = os.path.join(conf.folder_meta_analysis, 'across_genes')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        result_dict = get_project_groups("across_genes")
        prj_ls =  sum(result_dict.values(), [])
        run_analysis(prj_ls, saving_loc=outdir)


# if __name__ == "__main__":
#     run_meta_analysis(within_rep=True, apo_vs_holo=True, within_gene=True, across_genes=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--within-rep", action="store_true")
    parser.add_argument("--apo-vs-holo", action="store_true")
    parser.add_argument("--within-gene", action="store_true")
    parser.add_argument("--across-genes", action="store_true")
    args = parser.parse_args()

    run_meta_analysis(within_rep=args.within_rep, apo_vs_holo=args.apo_vs_holo,
                      within_gene=args.within_gene, across_genes=args.across_genes)