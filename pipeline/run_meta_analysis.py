"""Non-interactive Step 2 driver: pocket dataframes (2.1), then Global-ID clustering and
apo/holo comparison (2.2) for apo, holo and the combined run -- the same recipe as
step2_pocket_analysis.ipynb, as a single script for batch/SLURM use. Requires Step 1
(pipeline/analysis_pipeline.py) to have already finished for every replicate.

Step 2.1 always parses every PDB ID -- all_pockets/pocket_summary are meant to be the complete
dataset, regardless of what any one Global ID run is scoped to.

Step 2.2 (Global ID clustering) defaults to conf.REPRESENTATIVE_PDB_IDS (one PDB ID per gene)
rather than every PDB ID, because clustering is meant to answer a specific, scoped research
question -- comparing apo vs holo, comparing triplicate replicates, or (this default's purpose)
showcasing Global ID conservation/divergence across a biologically diverse set of genes -- not
an unscoped run across every solved structure of the same receptor, which mixes redundant
near-duplicates into one run rather than a deliberate comparison. (It also keeps memory
tractable: clustering the full 26-PDB-ID dataset needs 100+GB and has OOM-killed a node with
anon-rss over 230GB -- a consequence of an unscoped run, not the reason to scope one.) Pass
--pdb-ids to cluster a different subset, or --all-pdb-ids to explicitly opt into the unscoped,
memory-heavy run across everything.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf
from pipeline import pocket_dataframes as pdf
from pipeline.global_id_and_comparison import run_global_id_states
from scripts import run_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--states', nargs='+', default=['apo', 'holo', 'both'],
                         choices=['apo', 'holo', 'both'],
                         help="Global-ID runs to produce (default: apo holo both)")
    parser.add_argument('--no-reconcile-apo-holo', dest='reconcile_apo_holo', action='store_false',
                         default=True,
                         help="Skip match_states() apo<->holo Global ID reconciliation "
                              "(on by default when both 'apo' and 'holo' are requested)")
    parser.add_argument('--pdb-ids', nargs='+', default=None, metavar='PDB_ID',
                         help="Which PDB IDs to scope Global ID clustering (Step 2.2) to -- "
                              "Step 2.1's parse always covers every PDB ID regardless. Default: "
                              f"conf.REPRESENTATIVE_PDB_IDS, currently {conf.REPRESENTATIVE_PDB_IDS} "
                              "(one per gene). Mutually exclusive with --all-pdb-ids.")
    parser.add_argument('--all-pdb-ids', action='store_true',
                         help="Cluster every PDB ID together instead of the representative "
                              "subset -- an unscoped comparison across every solved structure of "
                              "the same receptor, and needs ~100+GB RAM for Step 2.2 (see module "
                              "docstring); has OOM-killed a 230GB+ node. Mutually exclusive with "
                              "--pdb-ids.")
    args = parser.parse_args()
    if args.pdb_ids and args.all_pdb_ids:
        parser.error("--pdb-ids and --all-pdb-ids are mutually exclusive")
    args.pdb_ids = None if args.all_pdb_ids else (args.pdb_ids or conf.REPRESENTATIVE_PDB_IDS)
    return args


def main():
    args = parse_args()
    bw_file_loc = os.path.join(conf.REFERENCE_DATA_DIR, 'TAARs_numbered')
    pdb_file_loc = os.path.join(conf.HOLO_RESULTS_DIR, 'holo8ITF', '1')

    isovalues = conf.require_float_isovalues(conf.ISOVALUES)
    for isovalue in isovalues:
        saving_loc = conf.isovalue_meta_analysis_dir(isovalue, isovalues)
        gid_root = conf.isovalue_meta_analysis_root(isovalue, isovalues)
        pocket_dirs = pdf.pocket_dirs_for(isovalue=isovalue, isovalues=isovalues)  # every PDB ID

        print(f"[Step 2.1] isovalue={isovalue}: parsing {len(pocket_dirs)} replicate(s) of "
              f"pockets/ output (every PDB ID) -> {saving_loc}")
        with run_summary.stage(*run_summary.AGGREGATE_KEY, 'meta_analysis'):
            tables = pdf.build_pocket_dataframes(pocket_dirs, saving_loc=saving_loc, isovalue=isovalue,
                                                 bw_file_loc=bw_file_loc, pdb_file_loc=pdb_file_loc)

        print(f"[Step 2.2] isovalue={isovalue}: assigning Global IDs ({', '.join(args.states)}) "
              f"scoped to {'every PDB ID' if args.pdb_ids is None else args.pdb_ids} -> {gid_root}")
        with run_summary.stage(*run_summary.AGGREGATE_KEY, 'global_id'):
            run_global_id_states(tables['all_pockets'], tables['pocket_summary'], states=args.states,
                                 source_loc=saving_loc, gid_root=gid_root, pdb_ids=args.pdb_ids,
                                 reconcile_apo_holo=args.reconcile_apo_holo)

    print("Step 2 complete.")


if __name__ == '__main__':
    main()
