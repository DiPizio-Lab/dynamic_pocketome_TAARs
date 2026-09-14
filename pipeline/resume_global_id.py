"""Re-runs Step 2.2 (Global ID clustering) alone, loading Step 2.1's already-written
all_pockets/pocket_summary tables from disk instead of re-parsing pockets/ output from scratch.

Verifies the loaded tables cover every replicate get_project_list() expects before running.
Pass --skip-completeness-check to bypass this.

Defaults Global ID clustering to conf.REPRESENTATIVE_PDB_IDS. Pass --pdb-ids for a different
subset, or --all-pdb-ids for every PDB ID.

Run this as `python pipeline/resume_global_id.py`, not from inside scripts/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf
from pipeline import pocket_io
from pipeline.analysis_pipeline import get_project_list
from pipeline.global_id_and_comparison import run_global_id_states
from scripts import run_summary


def _expected_prj_reps():
    return {(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in get_project_list()}


def _check_complete(pocket_summary, saving_loc):
    expected = _expected_prj_reps()
    found = set(zip(pocket_summary['prj'], pocket_summary['rep'].astype(str)))
    missing = expected - found
    if missing:
        raise SystemExit(
            f"pocket_summary loaded from {saving_loc} covers {len(found)}/{len(expected)} "
            f"replicates -- missing e.g. {sorted(missing)[:5]}. Refusing to run Global ID "
            f"clustering on incomplete data (pass --skip-completeness-check to override).")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--states', nargs='+', default=['apo', 'holo', 'both'],
                         choices=['apo', 'holo', 'both'],
                         help="Global-ID runs to (re)produce (default: apo holo both)")
    parser.add_argument('--no-reconcile-apo-holo', dest='reconcile_apo_holo', action='store_false',
                         default=True,
                         help="Skip match_states() apo<->holo Global ID reconciliation "
                              "(on by default when both 'apo' and 'holo' are requested)")
    parser.add_argument('--skip-completeness-check', action='store_true',
                         help="Don't verify the loaded tables cover every expected replicate "
                              "before running -- only for deliberate partial/test runs")
    parser.add_argument('--pdb-ids', nargs='+', default=None, metavar='PDB_ID',
                         help="Which PDB IDs to scope Global ID clustering to. Default: "
                              f"conf.REPRESENTATIVE_PDB_IDS, currently {conf.REPRESENTATIVE_PDB_IDS} "
                              "(one per gene). Mutually exclusive with --all-pdb-ids.")
    parser.add_argument('--all-pdb-ids', action='store_true',
                         help="Cluster every PDB ID together instead of the representative "
                              "subset -- needs ~100+GB RAM; has OOM-killed a 230GB+ node. "
                              "Mutually exclusive with --pdb-ids.")
    args = parser.parse_args()
    if args.pdb_ids and args.all_pdb_ids:
        parser.error("--pdb-ids and --all-pdb-ids are mutually exclusive")
    args.pdb_ids = None if args.all_pdb_ids else (args.pdb_ids or conf.REPRESENTATIVE_PDB_IDS)
    return args


def main():
    args = parse_args()
    isovalues = conf.require_float_isovalues(conf.ISOVALUES)
    for isovalue in isovalues:
        saving_loc = conf.isovalue_meta_analysis_dir(isovalue, isovalues)
        gid_root = conf.isovalue_meta_analysis_root(isovalue, isovalues)

        print(f"isovalue={isovalue}: loading all_pockets/pocket_summary from {saving_loc}")
        all_pockets = pocket_io.load_table(saving_loc, 'all_pockets')
        pocket_summary = pocket_io.load_table(saving_loc, 'pocket_summary')

        if not args.skip_completeness_check:
            _check_complete(pocket_summary, saving_loc)

        print(f"[Step 2.2] isovalue={isovalue}: assigning Global IDs ({', '.join(args.states)}) "
              f"scoped to {'every PDB ID' if args.pdb_ids is None else args.pdb_ids} -> {gid_root}")
        with run_summary.stage(*run_summary.AGGREGATE_KEY, 'global_id'):
            run_global_id_states(all_pockets, pocket_summary, states=args.states,
                                 source_loc=saving_loc, gid_root=gid_root, pdb_ids=args.pdb_ids,
                                 reconcile_apo_holo=args.reconcile_apo_holo)

    print("Global ID resume complete.")


if __name__ == '__main__':
    main()
