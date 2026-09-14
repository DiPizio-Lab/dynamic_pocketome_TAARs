"""Stages a Zenodo-ready export of Step 1's raw pocket-search output (mdpocket/ATClus, the
`pockets/` directory under every replicate) plus Step 2.2's Global ID clustering runs. Read-only
over output/, writes only under --out-dir (default: 0_zenodo_pocketome/).

Excluded from each replicate's pockets/: aligned_traj.xtc/.dcd, aligned_top.pdb (published
separately, see build_zenodo_export.py) and atclus/atclus.f/atclus.inc/atclus.o (static copies
of scripts/atclus/, identical across every replicate).

Layout
------
    0_zenodo_pocketome/<PDB_ID>.zip
        <PDB_ID>/<state>/<rep>/pockets/...   -- for state in (apo, holo), rep in (1, 2, 3)
    0_zenodo_pocketome/global_id_<run>.zip   -- one per Global ID run actually present
        global_id_<run>/...                  -- run in (apo, holo, combined)
    0_zenodo_pocketome/zenodo_pocketome_manifest.csv

Idempotent and incremental: a zip already present is left alone unless --force, and a PDB ID or
Global ID run whose source isn't done yet is skipped (reported in the manifest as
'missing-source').

Run this as `python pipeline/build_pocketome_zenodo_export.py`, not from inside scripts/.
"""
import argparse
import csv
import os
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf
from scripts import gene_selections as gene_sel

EXCLUDE_BASENAMES = {
    "aligned_traj.xtc", "aligned_traj.dcd", "aligned_top.pdb",
    "atclus", "atclus.f", "atclus.inc", "atclus.o",
}

GID_RUNS = ["apo", "holo", "combined"]


def pdb_ids():
    """Every PDB ID this dataset covers, from the hard-coded gene dict."""
    return sorted(gene_sel.gene_dict().keys())


def _pocket_search_done(pockets_dir):
    """True iff a *_descriptors.txt file exists in pockets_dir."""
    if not os.path.isdir(pockets_dir):
        return False
    return any(f.endswith("descriptors.txt") for f in os.listdir(pockets_dir))


def _replicate_pockets_dirs(pdb_id):
    """{(state, rep): pockets_dir_path} for every (state, rep) whose pocket-search has finished."""
    found = {}
    for state in ("apo", "holo"):
        results_dir = conf.APO_RESULTS_DIR if state == "apo" else conf.HOLO_RESULTS_DIR
        for rep in ("1", "2", "3"):
            pockets_dir = os.path.join(results_dir, f"{state}{pdb_id}", rep, "pockets")
            if _pocket_search_done(pockets_dir):
                found[(state, rep)] = pockets_dir
    return found


def zip_pdb(pdb_id, out_dir, dry_run, force, manifest):
    zip_path = os.path.join(out_dir, f"{pdb_id}.zip")
    reps = _replicate_pockets_dirs(pdb_id)

    if not reps:
        manifest.append({"kind": "pdb", "id": pdb_id, "status": "missing-source",
                         "n_files": 0, "size_bytes": 0, "destination": zip_path})
        return

    if not force and os.path.exists(zip_path):
        manifest.append({"kind": "pdb", "id": pdb_id, "status": "skipped",
                         "n_files": "", "size_bytes": os.path.getsize(zip_path),
                         "destination": zip_path})
        return

    print(f"{'[dry-run] would zip' if dry_run else 'zipping'} {pdb_id} "
         f"({len(reps)}/6 replicates found) -> {zip_path}")
    if dry_run:
        manifest.append({"kind": "pdb", "id": pdb_id, "status": "dry-run",
                         "n_files": "", "size_bytes": 0, "destination": zip_path})
        return

    os.makedirs(out_dir, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for (state, rep), pockets_dir in sorted(reps.items()):
            for root, _, files in os.walk(pockets_dir):
                for fname in files:
                    if fname in EXCLUDE_BASENAMES:
                        continue
                    fpath = os.path.join(root, fname)
                    arcname = os.path.join(pdb_id, state, rep, "pockets",
                                           os.path.relpath(fpath, pockets_dir))
                    zf.write(fpath, arcname)
                    n_files += 1
    manifest.append({"kind": "pdb", "id": pdb_id, "status": "written", "n_files": n_files,
                     "size_bytes": os.path.getsize(zip_path), "destination": zip_path})


def zip_gid_run(run, out_dir, dry_run, force, manifest):
    src_dir = os.path.join(conf.META_ANALYSIS_ROOT, f"global_ID_{run}")
    zip_path = os.path.join(out_dir, f"global_id_{run}.zip")

    if not os.path.isdir(src_dir) or not os.listdir(src_dir):
        manifest.append({"kind": "gid", "id": run, "status": "missing-source",
                         "n_files": 0, "size_bytes": 0, "destination": zip_path})
        return

    if not force and os.path.exists(zip_path):
        manifest.append({"kind": "gid", "id": run, "status": "skipped",
                         "n_files": "", "size_bytes": os.path.getsize(zip_path),
                         "destination": zip_path})
        return

    print(f"{'[dry-run] would zip' if dry_run else 'zipping'} Global ID run '{run}' -> {zip_path}")
    if dry_run:
        manifest.append({"kind": "gid", "id": run, "status": "dry-run",
                         "n_files": "", "size_bytes": 0, "destination": zip_path})
        return

    os.makedirs(out_dir, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.join(f"global_id_{run}", os.path.relpath(fpath, src_dir))
                zf.write(fpath, arcname)
                n_files += 1
    manifest.append({"kind": "gid", "id": run, "status": "written", "n_files": n_files,
                     "size_bytes": os.path.getsize(zip_path), "destination": zip_path})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(conf.PROJECT_ROOT, "0_zenodo_pocketome"),
                        help="Root of the staged export (default: PROJECT_ROOT/0_zenodo_pocketome)")
    parser.add_argument("--pdb-ids", nargs="+", default=None, metavar="PDB_ID",
                        help="Only these PDB IDs (default: all -- see pdb_ids())")
    parser.add_argument("--skip-gid", action="store_true",
                        help="Don't zip the Global ID runs, just the per-PDB pocket data")
    parser.add_argument("--force", action="store_true",
                        help="Re-zip even if a same-named zip already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be zipped without writing anything")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = []

    targets = args.pdb_ids if args.pdb_ids else pdb_ids()
    for pdb_id in sorted(targets):
        zip_pdb(pdb_id, args.out_dir, args.dry_run, args.force, manifest)

    if not args.skip_gid:
        for run in GID_RUNS:
            zip_gid_run(run, args.out_dir, args.dry_run, args.force, manifest)

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        manifest_path = os.path.join(args.out_dir, "zenodo_pocketome_manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["kind", "id", "status", "n_files",
                                                    "size_bytes", "destination"])
            writer.writeheader()
            writer.writerows(manifest)
        print(f"manifest: {manifest_path}")

    counts = defaultdict(int)
    bytes_written = 0
    n_zip_files = 0
    for row in manifest:
        counts[row["status"]] += 1
        if row["status"] in ("written", "skipped"):
            bytes_written += row["size_bytes"]
            n_zip_files += 1
    print(f"\n{len(manifest)} target(s): {counts['written']} written, {counts['skipped']} "
         f"already present, {counts['missing-source']} not ready yet")
    print(f"Zenodo upload would be {n_zip_files} file(s) (cap: 100), "
         f"{bytes_written / 1e9:.2f} GB (cap: 50 GB)")
    missing = [row for row in manifest if row["status"] == "missing-source"]
    if missing:
        print(f"Not ready yet: {', '.join(row['id'] for row in missing)}")


if __name__ == "__main__":
    main()
