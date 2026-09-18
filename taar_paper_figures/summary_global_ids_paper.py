"""One row per Global ID over the combined apo+holo run, built from
pocket_comparison_table.csv (one row per local pocket).
Required : pocket_comparison_table.csv/.parquet (pipeline/pocket_io.py - written by
    pipeline/global_id_and_comparison.py)
    columns used: Global ID, Local Pocket ID, prj, rep, gene, state,
                  median_interpolated_volume, is_orthosteric
Optional (best-effort, skipped if absent):
    first_frame_summary_df.csv   -> cx,cy,cz  (centroid). No longer produced by the
                                     pipeline (pipeline/pocket_dataframes.py writes
                                     all_pockets/pocket_summary instead)
    pocket_labels_ortho_def.csv  -> is_orthosteric (used only if is_orthosteric in the
                                     main table is empty - shouldn't happen
                                     pipeline/pocket_dataframes.py always populates it)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pipeline import pocket_io


def summarise(run_dir):
    pock_df = pocket_io.load_table(run_dir, "pocket_comparison_table")
    pock_df["Global ID"] = pd.to_numeric(pock_df["Global ID"], errors="coerce")
    pock_df = pock_df.dropna(subset=["Global ID"])
    pock_df["Global ID"] = pock_df["Global ID"].astype(int)
    if "state" not in pock_df.columns:
        pock_df["state"] = pock_df["prj"].str.extract(r"^(apo|holo)", expand=False)
    pock_df["experiment"] = pock_df["prj"] + "_" + pock_df["rep"].astype(str)

    rows = []
    for gid, g in pock_df.groupby("Global ID"):
        apo = g[g.state == "apo"]
        holo = g[g.state == "holo"]
        vol_apo = apo["median_interpolated_volume"].median()
        vol_holo = holo["median_interpolated_volume"].median()
        rows.append(dict(
            global_id=gid,
            n_local=g["Local Pocket ID"].nunique(),
            n_experiments=g["experiment"].nunique(),
            n_genes=g["gene"].nunique(),
            genes=",".join(sorted(g["gene"].dropna().unique())),
            n_apo_exp=apo["experiment"].nunique(),
            n_holo_exp=holo["experiment"].nunique(),
            present_in=("both" if len(apo) and len(holo)
                        else "apo only" if len(apo) else "holo only"),
            vol_apo=round(vol_apo, 1) if pd.notna(vol_apo) else np.nan,
            vol_holo=round(vol_holo, 1) if pd.notna(vol_holo) else np.nan,
            vol_delta_holo_minus_apo=(round(vol_holo - vol_apo, 1)
                                      if pd.notna(vol_apo) and pd.notna(vol_holo) else np.nan)))
    ortho = pd.DataFrame(rows).set_index("global_id").sort_index()

    if "is_orthosteric" in pock_df.columns and (pock_df["is_orthosteric"] == True).any():
        flag = pock_df.groupby("Global ID")["is_orthosteric"].apply(lambda s: (s == True).any())
        ortho["is_orthosteric"] = ortho.index.map(flag).fillna(False)
    else:
        ortho = _orthosteric_from_labels(ortho, run_dir)
    return ortho


def _orthosteric_from_labels(tbl, run_dir):
    f = os.path.join(run_dir, "pocket_labels_ortho_def.csv")
    if not os.path.isfile(f):
        print("no orthosteric info: is_orthosteric is empty and "
              "pocket_labels_ortho_def.csv not found - is_orthosteric is left blank")
        tbl["is_orthosteric"] = np.nan
        return tbl
    lab = pd.read_csv(f)
    gcol = next((c for c in ("Global ID", "global_id") if c in lab.columns), None)
    bcol = next((c for c in ("is_orthosteric", "is_binding_site") if c in lab.columns), None)
    if gcol is None or bcol is None:
        print(f"pocket_labels_ortho_def.csv found but columns unclear "
              f"({list(lab.columns)}) - is_orthosteric is left blank")
        tbl["is_orthosteric"] = np.nan
        return tbl
    lab[gcol] = pd.to_numeric(lab[gcol], errors="coerce")
    flag = lab.dropna(subset=[gcol]).groupby(lab[gcol].astype(int))[bcol].apply(
        lambda s: (s == True).any() if s.dtype != bool else s.any())
    tbl["is_orthosteric"] = tbl.index.map(flag).fillna(False)
    return tbl


def add_centroids(tbl, run_dir):
    first_frame_file = os.path.join(run_dir, "first_frame_summary_df.csv")
    if not os.path.isfile(first_frame_file):
        print("  [note] first_frame_summary_df.csv not found -- cx,cy,cz skipped")
        return tbl
    first_frame_df = pd.read_csv(first_frame_file)
    if "Global ID" not in first_frame_df.columns or not {"x", "y", "z"}.issubset(first_frame_df.columns):
        return tbl
    first_frame_df["Global ID"] = pd.to_numeric(first_frame_df["Global ID"], errors="coerce")
    first_frame_df = first_frame_df.dropna(subset=["Global ID"])
    first_frame_df["Global ID"] = first_frame_df["Global ID"].astype(int)
    centroid = first_frame_df.groupby("Global ID")[["x", "y", "z"]].mean().round(1)
    centroid.columns = ["cx", "cy", "cz"]
    return tbl.join(centroid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help=".../global_ID_combined")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pock_df = summarise(args.run_dir)
    pock_df = add_centroids(pock_df, args.run_dir)

    ortho = pock_df.get("is_orthosteric", pd.Series(False, index=pock_df.index)).fillna(False).astype(int)
    pock_df = (pock_df.assign(_o=ortho)
              .sort_values(["_o", "n_experiments", "n_genes"], ascending=False)
              .drop(columns="_o"))

    out = args.out or os.path.join(args.run_dir, "global_pocket_summary.csv")
    pock_df.to_csv(out)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pock_df.to_string())
    n_both = (pock_df["present_in"] == "both").sum()
    print(f"\nwrote {out}")
    print(f"headline: {len(pock_df)} global pockets | {n_both} in both states "
          f"| {len(pock_df) - n_both} state-specific")


if __name__ == "__main__":
    main()