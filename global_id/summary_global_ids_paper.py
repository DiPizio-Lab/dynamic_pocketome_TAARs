#!/usr/bin/env python3
"""
One row per Global ID over the COMBINED apo+holo run, built from
pocket_comparison_table.csv (one row per local pocket -- small, no 60GB file).

Required : pocket_comparison_table.csv
    columns used: Global ID, Local Pocket ID, prj, rep, gene, state,
                  median_interpolated_volume, is_binding_site (if populated)

Optional (best-effort, skipped if absent):
    first_frame_summary_df.csv   -> cx,cy,cz  (centroid; ~1000x subsample, small)
    pocket_labels_ortho_def.csv  -> is_orthosteric  (used only if is_binding_site
                                     in the main table is empty, which it was in
                                     the combined run)
"""
import os
import argparse
import numpy as np
import pandas as pd


def summarise(run_dir):
    p = pd.read_csv(os.path.join(run_dir, "pocket_comparison_table.csv"))
    p["Global ID"] = pd.to_numeric(p["Global ID"], errors="coerce")
    p = p.dropna(subset=["Global ID"])
    p["Global ID"] = p["Global ID"].astype(int)
    if "state" not in p.columns:
        p["state"] = p["prj"].str.extract(r"^(apo|holo)", expand=False)
    p["experiment"] = p["prj"] + "_" + p["rep"].astype(str)

    rows = []
    for gid, g in p.groupby("Global ID"):
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
    tbl = pd.DataFrame(rows).set_index("global_id").sort_index()

    if "is_binding_site" in p.columns and (p["is_binding_site"] == True).any():
        flag = p.groupby("Global ID")["is_binding_site"].apply(lambda s: (s == True).any())
        tbl["is_orthosteric"] = tbl.index.map(flag).fillna(False)
    else:
        tbl = _ortho_from_labels(tbl, run_dir)
    return tbl


def _ortho_from_labels(tbl, run_dir):
    f = os.path.join(run_dir, "pocket_labels_ortho_def.csv")
    if not os.path.isfile(f):
        print("  [note] no orthosteric info: is_binding_site is empty and "
              "pocket_labels_ortho_def.csv not found -- is_orthosteric left blank")
        tbl["is_orthosteric"] = np.nan
        return tbl
    lab = pd.read_csv(f)
    gcol = next((c for c in ("Global ID", "global_id") if c in lab.columns), None)
    bcol = next((c for c in lab.columns
                 if c.lower().replace(" ", "_") in
                 ("is_binding_site", "is_orthosteric", "orthosteric", "binding_site")), None)
    if gcol is None or bcol is None:
        print(f"  [note] pocket_labels_ortho_def.csv found but columns unclear "
              f"({list(lab.columns)}) -- is_orthosteric left blank")
        tbl["is_orthosteric"] = np.nan
        return tbl
    lab[gcol] = pd.to_numeric(lab[gcol], errors="coerce")
    flag = lab.dropna(subset=[gcol]).groupby(lab[gcol].astype(int))[bcol].apply(
        lambda s: (s == True).any() if s.dtype != bool else s.any())
    tbl["is_orthosteric"] = tbl.index.map(flag).fillna(False)
    return tbl


def add_centroids(tbl, run_dir):
    f = os.path.join(run_dir, "first_frame_summary_df.csv")
    if not os.path.isfile(f):
        print("  [note] first_frame_summary_df.csv not found -- cx,cy,cz skipped")
        return tbl
    s = pd.read_csv(f)
    if "Global ID" not in s.columns or not {"x", "y", "z"}.issubset(s.columns):
        return tbl
    s["Global ID"] = pd.to_numeric(s["Global ID"], errors="coerce")
    s = s.dropna(subset=["Global ID"])
    s["Global ID"] = s["Global ID"].astype(int)
    cen = s.groupby("Global ID")[["x", "y", "z"]].mean().round(1)
    cen.columns = ["cx", "cy", "cz"]
    return tbl.join(cen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help=".../global_ID_both_states")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tbl = summarise(args.run_dir)
    tbl = add_centroids(tbl, args.run_dir)

    ortho = tbl.get("is_orthosteric", pd.Series(False, index=tbl.index)).fillna(False).astype(int)
    tbl = (tbl.assign(_o=ortho)
              .sort_values(["_o", "n_experiments", "n_genes"], ascending=False)
              .drop(columns="_o"))

    out = args.out or os.path.join(args.run_dir, "global_pocket_summary.csv")
    tbl.to_csv(out)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(tbl.to_string())
    n_both = (tbl["present_in"] == "both").sum()
    print(f"\nwrote {out}")
    print(f"headline: {len(tbl)} global pockets | {n_both} in both states "
          f"| {len(tbl) - n_both} state-specific")


if __name__ == "__main__":
    main()