"""This script is a helper to find the frame of the pockets closest to their median volume,
exports them as a molecular nodes ready PDB. This is to assist in the visualisation of Chapter 4 - Global ID
For every pocket (apo + holo) in the combined run, find the frame closest to
its median_interpolated_volume and write that frame's cavity alpha spheres as a
single-frame, MN-readable PDB"""
import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

ROOT    = OUTPUT_DIR
SUMMARY = f"{ROOT}/global_ID_holo_only/pocket_comparison_table.csv"
OUT_DIR = f"{ROOT}/global_ID_holo_only/blender_median_frames"
ISO     = "3"   # mdpout_dens_iso_3-out-NN...


def pick_frame(descriptors, target):
    df = pd.read_csv(descriptors, sep=r"\s+")
    present = df[df["pock_volume"].astype(float) > 0]
    row = present.iloc[(present["pock_volume"] - target).abs().argmin()]
    return int(row["snapshot"]), float(row["pock_volume"])


def nth_block(mdpocket_pdb, frame):
    blocks, cur = [], []
    for line in open(mdpocket_pdb):
        if line.startswith("END"):
            blocks.append(cur); cur = []
        elif line.startswith(("ATOM", "HETATM")):
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks[frame - 1]


os.makedirs(OUT_DIR, exist_ok=True)
summary = pd.read_csv(SUMMARY)

written, skipped = 0, 0
for _, r in summary.iterrows():
    if pd.isna(r["median_interpolated_volume"]):
        skipped += 1; continue
    prj   = r["prj"]                                   # e.g. apo8ITF / holo8ITF
    rep   = str(r["rep"])
    gid   = int(r["Global ID"]) if pd.notna(r["Global ID"]) else "NA"
    nn    = re.search(r"_p(\d+)", r["Local Pocket ID"]).group(1).zfill(2)
    state = "apo" if prj.startswith("apo") else "holo"

    pdir  = f"{ROOT}/{state}_structures/results/{prj}/{rep}/pockets_dens"
    descr = f"{pdir}/mdpout_dens_iso_{ISO}-out-{nn}_descriptors.txt"
    mdp   = f"{pdir}/mdpout_dens_iso_{ISO}-out-{nn}_mdpocket.pdb"
    if not (os.path.isfile(descr) and os.path.isfile(mdp)):
        print(f"skip {prj} {rep} p{nn}: files missing"); skipped += 1; continue

    frame, vol = pick_frame(descr, float(r["median_interpolated_volume"]))
    lines = nth_block(mdp, frame)
    out = f"{OUT_DIR}/g{gid}_{prj}_{rep}_p{nn}.pdb"
    with open(out, "w") as f:
        f.write(f"REMARK global {gid} {prj} rep{rep} p{nn} frame {frame} "
                f"vol={vol:.2f} median={r['median_interpolated_volume']:.2f}\n")
        f.writelines(lines)
        f.write("END\n")
    written += 1

print(f"wrote {written} median-frame PDBs to {OUT_DIR} ({skipped} skipped)")