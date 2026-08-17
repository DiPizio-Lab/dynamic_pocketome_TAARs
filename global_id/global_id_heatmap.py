"""
The 'compare everything' panel for the combined apo+holo run. Built ONLY from
pocket_comparison_table.csv (no 60GB file). Uses output of global_id_wrapper_paper.py to generate a heatmap.

    rows    = Global ID (ubiquitous pockets on top; orthosteric flagged if the
              is_binding_site column is populated -- re-run the fixed wrapper for that)
    columns = every experiment, ordered gene -> state -> rep, so:
                * the 3 columns inside one gene+state block  = replicate reproducibility
                * the two state blocks inside one gene       = apo vs holo
                * the four gene super-blocks                 = cross-gene
    cell    = median pocket volume (viridis); blank grey = pocket absent
    dot     = >=2 local pockets fell into this global in that experiment (fusion)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SHORT_GENE_NAMES = {
    '8ITF': 'mTAAR9', '8IW4': 'mTAAR9', '8IW7': 'mTAAR9', '8IWM': 'mTAAR7f',
    '8JLJ': 'mTAAR1', '8JLK': 'mTAAR1', '8JLN': 'mTAAR9', '8JLO': 'hTAAR1',
    '8JLP': 'hTAAR1', '8JLQ': 'hTAAR1', '8JLR': 'hTAAR1', '8JSO': 'hTAAR1',
    '8PM2': 'mTAAR7f', '8W87': 'hTAAR1', '8W88': 'hTAAR1', '8W89': 'hTAAR1',
    '8W8A': 'hTAAR1', '8WC3': 'mTAAR1', '8WC4': 'mTAAR1', '8WC5': 'mTAAR1',
    '8WC6': 'mTAAR1', '8WC7': 'hTAAR1', '8WC8': 'hTAAR1', '8WC9': 'mTAAR9',
    '8WCB': 'mTAAR1', '8WCC': 'hTAAR1'}
STATE_ORDER = {"apo": 0, "holo": 1}


def load(run_dir=None, table=None):
    path = table or os.path.join(run_dir, "pocket_comparison_table.csv")
    pock_df = pd.read_csv(path)
    pock_df["Global ID"] = pd.to_numeric(pock_df["Global ID"], errors="coerce")
    pock_df = pock_df.dropna(subset=["Global ID"])
    pock_df["Global ID"] = pock_df["Global ID"].astype(int)
    pock_df = pock_df[pock_df["Global ID"] != 0]
    if "state" not in pock_df.columns:
        pock_df["state"] = pock_df["prj"].str.extract(r"^(apo|holo)", expand=False)
    if "gene" not in pock_df.columns or pock_df["gene"].isna().any():
        pock_df["gene"] = pock_df["prj"].str.replace(r"^(apo|holo)", "", regex=True).map(SHORT_GENE_NAMES)
    pock_df["rep"] = pock_df["rep"].astype(str)
    pock_df["experiment"] = pock_df["prj"] + "_" + pock_df["rep"]
    return pock_df


def build(pock_df):
    vol = pock_df.pivot_table(index="Global ID", columns="experiment",
                              values="median_interpolated_volume", aggfunc="median")
    cnt = pock_df.pivot_table(index="Global ID", columns="experiment",
                              values="Local Pocket ID", aggfunc="nunique")

    # experiment -> (gene, state, rep) for ordering + labels
    meta = (pock_df[["experiment", "gene", "state", "rep"]].drop_duplicates()
            .assign(rep_i=lambda d: d["rep"].astype(int),
                    st_i=lambda d: d["state"].map(STATE_ORDER))
            .sort_values(["gene", "st_i", "rep_i"]))
    col_order = meta["experiment"].tolist()
    vol = vol.reindex(columns=col_order)
    cnt = cnt.reindex(columns=col_order)

    # rows: numeric Global ID order (orthosteric still marked with a star)
    ortho = pd.Series(False, index=vol.index)
    if "is_binding_site" in pock_df.columns and (pock_df["is_binding_site"] == True).any():
        ortho = pock_df.groupby("Global ID")["is_binding_site"].apply(lambda s: (s == True).any())
        ortho = ortho.reindex(vol.index).fillna(False)
    row_order = sorted(vol.index)
    return vol.loc[row_order], cnt.loc[row_order], meta, ortho.loc[row_order]


def plot(vol, cnt, meta, ortho, out_png, out_pdf):
    genes = list(dict.fromkeys(meta["gene"]))  # preserves sorted order
    ncol, nrow = vol.shape[1], vol.shape[0]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#EFEFEF")  # absent pockets
    data = np.ma.masked_invalid(vol.values)

    fig, ax = plt.subplots(figsize=(0.42 * ncol + 3.2, 0.42 * nrow + 2.4))
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # fusion dots
    for i in range(nrow):
        for j in range(ncol):
            if cnt.values[i, j] is not None and not np.isnan(cnt.values[i, j]) and cnt.values[i, j] >= 2:
                ax.plot(j, i, marker="o", ms=3.2, mfc="white", mec="#222222", mew=0.4)

    # y ticks: global id, orthosteric in bold
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{'★ ' if o else ''}{gid}" for gid, o in zip(vol.index, ortho)],
                       fontsize=8, fontweight="normal")
    ax.set_ylabel("Global ID", fontsize=10)

    # x ticks: rep number at the base
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(meta["rep"].tolist(), fontsize=7)
    ax.tick_params(length=0)

    # block separators + gene/state labels
    edges_state, edges_gene = [], []
    prev_gene = prev_state = None
    gene_spans = {}
    for j, (_, r) in enumerate(meta.iterrows()):
        if r["gene"] != prev_gene and j != 0:
            edges_gene.append(j - 0.5)
        if (r["gene"], r["state"]) != (prev_gene, prev_state) and j != 0:
            edges_state.append(j - 0.5)
        gene_spans.setdefault(r["gene"], []).append(j)
        # state label centred over each state block
        prev_gene, prev_state = r["gene"], r["state"]
    for x in edges_state:
        ax.axvline(x, color="white", lw=1.2)
    for x in edges_gene:
        ax.axvline(x, color="#222222", lw=1.6)

    # state labels (apo/holo) just above the grid
    prev = None
    for j, (_, r) in enumerate(meta.iterrows()):
        key = (r["gene"], r["state"])
        if key != prev:
            block = meta.index[(meta["gene"] == r["gene"]) & (meta["state"] == r["state"])]
            js = [meta.index.get_loc(b) for b in block]
            ax.text(np.mean(js), -0.95, r["state"], ha="center", va="bottom",
                    fontsize=7.5, color="#444444")
            prev = key
    # gene labels above that
    for g, js in gene_spans.items():
        ax.text(np.mean(js), -2.35, g, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(nrow - 0.5, -3.1)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("median pocket volume (Å³)", fontsize=9)
    # legend proxies
    absent = plt.Rectangle((0, 0), 1, 1, fc="#EFEFEF")
    fus = plt.Line2D([0], [0], marker="o", ls="", mfc="white", mec="#222222", ms=6)
    ax.legend([absent, fus], ["pocket absent", "≥2 locals fused"],
              loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8)

    ax.set_title("Global pocket identity across every experiment (gene → state → rep)",
                 fontsize=11, loc="left", pad=40)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--run-dir", default=None)
    # ap.add_argument("--table", default=None)
    # ap.add_argument("--out", default="global_id_master_heatmap")
    # args = ap.parse_args()
    # if not args.run_dir and not args.table:
    #     ap.error("give --run-dir or --table")
    # pock_df = load(args.run_dir, args.table)
    pock_df = load(run_dir=os.path.join(os.getcwd(), "global_ID_both_states"))
    out_dir = os.path.join(os.getcwd(), "global_ID_both_states", "global_ID_heatmap")
    vol, cnt, meta, ortho = build(pock_df)
    plot(vol, cnt, meta, ortho,  out_dir + ".png", out_dir + ".pdf")
    print(f"{vol.shape[0]} global pockets x {vol.shape[1]} experiments")
    print(f"orthosteric flagged: {int(ortho.sum())}")
    print(f"wrote {out_dir}.png / .pdf")


if __name__ == "__main__":
    main()