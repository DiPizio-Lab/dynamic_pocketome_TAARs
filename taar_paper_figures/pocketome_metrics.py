"""apo-vs-holo comparison metrics for the allosteric pocketome.

Two complementary readouts, both computed per PDB ID from the pocket summary table:

  delta_pocket_count   how MANY allosteric pockets appear or vanish on ligand binding
  delta_median_volume  how BIG they are, as a percent change in the median pocket volume
  js_distance          how much the SHAPE of the size distribution is reorganised

count and volume can move independently: a receptor can keep the same number of pockets
while converting small ones into large ones, or gain pockets that are all small. The
JS distance catches reorganisations that both of the simple deltas miss."""
import numpy as np
import pandas as pd

CATEGORIES = ["Small (<250)", "Medium (250-500)", "Large (500-750)", "Very Large (>750)"]


def _entropy(p):
    """Shannon entropy in bits of a normalised distribution, 0 log 0 treated as 0"""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def js_distance(counts_a, counts_b):
    """Jensen-Shannon distance between two histograms of pocket size categories.

    Measures how differently the pockets are distributed over the size categories in two
    states, independent of how many pockets there are in total (both histograms are
    normalised to fractions first).

        JSD(P||Q) = H((P+Q)/2) - [H(P) + H(Q)] / 2
        d_JS      = sqrt(JSD)

    with H the Shannon entropy in bits.

    Chosen over the more familiar KL divergence for three reasons:
      symmetric   - apo vs holo and holo vs apo give the same number, which is what you
                    want when quantifying change rather than a direction of change.
      bounded     - with log2, d_JS lies in [0, 1] regardless of pocket count, so
                    structures with 9 and 24 pockets are directly comparable.
      finite      - KL is infinite whenever a category is populated in one state and
                    empty in the other. That happens constantly here ('Very Large' is
                    rare), so KL would be undefined for most structures.

    Interpretation: 0 means the size profile is identical between the two states; larger
    values mean ligand binding redistributes pockets across size classes. It says nothing
    about direction - pair it with delta_median_volume to see which way the shift goes.

    Returns np.nan if either state has no pockets."""
    p, q = np.asarray(counts_a, float), np.asarray(counts_b, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(np.sqrt(max(_entropy(m) - 0.5 * (_entropy(p) + _entropy(q)), 0.0)))


def category_histograms(df, pdb_col="pdb_id", state_col="state", rep_col="rep",
                        cat_col="volume_category"):
    """per (pdb, state, replicate) counts of pockets in each size category"""
    return (df.groupby([pdb_col, state_col, rep_col, cat_col], observed=True)
              .size().unstack(cat_col).reindex(columns=CATEGORIES).fillna(0.0))


def js_per_structure(df, pdb_order, **cols):
    """JS distance per structure, with replicate spread.

    Replicates are not paired across states (apo rep 1 is not a counterpart of holo
    rep 1), so every apo x holo replicate combination is evaluated and summarised:
    the median is the point estimate and the min/max give an honest spread. A wide
    spread means the effect is not resolved by three replicates.

    Returns (median, low, high) arrays aligned with pdb_order."""
    hist = category_histograms(df, **cols)
    med, lo, hi = [], [], []
    for pdb in pdb_order:
        try:
            apo, holo = hist.loc[(pdb, "apo")].values, hist.loc[(pdb, "holo")].values
        except KeyError:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        pairs = [js_distance(a, h) for a in apo for h in holo]
        pairs = [v for v in pairs if not np.isnan(v)]
        if pairs:
            med.append(float(np.median(pairs))); lo.append(min(pairs)); hi.append(max(pairs))
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    return np.array(med), np.array(lo), np.array(hi)


def delta_median_volume(df, pdb_order, volume_col="median_pock_volume_open",
                        pdb_col="pdb_id", state_col="state"):
    """percent change in the median allosteric pocket volume, (holo - apo) / apo * 100.

    Uses the per-pocket trajectory median as the input value, so it reports a change in
    typical pocket size rather than a change at the first frame."""
    med = df.groupby([pdb_col, state_col], observed=True)[volume_col].median()
    out = []
    for pdb in pdb_order:
        try:
            apo, holo = med.loc[(pdb, "apo")], med.loc[(pdb, "holo")]
        except KeyError:
            out.append(np.nan); continue
        out.append((holo - apo) / apo * 100.0 if apo else np.nan)
    return np.array(out, dtype=float)


def delta_pocket_count(df, pdb_order, pdb_col="pdb_id", state_col="state", rep_col="rep"):
    """change in the number of allosteric pockets, holo - apo, median over replicates"""
    per_rep = df.groupby([pdb_col, state_col, rep_col], observed=True).size()
    med = per_rep.groupby(level=[0, 1]).median()
    out = []
    for pdb in pdb_order:
        try:
            out.append(float(med.loc[(pdb, "holo")] - med.loc[(pdb, "apo")]))
        except KeyError:
            out.append(np.nan)
    return np.array(out, dtype=float)


def delta_category_fraction(df, pdb_order, pdb_col="pdb_id", state_col="state",
                            cat_col="volume_category"):
    """signed change in the FRACTION of pockets in each size category, holo - apo.

    This is the readable form of what js_distance compresses into one number: instead of
    'the profile moved by 0.35', it says 'small lost 18 points, medium gained 15, large
    gained 3'. Fractions are used rather than counts so a structure that simply has more
    pockets in one state does not register as a shape change - that is panel C's job.

    Each row sums to zero by construction (fractions in minus fractions out), so the
    positive and negative halves of a stacked bar are mirror images in total length.

    Returns an (n_structures, n_categories) array aligned with pdb_order / CATEGORIES."""
    out = np.full((len(pdb_order), len(CATEGORIES)), np.nan)
    for i, pdb in enumerate(pdb_order):
        sub = df[df[pdb_col] == pdb]
        fracs = {}
        for state in ("apo", "holo"):
            counts = np.array([(sub[(sub[state_col] == state)][cat_col] == c).sum()
                               for c in CATEGORIES], dtype=float)
            fracs[state] = counts / counts.sum() if counts.sum() else counts
        out[i] = fracs["holo"] - fracs["apo"]
    return out