"""
Orthosteric ("binding site") pocket classification. Computed here as
`is_binding_site` (pocket_comparison_table.csv, Step 2a and Step 3); Step 2b
merges it in and renames it to `is_orthosteric` for pocket_analysis_summary.csv
and everything downstream of that file.

A pocket is binding-site (orthosteric) if its own centroid (the mean x/y/z of
all its dummy-atom/alpha-sphere coordinates, across every frame) falls within
DEFAULT_DISTANCE_THRESHOLD (5 A) of its PDB's co-crystallized ligand centroid.

Apo pockets are scored against their *holo counterpart's* ligand centroid
(load_ligand_centroids() only reads holo_base, keyed by bare PDB ID; an apo
pocket's own apo/holo prefix is stripped before the lookup).
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
from config import HOLO_BASE_DIR

DEFAULT_HOLO_BASE = HOLO_BASE_DIR
DEFAULT_DISTANCE_THRESHOLD = 5.0  # Angstrom, pocket-centroid-to-ligand-centroid cutoff
DEFAULT_LIGAND_RESNAMES = ['LIG']


def extract_ligand_coords(pdb_path: str, resnames: List[str] = DEFAULT_LIGAND_RESNAMES) -> Optional[np.ndarray]:
    """
    Extract heavy-atom coordinates for the ligand from a PDB file.

    Reads both ATOM and HETATM records: the CHARMM-formatted structure.pdb files
    this pipeline works with record every residue -- ligand, protein, water,
    ions, lipid alike -- as ATOM, never HETATM, so filtering on record type
    alone would silently find nothing.

    Returns array of shape (n_atoms, 3), or None if the file has no residue
    named `resnames` -- always the case for an apo structure, since the ligand
    is stripped out before the MD system is built (see module docstring for how
    apo pockets are classified anyway).
    """
    coords = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            resname = line[17:20].strip()
            if resname not in resnames:
                continue

            atom_name = line[12:16].strip()
            if atom_name.startswith('H') or atom_name.startswith('D'):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue

    return np.array(coords) if coords else None


def load_ligand_centroids(holo_base: str = DEFAULT_HOLO_BASE,
                          ligand_resnames: List[str] = DEFAULT_LIGAND_RESNAMES) -> Dict[str, np.ndarray]:
    """
    Per bare PDB ID (no apo/holo prefix, e.g. '8ITF'): the co-crystallized
    ligand's centroid, extracted from that PDB's holo structure (the
    lowest-numbered replicate under holo_base/holo<PDBID>/<rep>/structure.pdb).
    """
    holo_path = Path(holo_base)
    centroids: Dict[str, np.ndarray] = {}

    if not holo_path.exists():
        print(f"WARNING: Holo base directory not found: {holo_base}")
        return centroids

    for pdb_dir in holo_path.iterdir():
        if not pdb_dir.is_dir():
            continue

        rep_dirs = sorted((d for d in pdb_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        pdb_file = next((d / "structure.pdb" for d in rep_dirs if (d / "structure.pdb").exists()), None)
        if pdb_file is None:
            continue

        ligand_coords = extract_ligand_coords(str(pdb_file), ligand_resnames)
        if ligand_coords is None or len(ligand_coords) == 0:
            continue

        bare_id = re.sub(r'^(apo|holo)', '', pdb_dir.name)
        centroids[bare_id] = ligand_coords.mean(axis=0)

    return centroids


def classify_binding_site(pockets_df: pd.DataFrame, holo_base: str = DEFAULT_HOLO_BASE,
                          distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
                          ligand_resnames: List[str] = DEFAULT_LIGAND_RESNAMES,
                          id_column: str = 'ID', coord_columns: List[str] = ['x', 'y', 'z'],
                          prj_column: str = 'prj',
                          saving_loc: Optional[str] = None,
                          filename: str = 'pocket_labels_ortho_def.csv') -> pd.DataFrame:
    """
    Classify every pocket in pockets_df as binding-site (orthosteric) or not.

    A pocket is binding-site if its own centroid (the mean of coord_columns
    across every one of its rows - every frame, every dummy atom/alpha sphere)
    is within distance_threshold of its PDB's ligand centroid (see
    load_ligand_centroids; apo pockets resolve to their holo counterpart's).

    Parameters
    ----------
    pockets_df : pd.DataFrame
        One row per (pocket, frame, alpha sphere) -- e.g.
        pipeline/pocket_dataframes.py's `all_pockets` right after
        _add_identity_columns(), before it's trimmed down to KEEP_COLS. Needs
        id_column, prj_column and coord_columns; 'rep' and 'pocket_number', if
        present, are carried through into the result for merging back.
    holo_base : str
        Path to the raw holo input tree (PDB+rep layout), used to derive each
        PDB's ligand centroid (see load_ligand_centroids).
    distance_threshold : float
        Pocket-centroid-to-ligand-centroid cutoff (Angstrom).
    id_column, coord_columns, prj_column : str / list[str]
        Column names for the per-pocket identifier, its x/y/z coordinates, and
        the project id (e.g. 'apo8ITF') used to look up the right ligand.
    saving_loc : str, optional
        If given, also writes the result to {saving_loc}/{filename} (Stage 3's
        fallback source for is_binding_site when it's missing from
        pocket_comparison_table.csv).

    Returns
    -------
    pd.DataFrame
        One row per pocket ID: id_column, prj_column, 'rep'/'pocket_number' (if
        present in pockets_df), 'distance_to_ligand_centroid', 'is_binding_site'.
    """
    ligand_centroids = load_ligand_centroids(holo_base, ligand_resnames=ligand_resnames)

    extra_cols = [c for c in ('rep', 'pocket_number') if c in pockets_df.columns]
    pocket_centroids = pockets_df.groupby(id_column)[coord_columns].mean()
    pocket_meta = pockets_df.groupby(id_column)[[prj_column] + extra_cols].first()

    rows = []
    for pocket_id, centroid in pocket_centroids.iterrows():
        prj = pocket_meta.loc[pocket_id, prj_column]
        bare_id = re.sub(r'^(apo|holo)', '', str(prj))
        ligand_centroid = ligand_centroids.get(bare_id)

        if ligand_centroid is None:
            distance = np.nan
            is_binding = False
        else:
            distance = float(np.linalg.norm(centroid.values - ligand_centroid))
            is_binding = distance <= distance_threshold

        row = {id_column: pocket_id, prj_column: prj,
              'distance_to_ligand_centroid': round(distance, 2) if not np.isnan(distance) else np.nan,
              'is_binding_site': is_binding}
        for c in extra_cols:
            row[c] = pocket_meta.loc[pocket_id, c]
        rows.append(row)

    result_df = pd.DataFrame(rows)

    if saving_loc is not None:
        result_df.to_csv(os.path.join(saving_loc, filename), index=False)

    return result_df
