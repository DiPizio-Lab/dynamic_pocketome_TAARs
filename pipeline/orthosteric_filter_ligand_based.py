"""
Orthosteric ("binding site") pocket classification -- single source of truth for
the underlying is_binding_site/is_orthosteric label used everywhere in the
pipeline (Step 2 and Step 3). Computed once here as `is_binding_site`
(pocket_comparison_table.csv, Step 2a and Step 3); Step 2b merges it in and
renames it to `is_orthosteric` for pocket_analysis_summary.csv and everything
downstream of that file (run_apo_holo_comparison.py, taar_paper_figures/) --
see the README for why the name differs between the two.

Definition: per holo PDB, the residues within RESIDUE_DISTANCE_THRESHOLD (5 A) of
the co-crystallized ligand's centroid define that PDB's orthosteric site. Apo
structures reuse their holo counterpart's site (both states are superposed into
one coordinate frame and share residue numbering, and apo has no ligand of its
own). A pocket is classified as binding-site (orthosteric) if at least
MIN_RESIDUE_OVERLAP (80%) of its own residues fall inside that site.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
from config import HOLO_BASE_DIR

DEFAULT_HOLO_BASE = HOLO_BASE_DIR
DEFAULT_LIGAND_RESNAMES = ['LIG']
RESIDUE_DISTANCE_THRESHOLD = 5.0   # Angstrom, residue-to-ligand-centroid cutoff
MIN_RESIDUE_OVERLAP = 0.8          # fraction of a pocket's own residues that must fall in the site


def extract_ligand_coords(pdb_path: str, resnames: List[str] = DEFAULT_LIGAND_RESNAMES) -> Optional[np.ndarray]:
    """
    Extract heavy atom coordinates for the ligand from a PDB file.
    Returns array of shape (n_atoms, 3) or None if not found.
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


def _protein_residue_atom_coords(pdb_path: str) -> Dict[str, List[np.ndarray]]:
    """residue_id (str) -> list of that residue's heavy-atom coordinates (ATOM records only,
    i.e. protein -- ligand/water/ions are HETATM and excluded)."""
    residues: Dict[str, List[np.ndarray]] = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue

            atom_name = line[12:16].strip()
            if atom_name.startswith('H') or atom_name.startswith('D'):
                continue

            resid = line[22:26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            residues.setdefault(resid, []).append(np.array([x, y, z]))

    return residues


def orthosteric_site_residues(holo_base: str = DEFAULT_HOLO_BASE,
                              ligand_resnames: List[str] = DEFAULT_LIGAND_RESNAMES,
                              distance_threshold: float = RESIDUE_DISTANCE_THRESHOLD) -> Dict[str, Set[str]]:
    """
    Per bare PDB ID (no apo/holo prefix, e.g. '8ITF'): the set of residue IDs within
    distance_threshold Angstrom of the ligand centroid, computed from that PDB's holo
    structure (the lowest-numbered replicate under holo_base/holo<PDBID>/<rep>/structure.pdb --
    the ligand pose at frame 0 is effectively the same across replicates of the same
    crystal structure).
    """
    holo_path = Path(holo_base)
    site_residues: Dict[str, Set[str]] = {}

    if not holo_path.exists():
        print(f"WARNING: Holo base directory not found: {holo_base}")
        return site_residues

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
        centroid = ligand_coords.mean(axis=0)

        near: Set[str] = set()
        for resid, atom_coords in _protein_residue_atom_coords(str(pdb_file)).items():
            distances = np.linalg.norm(np.array(atom_coords) - centroid, axis=1)
            if distances.min() <= distance_threshold:
                near.add(resid)

        bare_id = re.sub(r'^(apo|holo)', '', pdb_dir.name)
        site_residues[bare_id] = near

    return site_residues


def normalize_pocket_number(series: pd.Series) -> pd.Series:
    """Pocket numbers come out of the mdpocket filename parser as strings that may be
    zero-padded ('02') or embedded in an ID suffix ('_p24'); strip to the bare int so
    pocket tables merge reliably on (prj, rep, pocket_number)."""
    return series.astype(str).str.extract(r'(\d+)')[0].astype(int)


def classify_binding_site(pockets_df: pd.DataFrame, holo_base: str = DEFAULT_HOLO_BASE,
                          distance_threshold: float = RESIDUE_DISTANCE_THRESHOLD,
                          min_overlap: float = MIN_RESIDUE_OVERLAP,
                          saving_loc: Optional[str] = None,
                          filename: str = 'pocket_labels_ortho_def.csv') -> pd.DataFrame:
    """
    Classify every pocket in pockets_df as binding-site (orthosteric) or not.

    Parameters
    ----------
    pockets_df : pd.DataFrame
        One row per residue per pocket, as produced by
        MetaAnalysis.pock_file_parser()['res_to_pock_df']. Needs 'prj', 'rep',
        'pocket_number', 'residue_id' and, if present, 'residue_name' columns.
    holo_base : str
        Path to the raw holo input tree (PDB+rep layout), used to derive each PDB's
        orthosteric site (see orthosteric_site_residues).
    distance_threshold : float
        Residue-to-ligand-centroid cutoff (Angstrom) that defines a PDB's site.
    min_overlap : float
        Minimum fraction of a pocket's own residues that must fall in its PDB's
        site for the pocket to be classified as binding-site.
    saving_loc : str, optional
        If given, also writes the result to {saving_loc}/{filename} (Stage 3's
        fallback source for is_binding_site when it's missing from
        pocket_comparison_table.csv).

    Returns
    -------
    pd.DataFrame
        One row per (pocket_number, prj, rep): 'pocket_number', 'prj', 'rep',
        'isovalue', 'total_residues', 'residues', 'overlap_ratio', 'is_binding_site'.
    """
    site_residues = orthosteric_site_residues(holo_base, distance_threshold=distance_threshold)

    rows = []
    for (pock_no, prj, rep), df_pocket in pockets_df.groupby(['pocket_number', 'prj', 'rep']):
        df_pocket = df_pocket.reset_index(drop=True)
        res_ids = df_pocket['residue_id'].astype(str).tolist()
        if 'residue_name' in df_pocket.columns:
            unique_pairs = (df_pocket[['residue_name', 'residue_id']]
                            .drop_duplicates().sort_values('residue_id'))
            aa_list = [f"{row.residue_name}{int(row.residue_id)}" for _, row in unique_pairs.iterrows()]
        else:
            aa_list = sorted(set(res_ids), key=lambda x: int(x))

        bare_id = re.sub(r'^(apo|holo)', '', str(prj))
        site = site_residues.get(bare_id, set())

        pocket_residues = set(res_ids)
        overlap_ratio = len(pocket_residues & site) / len(pocket_residues) if pocket_residues else 0.0
        is_binding = overlap_ratio >= min_overlap

        isovalue = df_pocket['isovalue'].iloc[0] if 'isovalue' in df_pocket.columns else None
        rows.append({'pocket_number': pock_no, 'prj': prj, 'rep': rep, 'isovalue': isovalue,
                     'total_residues': len(pocket_residues), 'residues': aa_list,
                     'overlap_ratio': round(overlap_ratio, 2), 'is_binding_site': is_binding})

    result_df = pd.DataFrame(rows)

    if saving_loc is not None:
        result_df.to_csv(os.path.join(saving_loc, filename), index=False)

    return result_df
