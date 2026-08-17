"""
Orthosteric binding site filter for pocket analysis.
Adds a boolean column 'is_orthosteric' to pocket data based on proximity to ligand coordinates.
Since we have holo structures, the coordinates of the ligand in holo is used to filter for the pocket(s) that are within
a distance parameter range (5Angström) of the centroid of the ligand. This centroid is also used for the apo states.

"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
from config import HOLO_BASE_DIR

DEFAULT_HOLO_BASE = HOLO_BASE_DIR
DEFAULT_DISTANCE_THRESHOLD = 5.0  # Å - pocket centroid must be within this distance of ligand centroid
DEFAULT_LIGAND_RESNAMES = ['LIG']


def extract_ligand_coords(pdb_path: str, resnames: List[str] = ['LIG']) -> Optional[np.ndarray]:
    """
    Extract heavy atom coordinates for ligand from PDB file.
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

            # Skip hydrogens
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


def load_all_ligands(holo_base: str, resnames: List[str] = ['LIG']) -> Dict[str, np.ndarray]:
    """
    Load ligand coordinates for all available PDB IDs.
    Returns dict mapping PDB ID -> ligand centroid (x, y, z).
    """
    holo_path = Path(holo_base)
    ligands = {}

    if not holo_path.exists():
        print(f"WARNING: Holo base directory not found: {holo_base}")
        return ligands

    for pdb_dir in holo_path.iterdir():
        if not pdb_dir.is_dir():
            continue

        pdb_file = pdb_dir / "NEUTRAL_fis.pdb"
        if not pdb_file.exists():
            continue

        coords = extract_ligand_coords(str(pdb_file), resnames)
        if coords is not None and len(coords) > 0:
            ligands[pdb_dir.name] = coords.mean(axis=0)  # Store centroid

    return ligands


def parse_pocket_id(pocket_id: str) -> Optional[Dict]:
    """
    Parse pocket ID like 'apo8ITF_1_p24_i3' into components.

    Returns dict with: state, pdb_id, replicate, pocket_num
    """
    pattern = r'^(apo|holo)([A-Z0-9]+)_(\d+)_p(\d+)_i\d+$'
    match = re.match(pattern, pocket_id)

    if not match:
        return None

    return {'state': match.group(1),
        'pdb_id': match.group(2),
        'replicate': int(match.group(3)),
        'pocket_num': int(match.group(4))}


def add_orthosteric_column(
        df: pd.DataFrame,
        holo_base: str = DEFAULT_HOLO_BASE,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        ligand_resnames: List[str] = DEFAULT_LIGAND_RESNAMES,
        id_column: str = 'ID',
        coord_columns: List[str] = ['x', 'y', 'z'],
        verbose: bool = True) -> pd.DataFrame:
    """
    Add 'is_orthosteric' boolean column to pocket dataframe.
    A pocket is marked as orthosteric if its centroid is within
    distance_threshold of the ligand centroid.

    Parameters
    ----------
    df : pd.DataFrame
        Pocket data with ID column and coordinate columns
    holo_base : str
        Path to directory containing {PDBID}/NEUTRAL_fis.pdb files
    distance_threshold : float
        Maximum distance (Å) from pocket centroid to ligand centroid
    ligand_resnames : list
        Residue names to identify ligand in PDB files
    id_column : str
        Column name containing pocket IDs (e.g., 'apo8ITF_1_p24_i3')
    coord_columns : list
        Column names for x, y, z coordinates
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'is_orthosteric' column
    """
    df = df.copy()

    if verbose:
        print("Loading ligand coordinates...")
    ligands = load_all_ligands(holo_base, ligand_resnames)

    if verbose:
        print(f"  Found ligands for {len(ligands)} PDB IDs: {sorted(ligands.keys())}")

    if not ligands:
        print("WARNING: No ligands found. Adding is_orthosteric=False for all rows.")
        df['is_orthosteric'] = False
        return df

    # Calculate pocket centroids per unique ID
    if verbose:
        print("Calculating pocket centroids...")

    pocket_centroids = df.groupby(id_column)[coord_columns].mean()

    # Determine orthosteric status for each pocket ID
    if verbose:
        print("Classifying pockets...")

    orthosteric_ids = set()
    stats = {'orthosteric': 0, 'non_orthosteric': 0, 'no_ligand': 0, 'parse_error': 0}

    for pocket_id in pocket_centroids.index:
        parsed = parse_pocket_id(pocket_id)

        if parsed is None:
            stats['parse_error'] += 1
            continue

        pdb_id = parsed['pdb_id']

        if pdb_id not in ligands:
            stats['no_ligand'] += 1
            continue

        # Get pocket centroid and ligand centroid
        pocket_centroid = pocket_centroids.loc[pocket_id, coord_columns].values
        ligand_centroid = ligands[pdb_id]

        # Calculate distance
        distance = np.linalg.norm(pocket_centroid - ligand_centroid)

        if distance <= distance_threshold:
            orthosteric_ids.add(pocket_id)
            stats['orthosteric'] += 1
        else:
            stats['non_orthosteric'] += 1

    # Add column to dataframe
    df['is_orthosteric'] = df[id_column].isin(orthosteric_ids)

    if verbose:
        print(f"\nResults:")
        print(f"  Orthosteric pockets:     {stats['orthosteric']}")
        print(f"  Non-orthosteric pockets: {stats['non_orthosteric']}")
        print(f"  No ligand available:     {stats['no_ligand']}")
        print(f"  Parse errors:            {stats['parse_error']}")
        print(f"  Total rows marked True:  {df['is_orthosteric'].sum()}")

    return df


def filter_orthosteric(
        df: pd.DataFrame,
        holo_base: str = DEFAULT_HOLO_BASE,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        **kwargs) -> pd.DataFrame:
    """
    Convenience function: add column and filter to orthosteric pockets only.
    Returns only rows where is_orthosteric=True.
    """
    df = add_orthosteric_column(df, holo_base, distance_threshold, **kwargs)
    return df[df['is_orthosteric']].copy()

