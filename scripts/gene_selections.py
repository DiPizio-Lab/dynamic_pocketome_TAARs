""""this is a script to resolve which TAAR gene a project belongs to and load the per-gene residue
selections (binding site, ICL3) needed for the RMSD group selections in Step 1 of the pipeline.
The BW-equivalent residue numbers were derived once from the hTAAR1 binding site residues using the
BW numbering in reference_data/TAARs_numbered/ and are stored (not recomputed) in
reference_data/binding_site_residues.txt and reference_data/ICL3_definition.txt.
Note: mTAAR9 has only 26 binding site residues -- its TM5 is missing the residue at BW 5.46
(a one-residue deletion relative to the other three genes), so no equivalent resid exists for it.
ICL3_definition.txt lists the ICL3 residues plus the 2 immediately neighbouring residues (one on
each flanking side, i.e., the last TM5 and first TM6 residue) so the "CA without ICL3" RMSD
selection excludes the flexible loop junction along with the loop itself."""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`
import config as conf

GENE_DICT_PATH = os.path.join(conf.REFERENCE_DATA_DIR, 'hard_coded_gene_dict.txt')
BINDING_SITE_PATH = os.path.join(conf.REFERENCE_DATA_DIR, 'binding_site_residues.txt')
ICL3_PATH = os.path.join(conf.REFERENCE_DATA_DIR, 'ICL3_definition.txt')


def _load_resid_dict(path):
    """parses lines of the form GENE: "resid resid resid ..." into {gene: [resid, ...]}"""
    resid_dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            gene, resids = line.split(':', 1)
            resid_dict[gene.strip()] = [int(r) for r in resids.strip().strip('"').split()]
    return resid_dict


def gene_dict():
    """{PDB ID: gene}, e.g. {'8ITF': 'mTAAR9', ...} -- the single source of truth for PDB ID ->
    gene, read fresh from reference_data/hard_coded_gene_dict.txt every call (cheap, tiny file).
    Also used directly by pipeline/pocket_dataframes.py and taar_paper_figures/taar_style.py, so
    this file is the only place that mapping is allowed to live -- don't hardcode a copy elsewhere."""
    with open(GENE_DICT_PATH) as f:
        text = f.read()
    return ast.literal_eval(text.split('=', 1)[1].strip())


def gene_for_project(folder_name):
    """'apo8ITF' / 'holo8ITF' -> 'mTAAR9', using the hard coded PDBID -> gene dict"""
    pdb_id = re.sub(r'^(apo|holo)', '', folder_name)
    return gene_dict()[pdb_id]


# read once at import time
BINDING_SITE_RESIDS = _load_resid_dict(BINDING_SITE_PATH)
ICL3_RESIDS = _load_resid_dict(ICL3_PATH)


def group_selection_for(gene):
    """MDAnalysis group_selection dict (as used by BasicAnalysis.calc_rmsd) for the given gene:
    binding site CA, binding site side chains, and CA without ICL3"""
    binding_site_resids = ' '.join(str(r) for r in BINDING_SITE_RESIDS[gene])
    icl3_resids = ' '.join(str(r) for r in ICL3_RESIDS[gene])
    return {'binding site': [f'protein and name CA and (resid {binding_site_resids})'],
        'side chains of binding site': [f'protein and resid {binding_site_resids} and not name H* CA N O C'],
        'CA without ICL3': [f'protein and name CA and not (resid {icl3_resids})']}
