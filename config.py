"""Single place all scripts resolve project paths from.

PROJECT_ROOT defaults to the current working directory, so run scripts from the
repo root (or `export TAAR_ROOT=/path/to/taar-pocketome` first if that's not
convenient, e.g. when submitting jobs from a different working directory).
"""
import os

PROJECT_ROOT = os.environ.get("TAAR_ROOT", os.getcwd())
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Conventional subpaths used across the pipeline. All of these live under
# input/ or output/ and are created by the user (data is published separately,
# not shipped in this repo) or by the pipeline scripts themselves.
APO_STRUCTURES_DIR = os.path.join(INPUT_DIR, "apo_structures")
HOLO_STRUCTURES_DIR = os.path.join(INPUT_DIR, "holo_structures")
HOLO_BASE_DIR = os.path.join(HOLO_STRUCTURES_DIR, "input")
META_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "meta_analysis", "across_genes")
