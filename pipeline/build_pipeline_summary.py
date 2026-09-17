"""Reconstructs logs/pipeline_summary.csv (see scripts/run_summary.py) from the result files
already on disk, for a run that finished (or was interrupted) before that checkpoint CSV
existed, or to reconcile the CSV against ground truth at any later point.

Per-experiment stages (preprocessing, rmsd_calculation, pocket_detection, pocket_separation,
pocket_characterisation) are inferred per (state, pdb_id, replicate) under
input/{apo,holo}_structures/ from the presence of the files Step 1 (pipeline/analysis_pipeline.py
-> scripts/preprocessing.py, basic_analysis.py, pocket_analysis.py) is known to write. The three
dataset-wide stages (meta_analysis, global_id, optional_analyses) are inferred from Step 2/3's
output tables under output/meta_analysis/, recorded on the shared run_summary.AGGREGATE_KEY row.
These are only ever OK when the table's own 'prj'/'rep' columns cover every replicate
get_project_list() finds under input/{apo,holo}_structures/.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as conf
from pipeline.analysis_pipeline import get_project_list
from scripts import run_summary


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _stamp(mtime):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))


def _latest(*paths):
    """Newest mtime among paths that exist, or None if none exist."""
    mtimes = [m for m in (_mtime(p) for p in paths) if m is not None]
    return max(mtimes) if mtimes else None


def _newest(*mtimes):
    """Newest of several already-resolved mtimes (floats or None), or None if all are None."""
    present = [m for m in mtimes if m is not None]
    return max(present) if present else None


def _check_preprocessing(rep_dir):
    dry_pdb = os.path.join(rep_dir, 'dry_prot.pdb')
    aligned_top = os.path.join(rep_dir, 'aligned_top.pdb')
    aligned_xtc = os.path.join(rep_dir, 'aligned_traj.xtc')
    aligned_dcd = os.path.join(rep_dir, 'aligned_traj.dcd')
    aligned_traj = aligned_xtc if os.path.exists(aligned_xtc) else aligned_dcd
    have = [p for p in (dry_pdb, aligned_top, aligned_traj) if os.path.exists(p)]
    if not have:
        return 'MISSING', 'no preprocessing output found', None
    expected = {dry_pdb, aligned_top, aligned_traj}
    if len(have) < len(expected):
        missing = expected - set(have)
        return 'ERROR', f'incomplete, missing {sorted(os.path.basename(p) for p in missing)}', _latest(*have)
    return 'OK', '', _latest(*have)


def _check_rmsd(rep_dir):
    if not os.path.isdir(rep_dir):
        return 'MISSING', 'replicate directory not found', None
    rmsd_csvs = [f for f in os.listdir(rep_dir) if f.startswith('RMSD_') and f.endswith('.csv')]
    if not rmsd_csvs:
        return 'MISSING', 'no RMSD_*.csv found', None
    paths = [os.path.join(rep_dir, f) for f in rmsd_csvs]
    return 'OK', f'{len(rmsd_csvs)} RMSD table(s)', _latest(*paths)


def _iso_stem_candidates(isovalue):
    """Both filename stems mdpocket/ATClus output for `isovalue` can be found under, in the
    order Step 2.1 (pipeline/pocket_dataframes.py's _normalize_pocket_filenames) would produce
    them: the raw bare-float stem ATClus/mdpocket actually write ('mdpout_dens_iso_3.0'), and
    the dot-free stem Step 2.1 renames it to IN PLACE the first time it runs over a replicate
    ('mdpout_dens_iso_3' for an integer isovalue, 'mdpout_dens_iso_3_5' otherwise). A replicate
    Step 2 has already touched is found under the post-rename stem; one it hasn't is found
    under the name Step 1 originally wrote - checking only the raw stem (as this used to)
    reports every already-processed replicate as ERROR once Step 2 has renamed its files."""
    raw = f'mdpout_dens_iso_{isovalue}'
    iso_str = str(isovalue)
    normalized = f'mdpout_dens_iso_{int(isovalue)}' if isovalue.is_integer() \
        else f'mdpout_dens_iso_{iso_str.replace(".", "_")}'
    return [normalized, raw] if normalized != raw else [raw]


def _pocket_search_status(pockets_dir, isovalues):
    """(pocket_detection status/message/mtime, pocket_separation ..., pocket_characterisation ...)
    - mirrors scripts/pocket_analysis.py's PocketAnalysis._pocket_search_complete /
    _expected_pocket_count logic (ATClus .group file's reported count vs. separated PDBs vs.
    descriptors.txt)."""
    grid_file = os.path.join(pockets_dir, 'mdpout_dens_grid.dx')
    if not os.path.isdir(pockets_dir) or not os.path.exists(grid_file):
        missing = ('MISSING', 'no pockets/ output found', None)
        return missing, missing, missing
    detection = ('OK', '', _mtime(grid_file))

    sep_ok_all, char_ok_all = True, True
    sep_msgs, char_msgs = [], []
    latest_sep, latest_char = _mtime(grid_file), None
    for isovalue in isovalues:
        candidates = _iso_stem_candidates(isovalue)
        stem = next((s for s in candidates if os.path.exists(os.path.join(pockets_dir, f'{s}.group'))),
                    candidates[0])
        group_file = os.path.join(pockets_dir, f'{stem}.group')
        expected = None
        if os.path.exists(group_file):
            with open(group_file) as f:
                lines = [line for line in f if line.strip()]
            if lines:
                parts = lines[-1].split()
                if 'Group:' in parts:
                    expected = int(parts[parts.index('Group:') + 1])
        pocket_pdbs = [f for f in os.listdir(pockets_dir)
                       if f.startswith(f'{stem}-out-') and f.endswith('.pdb')
                       and 'atoms' not in f and not f.endswith('mdpocket.pdb')]
        if expected is None:
            sep_ok_all = False
            sep_msgs.append(f'iso {isovalue}: no .group file (ATClus never ran)')
        elif len(pocket_pdbs) < expected:
            sep_ok_all = False
            sep_msgs.append(f'iso {isovalue}: separated {len(pocket_pdbs)}/{expected} pockets')
        else:
            sep_msgs.append(f'iso {isovalue}: separated {len(pocket_pdbs)}/{expected} pockets')
            latest_sep = _newest(latest_sep, _latest(group_file, *[os.path.join(pockets_dir, p) for p in pocket_pdbs]))

        descriptor_paths = [os.path.join(pockets_dir, f"{p.split('.pdb')[0]}_descriptors.txt") for p in pocket_pdbs]
        have_descriptors = [p for p in descriptor_paths if os.path.exists(p)]
        if expected is None or len(pocket_pdbs) < expected or len(have_descriptors) < len(pocket_pdbs):
            char_ok_all = False
            char_msgs.append(f'iso {isovalue}: {len(have_descriptors)}/{len(pocket_pdbs)} pockets characterised')
        else:
            char_msgs.append(f'iso {isovalue}: {len(have_descriptors)}/{len(pocket_pdbs)} pockets characterised')
            latest_char = _newest(latest_char, _latest(*have_descriptors))

    separation = ('OK' if sep_ok_all else 'ERROR', '; '.join(sep_msgs), latest_sep)
    characterisation = ('OK' if char_ok_all else 'ERROR', '; '.join(char_msgs), latest_char)
    return detection, separation, characterisation


def backfill_per_experiment():
    isovalues = conf.require_float_isovalues(conf.ISOVALUES)
    n = 0
    for proj_rep in get_project_list():
        curr_rep = os.path.basename(proj_rep)
        folder_name = os.path.basename(os.path.dirname(proj_rep))
        state, pdb_id = run_summary.parse_experiment(folder_name)
        rep_dir = os.path.join(conf.results_dir_for(folder_name), folder_name, curr_rep)

        status, message, mtime = _check_preprocessing(rep_dir)
        run_summary.update_checkpoint(state, pdb_id, curr_rep, 'preprocessing', status, message,
                                      timestamp=_stamp(mtime) if mtime else None)

        status, message, mtime = _check_rmsd(rep_dir)
        run_summary.update_checkpoint(state, pdb_id, curr_rep, 'rmsd_calculation', status, message,
                                      timestamp=_stamp(mtime) if mtime else None)

        pockets_dir = os.path.join(rep_dir, conf.POCKETS_DIRNAME)
        (det_status, det_msg, det_mtime), (sep_status, sep_msg, sep_mtime), \
            (char_status, char_msg, char_mtime) = _pocket_search_status(pockets_dir, isovalues)
        run_summary.update_checkpoint(state, pdb_id, curr_rep, 'pocket_detection', det_status, det_msg,
                                      timestamp=_stamp(det_mtime) if det_mtime else None)
        run_summary.update_checkpoint(state, pdb_id, curr_rep, 'pocket_separation', sep_status, sep_msg,
                                      timestamp=_stamp(sep_mtime) if sep_mtime else None)
        run_summary.update_checkpoint(state, pdb_id, curr_rep, 'pocket_characterisation', char_status, char_msg,
                                      timestamp=_stamp(char_mtime) if char_mtime else None)
        n += 1
    return n


def _expected_prj_reps():
    """{(folder_name, replicate), ...} for every replicate under both input structure trees,
    e.g. {('apo8ITF', '1'), ...} - matches the 'prj'/'rep' columns Step 2 writes into
    output/meta_analysis/**/*.csv, so a checkpoint can check "does this table actually cover
    everything" instead of just "does this table exist". A partial Step 2 re-run (e.g. one
    interrupted after only a couple of replicates) still produces a real all_pockets.csv /
    pocket_comparison_table.csv - file existence alone can't tell that apart from a complete
    run."""
    return {(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in get_project_list()}


def _prj_reps_in_csv(path):
    """Distinct (prj, rep) pairs actually present in a Step 2 output CSV, or None if the file
    doesn't exist. These tables are small (one row per pocket/replicate, not per frame), so
    reading them in full here is cheap."""
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        return {(row['prj'], row['rep']) for row in csv.DictReader(f)}


def backfill_aggregate():
    key = run_summary.AGGREGATE_KEY
    expected_all = _expected_prj_reps()
    expected_by_state = {
        'apo': {pr for pr in expected_all if pr[0].startswith('apo')},
        'holo': {pr for pr in expected_all if pr[0].startswith('holo')},
        'both': expected_all,
    }

    all_pockets = os.path.join(conf.META_ANALYSIS_DIR, 'all_pockets.csv')
    pocket_summary = os.path.join(conf.META_ANALYSIS_DIR, 'pocket_summary.csv')
    mtime = _latest(all_pockets, pocket_summary)
    summary_prj_reps = _prj_reps_in_csv(pocket_summary)
    if summary_prj_reps is None or not os.path.exists(all_pockets):
        run_summary.update_checkpoint(*key, 'meta_analysis', 'MISSING',
                                      'all_pockets.csv / pocket_summary.csv not found under '
                                      f'{conf.META_ANALYSIS_DIR}')
    else:
        missing = expected_all - summary_prj_reps
        if missing:
            run_summary.update_checkpoint(*key, 'meta_analysis', 'ERROR',
                                          f'pocket_summary.csv covers {len(summary_prj_reps)}/{len(expected_all)} '
                                          f'replicates -- missing e.g. {sorted(missing)[:5]}',
                                          timestamp=_stamp(mtime))
        else:
            run_summary.update_checkpoint(*key, 'meta_analysis', 'OK',
                                          f'all_pockets.csv + pocket_summary.csv found, covering all '
                                          f'{len(expected_all)} replicates', timestamp=_stamp(mtime))

    gid_tables = {
        'apo': os.path.join(conf.META_ANALYSIS_ROOT, 'global_ID_apo', 'pocket_comparison_table.csv'),
        'holo': os.path.join(conf.META_ANALYSIS_ROOT, 'global_ID_holo', 'pocket_comparison_table.csv'),
        'both': os.path.join(conf.META_ANALYSIS_ROOT, 'global_ID_combined', 'pocket_comparison_table.csv'),
    }
    complete, incomplete, missing_files = {}, {}, []
    for name, path in gid_tables.items():
        prj_reps = _prj_reps_in_csv(path)
        if prj_reps is None:
            missing_files.append(name)
            continue
        gap = expected_by_state[name] - prj_reps
        if gap:
            incomplete[name] = (len(prj_reps), len(expected_by_state[name]))
        else:
            complete[name] = path

    if complete and not incomplete and not missing_files:
        run_summary.update_checkpoint(*key, 'global_id', 'OK', f'runs found: {", ".join(sorted(complete))}',
                                      timestamp=_stamp(_latest(*complete.values())))
    elif complete or incomplete:
        parts = [f'{name}: complete' for name in sorted(complete)]
        parts += [f'{name}: incomplete ({have}/{total} replicates)' for name, (have, total) in sorted(incomplete.items())]
        parts += [f'{name}: missing' for name in sorted(missing_files)]
        present_paths = list(complete.values()) + [gid_tables[name] for name in incomplete]
        run_summary.update_checkpoint(*key, 'global_id', 'ERROR', '; '.join(parts),
                                      timestamp=_stamp(_latest(*present_paths)) if present_paths else None)
    else:
        run_summary.update_checkpoint(*key, 'global_id', 'MISSING',
                                      f'no pocket_comparison_table.csv found under {conf.META_ANALYSIS_ROOT}')
    global_id_complete = bool(complete) and not incomplete and not missing_files

    figures_dir = os.path.join(conf.META_ANALYSIS_ROOT, 'paper_figures')
    figure_names = ['figure1_RMSD_combined_replicates.png', 'figure2_binding_site.png',
                    'figure3_allosteric_pocketome.png']
    figure_paths = [os.path.join(figures_dir, f) for f in figure_names]
    found_figs = [p for p in figure_paths if os.path.exists(p)]
    if not found_figs:
        run_summary.update_checkpoint(*key, 'optional_analyses', 'MISSING',
                                      f'no paper figures found in {figures_dir}')
    elif not global_id_complete:
        run_summary.update_checkpoint(*key, 'optional_analyses', 'ERROR',
                                      f'{len(found_figs)}/{len(figure_paths)} paper figures found in {figures_dir}, '
                                      'but Global ID data is incomplete (see global_id) -- figures reflect only a '
                                      'partial run', timestamp=_stamp(_latest(*found_figs)))
    else:
        status = 'OK' if len(found_figs) == len(figure_paths) else 'ERROR'
        run_summary.update_checkpoint(*key, 'optional_analyses', status,
                                      f'{len(found_figs)}/{len(figure_paths)} paper figures found in {figures_dir}',
                                      timestamp=_stamp(_latest(*found_figs)))


def main():
    n = backfill_per_experiment()
    backfill_aggregate()
    print(f'Backfilled {n} experiment(s) + the dataset-wide row into {run_summary.SUMMARY_CSV}')


if __name__ == '__main__':
    main()
