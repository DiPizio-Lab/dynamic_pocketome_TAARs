"""Per-experiment / per-stage checkpoint CSV, written to as the pipeline runs.

Every stage of the pipeline writes one cell of logs/pipeline_summary.csv as it starts (RUNNING) and again as it ends
(OK, or ERROR: <message>). A cell stuck on RUNNING after a job has ended means that process
was killed (OOM, walltime, node failure, ...) mid-stage.

Row key is (state, pdb_id, replicate) for the five per-experiment stages (preprocessing
through pocket_characterisation, all produced by Step 1 - pipeline/analysis_pipeline.py).
meta_analysis / global_id / optional_analyses are dataset-wide, not per-experiment (Step 2/3
run once over every replicate together), so they're recorded on one shared row keyed
AGGREGATE_KEY = ('ALL', 'ALL', 'ALL') instead of being repeated identically on every
experiment's row.
"""
import contextlib
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as conf

SUMMARY_CSV = os.path.join(conf.PROJECT_ROOT, 'logs', 'pipeline_summary.csv')

# Order here is the column order in the CSV and the pipeline order.
STAGES = [
    'preprocessing',
    'rmsd_calculation',
    'pocket_detection',
    'pocket_separation',
    'pocket_characterisation',
    'meta_analysis',
    'global_id',
    'optional_analyses',
]
KEY_FIELDS = ['state', 'pdb_id', 'replicate']
FIELDNAMES = KEY_FIELDS + STAGES + ['last_updated']

# Key used for the three dataset-wide stages (meta_analysis, global_id, optional_analyses),
# which run once over every experiment rather than per (state, pdb_id, replicate).
AGGREGATE_KEY = ('ALL', 'ALL', 'ALL')

_LOCK_TIMEOUT_S = 120       # a held lock older than this is assumed to be left over from a
                            # crashed process, not a slow writer - every write here is a small
                            # in-memory CSV rewrite, never anything that legitimately takes
                            # anywhere near this long.
_LOCK_POLL_S = 0.2


def parse_experiment(curr_proj):
    """('apo8ITF' | 'holo8ITF', ...) --> ('apo', '8ITF') | ('holo', '8ITF') - the inverse of
    the state+pdb_id concatenation used for output folder names throughout the pipeline
    (see config.results_dir_for)."""
    if curr_proj.startswith('apo'):
        return 'apo', curr_proj[len('apo'):]
    if curr_proj.startswith('holo'):
        return 'holo', curr_proj[len('holo'):]
    raise ValueError(f"Project folder name {curr_proj!r} doesn't start with 'apo' or 'holo'")


@contextlib.contextmanager
def _locked(path):
    lock_path = path + '.lock'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    waited = 0.0
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                stale = (time.time() - os.path.getmtime(lock_path)) > _LOCK_TIMEOUT_S
            except OSError:
                stale = False  # lock file vanished between the failed open and the stat
            if stale:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            time.sleep(_LOCK_POLL_S)
            waited += _LOCK_POLL_S
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _read_rows(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        rows = {(r['state'], r['pdb_id'], r['replicate']): r for r in csv.DictReader(f)}
    return rows


def _write_rows(path, rows):
    # Sorted for a stable, diffable file: aggregate row first, then state/pdb_id/replicate.
    def sort_key(k):
        return (0, '', '', '') if k == AGGREGATE_KEY else (1,) + k

    ordered = sorted(rows.values(), key=lambda r: sort_key((r['state'], r['pdb_id'], r['replicate'])))
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, '') for field in FIELDNAMES})
    os.replace(tmp_path, path)


def update_checkpoint(state, pdb_id, replicate, stage, status, message='', timestamp=None):
    """Upserts one (state, pdb_id, replicate) row's `stage` cell in logs/pipeline_summary.csv.
    status is typically 'RUNNING', 'OK', 'SKIPPED' or 'ERROR' - combined with `message` as
    'STATUS: message' when a message is given (e.g. an exception summary). Safe to call
    concurrently from several processes.
    timestamp overrides 'last_updated' (default: now) - used by build_pipeline_summary.py to
    record a result file's own mtime instead of the backfill's run time, so a stale stage is
    visibly stale rather than looking freshly checked."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}, must be one of {STAGES}")
    key = (state, pdb_id, replicate)
    cell = status if not message else f'{status}: {message}'
    with _locked(SUMMARY_CSV):
        rows = _read_rows(SUMMARY_CSV)
        row = rows.get(key, {field: '' for field in FIELDNAMES})
        row['state'], row['pdb_id'], row['replicate'] = key
        row[stage] = cell
        row['last_updated'] = timestamp or time.strftime('%Y-%m-%d %H:%M:%S')
        rows[key] = row
        _write_rows(SUMMARY_CSV, rows)


@contextlib.contextmanager
def stage(state, pdb_id, replicate, stage_name):
    """Marks `stage_name` RUNNING on entry and OK/ERROR on exit, e.g.:
        with run_summary.stage(state, pdb_id, rep, 'preprocessing'):
            preprocess_obj.make_selection_n_save(...)
    A cell left on RUNNING (never updated to OK/ERROR) means the process was killed mid-stage.
    Re-raises whatever the wrapped block raised - callers that want to continue with the next
    experiment after a failure should catch around the `with` block, not inside it, so the
    ERROR is recorded before control returns to the caller."""
    update_checkpoint(state, pdb_id, replicate, stage_name, 'RUNNING')
    try:
        yield
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        update_checkpoint(state, pdb_id, replicate, stage_name, 'ERROR', message)
        raise
    else:
        update_checkpoint(state, pdb_id, replicate, stage_name, 'OK')


def format_exception_message(exc):
    """One-line 'ExcType: message' summary for logging an already-caught exception, matching
    the message stage() records for one it re-raises."""
    return f'{type(exc).__name__}: {exc}'
