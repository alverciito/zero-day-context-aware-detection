#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
warmup_audit.py -- T0 label-only audit of the JISAS-D-26-02300 revision.

Scope: ONLY labels, timestamps, exact row hashes and counts.
It never reads model scores and never computes performance metrics.

For every dataset it rebuilds
  * the ORIGINAL stream (experiments/frames/*): file order of the original scripts,
    per-file data_loader, grant_first_label (300k label-selected copies), KNAD read with
    header=0 (1-row misalignment). The pure numpy/pandas functions data_loader,
    clean_csv_headers, grant_first_label and regularize_data are IMPORTED from
    experiments/frames/launch_all.py through the AST (TensorFlow is never imported);
  * the CORRECTED stream, in every variant that was considered;
and reports per region (training, original post-processing reference, warm-up 5/10/20 %
of the execution segment, evaluation remainder): size, % malicious, first malicious
index, attack types, L=800 context windows, copies/originals, exact duplicates with
training, timestamp monotonicity and file/day block positions. It writes
stream_cuts.json with the corrected cuts.

Writes only to --out-dir and --cache-dir; refuses the read-only data directories.
Idempotent: every expensive stage is cached (keyed by source size/mtime, sample size and
cache version); per-dataset DONE markers; atomic writes; stale-lock recovery.

Examples
  python warmup_audit.py --dataset all --pipeline both --out-dir OUT --sample-rows 50000
  python warmup_audit.py --dataset all --pipeline both --out-dir OUT --workers 3 --log LOG
"""
import argparse
import ast
import hashlib
import json
import logging
import os
import struct
import sys
import time
import traceback
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/acid137/Projects/review-zd')
DS = REPO / 'data' / 'datasets'
FRAMES = REPO / 'experiments' / 'frames'
LAUNCH_ALL = FRAMES / 'launch_all.py'
READ_ONLY = [DS, REPO / 'data' / 'original_results', REPO / 'data' / 'remote_chiquitin', Path('/mnt/hdd8tb')]
DEFAULT_CACHE = REPO / 'data' / 'cache' / 't0'
REMOTE_TARGET = REPO / 'data' / 'remote_chiquitin' / 'detectors_2025' / 'data'
ORIG_ACI_TARGET = REPO / 'data' / 'original_results' / 'ACI-IOT-2023' / 'target.npy'

CACHE_VERSION = 1
L_SEQ = 800
WARM_PCTS = (5, 10, 20)
TS_NA = np.iinfo(np.int64).min

KNAD_ATTACKS = ['Active Wiretap', 'Fuzzing', 'OS Scan', 'SSDP Flood', 'SSL Renegotiation', 'Video Injection']
CIC2017_CHRONO = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
]
CIC2018_ALL = ['02-14-2018.csv', '02-15-2018.csv', '02-16-2018.csv', '02-20-2018.csv', '02-21-2018.csv',
               '02-22-2018.csv', '02-23-2018.csv', '02-28-2018.csv', '03-01-2018.csv', '03-02-2018.csv']
CIC2018_EXCLUDED_ORIG = ['02-20-2018.csv', '02-28-2018.csv', '03-01-2018.csv']
CIC2018_DAYS7 = [f for f in CIC2018_ALL if f not in CIC2018_EXCLUDED_ORIG]
ACI_FILE = 'ACI-IoT-2023.csv'
ACI_RANGE = ('2023-10-30', '2023-11-04')
ACI_DAYS = ['2023-10-30', '2023-10-31', '2023-11-01', '2023-11-02', '2023-11-03']

DROP_CORR = {
    'cic2017': [],
    'cic2018': ['Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP'],
    'aci': ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'],
    'knad': [],
}
_PREPROC_COMMON = ('todo objeto ajustado (sustitución de inf/NaN, estadísticos de normalización, Feature Map, '
                   'vocabularios categóricos, pesos) se ajusta SOLO con las filas de entrenamiento del stream del '
                   'modelo [0, train_end); nada se ajusta con la ejecución (sin fillna/inf->máx. del fichero ni '
                   'z-score global como en launch_all.py:98-120,150-153). ')
PREPROC_RULE = {
    'cic2017': _PREPROC_COMMON + 'Features no finitas: sustituir con valores derivados del entrenamiento (C5).',
    'cic2018': _PREPROC_COMMON + 'Features no finitas y cabeceras repetidas: exclusión o sustitución con valores del entrenamiento (C5).',
    'aci': _PREPROC_COMMON + ('Connection Type: vocabulario del entrenamiento, categoría no vista -> código reservado; sin LabelEncoder '
                              'de identificadores ni desbordamiento uint8 (identificadores eliminados).'),
    'knad': _PREPROC_COMMON + 'Carga con header=None; sin transformación en carga.',
}
LABEL_COL = {'cic2017': ' Label', 'cic2018': 'Label', 'aci': 'Label'}
BENIGN = {'cic2017': 'BENIGN', 'cic2018': 'Benign', 'aci': 'Benign', 'knad': 'Benign'}
ORIG_EXPECTED_N = {'aci': 1_531_411, 'cic2017': 3_130_743, 'cic2018': 7_640_018,
                   'knad_active_wiretap': 2_278_688, 'knad_fuzzing': 2_244_138, 'knad_os_scan': 1_697_850,
                   'knad_ssdp_flood': 4_077_265, 'knad_ssl_renegotiation': 2_207_570,
                   'knad_video_injection': 2_472_400}
NEG0 = b'-0.000000000000000000e+00'
POS0 = b'0.000000000000000000e+00'

log = logging.getLogger('warmup_audit')


class Consts:
    def __init__(self, div=1):
        self.div = int(div)
        self.FM = 50_000 // self.div
        self.AE = 150_000 // self.div
        self.TRAIN = self.FM + self.AE
        self.COPIES = 300_000 // self.div
        self.REF = (100_000 // self.div, 200_000 // self.div)
        self.CIC2018_SKIP_LINE = 1_000_000

    def as_dict(self):
        return {'scale_div': self.div, 'FM': self.FM, 'AE': self.AE, 'TRAIN': self.TRAIN, 'COPIES': self.COPIES,
                'REF_PP': list(self.REF), 'L': L_SEQ, 'WARM_PCTS': list(WARM_PCTS),
                'CIC2018_SKIP_LINE': self.CIC2018_SKIP_LINE}


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not serialisable: {type(o)}')


def _norm_json(obj):
    return json.loads(json.dumps(obj, default=jsonable))


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def dump_json(path, obj):
    write_atomic(path, json.dumps(obj, indent=1, ensure_ascii=False, default=jsonable))


def assert_writable(p):
    p = Path(p).resolve()
    for ro in READ_ONLY:
        try:
            p.relative_to(ro.resolve())
        except ValueError:
            continue
        raise SystemExit(f'Refusing to write under read-only path {ro}: {p}')


def file_sig(p):
    st = Path(p).stat()
    return {'path': str(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def script_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class FileLock:
    """O_EXCL lock file with the owner's pid; a lock whose pid is dead is taken over."""

    def __init__(self, target):
        self.path = Path(str(target) + '.lock')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().strip() or 0)
                    age = time.time() - self.path.stat().st_mtime
                except (OSError, ValueError):
                    pid, age = 0, 0.0
                if (pid and _pid_alive(pid)) or (pid == 0 and age < 30):
                    time.sleep(2)
                    continue
                log.warning('stale lock %s (pid %s): taking over', self.path, pid)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def cached_npz(path, deps, compute):
    """Load arrays+info from cache if deps match, else compute under a lock and store atomically."""
    path = Path(path)
    meta = path.with_suffix('.json')
    deps = _norm_json(deps)

    def _load():
        if path.exists() and meta.exists():
            try:
                m = json.loads(meta.read_text())
            except json.JSONDecodeError:
                return None
            if m.get('deps') == deps:
                with np.load(path, allow_pickle=False) as z:
                    return {k: z[k] for k in z.files}, m.get('info', {})
        return None

    r = _load()
    if r is not None:
        return r
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path):
        r = _load()
        if r is not None:
            return r
        t0 = time.time()
        arrays, info = compute()
        info = dict(info)
        info['compute_s'] = round(time.time() - t0, 1)
        tmp = path.with_name(f'.{path.stem}.tmp{os.getpid()}.npz')
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        dump_json(meta, {'deps': deps, 'info': info, 'created': now_iso()})
        log.info('cache written %s (%.1fs)', path, info['compute_s'])
        return arrays, _norm_json(info)


def iso_sec(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return None
        return str(np.datetime64(int(round(float(v) * 1e6)), 'us'))
    v = int(v)
    if v == TS_NA:
        return None
    return str(np.datetime64(v, 's'))


def _hms(s):
    s = int(s)
    return f'{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}'


def ts_valid(t):
    return ~np.isnan(t) if t.dtype.kind == 'f' else (t != TS_NA)


def setup_logging(logfile=None):
    fmt = logging.Formatter('%(asctime)s %(processName)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
        fh.setFormatter(fmt)
        root.addHandler(fh)


def ds_key(ds, attack=None):
    return f'knad_{attack.lower().replace(" ", "_")}' if ds == 'knad' else ds


class _PdShim:
    """Forwards to pandas; forces nrows=<sample> in read_csv (sample mode only)."""

    def __init__(self, nrows):
        self._nrows = nrows

    def __getattr__(self, name):
        return getattr(pd, name)

    def read_csv(self, *a, **k):
        if self._nrows:
            k['nrows'] = self._nrows
        return pd.read_csv(*a, **k)


def load_original():
    src = LAUNCH_ALL.read_text()
    tree = ast.parse(src)
    want = ('data_loader', 'clean_csv_headers', 'regularize_data', 'grant_first_label')
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    missing = set(want) - {n.name for n in body}
    if missing:
        raise RuntimeError(f'functions not found in {LAUNCH_ALL}: {missing}')
    from sklearn.preprocessing import LabelEncoder
    ns = {'np': np, 'pd': pd, 'LabelEncoder': LabelEncoder, '__name__': 'launch_all_extracted'}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LAUNCH_ALL), 'exec'), ns)
    ns['__sha256__'] = hashlib.sha256(src.encode()).hexdigest()
    return ns


def literal_assign(script, name):
    tree = ast.parse(Path(script).read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise RuntimeError(f'{name} not found in {script}')


def orig_files(ds):
    script = {'cic2017': FRAMES / 'CIC2017' / 'cic2017.py', 'cic2018': FRAMES / 'CIC2018' / 'cic2018.py',
              'aci': FRAMES / 'ACI-IOT-2023' / 'aci_iot_2023_experiment.py'}[ds]
    return [Path(p).name for p in literal_assign(script, 'PATHS')]


def dataset_dir(ds):
    return {'cic2017': DS / 'CIC2017', 'cic2018': DS / 'CIC2018', 'aci': DS / 'ACI-IOT-2023'}[ds]


def labels_cache_path(cache, ds, name):
    return Path(cache) / 'labels' / ds / (Path(name).stem.replace(' ', '_') + '.npz')


def labels_csv(ds, path, nrows, cache):
    path = Path(path)
    deps = {'kind': f'labels_{ds}', 'v': CACHE_VERSION, 'src': file_sig(path), 'nrows': nrows}
    if ds == 'aci':
        deps['rule_v'] = 2

    def compute():
        info = {'file': path.name}
        if ds == 'cic2017':
            df = pd.read_csv(path, usecols=[LABEL_COL[ds]], dtype=str, nrows=nrows or None)
            lab = df[LABEL_COL[ds]]
            arrays = {}
        else:
            df = pd.read_csv(path, usecols=['Timestamp', 'Label'], dtype=str, nrows=nrows or None)
            lab = df['Label']
        n = len(df)
        info['n_rows'] = n
        codes, vocab = pd.factorize(lab.fillna('<NA>'), sort=True)
        arrays = {'codes': codes.astype(np.int16)}
        info['vocab'] = [str(v) for v in vocab]
        info['label_counts'] = {str(k): int(v) for k, v in lab.fillna('<NA>').value_counts().items()}
        if ds == 'cic2018':
            header = (lab == 'Label').to_numpy()
            clock = pd.to_datetime(df['Timestamp'].where(~header), format='%d/%m/%Y %H:%M:%S', errors='coerce')
            nat = clock.isna().to_numpy() & ~header
            hour = clock.dt.hour.to_numpy()
            shift = (hour >= 1) & (hour <= 7)
            ts24 = clock + pd.to_timedelta(np.where(shift, 12, 0), unit='h')
            mm, dd, yy = path.stem.split('-')
            file_day = pd.Timestamp(int(yy), int(mm), int(dd))
            other = (ts24.dt.normalize() != file_day).to_numpy() & ~nat & ~header
            excl = np.zeros(n, np.int8)
            excl[other] = 2
            excl[nat] = 3
            excl[header] = 1
            arrays['ts'] = ts24.to_numpy().astype('datetime64[s]').astype(np.int64)
            arrays['ts_raw'] = clock.to_numpy().astype('datetime64[s]').astype(np.int64)
            arrays['excl'] = excl
            info.update({
                'timestamp_rule': '12-hour clock without AM/PM: hours 01-07 -> +12 h; 08-11 AM; 12 = noon',
                'n_header_repeat': int(header.sum()), 'header_repeat_rows': np.flatnonzero(header)[:50].tolist(),
                'n_nat': int(nat.sum()), 'n_other_date': int(other.sum()),
                'other_dates': {str(k): int(v) for k, v in ts24[other].dt.date.value_counts().items()},
                'n_hour_shift_pm': int(shift.sum()),
                'hour_hist_raw_clock': {int(h): int(c) for h, c in clock.dt.hour.dropna().astype(int).value_counts().sort_index().items()},
            })
            ok = excl == 0
            sod = (arrays['ts'][ok] % 86400)
            g = pd.DataFrame({'lab': lab[ok].to_numpy(), 's': sod}).groupby('lab')['s'].agg(['min', 'max', 'size'])
            info['label_windows_24h'] = {str(k): [_hms(r['min']), _hms(r['max']), int(r['size'])] for k, r in g.iterrows()}
        elif ds == 'aci':
            ts = df['Timestamp']
            dash = ts.str.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').fillna(False).to_numpy(bool)
            ampm = ts.str.fullmatch(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [AP]M').fillna(False).to_numpy(bool)
            ydm = pd.to_datetime(ts.where(dash), format='%Y-%d-%m %H:%M:%S', errors='coerce')
            ymd = pd.to_datetime(ts.where(dash), format='%Y-%m-%d %H:%M:%S', errors='coerce')
            dmy = pd.to_datetime(ts.where(ampm), format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
            lo, hi = pd.Timestamp(ACI_RANGE[0]), pd.Timestamp(ACI_RANGE[1])
            in_ydm = ((ydm >= lo) & (ydm < hi)).to_numpy(bool)
            in_ymd = ((ymd >= lo) & (ymd < hi)).to_numpy(bool)
            in_dmy = ((dmy >= lo) & (dmy < hi)).to_numpy(bool)
            ydm_ok = ydm.notna().to_numpy(bool)
            ymd_ok = ymd.notna().to_numpy(bool)
            fmt = np.zeros(n, np.int8)
            fmt[dash & in_ymd & ~in_ydm & ~ydm_ok] = 1
            fmt[dash & in_ymd & ~in_ydm & ydm_ok] = 2
            fmt[dash & in_ydm & ~in_ymd & ~ymd_ok] = 3
            fmt[dash & in_ydm & ~in_ymd & ymd_ok] = 4
            fmt[ampm & in_dmy] = 5
            fmt[dash & in_ydm & in_ymd] = 6
            fmt[(fmt == 0)] = 7
            t = ymd.where(pd.Series(np.isin(fmt, [1, 2])), ydm.where(pd.Series(np.isin(fmt, [3, 4])), dmy.where(pd.Series(fmt == 5))))
            arrays['ts'] = t.to_numpy().astype('datetime64[s]').astype(np.int64)
            excl = np.zeros(n, np.int8)
            excl[fmt == 6] = 4
            excl[fmt == 7] = 5
            arrays['excl'] = excl
            arrays['fmt'] = fmt
            names = {1: 'yyyy-mm-dd (unambiguous)', 2: 'yyyy-mm-dd (alt. reading out of range)',
                     3: 'yyyy-dd-mm (unambiguous)', 4: 'yyyy-dd-mm (syntactically ambiguous; resolved by capture range)',
                     5: 'dd/mm/yyyy hh:mm:ss AM/PM', 6: 'ambiguous, both readings in range (excluded)',
                     7: 'unparseable / out of range (excluded)'}
            info['timestamp_rule'] = (f'dash dates read as yyyy-dd-mm or yyyy-mm-dd, whichever falls in the documented '
                                      f'capture range [{ACI_RANGE[0]}, {ACI_RANGE[1]}); AM/PM strings as dd/mm/yyyy 12 h')
            info['format_counts'] = {names[int(k)]: int(v) for k, v in zip(*np.unique(fmt, return_counts=True))}
            mdy = pd.to_datetime(ts.where(ampm), format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            in_mdy = ((mdy >= lo) & (mdy < hi)).to_numpy(bool)
            info['ampm_alt_mmdd_reading'] = {'n_rows_alt_reading_valid_date': int((ampm & in_dmy & mdy.notna().to_numpy(bool)).sum()),
                                             'n_rows_alt_reading_in_range': int((ampm & in_mdy).sum()),
                                             'n_rows_both_readings_in_range': int((ampm & in_mdy & in_dmy).sum())}
            info['raw_date_prefix_counts'] = {str(k): int(v) for k, v in ts.str[:10].value_counts().items()}
            ok = excl == 0
            tsec = arrays['ts'][ok]
            d = pd.DataFrame({'day': tsec // 86400, 'lab': lab[ok].to_numpy(), 's': tsec % 86400})
            g = d.groupby(['day', 'lab'])['s'].agg(['min', 'max', 'size'])
            info['label_windows'] = {f'{np.datetime64(int(a), "D")}|{b}': [_hms(r['min']), _hms(r['max']), int(r['size'])]
                                     for (a, b), r in g.iterrows()}
            gh = d.assign(h=d['s'] // 3600).groupby(['day', 'h', 'lab']).size()
            info['label_counts_day_hour'] = {f'{np.datetime64(int(a), "D")}|{int(h):02d}h|{b}': int(v) for (a, h, b), v in gh.items()}
        return arrays, info

    return cached_npz(labels_cache_path(cache, ds, path.name), deps, compute)


def labels_knad(attack, nrows, cache):
    d = DS / 'KITSUNE' / attack
    stem = attack.replace(' ', '_')
    lp = d / f'{stem}_labels.csv'
    deps = {'kind': 'labels_knad', 'v': CACHE_VERSION, 'src': file_sig(lp), 'nrows': nrows}

    def compute():
        y = pd.read_csv(lp, usecols=['x'], nrows=nrows or None)['x'].to_numpy()
        u = np.unique(y)
        if not set(u.tolist()) <= {0, 1}:
            raise ValueError(f'unexpected KNAD labels {u[:10]}')
        return {'codes': y.astype(np.int16)}, {'file': lp.name, 'n_rows': int(len(y)), 'n_attack': int(y.sum()),
                                              'vocab': ['Benign', attack]}

    return cached_npz(Path(cache) / 'labels' / 'knad' / f'{stem}.npz', deps, compute)


def _read_pcap(f, nmax):
    out = array('d')
    h = f.read(24)
    magic_le = struct.unpack('<I', h[:4])[0]
    if magic_le in (0xa1b2c3d4, 0xa1b23c4d):
        e = '<'
        nano = magic_le == 0xa1b23c4d
    else:
        e = '>'
        nano = struct.unpack('>I', h[:4])[0] == 0xa1b23c4d
    frac = 1e-9 if nano else 1e-6
    while True:
        r = f.read(16)
        if len(r) < 16:
            break
        s, sub, incl, _ = struct.unpack(e + 'IIII', r)
        f.seek(incl, 1)
        out.append(s + sub * frac)
        if nmax and len(out) >= nmax:
            break
    return np.frombuffer(out, dtype=np.float64).copy(), {'format': 'pcap', 'nanosecond': nano}


def _read_pcapng(f, nmax):
    out = array('d')
    e = '<'
    tsres = {}
    n_if = 0
    n_spb = 0
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        if struct.unpack('<I', hdr[:4])[0] == 0x0A0D0D0A:
            bom = f.read(4)
            e = '<' if struct.unpack('<I', bom)[0] == 0x1A2B3C4D else '>'
            blen = struct.unpack(e + 'I', hdr[4:])[0]
            f.seek(blen - 12, 1)
            tsres, n_if = {}, 0
            continue
        bt, blen = struct.unpack(e + 'II', hdr)
        if bt == 6:
            iface, hi, lo = struct.unpack(e + 'III', f.read(12))
            out.append(((hi << 32) | lo) * tsres.get(iface, 1e-6))
            f.seek(blen - 20, 1)
            if nmax and len(out) >= nmax:
                break
        elif bt == 1:
            body = f.read(blen - 8)
            res = 1e-6
            opts = body[8:-4]
            i = 0
            while i + 4 <= len(opts):
                code, olen = struct.unpack(e + 'HH', opts[i:i + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    v = opts[i + 4]
                    res = 2.0 ** -(v & 0x7f) if v & 0x80 else 10.0 ** -v
                i += 4 + ((olen + 3) & ~3)
            tsres[n_if] = res
            n_if += 1
        elif bt == 2:
            iface, _, hi, lo = struct.unpack(e + 'HHII', f.read(12))
            out.append(((hi << 32) | lo) * tsres.get(iface, 1e-6))
            f.seek(blen - 20, 1)
        elif bt == 3:
            out.append(float('nan'))
            n_spb += 1
            f.seek(blen - 8, 1)
        else:
            f.seek(blen - 8, 1)
    return np.frombuffer(out, dtype=np.float64).copy(), {'format': 'pcapng', 'if_tsresol': tsres, 'n_simple_packet_blocks': n_spb}


def pcap_knad(attack, nrows, cache):
    d = DS / 'KITSUNE' / attack
    cands = sorted(d.glob('*_pcap.pcap*'))
    if not cands:
        raise FileNotFoundError(f'no pcap in {d}')
    p = cands[0]
    stem = attack.replace(' ', '_')
    deps = {'kind': 'pcap_ts', 'v': CACHE_VERSION, 'src': file_sig(p), 'nrows': nrows}

    def compute():
        with open(p, 'rb', buffering=1 << 20) as f:
            ng = f.read(4) == b'\x0a\x0d\x0d\x0a'
            f.seek(0)
            ts, info = _read_pcapng(f, nrows) if ng else _read_pcap(f, nrows)
        info.update({'file': p.name, 'n_packets': int(len(ts))})
        info['if_tsresol'] = {str(k): v for k, v in info.get('if_tsresol', {}).items()}
        return {'ts': ts}, info

    return cached_npz(Path(cache) / 'pcap' / f'{stem}.npz', deps, compute)


def hash_matrix(a):
    a = np.array(a, dtype=np.float64, copy=True, order='C') + 0.0
    a[np.isnan(a)] = np.nan
    return pd.util.hash_pandas_object(pd.DataFrame(a, copy=False), index=False).to_numpy(np.uint64)


def csv_header(path):
    return pd.read_csv(path, nrows=0).columns.tolist()


def corrected_feature_cols(ds):
    """Canonical corrected feature list (same order for every file of the dataset)."""
    d = dataset_dir(ds)
    if ds == 'cic2017':
        hdrs = [csv_header(d / f) for f in CIC2017_CHRONO]
        base = pd.read_csv(d / CIC2017_CHRONO[0], nrows=0).columns.tolist()
        same = all(h == hdrs[0] for h in hdrs)
        return [c for c in base if c != LABEL_COL[ds]], {'identical_headers': same}
    if ds == 'cic2018':
        hdrs = {f: csv_header(d / f) for f in CIC2018_ALL}
        common = [c for c in hdrs[CIC2018_ALL[0]] if all(c in h for h in hdrs.values())]
        feats = [c for c in common if c not in DROP_CORR[ds] + ['Label']]
        extra = {f: [c for c in h if c not in common] for f, h in hdrs.items() if [c for c in h if c not in common]}
        return feats, {'extra_columns_by_file': extra}
    if ds == 'aci':
        h = csv_header(d / ACI_FILE)
        return [c for c in h if c not in DROP_CORR[ds] + ['Label']], {}
    raise ValueError(ds)


def hash_orig_csv(ds, fname, nrows, cache):
    path = dataset_dir(ds) / fname
    skip = 1_000_000 if ds == 'cic2018' else None
    deps = {'kind': f'hash_orig_{ds}', 'v': CACHE_VERSION, 'src': file_sig(path), 'nrows': nrows,
            'skip': skip, 'launch_all_sha': hashlib.sha256(LAUNCH_ALL.read_bytes()).hexdigest()}

    def compute():
        ns = load_original()
        ns['pd'] = _PdShim(nrows)
        try:
            (x, y), (enc, cats) = ns['data_loader'](str(path), skip=skip)
        finally:
            ns['pd'] = pd
        x = np.asarray(x, dtype=np.float64)
        cats = np.asarray(cats, dtype=bool)
        out = {'h_loader': hash_matrix(x)}
        info = {'file': fname, 'n_rows': int(len(x)), 'dim_loader': int(x.shape[1]), 'n_categorical': int(cats.sum()),
                'label_classes': [str(c) for k, e in enc.items() if k in LABEL_COL.values() for c in e.classes_]}
        if cats.any():
            xc = x[:, cats]
            model = np.concatenate([x[:, ~cats], xc.astype(np.uint8).astype(np.float64)], axis=1)
            out['h_model'] = hash_matrix(model)
            info['dim_model_after_unpackbits'] = int((~cats).sum() + 8 * cats.sum())
            info['categorical_codes'] = [{'n_codes': int(len(np.unique(xc[:, j]))),
                                          'n_after_uint8': int(len(np.unique(xc[:, j].astype(np.uint8))))}
                                         for j in range(xc.shape[1])]
        return out, info

    return cached_npz(Path(cache) / 'hash' / f'orig_{ds}' / (Path(fname).stem + '.npz'), deps, compute)


def hash_corr_csv(ds, fname, feats, nrows, cache):
    path = dataset_dir(ds) / fname
    deps = {'kind': f'hash_corr_{ds}', 'v': CACHE_VERSION, 'src': file_sig(path), 'nrows': nrows, 'feats': feats}

    def _numeric(df, info):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                if c == 'Connection Type':
                    codes, uniq = pd.factorize(df[c], sort=True)
                    df[c] = codes
                    info['connection_type_values'] = [str(u) for u in uniq]
                else:
                    num = pd.to_numeric(df[c], errors='coerce')
                    info['coerced_non_numeric'][c] = info['coerced_non_numeric'].get(c, 0) + int((num.isna() & df[c].notna()).sum())
                    df[c] = num
        a = df[feats].to_numpy(np.float64)
        nonfinite = (~np.isfinite(a)).sum(axis=1).astype(np.int16)
        return hash_matrix(a), nonfinite

    def compute():
        info = {'file': fname, 'n_features': len(feats), 'coerced_non_numeric': {}}
        if ds == 'cic2017':
            df = pd.read_csv(path, nrows=nrows or None)
            df = df.drop(columns=[LABEL_COL[ds]])
            h, nonfinite = _numeric(df, info)
        elif ds == 'cic2018':
            hs, nfs = [], []
            for chunk in pd.read_csv(path, usecols=feats, nrows=nrows or None, chunksize=1_000_000):
                h_, nf_ = _numeric(chunk, info)
                hs.append(h_)
                nfs.append(nf_)
                del chunk
            h, nonfinite = np.concatenate(hs), np.concatenate(nfs)
        else:
            df = pd.read_csv(path, usecols=feats, nrows=nrows or None)
            h, nonfinite = _numeric(df, info)
        info['n_rows'] = int(len(h))
        info['rows_with_nonfinite'] = int((nonfinite > 0).sum())
        return {'h': h, 'nonfinite': nonfinite}, info

    return cached_npz(Path(cache) / 'hash' / f'corr_{ds}' / (Path(fname).stem + '.npz'), deps, compute)


def _norm_neg0_line(line):
    return b','.join(POS0 if f == NEG0 else f for f in line.split(b','))


def hash_knad(attack, nrows, cache, check_values=False):
    d = DS / 'KITSUNE' / attack
    stem = attack.replace(' ', '_')
    p = d / f'{stem}_dataset.csv'
    deps = {'kind': 'hash_knad_text', 'v': CACHE_VERSION, 'src': file_sig(p), 'nrows': nrows, 'check': check_values}

    def compute():
        buf = bytearray()
        n = n_neg0 = 0
        nfields = set()
        with open(p, 'rb', buffering=1 << 24) as f:
            for line in f:
                if nrows and n >= nrows:
                    break
                line = line.rstrip(b'\r\n')
                if NEG0 in line:
                    n_neg0 += 1
                    line = _norm_neg0_line(line)
                if n < 1000:
                    nfields.add(line.count(b',') + 1)
                buf += hashlib.blake2b(line, digest_size=8).digest()
                n += 1
        h = np.frombuffer(bytes(buf), dtype='<u8').astype(np.uint64)
        info = {'file': p.name, 'n_lines': int(n), 'n_lines_with_neg_zero': int(n_neg0),
                'fields_first_1000_lines': sorted(nfields),
                'method': 'blake2b-64 of the CSV text line (numpy %.18e format: text equality == value equality; -0 normalised)'}
        if check_values:
            v = pd.read_csv(p, header=None, nrows=n, dtype=np.float64)
            hv = hash_matrix(v.to_numpy())
            pairs = len(np.unique(np.stack([h, hv], 1), axis=0))
            info['value_hash_check'] = {'unique_text': int(len(np.unique(h))), 'unique_value': int(len(np.unique(hv))),
                                        'unique_pairs': int(pairs),
                                        'equivalent': bool(pairs == len(np.unique(h)) == len(np.unique(hv)))}
        return {'h': h}, info

    return cached_npz(Path(cache) / 'hash' / 'knad' / f'{stem}.npz', deps, compute)


def _job(kind, *a):
    """Process-pool entry point (each worker sets up its own logging)."""
    logfile = os.environ.get('WARMUP_AUDIT_LOG') or None
    if not logging.getLogger().handlers:
        setup_logging(logfile)
    t0 = time.time()
    try:
        if kind == 'hash_orig':
            _, info = hash_orig_csv(*a)
        elif kind == 'hash_corr':
            _, info = hash_corr_csv(*a)
        elif kind == 'knad_prep':
            attack, nrows, cache, do_pcap, do_hash, check = a
            _, info = labels_knad(attack, nrows, cache)
            if do_pcap:
                pcap_knad(attack, nrows, cache)
            if do_hash:
                hash_knad(attack, nrows, cache, check)
        else:
            raise ValueError(kind)
        log.info('job %s %s done (%.1fs)', kind, a[:2], time.time() - t0)
        return None
    except Exception:
        return f'{kind} {a[:2]}: {traceback.format_exc()}'


def run_jobs(jobs, workers):
    if not jobs:
        return
    errs = []
    if workers <= 1:
        for j in jobs:
            e = _job(*j)
            if e:
                errs.append(e)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_job, *j) for j in jobs]
            for fut in futures:
                e = fut.result()
                if e:
                    errs.append(e)
    for e in errs:
        log.error(e)
    if errs:
        raise RuntimeError(f'{len(errs)} job(s) failed')


class Source:
    pass


def assemble_csv(ds, args, cache, need_orig_hash, need_corr_hash):
    files = {'cic2017': CIC2017_CHRONO, 'cic2018': CIC2018_ALL, 'aci': [ACI_FILE]}[ds]
    d = dataset_dir(ds)
    S = Source()
    S.ds, S.files, S.benign = ds, list(files), BENIGN[ds]
    per = []
    for f in files:
        log.info('[%s] labels %s', ds, f)
        per.append(labels_csv(ds, d / f, args.sample_rows, cache))
    S.n_phys = [int(p[1]['n_rows']) for p in per]
    S.offsets = np.concatenate([[0], np.cumsum(S.n_phys)]).astype(np.int64)
    S.vocab = sorted(set(v for p in per for v in p[1]['vocab']))
    lut = [np.array([S.vocab.index(v) for v in p[1]['vocab']], dtype=np.int16) for p in per]
    S.lcode = np.concatenate([lut[i][p[0]['codes']] for i, p in enumerate(per)]).astype(np.int16)
    S.ybin = (S.lcode != S.vocab.index(S.benign)).astype(np.int8) if S.benign in S.vocab else np.ones(len(S.lcode), np.int8)
    if 'Label' in S.vocab:
        S.ybin[S.lcode == S.vocab.index('Label')] = 0
    S.ts = np.concatenate([p[0]['ts'] for p in per]) if ds != 'cic2017' else None
    S.ts_raw = np.concatenate([p[0]['ts_raw'] for p in per]) if ds == 'cic2018' else None
    S.excl = np.concatenate([p[0]['excl'] for p in per]) if ds != 'cic2017' else np.zeros(len(S.lcode), np.int8)
    S.file_info = [p[1] for p in per]
    S.hashes = {}
    S.hash_info = {}
    S.nonfinite = None
    N = int(S.offsets[-1])
    feats, feat_info = (None, {})
    if need_corr_hash:
        feats, feat_info = corrected_feature_cols(ds)
    S.corr_features = feats
    S.corr_feature_info = feat_info
    jobs = []
    ofiles = orig_files(ds)
    if need_orig_hash:
        jobs += [('hash_orig', ds, f, args.sample_rows, str(cache)) for f in ofiles]
    if need_corr_hash:
        jobs += [('hash_corr', ds, f, feats, args.sample_rows, str(cache)) for f in files]
    run_jobs(jobs, args.workers)
    if need_orig_hash:
        H = {'orig_loader': np.zeros(N, np.uint64)}
        infos = []
        for f in ofiles:
            arr, info = hash_orig_csv(ds, f, args.sample_rows, cache)
            fi = S.files.index(f)
            rows = _orig_rows(ds, S.n_phys[fi])
            if len(rows) != len(arr['h_loader']):
                raise RuntimeError(f'{f}: data_loader rows {len(arr["h_loader"])} != expected {len(rows)}')
            H['orig_loader'][S.offsets[fi] + rows] = arr['h_loader']
            if 'h_model' in arr:
                H.setdefault('orig_model', np.zeros(N, np.uint64))[S.offsets[fi] + rows] = arr['h_model']
            infos.append(info)
        S.hashes.update(H)
        S.hash_info['orig'] = infos
    if need_corr_hash:
        H = np.zeros(N, np.uint64)
        NF = np.zeros(N, np.int16)
        infos = []
        for fi, f in enumerate(files):
            arr, info = hash_corr_csv(ds, f, feats, args.sample_rows, cache)
            if len(arr['h']) != S.n_phys[fi]:
                raise RuntimeError(f'{f}: corrected rows {len(arr["h"])} != label rows {S.n_phys[fi]}')
            H[S.offsets[fi]:S.offsets[fi + 1]] = arr['h']
            NF[S.offsets[fi]:S.offsets[fi + 1]] = arr['nonfinite']
            infos.append(info)
        S.hashes['corr'] = H
        S.nonfinite = NF
        S.hash_info['corr'] = infos
    return S


def _orig_rows(ds, n_phys):
    rows = np.arange(n_phys, dtype=np.int64)
    if ds == 'cic2018' and (1_000_000 - 1) < n_phys:
        rows = np.delete(rows, 1_000_000 - 1)
    return rows


def assemble_knad(attack, args, cache, need_hash, need_pcap):
    S = Source()
    S.ds, S.attack, S.files, S.benign = 'knad', attack, [attack], 'Benign'
    arr, info = labels_knad(attack, args.sample_rows, cache)
    S.lcode = arr['codes'].astype(np.int16)
    S.ybin = S.lcode.astype(np.int8)
    S.vocab = ['Benign', attack]
    S.n_phys = [len(S.lcode)]
    S.offsets = np.array([0, len(S.lcode)], np.int64)
    S.excl = np.zeros(len(S.lcode), np.int8)
    S.ts_raw = None
    S.file_info = [info]
    S.hashes, S.hash_info, S.nonfinite = {}, {}, None
    S.corr_features, S.corr_feature_info = None, {}
    S.ts = None
    S.n_feature_lines = None
    if need_pcap:
        parr, pinfo = pcap_knad(attack, args.sample_rows, cache)
        ts = parr['ts']
        S.pcap_info = pinfo
        if len(ts) < len(S.lcode):
            ts = np.concatenate([ts, np.full(len(S.lcode) - len(ts), np.nan)])
        S.ts = ts[:len(S.lcode)]
    if need_hash:
        harr, hinfo = hash_knad(attack, args.sample_rows, cache, bool(args.sample_rows))
        h = harr['h']
        S.n_feature_lines = len(h)
        H = np.zeros(len(S.lcode), np.uint64)
        k = min(len(h), len(H))
        H[:k] = h[:k]
        S.hashes['knad_text'] = H
        S.hash_info['knad'] = hinfo
    return S


class Stream:
    def __init__(self, name, gid, ybin, lcode, train_end, is_copy=None, meta=None):
        self.name = name
        self.gid = np.asarray(gid, np.int64)
        self.ybin = np.asarray(ybin, np.int8)
        self.lcode = np.asarray(lcode, np.int16)
        self.N = len(self.gid)
        self.train_end = int(train_end)
        self.is_copy = is_copy
        self.meta = meta or {}


def build_orig_csv(S, C, ns):
    files = orig_files(S.ds)
    ids, ycodes = [], []
    for f in files:
        fi = S.files.index(f)
        rows = _orig_rows(S.ds, S.n_phys[fi])
        g = S.offsets[fi] + rows
        codes = S.lcode[g].astype(np.int64)
        present = np.unique(codes)
        lut = np.full(len(S.vocab), -1, np.int64)
        lut[present] = np.arange(len(present))
        ids.append(g)
        ycodes.append(lut[codes])
    ids = np.concatenate(ids)
    y = np.concatenate(ycodes)
    new_x, new_y = ns['grant_first_label'](ids.reshape(-1, 1), y.copy(), C.COPIES)
    gid = new_x[:, 0].astype(np.int64)
    _, yb = ns['regularize_data'](np.zeros((len(new_y), 1)), new_y.copy(), [False])
    lcode = S.lcode[gid]
    y0 = S.vocab[int(lcode[0])]
    consistent = bool(np.array_equal(yb.astype(np.int8), S.ybin[gid]))
    n_target = int(sum(S.n_phys[S.files.index(f)] for f in files))
    trim = 0 if S.ds == 'aci' else max(0, len(gid) - n_target)
    meta = {'files_in_order': files, 'y0_label': y0, 'binarisation_equals_label_ne_benign': consistent,
            'n_before_copies': int(len(ids)), 'copies': C.COPIES,
            'copies_source_positions_in_uncopied_stream': [int(np.flatnonzero(y == y[0])[0]), int(np.flatnonzero(y == y[0])[C.COPIES - 1])],
            'postprocess_trim': int(trim), 'launch_all_sha256': ns['__sha256__']}
    copy_src = gid[:C.COPIES]
    fidx = np.searchsorted(S.offsets, copy_src, side='right') - 1
    meta['copies_by_file'] = {S.files[int(k)]: int(v) for k, v in zip(*np.unique(fidx, return_counts=True))}
    is_copy = np.zeros(len(gid), bool)
    is_copy[:C.COPIES] = True
    return Stream('orig', gid, yb, lcode, C.TRAIN, is_copy, meta)


def build_orig_knad(S, C):
    m_lab = len(S.lcode)
    m_feat = S.n_feature_lines if S.n_feature_lines is not None else m_lab
    n = min(m_feat - 1, m_lab)
    gid = np.minimum(np.arange(1, n + 1, dtype=np.int64), m_lab - 1)
    y = S.lcode[:n]
    meta = {'feature_lines': int(m_feat), 'label_rows': int(m_lab), 'n_feature_lines_verified': S.n_feature_lines is not None,
            'alignment': 'x[i] = feature line i+1, y[i] = label i (header=0 + y[:len(x)])', 'postprocess_trim': 0}
    return Stream('orig', gid, y.astype(np.int8), y, C.TRAIN, None, meta)


def corrected_variants(ds, attack=None):
    if ds == 'cic2017':
        return {'mon_first': {'files': CIC2017_CHRONO, 'sort': None, 'train_rule': 'first_n_rows', 'default': True,
                              'order': 'capture days in chronological order (Mon, Tue, Wed, Thu-morning WebAttacks, '
                                       'Thu-afternoon Infiltration, Fri-morning, Fri-afternoon PortScan, Fri-afternoon DDoS); '
                                       'CSV row order inside each file (no timestamp)'}}
    if ds == 'cic2018':
        return {
            'days7': {'files': CIC2018_DAYS7, 'sort': 'ts', 'train_rule': 'first_n_benign', 'default': True, 'chronological': True,
                      'order': 'corrected 24-h timestamp; ties: file (chronological) then CSV row'},
            'days10': {'files': CIC2018_ALL, 'sort': 'ts', 'train_rule': 'first_n_benign', 'chronological': True,
                       'order': 'corrected 24-h timestamp; ties: file (chronological) then CSV row'},
            'days7_rawclock': {'files': CIC2018_DAYS7, 'sort': 'ts_raw', 'train_rule': 'first_n_benign',
                               'not_recommended': True,
                               'order': 'RAW 12-hour clock as written in the CSV (documentation only: wrong order)'},
        }
    if ds == 'aci':
        order_bd = ['2023-10-31', '2023-11-03', '2023-10-30', '2023-11-01', '2023-11-02']
        return {
            'd7_literal': {'files': [ACI_FILE], 'sort': 'ts', 'train_rule': 'first_n_benign', 'default': True, 'chronological': True,
                           'order': 'resolved timestamp; ties: CSV row'},
            'oct31_train_chrono': {'files': [ACI_FILE], 'sort': 'day_order', 'chronological': True,
                                   'day_order': ['2023-10-31', '2023-11-01', '2023-11-02', '2023-11-03'],
                                   'drop_unlisted_days': True, 'train_rule': 'days', 'train_days': ['2023-10-31'],
                                   'deviation_train_size': True,
                                   'order': 'resolved timestamp (chronological); 30 Oct dropped (before the training day); '
                                            'ties: CSV row'},
            'benign_days_first': {'files': [ACI_FILE], 'sort': 'day_order', 'day_order': order_bd,
                                  'train_rule': 'first_n_benign', 'not_chronological': True,
                                  'order': 'days ' + ' > '.join(order_bd) + '; resolved timestamp inside each day; ties: CSV row'},
            'benign_days_only': {'files': [ACI_FILE], 'sort': 'day_order', 'day_order': order_bd,
                                 'train_rule': 'days', 'train_days': ['2023-10-31', '2023-11-03'],
                                 'not_chronological': True, 'deviation_train_size': True,
                                 'order': 'days ' + ' > '.join(order_bd) + '; resolved timestamp inside each day; ties: CSV row'},
        }
    if ds == 'knad':
        return {'capture_order': {'files': [attack], 'sort': None, 'train_rule': 'first_n_rows', 'default': True,
                                  'order': 'capture (pcap/CSV) order; AfterImage features are order-dependent and are not re-sorted'}}
    raise ValueError(ds)


def _day_index(ts):
    return np.where(ts == TS_NA, -1, ts // 86400)


def build_corrected(S, variant, spec, C):
    fis = [S.files.index(f) for f in spec['files']]
    g = np.concatenate([S.offsets[fi] + np.arange(S.n_phys[fi], dtype=np.int64) for fi in fis])
    excl = S.excl[g]
    excl_names = {1: 'repeated header row', 2: 'timestamp outside the capture day', 3: 'unparseable timestamp',
                  4: 'ambiguous date (both readings in range)', 5: 'unparseable/out-of-range timestamp'}
    excluded = {}
    for code in np.unique(excl[excl != 0]):
        sel = g[excl == code]
        excluded[excl_names[int(code)]] = {'n': int(len(sel)), 'types': _types(S, sel), 'n_mal': int(S.ybin[sel].sum())}
    g = g[excl == 0]
    meta = {'variant': variant, 'files': spec['files'], 'order': spec['order'], 'excluded_before_sort': excluded,
            'train_rule': spec['train_rule'],
            'flags': {k: spec[k] for k in ('not_recommended', 'not_chronological', 'deviation_train_size', 'chronological') if k in spec}}
    sort = spec['sort']
    tskey = None
    daykey = None
    if sort == 'ts':
        g = g[np.lexsort((g, S.ts[g]))]
        tskey, daykey = S.ts[g], _day_index(S.ts[g])
    elif sort == 'ts_raw':
        g = g[np.lexsort((g, S.ts_raw[g]))]
        tskey, daykey = S.ts_raw[g], _day_index(S.ts_raw[g])
    elif sort == 'day_order':
        day = _day_index(S.ts[g])
        ords = [int(np.datetime64(dd, 'D').astype(np.int64)) for dd in spec['day_order']]
        rank = np.full(len(g), len(ords), np.int64)
        for r, o in enumerate(ords):
            rank[day == o] = r
        meta['rows_in_unlisted_days'] = int((rank == len(ords)).sum())
        if spec.get('drop_unlisted_days'):
            drop = rank == len(ords)
            if drop.any():
                excluded['day not used (not in day_order)'] = {'n': int(drop.sum()), 'types': _types(S, g[drop]),
                                                               'n_mal': int(S.ybin[g[drop]].sum()),
                                                               'days': sorted({str(np.datetime64(int(x), 'D')) for x in np.unique(day[drop]) if x >= 0})}
            g, rank = g[~drop], rank[~drop]
        order = np.lexsort((g, S.ts[g], rank))
        g = g[order]
        tskey, daykey = S.ts[g], rank[order]
    ben = S.ybin[g] == 0
    rule = spec['train_rule']
    n = len(g)
    if rule == 'first_n_rows':
        if n <= C.TRAIN:
            raise ValueError(f'stream too short ({n}) for {C.TRAIN} training rows')
        t_end = C.TRAIN
        train_sel = np.arange(t_end)
    elif rule == 'first_n_benign':
        cb = np.cumsum(ben)
        if len(cb) == 0 or cb[-1] < C.TRAIN:
            raise ValueError(f'only {int(cb[-1]) if len(cb) else 0} benign rows < {C.TRAIN}')
        t_end = int(np.searchsorted(cb, C.TRAIN)) + 1
        train_sel = np.flatnonzero(ben[:t_end])
    elif rule == 'days':
        ords = spec['train_days']
        t_end = int((daykey < len(ords)).sum())
        if t_end == 0:
            raise ValueError('no rows in the training days')
        train_sel = np.arange(t_end)
    else:
        raise ValueError(rule)
    disc = np.flatnonzero(~ben[:t_end]) if rule == 'first_n_benign' else np.array([], np.int64)
    if tskey is not None:
        last, lastday = tskey[t_end - 1], daykey[t_end - 1]
        stop = np.flatnonzero((tskey[t_end:] > last) | (daykey[t_end:] != lastday))
        j = t_end + (int(stop[0]) if len(stop) else n - t_end)
    else:
        j = t_end
    if j >= n:
        raise ValueError('no execution rows left after training')
    guard = np.arange(t_end, j)
    model_gid = np.concatenate([g[train_sel], g[j:]])
    train_eff = len(train_sel)
    tr_mm, dc_mm, ex_mm = _ts_minmax(S, g[:t_end]), _ts_minmax(S, g[disc]), _ts_minmax(S, g[j:])
    meta['cuts'] = {
        'sorted_n_after_exclusions': int(n),
        'train_interval_sorted': [0, int(t_end)],
        'train_rows_kept': int(train_eff),
        'train_size_deviation': bool(train_eff != C.TRAIN),
        'fm_model': [0, int(min(C.FM, train_eff))], 'ae_model': [int(min(C.FM, train_eff)), int(train_eff)],
        'train_ts_first': tr_mm[0], 'train_ts_last': tr_mm[1],
        'n_attack_in_train_kept': int(S.ybin[g[train_sel]].sum()),
        'discarded_label_filter': {'n': int(len(disc)), 'types': _types(S, g[disc]),
                                   'ts_min': dc_mm[0], 'ts_max': dc_mm[1],
                                   'sorted_idx_min': int(disc.min()) if len(disc) else None,
                                   'sorted_idx_max': int(disc.max()) if len(disc) else None},
        'guard_ties_dropped': {'n': int(len(guard)), 'types': _types(S, g[guard])},
        'exec_start_sorted': int(j), 'exec_start_model': int(train_eff), 'N_model': int(len(model_gid)),
        'exec_ts_first': ex_mm[0], 'exec_ts_last': ex_mm[1],
    }
    if S.ts is not None:
        ttr = S.ts[g[train_sel]]
        tex = S.ts[g[j:]]
        vt, ve = ts_valid(ttr), ts_valid(tex)
        if vt.any() and ve.any():
            meta['cuts']['n_exec_rows_ts_le_train_max'] = int((tex[ve] <= ttr[vt].max()).sum())
            meta['cuts']['n_exec_rows_ts_lt_train_min'] = int((tex[ve] < ttr[vt].min()).sum())
    if spec.get('chronological'):
        n_le = meta['cuts'].get('n_exec_rows_ts_le_train_max')
        if n_le is None or n_le != 0:
            raise AssertionError(f'{variant}: chronological variant with {n_le} execution rows at or before the last training timestamp')
    lc = S.lcode[model_gid]
    return Stream(variant, model_gid, S.ybin[model_gid], lc, train_eff, None, meta), g


def _types(S, gids):
    if len(gids) == 0:
        return {}
    c = np.bincount(S.lcode[gids].astype(np.int64), minlength=len(S.vocab))
    return {S.vocab[i]: int(c[i]) for i in np.argsort(-c) if c[i] > 0}


def _ts_minmax(S, gids):
    if S.ts is None or len(gids) == 0:
        return None, None
    t = S.ts[gids]
    v = ts_valid(t)
    if not v.any():
        return None, None
    return iso_sec(t[v].min()), iso_sec(t[v].max())


def regions_for(st, pipeline, C, trim=0):
    R = {'train': (0, st.train_end), 'train_fm': (0, min(C.FM, st.train_end)), 'train_ae': (min(C.FM, st.train_end), st.train_end)}
    if pipeline == 'orig':
        R['ref_pp_orig'] = (trim + C.REF[0], trim + C.REF[1])
    E = st.N - st.train_end
    R['exec'] = (st.train_end, st.N)
    for p in WARM_PCTS:
        e = st.train_end + (E * p) // 100
        R[f'warm{p:02d}'] = (st.train_end, e)
        R[f'eval_after{p:02d}'] = (e, st.N)
    return R


def region_stats(st, S, a, b, ctx):
    a = int(max(0, min(a, st.N)))
    b = int(max(a, min(b, st.N)))
    n = b - a
    r = {'start': a, 'end': b, 'n': n}
    if n == 0:
        return r
    yb = st.ybin[a:b]
    nm = int(yb.sum())
    r['n_mal'] = nm
    r['pct_mal'] = 100.0 * nm / n
    if nm:
        k = int(np.argmax(yb))
        r['first_mal_abs'], r['first_mal_rel'] = a + k, k
    else:
        r['first_mal_abs'] = r['first_mal_rel'] = None
    c = np.bincount(st.lcode[a:b].astype(np.int64), minlength=len(S.vocab))
    r['types'] = {S.vocab[i]: int(c[i]) for i in np.argsort(-c) if c[i] > 0}
    P = ctx['prefix']
    i = np.arange(a, b, dtype=np.int64)
    lo = np.maximum(0, i - (L_SEQ - 1))
    r['pct_windows_any_mal'] = 100.0 * float(((P[i + 1] - P[lo]) > 0).mean())
    r['n_windows_full_context'] = int(max(0, b - max(a, L_SEQ - 1)))
    r['n_windows_crossing_start'] = int(min(n, L_SEQ - 1)) if a > 0 else 0
    if S.ts is not None:
        t = S.ts[st.gid[a:b]]
        v = ts_valid(t)
        if v.any():
            r['ts_min'], r['ts_max'] = iso_sec(t[v].min()), iso_sec(t[v].max())
            if t.dtype.kind != 'f':
                days, cnt = np.unique(t[v] // 86400, return_counts=True)
                r['days'] = {str(np.datetime64(int(dd), 'D')): int(cc) for dd, cc in zip(days, cnt)}
    if st.is_copy is not None:
        r['n_copies'] = int(st.is_copy[a:b].sum())
        orig_of_train = np.isin(st.gid[a:b], ctx['train_gid_unique']) & ~st.is_copy[a:b]
        r['n_originals_of_train_copies'] = int(orig_of_train.sum())
        r['n_originals_of_any_copy'] = int((np.isin(st.gid[a:b], ctx['copy_gid_unique']) & ~st.is_copy[a:b]).sum())
    else:
        same = np.isin(st.gid[a:b], ctx['train_gid_unique'])
        r['n_training_rows_re_evaluated'] = int(same.sum()) if a >= st.train_end else None
    if S.nonfinite is not None and st.name != 'orig':
        r['rows_with_nonfinite_features'] = int((S.nonfinite[st.gid[a:b]] > 0).sum())
    dups = {}
    same = np.isin(st.gid[a:b], ctx['train_gid_unique']) if ctx['hash_names'] else None
    for hn in ctx['hash_names']:
        H = S.hashes[hn]
        h = H[st.gid[a:b]]
        in_tr = np.isin(h, ctx['H_train'][hn])
        dd = {'n': int(in_tr.sum()), 'n_attack': int((in_tr & (yb == 1)).sum()),
              'n_same_source_row': int((in_tr & same).sum()), 'n_natural': int((in_tr & ~same).sum())}
        if 'H_warm10' in ctx and a >= ctx['warm10_end']:
            in_w = np.isin(h, ctx['H_warm10'][hn])
            dd['n_dup_warm10'] = int(in_w.sum())
            dd['n_dup_warm10_attack'] = int((in_w & (yb == 1)).sum())
        dups[hn] = dd
    if dups:
        r['dup_with_train'] = dups
    return r


def stream_stats(st, S, ctx):
    s = {'N': st.N, 'n_mal': int(st.ybin.sum()), 'pct_mal': 100.0 * float(st.ybin.mean()) if st.N else None}
    s['first_mal'] = int(np.argmax(st.ybin)) if st.ybin.any() else None
    c = np.bincount(st.lcode.astype(np.int64), minlength=len(S.vocab))
    s['types'] = {S.vocab[i]: int(c[i]) for i in np.argsort(-c) if c[i] > 0}
    if S.ts is not None:
        t = S.ts[st.gid]
        v = ts_valid(t)
        tv = t[v].astype(np.float64)
        pos = np.flatnonzero(v)
        ts = {'n_valid': int(v.sum()), 'n_missing': int((~v).sum())}
        if len(tv) > 1:
            d = np.diff(tv)
            neg = d < 0
            ts.update({'min': iso_sec(t[v].min()), 'max': iso_sec(t[v].max()),
                       'monotonic_non_decreasing': bool(not neg.any()), 'n_neg_steps': int(neg.sum()),
                       'pct_neg_steps': 100.0 * float(neg.mean()),
                       'max_backward_jump_s': float(-d.min()) if neg.any() else 0.0,
                       'n_forward_gaps_gt_600s': int((d > 600).sum())})
            top = np.argsort(-d)[:5]
            ts['top_forward_gaps'] = [{'pos': int(pos[k + 1]), 'from': iso_sec(t[pos[k]]), 'to': iso_sec(t[pos[k + 1]]),
                                       'gap_s': float(d[k])} for k in top if d[k] > 0]
            if neg.any():
                bk = np.argsort(d)[:3]
                ts['top_backward_jumps'] = [{'pos': int(pos[k + 1]), 'from': iso_sec(t[pos[k]]), 'to': iso_sec(t[pos[k + 1]]),
                                             'jump_s': float(d[k])} for k in bk if d[k] < 0]
        s['timestamps'] = ts
    fidx = np.searchsorted(S.offsets, st.gid, side='right') - 1
    blocks = []
    for k in np.unique(fidx):
        p = np.flatnonzero(fidx == k)
        blocks.append({'file': S.files[int(k)], 'first_pos': int(p[0]), 'last_pos': int(p[-1]), 'n': int(len(p)),
                       'n_mal': int(st.ybin[p].sum()), 'n_runs': int(1 + (np.diff(p) > 1).sum()),
                       'n_copies': int(st.is_copy[p].sum()) if st.is_copy is not None else None})
    s['blocks_by_file'] = sorted(blocks, key=lambda x: x['first_pos'])
    if S.ts is not None and S.ts.dtype.kind != 'f':
        t = S.ts[st.gid]
        day = _day_index(t)
        dblocks = []
        for k in np.unique(day):
            p = np.flatnonzero(day == k)
            dblocks.append({'day': str(np.datetime64(int(k), 'D')) if k >= 0 else 'NA', 'first_pos': int(p[0]),
                            'last_pos': int(p[-1]), 'n': int(len(p)), 'n_mal': int(st.ybin[p].sum()),
                            'n_runs': int(1 + (np.diff(p) > 1).sum())})
        s['blocks_by_day'] = sorted(dblocks, key=lambda x: x['first_pos'])
    if ctx['hash_names']:
        ug, first = np.unique(st.gid, return_index=True)
    for hn in ctx['hash_names']:
        h = S.hashes[hn][ug]
        df = pd.DataFrame({'h': h, 'y': st.ybin[first]})
        g = df.groupby('h')['y'].agg(['nunique', 'size'])
        conf = g['nunique'] > 1
        s.setdefault('duplicates', {})[hn] = {
            'n_distinct_source_rows': int(len(ug)), 'n_distinct_contents': int(len(g)),
            'n_rows_duplicating_an_earlier_content': int(len(ug) - len(g)),
            'n_groups_conflicting_labels': int(conf.sum()), 'n_rows_in_conflicting_groups': int(g.loc[conf, 'size'].sum()),
            'n_unique_training_contents': int(len(ctx['H_train'][hn]))}
    return s


def analyse(st, S, pipeline, C, trim=0):
    ctx = {'prefix': np.concatenate([[0], np.cumsum(st.ybin, dtype=np.int64)])}
    ctx['train_gid_unique'] = np.unique(st.gid[:st.train_end])
    if st.is_copy is not None:
        ctx['copy_gid_unique'] = np.unique(st.gid[st.is_copy])
    names = [hn for hn in (['orig_loader', 'orig_model', 'knad_text'] if pipeline == 'orig' else ['corr', 'knad_text'])
             if hn in S.hashes]
    ctx['hash_names'] = names
    ctx['H_train'] = {hn: np.unique(S.hashes[hn][st.gid[:st.train_end]]) for hn in names}
    R = regions_for(st, pipeline, C, trim)
    ctx['warm10_end'] = R['warm10'][1]
    ctx['H_warm10'] = {hn: np.unique(S.hashes[hn][st.gid[R['warm10'][0]:R['warm10'][1]]]) for hn in names}
    out = {'pipeline': pipeline, 'variant': st.name, 'meta': st.meta, 'stream': stream_stats(st, S, ctx), 'regions': {}}
    for k, (a, b) in R.items():
        out['regions'][k] = region_stats(st, S, a, b, ctx)
    if st.name != 'orig':
        tr = out['regions']['train']
        assert tr.get('n_mal', 0) == st.meta['cuts']['n_attack_in_train_kept']
        for k in ('exec', 'warm10', 'eval_after10'):
            assert (out['regions'][k].get('n_training_rows_re_evaluated') or 0) == 0, 'training row re-evaluated'
    return out


def orig_checks(key, S, st, C, sample):
    chk = {}
    if sample:
        return {'skipped': 'sample mode'}
    exp = ORIG_EXPECTED_N.get(key)
    chk['N_matches_original_npy_length'] = {'expected': exp, 'got': st.N, 'ok': st.N == exp}
    try:
        if key == 'aci' and ORIG_ACI_TARGET.exists():
            t = np.load(ORIG_ACI_TARGET)
            chk['equal_to_original_target_npy'] = bool(len(t) == st.N and np.array_equal(t.astype(np.int8), st.ybin))
        rdir = {'aci': 'aci_iot_2023', 'cic2017': 'cic2017', 'cic2018': 'cic2018'}.get(key, key.replace('knad_', ''))
        rp = REMOTE_TARGET / rdir / 'target.npy'
        if rp.exists():
            t = np.load(rp, allow_pickle=True).astype(np.int64)
            if key == 'cic2017':
                ref = st.ybin[C.COPIES:]
            elif key == 'cic2018':
                g = np.concatenate([S.offsets[S.files.index(f)] + np.arange(S.n_phys[S.files.index(f)]) for f in orig_files('cic2018')])
                ref = S.ybin[g]
            elif key == 'aci':
                ref = st.ybin
            else:
                ref = S.ybin
            k = min(len(t), len(ref))
            chk['remote_target_npy'] = {'path': str(rp), 'len_target': int(len(t)), 'len_ref': int(len(ref)),
                                        'n_mismatch_common_prefix': int((t[:k] != ref[:k]).sum())}
    except Exception as e:
        chk['error'] = repr(e)
    return chk


def _fmt_types(d, k=4):
    items = [(a, b) for a, b in d.items() if a not in ('Benign', 'BENIGN')]
    return ', '.join(f'{a} {b:,}' for a, b in items[:k]) + (' ...' if len(items) > k else '') if items else 'none'


def declarar(key, ds, variant, spec, res, C, attack=None):
    m = res['meta']
    cu = m['cuts']
    R = res['regions']
    s = res['stream']
    ex = R['exec']
    Nx = ex['n']
    pct = ex.get('pct_mal', 0.0)
    dup = (R['exec'].get('dup_with_train') or {})
    dup_txt = ''
    for hn, dd in dup.items():
        dup_txt = (f" {dd['n']:,} evaluated rows ({dd['n_attack']:,} labelled as attacks) have a feature vector identical to "
                   f"some training row (natural duplicates; no row is duplicated by construction).")
    disc = cu['discarded_label_filter']
    if ds == 'cic2017':
        mon = next((b for b in s['blocks_by_file'] if b['file'].startswith('Monday')), None)
        return (f"CIC-IDS2017 (MachineLearningCVE CSV release; {s['N']:,} rows, which are bidirectional flows rather than packets) "
                f"has no timestamp, IP address or flow identifier. The eight CSV files were concatenated in chronological order of "
                f"capture (Monday to Friday; Thursday morning before Thursday afternoon; Friday morning, then the PortScan and DDoS "
                f"afternoon files); the row order inside each file was kept and cannot be verified to be chronological. The first "
                f"{C.TRAIN:,} flows (Monday; feature mapping: first {C.FM:,}; autoencoders: next {C.AE:,}) are used for training, "
                f"{cu['n_attack_in_train_kept']:,} of them labelled as attacks, so no label-based filtering was applied. Evaluation "
                f"starts at the next flow and training flows are never evaluated ({Nx:,} flows, {pct:.2f} % malicious). Monday occupies "
                f"stream positions [0, {mon['last_pos'] + 1:,})." + dup_txt)
    if ds == 'cic2018':
        days = ', '.join(f[:5] for f in spec['files'])
        other =m['excluded_before_sort'].get('timestamp outside the capture day', {}).get('n', 0)
        rep = m['excluded_before_sort'].get('repeated header row', {}).get('n', 0)
        shift = res.get('_n_hour_shift', 0)
        extra = '' if len(spec['files']) == 10 else f" (02-20, 02-28 and 03-01 were not used)"
        filt = (f"{disc['n']:,} attack flows in that interval ({_fmt_types(disc['types'])}) were removed by label and are neither used "
                f"for training nor evaluated." if disc['n'] else
                "that interval contains no attack flow according to the labels, so the label filter removed no row (the choice of the "
                "interval is nevertheless informed by the labels).")
        if spec.get('not_recommended'):
            return ('NOT RECOMMENDED (documentation of the raw 12-hour clock ordering): ' + filt)
        return (f"CSE-CIC-IDS2018: {len(spec['files'])} of the 10 daily CSV files ({days}-2018){extra}; "
                "rows are flows. The CSV timestamps use a 12-hour clock without AM/PM designator; hours "
                f"01-07 were mapped to 13-19 h ({shift:,} rows), which aligns the attack labels with the "
                "documented attack schedule. Rows were sorted by the corrected timestamp (1-s resolution; "
                f"ties broken by file and CSV row). {other:,} rows with a timestamp outside the capture "
                f"day (year 1970) and {rep:,} repeated header row(s) were removed; Timestamp"
                f"{', Flow ID, Src IP, Src Port and Dst IP (present in 02-20 only) are' if len(spec['files']) == 10 else ' is'}"
                f" not used as a feature. Training uses the first {cu['train_rows_kept']:,} benign flows "
                f"in time order ({cu['train_ts_first']} to {cu['train_ts_last']}); {filt} "
                f"{cu['guard_ties_dropped']['n']:,} flows sharing the timestamp of the last training flow "
                "were dropped. Evaluation starts at the first flow strictly later than the last training "
                f"flow ({cu['exec_ts_first']}): {Nx:,} flows, {pct:.2f} % malicious." + dup_txt)
    if ds == 'aci':
        amb = res.get('_n_ambiguous_resolved', 0)
        amb2 = res.get('_n_ampm_alt_valid', 0)
        head = (f"ACI-IoT-2023: Flow ID, Src IP and Dst IP (identifiers) and Timestamp are not used as features. Timestamps appear "
                f"in three textual forms; {amb:,} rows written as yyyy-dd-mm (also syntactically valid as yyyy-mm-dd) and {amb2:,} rows "
                f"written as dd/mm/yyyy with a 12-hour clock (also valid as mm/dd/yyyy) were resolved with the documented capture "
                f"period (30 October - 3 November 2023); no row remained ambiguous. ")
        if variant == 'oct31_train_chrono':
            ex_d = m['excluded_before_sort'].get('day not used (not in day_order)', {})
            body = (f"Rows were sorted by timestamp. Training uses all {cu['train_rows_kept']:,} flows of the first benign capture day "
                    f"(31 October; {cu['n_attack_in_train_kept']:,} labelled as attacks, no row-level filtering), which deviates from the "
                    f"{C.TRAIN:,} training rows of the original configuration (feature mapping {min(C.FM, cu['train_rows_kept']):,}, "
                    f"autoencoders {max(0, cu['train_rows_kept'] - C.FM):,}). The {ex_d.get('n', 0):,} flows of 30 October, recorded before "
                    f"the training day ({_fmt_types(ex_d.get('types', {}), 6)}), are not used. Evaluation covers the later days in time "
                    f"order ({cu['exec_ts_first']} to {cu['exec_ts_last']}; every evaluated flow is later than every training flow): "
                    f"{Nx:,} flows, {pct:.2f} % malicious ({_fmt_types(ex['types'], 7)}).")
        elif variant == 'd7_literal':
            body = (f"Rows were sorted by timestamp (ties: CSV row). Training uses the first {cu['train_rows_kept']:,} benign flows in "
                    f"time order ({cu['train_ts_first']} to {cu['train_ts_last']}); {disc['n']:,} attack flows in that interval "
                    f"({_fmt_types(disc['types'], 6)}) were removed by label and are not evaluated. Evaluation covers {Nx:,} flows "
                    f"({cu['exec_ts_first']} to {cu['exec_ts_last']}) with {pct:.2f} % malicious flows ({_fmt_types(ex['types'], 6)}).")
        elif variant == 'benign_days_first':
            body = (f"The two benign capture days (31 October and 3 November) were placed first, followed by the attack days in "
                    f"chronological order (30 October, 1 and 2 November); the stream is therefore NOT chronological across days and "
                    f"training includes a day recorded after the evaluated days. Training uses the first {cu['train_rows_kept']:,} benign "
                    f"flows in this order; {disc['n']:,} attack flows in the training interval ({_fmt_types(disc['types'], 6)}) were "
                    f"removed by label and are not evaluated. Evaluation: {Nx:,} flows, {pct:.2f} % malicious ({_fmt_types(ex['types'], 6)}).")
        else:
            body = (f"Training uses all {cu['train_rows_kept']:,} flows of the two benign capture days (31 October and 3 November; "
                    f"{cu['n_attack_in_train_kept']:,} labelled as attacks), which deviates from the {C.TRAIN:,} training rows of the "
                    f"original configuration (feature mapping {C.FM:,}, autoencoders {cu['train_rows_kept'] - C.FM:,}); the attack days "
                    f"follow in chronological order (the stream is NOT chronological across days). Evaluation: {Nx:,} flows, "
                    f"{pct:.2f} % malicious ({_fmt_types(ex['types'], 6)}).")
        return head + body + dup_txt
    if ds == 'knad':
        tsx = s.get('timestamps', {})
        tstxt = (f" (pcap timestamps: {tsx.get('n_neg_steps', 0):,} backward steps, maximum {1e6 * (tsx.get('max_backward_jump_s') or 0):,.0f} us)"
                 if tsx else '')
        first = s['first_mal']
        first_txt = (f"the first attack packet is at index {first:,}" if first is not None
                     else "no packet of the stream is labelled as an attack")
        clean = cu['n_attack_in_train_kept'] == 0
        return (f"Kitsune {attack}: the AfterImage feature CSV has no header and is now read as such (the submitted code consumed the "
                f"first packet as a header, which shifted features and labels by one packet). Packets are kept in capture order{tstxt}. "
                f"Training uses packets [0, {C.TRAIN:,}) (feature mapping {C.FM:,}, autoencoders {C.AE:,}); {first_txt}, so training "
                f"{'is attack-free' if clean else 'contains ' + format(cu['n_attack_in_train_kept'], ',') + ' attack packets'} according "
                f"to the labels without any row-level filtering, relying on the dataset's attack-free prelude. Evaluation starts at "
                f"packet {C.TRAIN:,}: {Nx:,} packets, {pct:.2f} % malicious." + dup_txt)
    return ''


def cuts_entry(key, ds, variant, spec, res, C, attack=None):
    m = res['meta']
    cu = m['cuts']
    R = res['regions']
    drop = DROP_CORR[ds if ds != 'knad' else 'knad']
    e = {
        'ficheros_o_dias': spec['files'],
        'orden': {'clave': spec['order'], 'desempate_estable': 'fichero (orden de la lista) y fila del CSV' if spec['sort'] else 'orden del fichero'},
        'filas_excluidas_antes_de_ordenar': m['excluded_before_sort'],
        'columnas_eliminadas': drop if ds != 'aci' else drop + ['(Src Port: se mantiene)'],
        'preprocesado': PREPROC_RULE[ds],
        'entrenamiento': {
            'regla': {'first_n_rows': f'primeras {C.TRAIN} filas del stream', 'first_n_benign': f'primeras {C.TRAIN} filas benignas en el orden del stream (filtro por etiqueta)',
                      'days': 'todas las filas de los días benignos (desviación del tamaño de entrenamiento)'}[spec['train_rule']],
            'intervalo_ordenado': cu['train_interval_sorted'], 'filas_mantenidas': cu['train_rows_kept'],
            'fm_modelo': cu['fm_model'], 'ae_modelo': cu['ae_model'], 'ts_primero': cu['train_ts_first'], 'ts_ultimo': cu['train_ts_last'],
            'ataques_en_entrenamiento': cu['n_attack_in_train_kept'], 'desviacion_tamano': cu['train_size_deviation'],
        },
        'descartadas_por_filtro_de_etiqueta': cu['discarded_label_filter'],
        'guarda_empates_timestamp': cu['guard_ties_dropped'],
        'evaluacion': {'inicio_ordenado': cu['exec_start_sorted'], 'inicio_modelo': cu['exec_start_model'], 'fin_modelo': cu['N_model'],
                       'ts_primero': cu['exec_ts_first'], 'ts_ultimo': cu['exec_ts_last'], 'n_filas': R['exec']['n'],
                       'pct_malicioso': R['exec'].get('pct_mal'), 'tipos': R['exec'].get('types'),
                       'filas_ejec_con_ts_<=_max_entrenamiento': cu.get('n_exec_rows_ts_le_train_max'),
                       'filas_de_entrenamiento_reevaluadas': R['exec'].get('n_training_rows_re_evaluated')},
        'warmup_modelo': {str(p): [R[f'warm{p:02d}']['start'], R[f'warm{p:02d}']['end']] for p in WARM_PCTS},
        'warmup_pct_malicioso': {str(p): R[f'warm{p:02d}'].get('pct_mal') for p in WARM_PCTS},
        'duplicados_exactos_entrenamiento_vs_ejecucion': {hn: dd for hn, dd in (R['exec'].get('dup_with_train') or {}).items()} or None,
        'flags': m.get('flags', {}),
        'declarar': declarar(key, ds, variant, spec, res, C, attack),
    }
    return e


def run_dataset(ds, attack, args, C, cache, out_dir, ns_holder):
    key = ds_key(ds, attack)
    do_orig = args.pipeline in ('orig', 'both')
    do_corr = args.pipeline in ('corrected', 'both')
    t0 = time.time()
    res = {'key': key, 'dataset': ds, 'attack': attack, 'generated': now_iso(), 'script_sha256': script_sha(),
           'mode': 'sample' if args.sample_rows else 'full', 'sample_rows': args.sample_rows,
           'stages': {'labels': True, 'hashes': not args.skip_hashes, 'pcap': (ds == 'knad' and not args.skip_pcap)},
           'constants': C.as_dict(), 'orig': None, 'corrected': {}, 'errors': {}}
    if ds == 'knad':
        S = assemble_knad(attack, args, cache, not args.skip_hashes, not args.skip_pcap)
    else:
        S = assemble_csv(ds, args, cache, do_orig and not args.skip_hashes, do_corr and not args.skip_hashes)
    res['source'] = {'files': S.files, 'n_rows_by_file': S.n_phys, 'vocab': S.vocab, 'file_info': S.file_info,
                     'hash_info': S.hash_info, 'corrected_features': S.corr_features, 'corrected_feature_info': S.corr_feature_info,
                     'pcap_info': getattr(S, 'pcap_info', None)}
    log.info('[%s] sources assembled (%d rows, %.1fs)', key, int(S.offsets[-1]), time.time() - t0)
    if do_orig:
        try:
            if ds == 'knad':
                st = build_orig_knad(S, C)
            else:
                if ns_holder.get('ns') is None:
                    ns_holder['ns'] = load_original()
                st = build_orig_csv(S, C, ns_holder['ns'])
            r = analyse(st, S, 'orig', C, st.meta.get('postprocess_trim', 0))
            r['checks'] = orig_checks(key, S, st, C, bool(args.sample_rows))
            res['orig'] = r
            log.info('[%s] orig N=%d %%mal=%.2f (%.1fs)', key, st.N, r['stream']['pct_mal'], time.time() - t0)
        except Exception as e:
            res['errors']['orig'] = f'{e!r}'
            log.error('[%s] orig failed: %s\n%s', key, e, traceback.format_exc())
    if do_corr:
        for variant, spec in corrected_variants(ds, attack).items():
            try:
                st, sorted_g = build_corrected(S, variant, spec, C)
                r = analyse(st, S, 'corrected', C)
                if ds == 'cic2018':
                    fis = [S.files.index(f) for f in spec['files']]
                    r['_n_hour_shift'] = int(sum(S.file_info[i].get('n_hour_shift_pm', 0) for i in fis))
                if ds == 'aci':
                    fc = S.file_info[0].get('format_counts', {})
                    r['_n_ambiguous_resolved'] = int(sum(v for k, v in fc.items() if 'syntactically ambiguous' in k))
                    r['_n_ampm_alt_valid'] = int(S.file_info[0].get('ampm_alt_mmdd_reading', {}).get('n_rows_alt_reading_valid_date', 0))
                r['spec'] = spec
                r['stream_cuts'] = cuts_entry(key, ds, variant, spec, r, C, attack)
                sp = Path(cache) / 'streams' / f'{key}__{variant}.npz'
                sp.parent.mkdir(parents=True, exist_ok=True)
                tmp = sp.with_name(f'.{sp.stem}.tmp{os.getpid()}.npz')
                np.savez(tmp, gid=st.gid, train_end=np.int64(st.train_end), file_offsets=S.offsets)
                os.replace(tmp, sp)
                r['stream_index_file'] = str(sp)
                res['corrected'][variant] = r
                log.info('[%s] corrected/%s N=%d train_end=%d exec %%mal=%.2f (%.1fs)', key, variant, st.N, st.train_end,
                         r['regions']['exec'].get('pct_mal', float('nan')), time.time() - t0)
            except Exception as e:
                tag = 'errors' if spec.get('default') else 'warnings'
                res.setdefault(tag, {})[f'corrected/{variant}'] = f'{e!r}'
                log.log(logging.ERROR if spec.get('default') else logging.WARNING,
                        '[%s] corrected/%s failed: %s\n%s', key, variant, e, traceback.format_exc())
    res['elapsed_s'] = round(time.time() - t0, 1)
    return res


def _n(x):
    return '—' if x is None else f'{int(x):,}'.replace(',', '.')


def _p(x, d=2):
    return '—' if x is None else f'{x:.{d}f}'.replace('.', ',')


def _types_short(d, k=3):
    items = [(a, b) for a, b in (d or {}).items() if a not in ('Benign', 'BENIGN')]
    if not items:
        return '—'
    return '; '.join(f'{a} {_n(b)}' for a, b in items[:k]) + (' …' if len(items) > k else '')


REGION_ROWS = ['train', 'ref_pp_orig', 'warm05', 'warm10', 'warm20', 'eval_after10', 'eval_after20', 'exec']
REGION_LABEL = {'train': 'Entrenamiento [0,200k)', 'ref_pp_orig': 'Referencia post-proceso orig.', 'warm05': 'Warm-up 5 %',
                'warm10': 'Warm-up 10 %', 'warm20': 'Warm-up 20 %', 'eval_after10': 'Evaluación tras 10 %',
                'eval_after20': 'Evaluación tras 20 %', 'exec': 'Ejecución completa'}


def region_table(r, pipeline):
    hn = next(iter((r['regions']['exec'].get('dup_with_train') or {}).keys()), None)
    lines = ['| Región | [inicio, fin) | N | % mal. | 1.er mal. (abs / rel) | Tipos (top 3) | % vent. L=800 con algún mal. | '
             + ('Copias / originales de copias de entrenamiento | ' if pipeline == 'orig' else 'Filas no finitas | ')
             + f'Dup. exactos con entrenamiento (n / ataque{" / naturales" if hn else ""}) | Dup. con warm-up 10 % |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for k in REGION_ROWS:
        if k not in r['regions']:
            continue
        g = r['regions'][k]
        if g['n'] == 0:
            lines.append(f"| {REGION_LABEL[k]} | [{_n(g['start'])}, {_n(g['end'])}) | 0 | — | — | — | — | — | — | — |")
            continue
        cp = (f"{_n(g.get('n_copies'))} / {_n(g.get('n_originals_of_train_copies'))}" if pipeline == 'orig'
              else _n(g.get('rows_with_nonfinite_features')))
        dd = (g.get('dup_with_train') or {}).get(hn) if hn else None
        dtxt = f"{_n(dd['n'])} / {_n(dd['n_attack'])} / {_n(dd['n_natural'])}" if dd else 'no calculado'
        wtxt = f"{_n(dd.get('n_dup_warm10'))}" if dd and 'n_dup_warm10' in dd else '—'
        fm = '—' if g.get('first_mal_abs') is None else f"{_n(g['first_mal_abs'])} / {_n(g['first_mal_rel'])}"
        lines.append(f"| {REGION_LABEL[k]} | [{_n(g['start'])}, {_n(g['end'])}) | {_n(g['n'])} | {_p(g.get('pct_mal'))} | {fm} | "
                     f"{_types_short(g.get('types'))} | {_p(g.get('pct_windows_any_mal'), 1)} | {cp} | {dtxt} | {wtxt} |")
    return '\n'.join(lines)


def write_outputs(out_dir, C, args):
    jdir = Path(out_dir) / 'json'
    results = []
    for p in sorted(jdir.glob('*.json')):
        try:
            results.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            log.warning('unreadable %s', p)
    order = ['cic2017', 'cic2018', 'aci'] + [ds_key('knad', a) for a in KNAD_ATTACKS]
    results.sort(key=lambda r: order.index(r['key']) if r['key'] in order else 99)
    cuts = {'_meta': {'generated': now_iso(), 'script': str(Path(__file__).resolve()), 'script_sha256': script_sha(),
                      'decision': 'cortes del stream corregido, por variante',
                      'regla_11': 'solo etiquetas, timestamps, hashes y recuentos; ninguna métrica de rendimiento',
                      'coordenadas': 'intervalo_ordenado/inicio_ordenado: índice en el stream ordenado tras exclusiones; '
                                     '*_modelo: índice en el stream que ve el modelo (entrenamiento [0,train) + ejecución)',
                      'mode': sorted({r['mode'] for r in results}), 'stages': {r['key']: r['stages'] for r in results},
                      'constants': C.as_dict()}}
    rows_csv = []
    for r in results:
        entry = {'variantes': {}, 'errores': r.get('errors', {})}
        for v, cr in (r.get('corrected') or {}).items():
            entry['variantes'][v] = dict(cr['stream_cuts'], stream_index_file=cr.get('stream_index_file'))
            if cr['spec'].get('default'):
                entry['variante_por_defecto_D7'] = v
        if r['dataset'] == 'aci':
            entry['estado'] = 'la variante literal deja la evaluación casi sin ataques'
        elif r['dataset'] == 'cic2018':
            entry['estado'] = 'variantes de 7 y de 10 días; reloj de 12 h pasado a 24 h'
        else:
            entry['estado'] = 'cortes auditados'
        cuts[r['key']] = entry
        for pipe, blocks in (('orig', {'orig': r['orig']} if r.get('orig') else {}), ('corrected', r.get('corrected') or {})):
            for v, rr in blocks.items():
                for reg, g in rr['regions'].items():
                    row = {'dataset': r['key'], 'pipeline': pipe, 'variant': v, 'region': reg}
                    for k2, v2 in g.items():
                        if isinstance(v2, dict):
                            if k2 == 'dup_with_train':
                                for hn, dd in v2.items():
                                    for k3, v3 in dd.items():
                                        row[f'dup_{hn}_{k3}'] = v3
                            elif k2 == 'types':
                                row['types'] = json.dumps(v2, ensure_ascii=False)
                            continue
                        row[k2] = v2
                    rows_csv.append(row)
    dump_json(Path(out_dir) / 'stream_cuts.json', cuts)
    if rows_csv:
        df = pd.DataFrame(rows_csv)
        tmp = Path(out_dir) / f'.warmup_regions.csv.tmp{os.getpid()}'
        df.to_csv(tmp, index=False)
        os.replace(tmp, Path(out_dir) / 'warmup_regions.csv')
    md = [f'# Tablas T0 (generadas por `warmup_audit.py`, {now_iso()})', '',
          'Solo etiquetas, timestamps, hashes y recuentos. No editar a mano: se regeneran.', '',
          f"Constantes: {json.dumps(C.as_dict())}. Modo: {', '.join(sorted({r['mode'] for r in results}))}.", '']
    md += ['## Resumen (una fila por dataset y pipeline)', '',
           '| Dataset | Pipeline / variante | N stream | % mal. total | % mal. entrenamiento | % mal. warm-up 5/10/20 % | 1.er mal. en ejecución (rel.) | % mal. evaluación tras 10 % | Dup. exactos entrenamiento vs ejecución | Pasos negativos de timestamp |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        blocks = []
        if r.get('orig'):
            blocks.append(('orig', r['orig']))
        for v, cr in (r.get('corrected') or {}).items():
            blocks.append((f'corregido / {v}', cr))
        for name, rr in blocks:
            R = rr['regions']
            s = rr['stream']
            hn = next(iter((R['exec'].get('dup_with_train') or {}).keys()), None)
            dd = R['exec']['dup_with_train'][hn]['n'] if hn else None
            ts = s.get('timestamps') or {}
            md.append(f"| {r['key']} | {name} | {_n(s['N'])} | {_p(s['pct_mal'])} | {_p(R['train'].get('pct_mal'))} | "
                      f"{' / '.join(_p(R[f'warm{p:02d}'].get('pct_mal')) for p in WARM_PCTS)} | {_n(R['exec'].get('first_mal_rel'))} | "
                      f"{_p(R['eval_after10'].get('pct_mal'))} | {_n(dd) if hn else 'no calculado'} | "
                      f"{_n(ts.get('n_neg_steps')) if ts else '—'} |")
    md.append('')
    for r in results:
        md += [f"## {r['key']}", '', f"Modo {r['mode']}; etapas {r['stages']}; {r.get('elapsed_s')} s. Errores: {r.get('errors') or 'ninguno'}.", '']
        if r.get('orig'):
            o = r['orig']
            md += ['### Pipeline original', '', f"Meta: `{json.dumps({k: v for k, v in o['meta'].items() if k != 'launch_all_sha256'}, ensure_ascii=False)}`", '',
                   f"Comprobaciones: `{json.dumps(o.get('checks'), ensure_ascii=False)}`", '', region_table(o, 'orig'), '']
            md += _stream_md(o['stream'])
        for v, cr in (r.get('corrected') or {}).items():
            md += [f'### Corregido / {v}' + (' (por defecto)' if cr['spec'].get('default') else ''), '',
                   f"Orden: {cr['spec']['order']}. Flags: {cr['meta'].get('flags')}.", '',
                   f"Cortes: `{json.dumps(cr['meta']['cuts'], ensure_ascii=False)}`", '',
                   f"Excluidas antes de ordenar: `{json.dumps(cr['meta']['excluded_before_sort'], ensure_ascii=False)}`", '',
                   region_table(cr, 'corrected'), '']
            md += _stream_md(cr['stream'])
            md += ['Texto a declarar (EN):', '', '> ' + cr['stream_cuts']['declarar'], '']
    write_atomic(Path(out_dir) / 'warmup_tables.md', '\n'.join(md))


def _stream_md(s):
    md = []
    ts = s.get('timestamps')
    if ts:
        md.append(f"Timestamps: válidos {_n(ts.get('n_valid'))}, ausentes {_n(ts.get('n_missing'))}, monótono={ts.get('monotonic_non_decreasing')}, "
                  f"pasos negativos {_n(ts.get('n_neg_steps'))} ({_p(ts.get('pct_neg_steps'))} %), máx. salto atrás {ts.get('max_backward_jump_s')} s, "
                  f"huecos > 10 min {_n(ts.get('n_forward_gaps_gt_600s'))}; rango {ts.get('min')} – {ts.get('max')}.")
        md.append('')
    md += ['| Bloque (fichero) | Posición [primera, última] | N | N mal. | Tramos contiguos | Copias |', '|---|---|---|---|---|---|']
    for b in s['blocks_by_file']:
        md.append(f"| {b['file']} | [{_n(b['first_pos'])}, {_n(b['last_pos'])}] | {_n(b['n'])} | {_n(b['n_mal'])} | {_n(b['n_runs'])} | {_n(b.get('n_copies'))} |")
    md.append('')
    if s.get('blocks_by_day') and len(s['blocks_by_day']) > 1:
        md += ['| Día (timestamp) | Posición [primera, última] | N | N mal. | Tramos contiguos |', '|---|---|---|---|---|']
        for b in s['blocks_by_day']:
            md.append(f"| {b['day']} | [{_n(b['first_pos'])}, {_n(b['last_pos'])}] | {_n(b['n'])} | {_n(b['n_mal'])} | {_n(b['n_runs'])} |")
        md.append('')
    if s.get('duplicates'):
        md.append(f"Duplicados en el stream: `{json.dumps(s['duplicates'])}`")
        md.append('')
    return md


def expand_datasets(spec):
    out = []
    for item in spec.split(','):
        item = item.strip()
        if item == 'all':
            out += [('cic2017', None), ('cic2018', None), ('aci', None)] + [('knad', a) for a in KNAD_ATTACKS]
        elif item == 'knad':
            out += [('knad', a) for a in KNAD_ATTACKS]
        elif item.startswith('knad:'):
            want = item[5:].lower().replace('_', ' ')
            m = [a for a in KNAD_ATTACKS if a.lower() == want]
            if not m:
                raise SystemExit(f'unknown KNAD attack {item}; choose from {KNAD_ATTACKS}')
            out.append(('knad', m[0]))
        elif item in ('cic2017', 'cic2018', 'aci'):
            out.append((item, None))
        else:
            raise SystemExit(f'unknown dataset {item}')
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='cic2017 | cic2018 | aci | knad | knad:<attack> | all (comma-separated allowed)')
    ap.add_argument('--pipeline', choices=['orig', 'corrected', 'both'], default='both')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cache-dir', default=str(DEFAULT_CACHE))
    ap.add_argument('--sample-rows', type=int, default=0, help='read only the first N data rows of every file (quick test)')
    ap.add_argument('--scale-div', type=int, default=None, help='divide FM/AE/copies/reference sizes (default 50 with --sample-rows, else 1)')
    ap.add_argument('--skip-hashes', action='store_true', help='labels/timestamps only (no exact-duplicate stage)')
    ap.add_argument('--skip-pcap', action='store_true', help='do not read KNAD pcap timestamps')
    ap.add_argument('--workers', type=int, default=3, help='processes for hashing/pcap stages')
    ap.add_argument('--log', default=None, help='append progress log to this file')
    ap.add_argument('--force', action='store_true', help='ignore DONE markers (caches are still reused)')
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    cache_base = Path(args.cache_dir)
    assert_writable(out_dir)
    assert_writable(cache_base)
    if args.log:
        assert_writable(args.log)
        os.environ['WARMUP_AUDIT_LOG'] = str(args.log)
    setup_logging(args.log)
    div = args.scale_div if args.scale_div is not None else (50 if args.sample_rows else 1)
    C = Consts(div)
    cache = cache_base / ('full' if not args.sample_rows else f'sample{args.sample_rows}')
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = expand_datasets(args.dataset)
    log.info('warmup_audit start: datasets=%s pipeline=%s sample=%s div=%s hashes=%s pcap=%s cache=%s out=%s',
             [ds_key(d, a) for d, a in datasets], args.pipeline, args.sample_rows, div, not args.skip_hashes,
             not args.skip_pcap, cache, out_dir)
    t_all = time.time()
    knad = [a for d, a in datasets if d == 'knad']
    todo = []
    for a in knad:
        if not _done(out_dir, ds_key('knad', a), args, C) or args.force:
            todo.append(('knad_prep', a, args.sample_rows, str(cache), not args.skip_pcap, not args.skip_hashes, bool(args.sample_rows)))
    run_jobs(todo, args.workers)
    ns_holder = {'ns': None}
    failures = 0
    for d, a in datasets:
        key = ds_key(d, a)
        if _done(out_dir, key, args, C) and not args.force:
            log.info('[%s] DONE marker present with same arguments: skipped', key)
            continue
        try:
            res = run_dataset(d, a, args, C, cache, out_dir, ns_holder)
            dump_json(out_dir / 'json' / f'{key}.json', res)
            if not res['errors']:
                write_atomic(out_dir / 'done' / f'{key}.DONE', json.dumps(_done_sig(key, args, C)))
            else:
                failures += 1
        except Exception as e:
            failures += 1
            log.error('[%s] failed: %s\n%s', key, e, traceback.format_exc())
    write_outputs(out_dir, C, args)
    log.info('warmup_audit end: %.1fs, datasets with errors: %d; outputs in %s', time.time() - t_all, failures, out_dir)
    return 1 if failures else 0


def _done_sig(key, args, C):
    return {'key': key, 'pipeline': args.pipeline, 'sample_rows': args.sample_rows, 'consts': C.as_dict(),
            'hashes': not args.skip_hashes, 'pcap': not args.skip_pcap, 'script_sha256': script_sha()}


def _done(out_dir, key, args, C):
    p = Path(out_dir) / 'done' / f'{key}.DONE'
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()) == _norm_json(_done_sig(key, args, C))
    except json.JSONDecodeError:
        return False


if __name__ == '__main__':
    sys.exit(main())
