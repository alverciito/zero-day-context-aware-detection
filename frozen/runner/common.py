#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""common.py - shared library of the job runner (standard library only): paths, constants, seeds, manifest, locks and commit."""
import contextlib
import csv
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path

MAIN_REPO = Path(os.environ.get("JISA_MAIN_REPO", "/home/acid137/Projects/review-zd"))
RUNNER_DIR = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = RUNNER_DIR.parents[1]
DATA_ROOT = MAIN_REPO / "data" / "datasets"
CACHE_ROOT = MAIN_REPO / "data" / "cache" / "streams"
TEST_ROOT_BASE = MAIN_REPO / "data" / "cache" / "_tests"
ORIG_RESULTS = MAIN_REPO / "data" / "original_results"
RESULTS_ROOT = MAIN_REPO / "results" / "revision"
LOGS_ROOT = MAIN_REPO / "logs"
MANIFEST = MAIN_REPO / "revision" / "experiments" / "manifest.csv"
FORBIDDEN_WRITE = ["/mnt/hdd8tb", str(DATA_ROOT), str(ORIG_RESULTS), str(MAIN_REPO / "data" / "remote_chiquitin")]

FREEZE = dt.datetime(2026, 10, 7, 12, 0, 0)
FREEZE_DAY_START = dt.datetime(2026, 10, 7, 0, 0, 0)
FREEZE_MARGIN = 1.2

OK, FAILED, INVALID, BLOCKED, ANOMALY, LOCKED, INTERRUPTED, DIRTY, ENV, OOM, HALT, DUPLICATE = range(12)
CODE_NAMES = {OK: "OK", FAILED: "FAILED", INVALID: "INVALID", BLOCKED: "BLOCKED", ANOMALY: "ANOMALY",
              LOCKED: "LOCKED", INTERRUPTED: "INTERRUPTED", DIRTY: "DIRTY", ENV: "ENV", OOM: "OOM",
              HALT: "HALT", DUPLICATE: "DUPLICATE"}
MAX_FAILS = 3

MANIFEST_COLUMNS = ["job_id", "block", "priority", "dataset", "variant", "ar", "paradigm", "pipeline",
                    "pipeline_version", "window", "k_clusters", "grouping", "partition_seed", "warmup_frac",
                    "contamination", "decontaminate", "seed", "rep", "save_ckpt", "timing_ref", "notes"]
GATES_MANIFEST_COLS = ["job_id", "block", "priority", "dataset", "variant", "paradigm", "window",
                       "k_clusters", "grouping", "partition_seed", "warmup_frac", "contamination",
                       "decontaminate", "seed", "save_ckpt", "timing_ref", "notes"]
FIXED_VARIANTS = ["mlp_ctx_fix", "mlp_fix", "mlp_ctx_fix_e5", "mlp_fix_e5"]
CONTEXT_VARIANTS = ["mlp_ctx", "mlp_deep_ctx"] + FIXED_VARIANTS
EPS_VARIANTS = ["original_eps"]
NUMPY_VARIANTS = ("original", "original_eps")
VARIANTS = ["conv1d", "conv2d", "lstm", "mha", "mlp", "original", "stat"] + CONTEXT_VARIANTS + EPS_VARIANTS
KERAS_VARIANTS = ["conv1d", "conv2d", "lstm", "mha", "mlp"] + CONTEXT_VARIANTS
WINDOWED_VARIANTS = ["conv1d", "conv2d", "lstm", "mha"] + CONTEXT_VARIANTS
FAKE_VARIANTS = ["_fake_ok", "_fake_fail", "_fake_nan", "_fake_kill", "_fake_count", "_fake_oom", "_fake_halt",
                 "_fake_scorefail", "_fake_ckptfail", "_fake_sleep", "_fake_dup", "_fake_gpukernel"]
KNAD_ATTACKS = {
    "Active_Wiretap": "Active Wiretap", "Fuzzing": "Fuzzing", "OS_Scan": "OS Scan",
    "SSDP_Flood": "SSDP Flood", "SSL_Renegotiation": "SSL Renegotiation", "Video_Injection": "Video Injection"}
OTHER_DATASETS = ["CIC2017", "CIC2018", "ACI-IOT-2023"]
PIPELINES = {"orig": ["v1"], "corrected": ["v1"]}
SEEDS = {"A": 1234, "B": 5678, "C": 9012}
KNOWN_BLOCKS_V1 = ["X2", "S-w1", "S-w2", "S-w4", "S-L2000", "T1-A-KNAD"]
KNOWN_BLOCKS = ["T8-EPS", "T8-EP5", "T8-EP5B"] + KNOWN_BLOCKS_V1 + [f"T1-{s}-{c}" for s in "ABC" for c in ("KNAD", "ACI", "CIC2017", "CIC2018")
                                  if f"T1-{s}-{c}" != "T1-A-KNAD"] + [
    "T2-min", "T2-ext", "T2-B", "T3-min-a", "T3-min-b", "T3-min-c", "T3-ext-a", "T3-ext-b", "T4-E"] + [
    "T7-MLPW", "T7-MLPDEEP",
    "T8-CTX", "T8-BASE"]
BLOCK_TEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

CORRECTED_NOT_IMPLEMENTED = (
    "pipeline 'corrected' para {ds} no implementado: D-7 exige {req}, sin copias grant_first_label, tramo de "
    "entrenamiento contiguo, anterior y excluido de la evaluacion, preprocesado ajustado solo con entrenamiento y "
    "sin Flow ID/IPs; los cortes salen de la auditoria T0 (revision/analysis/t0/stream_cuts.json).")
CORRECTED_REQ = {"CIC2017": "el lunes como preludio y orden cronologico de los dias",
                 "CIC2018": "orden por Timestamp y tramo benigno inicial filtrado y declarado",
                 "ACI-IOT-2023": "orden por Timestamp y tramo benigno inicial filtrado y declarado"}
CORRECTED_SUPPORTED_KINDS = ("knad", "cic2017", "cic2018", "aci")
CORRECTED_VARIANT = {"CIC2017": "mon_first", "CIC2018": "days7", "ACI-IOT-2023": "oct31_train_chrono"}
CORRECTED_TRAIN_ROWS = {"KNAD": 200_000, "CIC2017": 200_000, "CIC2018": 200_000, "ACI-IOT-2023": 72_264}
CORRECTED_EXTRA_DROP = {"ACI-IOT-2023": ["Src Port"]}
CORRECTED_CATEGORICAL = {"ACI-IOT-2023": ["Connection Type"]}
CORRECTED_FORBIDDEN_COLUMNS = ("Flow ID", "Src IP", "Dst IP", "Source IP", "Destination IP", "Timestamp")
CORRECTED_EXPECTED = {
    "CIC2017": {"n_rows": 2_830_743, "train_end": 200_000, "first_attack_index": 541_265, "n_attack_train": 0,
                "discarded_label_filter": 0, "guard_ties_dropped": 0, "excluded_before_sort": 0,
                "n_attack": 557_646, "dim": 78},
    "CIC2018": {"n_rows": 7_340_007, "train_end": 200_000, "first_attack_index": 231_481, "n_attack_train": 0,
                "discarded_label_filter": 0, "guard_ties_dropped": 3, "excluded_before_sort": 15,
                "n_attack": 2_010_110, "dim": 78},
    "ACI-IOT-2023": {"n_rows": 554_116, "train_end": 72_264, "first_attack_index": 72_403, "n_attack_train": 0,
                     "discarded_label_filter": 0, "guard_ties_dropped": 0, "excluded_before_sort": 677_295,
                     "n_attack": 311_845, "dim": 86},
}
CORRECTED_DUP_REFERENCE = {
    "CIC2017": {"n": 7_678, "n_attack": 351, "calib10": 4_693, "source": "t0/stream_cuts.json + plan_support/label_counts.json"},
    "CIC2018": {"n": 384_460, "n_attack": 1, "calib10": 45_563, "source": "t0/stream_cuts.json + plan_support/label_counts.json"},
    "ACI-IOT-2023": {"n": 0, "n_attack": 0, "calib10": 0,
                     "source": "runner v2 pipelines.py --summary 2026-09-16 sin Src Port; T0 con Src Port: 0"},
}
CALIB_FRACS = (5, 10, 20)
PRIMARY_CALIB_PCT = 10
CORRECTED_AE_DEVIATIONS = {"ACI-IOT-2023": "entrenamiento de ACI = 31-oct completo (FM 50.000 + AE 22.264)"}
LABEL_COUNTS_REL = "revision/analysis/plan_support/label_counts.json"


class InvalidJob(Exception):
    """Fila no valida o configuracion no soportada -> INVALID (no suma fallo)."""


class EnvError(Exception):
    """Entorno no apto (GPU, git, rutas, LD_LIBRARY_PATH, precondiciones) -> ENV (no suma fallo)."""


class HaltError(Exception):
    """Error sistematico que afectaria a otros jobs (stream, cache, config congelada) -> HALT: no suma fallo y
    el worker se detiene (DESIGN §3, S1)."""


class PipelineError(HaltError):
    pass


class CacheMismatch(HaltError):
    pass


class CacheIntegrityError(HaltError):
    pass


class FrozenConfigError(HaltError):
    pass


class NonExecutableCell(InvalidJob):
    """Celda que el codigo original no puede completar de forma determinista (p. ej. el hot-fix de dbscan.py:58-63 no
    termina o hace pop de una lista vacia con el k ablacionado). Se detecta ANTES de que el original se cuelgue: run_job
    lo registra como .state/X/nonexec_N.json y sale INVALID sin sumar fallo; el worker no lo vuelve a reclamar (celda
    perdida, analysis_plan.md §12.2)."""


class Interrupted(BaseException):
    pass


def now():
    """Hora actual; JISA_NOW='YYYY-mm-dd HH:MM' la simula (solo tests)."""
    v = os.environ.get("JISA_NOW")
    return dt.datetime.strptime(v, "%Y-%m-%d %H:%M") if v else dt.datetime.now()


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def ts_compact():
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def hostname():
    return socket.gethostname()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json_atomic(path, obj):
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    try:
        import numpy as np
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def fsync_dir(path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def fsync_tree(path):
    """fsync every regular file and directory under `path` before DONE is written; returns the number of files."""
    n = 0
    for root, dirs, files in os.walk(str(path)):
        for f in files:
            p = os.path.join(root, f)
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
                n += 1
            finally:
                os.close(fd)
        fsync_dir(root)
    return n


def parse_bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes"):
        return True
    if s in ("0", "false", "f", "no", ""):
        return False
    raise ValueError(f"booleano no valido: {v!r}")


def realpath(p):
    return os.path.realpath(str(p))


def is_under(path, root):
    rp, rr = realpath(path), realpath(root)
    return rp == rr or rp.startswith(rr.rstrip("/") + "/")


def check_not_forbidden(path, what="ruta"):
    for f in FORBIDDEN_WRITE:
        if is_under(path, f):
            raise EnvError(f"{what} {path} esta bajo una ruta prohibida para escribir ({f})")


def tail_text(path, n_bytes=4000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n_bytes))
            return fh.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


OOM_RE = re.compile(r"ResourceExhausted|OOM when allocating|CUDA_ERROR_OUT_OF_MEMORY|out of memory|failed to allocate",
                    re.I)


def looks_like_oom(exc_or_text):
    if isinstance(exc_or_text, BaseException):
        names = [c.__name__ for c in type(exc_or_text).__mro__]
        if any("ResourceExhausted" in n for n in names):
            return True
        exc_or_text = f"{type(exc_or_text).__name__}: {exc_or_text}"
    return bool(OOM_RE.search(str(exc_or_text or "")))


GPU_KERNEL_RE = re.compile(r"CUDA_ERROR_(?:INVALID_IMAGE|NO_BINARY_FOR_GPU|INVALID_PTX|UNSUPPORTED_PTX_VERSION)|"
                           r"cuLaunchKernel|cuModuleLoadData")


def oom_kill_counters(test_file=None):
    """Counters of the kernel OOM killer (/proc/vmstat and the cgroup v2 memory.events of this process); None if unreadable."""
    if test_file:
        try:
            return {"test_file": int(Path(test_file).read_text().strip() or 0)}
        except (OSError, ValueError):
            return {"test_file": 0}
    out = {"vmstat": None, "cgroup": None}
    try:
        for ln in Path("/proc/vmstat").read_text().splitlines():
            if ln.startswith("oom_kill "):
                out["vmstat"] = int(ln.split()[1])
    except (OSError, ValueError):
        pass
    try:
        rel = [ln.split(":", 2)[2] for ln in Path("/proc/self/cgroup").read_text().splitlines() if ln.startswith("0::")]
        if rel:
            out["cgroup_path"] = rel[0]
            for ln in (Path("/sys/fs/cgroup") / rel[0].lstrip("/") / "memory.events").read_text().splitlines():
                if ln.startswith("oom_kill "):
                    out["cgroup"] = int(ln.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return out


def oom_kill_delta(before, after):
    """{contador: incremento} de los contadores que subieron entre dos lecturas de oom_kill_counters."""
    return {k: after[k] - before[k] for k in ("vmstat", "cgroup", "test_file")
            if isinstance((before or {}).get(k), int) and isinstance((after or {}).get(k), int) and after[k] > before[k]}


def looks_like_gpu_kernel_error(exc_or_text):
    if isinstance(exc_or_text, BaseException):
        exc_or_text = f"{type(exc_or_text).__name__}: {exc_or_text}"
    return bool(GPU_KERNEL_RE.search(str(exc_or_text or "")))


def proc_start_ticks(pid):
    """Campo 22 de /proc/<pid>/stat (arranque en ticks): distingue un PID reutilizado."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        rest = raw[raw.rindex(")") + 2:].split()
        return int(rest[19])
    except Exception:
        return None


def proc_cmdline(pid):
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def pid_alive(pid, start_ticks=None, needle=None):
    """Vivo = /proc/<pid> existe, el arranque coincide (si se conoce) y la linea de orden contiene `needle`."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if not Path(f"/proc/{pid}").exists():
        return False
    if start_ticks is not None:
        st = proc_start_ticks(pid)
        if st is not None and int(st) != int(start_ticks):
            return False
    if needle and needle not in proc_cmdline(pid):
        return False
    return True


def other_live_workers(gpu, exclude_pid=None):
    res = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == (exclude_pid or -1):
            continue
        c = proc_cmdline(p.name)
        if "worker.py" in c and "revision/experiments" in c and re.search(rf"--gpu[ =]{re.escape(str(gpu))}(\s|$)", c):
            res.append(int(p.name))
    return res


def gpu_compute_apps():
    """nvidia-smi --query-compute-apps (solo lectura). Devuelve lista o texto de error."""
    if shutil.which("nvidia-smi") is None:
        return {"error": "nvidia-smi no disponible"}
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        apps = []
        for ln in p.stdout.strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                apps.append({"pid": int(parts[0]), "name": parts[1], "used_mib": parts[2],
                             "cmdline": proc_cmdline(parts[0])[:200]})
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return {"apps": apps, "gpus": q.stdout.strip().splitlines()}
    except Exception as e:
        return {"error": str(e)[:200]}


def ds_slug(dataset):
    if dataset.upper().startswith("KNAD:"):
        return "knad-" + dataset.split(":", 1)[1].lower()
    return dataset.lower()


def derive_job_id(row):
    """{block}__{ds}__{variant}__ar{T|F}__{pipeline}.{version}__L{window}__k{k}[__g{grouping}-p{partition_seed}]
    [__c{contamination}]__s{seed}__r{rep}. Los segmentos entre corchetes solo aparecen con valores distintos de los
    neutros (grouping=dbscan, contamination=0), asi que toda fila v1 conserva su job_id (analysis_plan.md §7.3:
    partition_seed en el job_id; §8.5: c distingue las filas de T4-E)."""
    ar = parse_bool(row["ar"])
    jid = (f"{row['block']}__{ds_slug(row['dataset'])}__{row['variant']}__ar{'T' if ar else 'F'}__"
           f"{row['pipeline']}.{row['pipeline_version']}__L{row['window']}__k{row['k_clusters']}")
    grouping = str(row.get("grouping", "dbscan")).strip()
    if grouping != "dbscan":
        jid += f"__g{grouping}-p{row['partition_seed']}"
    contamination = str(row.get("contamination", "0")).strip()
    if contamination not in ("0", ""):
        jid += f"__c{contamination}"
    return jid + f"__s{row['seed']}__r{row['rep']}"


def read_manifest(path, retries=3):
    last = None
    for _ in range(retries):
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                header = [h.strip() for h in (rdr.fieldnames or [])]
                missing = [c for c in MANIFEST_COLUMNS if c not in header]
                if missing:
                    raise ValueError(f"manifiesto sin columnas {missing}")
                return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rdr
                        if any((v or "").strip() for v in r.values())]
        except (OSError, ValueError, csv.Error) as e:
            last = e
            time.sleep(1)
    raise last


def manifest_row(path, job_id):
    for r in read_manifest(path):
        if r.get("job_id") == job_id:
            return r
    return None


def block_family(block):
    """Family of a block; T7 and T8 are families of their own, outside the confirmatory analyses."""
    m = re.match(r"^(X2|T[1-8]|S|B)(?![0-9A-Za-z])", block or "")
    return m.group(1) if m else (block or "?")


def block_selected(block, selected):
    if not selected:
        return True
    return any(block == b or block.startswith(b + "-") for b in selected)


class Spec:
    """Fila del manifiesto validada."""

    def __init__(self, row, test_mode):
        self.row = dict(row)
        self.job_id = row["job_id"]
        self.block = row["block"]
        self.family = block_family(self.block)
        self.priority = int(row["priority"])
        self.dataset = row["dataset"]
        self.variant = row["variant"]
        self.ar = parse_bool(row["ar"])
        self.pipeline = row["pipeline"]
        self.pipeline_version = row["pipeline_version"]
        self.window = int(row["window"])
        self.k = int(row["k_clusters"])
        self.grouping = row["grouping"]
        self.partition_seed = None if row["partition_seed"] in ("NA", "") else int(row["partition_seed"])
        self.contamination_pct = int(row["contamination"])
        self.seed = None if row["seed"] in ("None", "") else row["seed"]
        self.seed_int = SEEDS.get(self.seed) if self.seed else None
        self.rep = int(row["rep"])
        self.save_ckpt = parse_bool(row["save_ckpt"])
        self.timing_ref = parse_bool(row["timing_ref"])
        self.notes = row.get("notes", "")
        self.test_mode = test_mode
        if self.dataset.upper().startswith("KNAD:"):
            self.kind = "knad"
            self.attack = self.dataset.split(":", 1)[1]
            self.attack_dir = KNAD_ATTACKS.get(self.attack)
        else:
            self.kind = {"CIC2017": "cic2017", "CIC2018": "cic2018", "ACI-IOT-2023": "aci"}.get(self.dataset)
            self.attack = self.attack_dir = None
        self.is_fake = self.variant in FAKE_VARIANTS
        self.mode = "fake" if self.is_fake else ("probe" if self.family == "S" else "full")

    def public(self):
        return {k: v for k, v in self.__dict__.items() if k != "row"}

    def ablation(self):
        """Factor ablacionado de la fila (None fuera de T2-T4) con su valor y la referencia de T1-A (plan §7.1)."""
        rule = ABLATION_BLOCKS.get(self.block)
        if not rule:
            return None
        f = rule["factor"]
        val = {"window": self.window, "k": self.k, "contamination": self.contamination_pct,
               "grouping": {"grouping": self.grouping, "partition_seed": self.partition_seed}}[f]
        return {"block": self.block, "factor": f, "value": val, "reference": ABLATION_REFERENCE[f],
                "reference_block": ablation_reference_block(self.block, self.dataset),
                "plan": rule["plan"], "only_this_factor_changes": True}


def ablation_reference_block(block, dataset):
    """Bloque T1 de la celda de referencia de una ablacion: la MISMA semilla que el bloque (§7.1: T1-A con semilla A;
    T2-B, con semilla B, se compara con T1-B) y el corpus del dataset. None fuera de T2-T4."""
    rule = ABLATION_BLOCKS.get(block)
    if not rule:
        return None
    ds = str(dataset)
    corpus = "KNAD" if ds.upper().startswith("KNAD:") else {"ACI-IOT-2023": "ACI"}.get(ds, ds)
    return f"T1-{rule['seed']}-{corpus}"


T1_BLOCK_RE = re.compile(r"^T1-([ABC])-(KNAD|ACI|CIC2017|CIC2018)$")
T1_CORPUS_DATASETS = {"ACI": ["ACI-IOT-2023"], "CIC2017": ["CIC2017"], "CIC2018": ["CIC2018"]}
T2_WINDOWS = (200, 2000)
T3_KS = (1, 2, 8)
T3_PARTITION_SEEDS = (101, 202, 303)
T4E_CONTAMINATION_PCT = (5, 20)
ABLATION_DATASETS = ("KNAD:Active_Wiretap", "CIC2017", "KNAD:OS_Scan")
ABLATION_REFERENCE = {"window": 800, "k": 4, "grouping": {"grouping": "dbscan", "partition_seed": None},
                      "contamination": 0}
ABLATION_BLOCKS = {
    "T2-min": {"factor": "window", "seed": "A", "variants": (("lstm", True), ("mha", True)), "plan": "§7.2 T2-min"},
    "T2-ext": {"factor": "window", "seed": "A", "variants": (("lstm", False), ("mha", False)), "plan": "§7.2 T2-ext"},
    "T2-B": {"factor": "window", "seed": "B", "variants": (("lstm", True), ("mha", True)), "plan": "§7.2 T2-B"},
    "T3-min-a": {"factor": "k", "seed": "A", "variants": (("original", True), ("stat", True), ("mlp", True)),
                 "all_datasets": True, "plan": "§7.3 T3-min-a"},
    "T3-min-b": {"factor": "grouping", "seed": "A", "variants": (("original", True), ("stat", True), ("mlp", True)),
                 "plan": "§7.3 T3-min-b"},
    "T3-min-c": {"factor": "k", "seed": "A", "variants": (("lstm", True),), "plan": "§7.3 T3-min-c"},
    "T3-ext-a": {"factor": "grouping", "seed": "A", "variants": (("lstm", True),), "plan": "§7.3 T3-ext-a"},
    "T3-ext-b": {"factor": "k", "seed": "A", "variants": (("mha", True),), "plan": "§7.3 T3-ext-b"},
    "T4-E": {"factor": "contamination", "seed": "A", "variants": (("original", True), ("mlp", True), ("lstm", True)),
             "datasets": ("KNAD:Active_Wiretap",), "plan": "§8.5 T4-E"},
}

ALL9_DATASETS = [f"KNAD:{a}" for a in sorted(KNAD_ATTACKS)] + list(OTHER_DATASETS)
CTX_BLOCKS = {
    "T7-MLPW": {"variant": "mlp_ctx", "seed": "A", "datasets": None,
                "paradigms": (True, False), "decision": "MLP con ventana",
                "note": "el MLP publicado con la ventana aplanada (un unico factor: la entrada)"},
    "T7-MLPDEEP": {"variant": "mlp_deep_ctx", "seed": "A", "datasets": ("KNAD:Active_Wiretap", "CIC2017"),
                   "paradigms": (True, False), "decision": "Deep MLP con ventana",
                   "note": "la Deep MLP de la Tabla 3 con ventana (deep_mlp_ae.py)"},
}
CTX_BLOCKS.update({
    "T8-CTX": {"variant": "mlp_ctx_fix", "seed": None, "datasets": None,
               "paradigms": (True, False), "decision": "par re-entrenado",
               "note": "el MLP con ventana con la regla Pi de lr por capa; unico factor cambiado frente a T7-MLPW"},
    "T8-BASE": {"variant": "mlp_fix", "seed": None, "datasets": None,
                "paradigms": (True, False), "decision": "control del protocolo",
                "note": "el MLP POR REGISTRO bajo el mismo protocolo; la regla Pi lo deja invariante, asi que el "
                        "contraste con T8-CTX aisla la ventana"},
})
CTX_BLOCKS.update({
    "T8-EPS": {"variant": "original_eps", "seed": "A", "datasets": None,
               "paradigms": (True,),
               "decision": "sensibilidad del estabilizador",
               "note": "el dA del envio con eps 1e-16 en las dos fases, como el dA de Mirsky y como las otras "
                       "seis redes; unico factor cambiado frente a C10"},
})
CTX_BLOCKS.update({
    "T8-EP5": {"variant": "mlp_ctx_fix_e5", "seed": "A",
               "datasets": ("ACI-IOT-2023", "CIC2017", "CIC2018",
                            "KNAD:Active_Wiretap", "KNAD:Fuzzing", "KNAD:OS_Scan",
                            "KNAD:SSDP_Flood", "KNAD:SSL_Renegotiation",
                            "KNAD:Video_Injection"),
               "paradigms": (False,), "decision": "ablacion de epocas",
               "note": "cinco epocas sobre el par ya inicializado por contencion; mide si el empate del par "
                       "re-entrenado es falta de presupuesto de entrenamiento y no falta de capacidad"},
    "T8-EP5B": {"variant": "mlp_fix_e5", "seed": "A",
                "datasets": ("ACI-IOT-2023", "CIC2017", "CIC2018",
                             "KNAD:Active_Wiretap", "KNAD:Fuzzing", "KNAD:OS_Scan",
                             "KNAD:SSDP_Flood", "KNAD:SSL_Renegotiation",
                             "KNAD:Video_Injection"),
                "paradigms": (False,), "decision": "control de la ablacion de epocas",
                "note": "el control sin ventana con el mismo presupuesto, para que el contraste siga aislando "
                        "la ventana y no el numero de pasos"},
})
CTX_VARIANT_BLOCK = {r["variant"]: b for b, r in CTX_BLOCKS.items()}
EXPLORATORY_FAMILIES = ("T7", "T8")
EXCLUSIVE_VARIANTS = {"mlp_deep_ctx": 1}


def validate_row(row, test_mode=False):
    """Devuelve Spec o lanza InvalidJob. No abre datos ni importa TF."""
    if row is None:
        raise InvalidJob("el job_id no esta en el manifiesto")
    miss = [c for c in MANIFEST_COLUMNS if c not in row]
    if miss:
        raise InvalidJob(f"faltan columnas {miss}")
    try:
        jid = derive_job_id(row)
    except Exception as e:
        raise InvalidJob(f"job_id no derivable: {e}")
    if row["job_id"] != jid:
        raise InvalidJob(f"job_id {row['job_id']!r} != derivado {jid!r} (edicion a mano del manifiesto)")
    if not JOB_ID_RE.match(jid):
        raise InvalidJob(f"job_id con caracteres no permitidos: {jid!r}")
    block = row["block"]
    if block not in KNOWN_BLOCKS and not (test_mode and BLOCK_TEST_RE.match(block)):
        raise InvalidJob(f"bloque desconocido {block!r} (conocidos: {KNOWN_BLOCKS})")
    try:
        s = Spec(row, test_mode)
    except (ValueError, KeyError) as e:
        raise InvalidJob(f"valor no valido: {e}")
    for col in ("window", "k_clusters", "contamination") + (("partition_seed",) if row["partition_seed"] != "NA" else ()):
        if str(row[col]) != str(int(row[col])):
            raise InvalidJob(f"{col}={row[col]!r} no canonico (se espera {int(row[col])!r}; el job_id depende de el)")
    if s.kind is None or (s.kind == "knad" and s.attack_dir is None):
        raise InvalidJob(f"dataset desconocido o no copiado: {s.dataset!r} (KNAD: {sorted(KNAD_ATTACKS)})")
    if s.variant not in VARIANTS:
        if s.variant in FAKE_VARIANTS and not test_mode:
            raise InvalidJob(f"variante de test {s.variant!r} sin --test-mode")
        if s.variant not in FAKE_VARIANTS:
            raise InvalidJob(f"variante desconocida {s.variant!r}")
    if row["paradigm"] != ("AR" if s.ar else "TSR"):
        raise InvalidJob(f"paradigm {row['paradigm']!r} no casa con ar={s.ar}")
    if s.pipeline not in PIPELINES or s.pipeline_version not in PIPELINES[s.pipeline]:
        raise InvalidJob(f"pipeline {s.pipeline}.{s.pipeline_version} no soportado")
    if s.seed is not None and s.seed not in SEEDS:
        raise InvalidJob(f"semilla {s.seed!r} no declarada ({sorted(SEEDS)} o None)")
    if s.rep < 1:
        raise InvalidJob("rep >= 1")
    for col, val in (("warmup_frac", "NA"), ("decontaminate", "0")):
        if row[col] != val:
            raise InvalidJob(f"{col}={row[col]!r}: solo {val!r}. La fraccion de warm-up (T4-P-c) y la descontaminacion "
                             f"(T4-P-a) son post-proceso sobre scores.npy (analysis_plan.md §8.2, §8.4), no un job")
    rule = ABLATION_BLOCKS.get(block)
    factor = rule["factor"] if rule else None
    if s.family == "S":
        if s.save_ckpt:
            raise InvalidJob("la sonda S no produce scores ni checkpoints (save_ckpt=0)")
        if s.variant not in KERAS_VARIANTS and not s.is_fake:
            raise InvalidJob("la sonda S solo mide variantes Keras")
    elif factor == "window":
        if s.window not in T2_WINDOWS:
            raise InvalidJob(f"window={s.window}: {block} solo admite L en {list(T2_WINDOWS)} (analysis_plan.md §7.2; "
                             f"L = 800 es la referencia de T1-A)")
    elif s.window != 800 and not s.is_fake:
        raise InvalidJob(f"window={s.window}: fuera de S y T2 exige 800 (CONFIG['sequence_length'])")
    if factor == "k":
        if s.k not in T3_KS or s.grouping != "dbscan" or row["partition_seed"] != "NA":
            raise InvalidJob(f"{block}: k_clusters en {list(T3_KS)} con grouping=dbscan y partition_seed=NA "
                             f"(analysis_plan.md §7.3); fila k={s.k} grouping={s.grouping} "
                             f"partition_seed={row['partition_seed']}")
    elif factor == "grouping":
        if s.k != 4 or s.grouping != "random" or s.partition_seed not in T3_PARTITION_SEEDS:
            raise InvalidJob(f"{block}: grouping=random con k_clusters=4 y partition_seed en {list(T3_PARTITION_SEEDS)} "
                             f"(analysis_plan.md §7.3); fila k={s.k} grouping={s.grouping} "
                             f"partition_seed={row['partition_seed']}")
    else:
        if s.k != 4 or s.grouping != "dbscan":
            raise InvalidJob(f"k_clusters={s.k}, grouping={s.grouping}: fuera de T3 solo k_clusters=4 y "
                             "grouping=dbscan (CONFIG congelado)")
        if row["partition_seed"] != "NA":
            raise InvalidJob(f"partition_seed={row['partition_seed']!r}: solo en T3-min-b y T3-ext-a (analysis_plan.md §7.3)")
    if factor == "contamination":
        if s.contamination_pct not in T4E_CONTAMINATION_PCT:
            raise InvalidJob(f"contamination={row['contamination']!r}: T4-E solo admite c en {list(T4E_CONTAMINATION_PCT)} "
                             f"(% de las filas AE [fm, T); analysis_plan.md §8.5)")
    elif row["contamination"] != "0":
        raise InvalidJob(f"contamination={row['contamination']!r}: solo T4-E contamina el entrenamiento (analysis_plan.md "
                         f"§8.5); la contaminacion de la calibracion (T4-P-b) es post-proceso")
    if rule and not s.is_fake:
        if (s.variant, s.ar) not in rule["variants"]:
            raise InvalidJob(f"{block}: variante {s.variant} ar={s.ar} fuera de {list(rule['variants'])} ({rule['plan']})")
        if s.seed != rule["seed"]:
            raise InvalidJob(f"{block}: semilla {s.seed} != {rule['seed']} ({rule['plan']}; §7.1)")
        allowed = rule.get("datasets") or (None if rule.get("all_datasets") else ABLATION_DATASETS)
        if allowed and s.dataset not in allowed and not test_mode:
            raise InvalidJob(f"{block}: dataset {s.dataset} fuera de {list(allowed)} ({rule['plan']}; §7.1)")
    ctx = CTX_BLOCKS.get(block)
    if s.variant in (CONTEXT_VARIANTS + EPS_VARIANTS) and not s.is_fake:
        want = CTX_VARIANT_BLOCK[s.variant]
        if block != want and not test_mode:
            raise InvalidJob(f"variante {s.variant!r} solo en el bloque {want} ([EXPLORATORIO]); fila en {block}")
    if ctx and not s.is_fake:
        if s.variant != ctx["variant"]:
            raise InvalidJob(f"{block}: variante {s.variant} != {ctx['variant']} ({ctx['decision']})")
        if ctx["seed"] is not None and s.seed != ctx["seed"]:
            raise InvalidJob(f"{block}: semilla {s.seed} != {ctx['seed']} ({ctx['decision']})")
        if ctx["seed"] is None and s.seed not in SEEDS:
            raise InvalidJob(f"{block}: semilla {s.seed!r} no declarada ({sorted(SEEDS)}) ({ctx['decision']})")
        if s.ar not in ctx["paradigms"]:
            raise InvalidJob(f"{block}: paradigma ar={s.ar} fuera de {list(ctx['paradigms'])} ({ctx['decision']})")
        allowed = ctx["datasets"] or ALL9_DATASETS
        if s.dataset not in allowed and not test_mode:
            raise InvalidJob(f"{block}: dataset {s.dataset} fuera de {list(allowed)} ({ctx['decision']})")
    m_t1 = T1_BLOCK_RE.match(block)
    if m_t1 and not s.is_fake:
        letter, corpus = m_t1.groups()
        if s.seed != letter:
            raise InvalidJob(f"{block}: semilla {s.seed} != {letter} (analysis_plan.md §4, §13.2)")
        ok_ds = s.kind == "knad" if corpus == "KNAD" else s.dataset in T1_CORPUS_DATASETS[corpus]
        if not ok_ds:
            raise InvalidJob(f"{block}: dataset {s.dataset} no pertenece al corpus {corpus}")
        if letter != "A" and s.variant in ("original", "stat"):
            raise InvalidJob(f"{block}: original y stat son deterministas y se ejecutan una sola vez, en T1-A "
                             f"(analysis_plan.md §4.3, §13.2)")
    if s.variant == "stat" and s.save_ckpt:
        raise InvalidJob("stat con save_ckpt=1: KitNET.save falla en stat (kitnet.py:227); parametros en "
                         "model_stats.npz")
    if s.variant == "original_eps" and s.save_ckpt:
        raise InvalidJob("original_eps con save_ckpt=1: KitNET.save lo manda a la rama Keras (kitnet.py:219, codigo "
                         "congelado) y falla; parametros en model_stats.npz")
    if s.family == "X2" and s.pipeline != "orig":
        raise InvalidJob("X2 solo usa el pipeline orig (el pipeline original tal cual solo para X2/regresion)")
    if s.family in ("S", "T1", "T2", "T3", "T4", "T7") and s.pipeline != "corrected" and not test_mode:
        raise InvalidJob("S, T1-T4 y T7 usan el pipeline corrected")
    return s


def check_supported(spec):
    """The corrected pipeline exists for KNAD, CIC2017, CIC2018 and ACI; for any other dataset it raises NotImplementedError."""
    if spec.pipeline == "corrected" and spec.kind not in CORRECTED_SUPPORTED_KINDS:
        raise NotImplementedError(CORRECTED_NOT_IMPLEMENTED.format(
            ds=spec.dataset, req=CORRECTED_REQ.get(spec.dataset, "un stream corregido auditado en T0")))


def corrected_train_rows(spec_or_dataset):
    """Training rows T = fm + ad of the corrected stream of a dataset (CORRECTED_TRAIN_ROWS; 72,264 for ACI)."""
    ds = getattr(spec_or_dataset, "dataset", spec_or_dataset)
    return CORRECTED_TRAIN_ROWS["KNAD" if str(ds).upper().startswith("KNAD:") else ds]


class JobPaths:
    def __init__(self, results_root, job_id):
        self.root = Path(results_root)
        self.job_id = job_id
        self.final = self.root / job_id
        self.done = self.final / "DONE"
        self.lock = self.root / ".locks" / job_id
        self.state = self.root / ".state" / job_id
        self.tmp_root = self.root / ".tmp"

    def new_tmp(self, host, pid):
        """Create the attempt directory; a leftover with the same host and PID is moved to .state if it holds scores.npy, removed otherwise."""
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        t = self.tmp_root / f"{self.job_id}.{host}.{pid}"
        action = None
        if t.exists():
            if (t / "scores.npy").exists():
                with state_flock(self.root, self.job_id):
                    k = _next_index(self.state, "attempt")
                    dest = self.state / f"attempt_{k}_{host}_{pid}_pidreuse"
                    os.rename(t, dest)
                write_json_atomic(dest / "adopted.json", {"from": str(t), "by_pid": pid, "ts": now_iso(),
                                                          "reason": "intento con scores.npy y el mismo host.PID "
                                                                    "(PID reutilizado): se conserva"})
                action = {"action": "moved_pid_reuse", "from": str(t), "to": str(dest)}
            else:
                shutil.rmtree(t)
                action = {"action": "deleted_no_scores_pid_reuse", "path": str(t)}
        t.mkdir()
        return t, action


def acquire_lock(lock_dir, info):
    lock_dir = Path(lock_dir)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        return None
    pid = os.getpid()
    ordered = {"pid": pid, "host": hostname(), "start_ticks": proc_start_ticks(pid)}
    ordered.update({k: v for k, v in info.items() if k not in ordered})
    try:
        write_json_atomic(lock_dir / "info.json", ordered)
        write_text_atomic(lock_dir / "heartbeat", f"pid={pid} host={hostname()} ts={now_iso()} phase=lock row=0/0\n")
    except BaseException:
        _remove_lock_dir(lock_dir)
        raise
    return lock_dir


def _remove_lock_dir(lock_dir, attempts=5):
    """Remove the files of a lock and its directory, retrying on ENOTEMPTY; True when the directory no longer exists."""
    lock_dir = Path(lock_dir)
    for i in range(attempts):
        try:
            for p in list(lock_dir.iterdir()):
                try:
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink()
                except FileNotFoundError:
                    pass
            os.rmdir(lock_dir)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(0.05 * (i + 1))
    return not lock_dir.exists()


def read_lock_info(lock_dir):
    return read_json(Path(lock_dir) / "info.json")


def heartbeat_age_s(lock_dir):
    lock_dir = Path(lock_dir)
    mt = []
    for p in (lock_dir / "heartbeat", lock_dir / "info.json", lock_dir):
        try:
            mt.append(p.stat().st_mtime)
        except OSError:
            pass
    return (time.time() - max(mt)) if mt else None


def lock_status(lock_dir, host=None):
    host = host or hostname()
    info = read_lock_info(lock_dir) or {}
    age = heartbeat_age_s(lock_dir)
    st = {"info": info, "host": info.get("host"), "pid": info.get("pid"), "start_ticks": info.get("start_ticks"),
          "hb_age_s": age, "same_host": info.get("host") == host, "alive": None, "heartbeat": ""}
    try:
        st["heartbeat"] = (Path(lock_dir) / "heartbeat").read_text().strip()
    except OSError:
        pass
    if not info:
        st["alive"] = True if (age is not None and age < 120) else None
    elif st["same_host"]:
        st["alive"] = pid_alive(info.get("pid"), info.get("start_ticks"), needle="run_job.py")
    return st


def release_lock(lock_dir, pid=None, start_ticks=None):
    """Libera SOLO si info.json es del proceso (pid y arranque). Devuelve True si lo libero."""
    lock_dir = Path(lock_dir)
    pid = os.getpid() if pid is None else pid
    start_ticks = proc_start_ticks(pid) if start_ticks is None else start_ticks
    info = read_lock_info(lock_dir)
    if not info or int(info.get("pid", -1)) != int(pid) or (
            info.get("start_ticks") is not None and start_ticks is not None and int(info["start_ticks"]) != int(start_ticks)):
        return False
    return _remove_lock_dir(lock_dir)


def recover_lock(results_root, job_id, expected_pid, expected_ticks, reason, log=None):
    """Mueve .locks/X a .state/X/lock_obsoleto_<ts>_<pid> (atomico: gana un solo worker). Tras mover comprueba
    que es el lock que se vio muerto (pid y arranque); si no, lo devuelve (S3). Devuelve el destino o None."""
    P = JobPaths(results_root, job_id)
    P.state.mkdir(parents=True, exist_ok=True)
    base = f"lock_obsoleto_{ts_compact()}_{os.getpid()}"
    dest = None
    for k in range(1000):
        cand = P.state / (base if k == 0 else f"{base}_{k}")
        if cand.exists():
            continue
        try:
            os.rename(P.lock, cand)
            dest = cand
            break
        except FileNotFoundError:
            return None
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
                continue
            if log:
                log(f"no se pudo mover el lock de {job_id}: {e}")
            return None
    if dest is None:
        return None
    moved = read_lock_info(dest) or {}
    same = (str(moved.get("pid")) == str(expected_pid) and
            (expected_ticks is None or moved.get("start_ticks") is None or str(moved.get("start_ticks")) == str(expected_ticks)))
    if not same:
        try:
            os.rename(dest, P.lock)
            if log:
                log(f"lock de {job_id} cambiado entre la comprobacion y el movimiento (pid {moved.get('pid')} != "
                    f"{expected_pid}): devuelto a .locks")
        except OSError as e:
            if log:
                log(f"ERROR: lock ajeno de {job_id} movido por carrera y no devuelto ({e}); queda en {dest}")
            write_json_atomic(dest / "recovered.json", {"by_pid": os.getpid(), "host": hostname(), "ts": now_iso(),
                                                        "reason": "carrera: no era el lock visto muerto",
                                                        "not_returned": str(e)})
        return None
    write_json_atomic(dest / "recovered.json", {"by_pid": os.getpid(), "host": hostname(), "ts": now_iso(),
                                                "reason": reason, "expected_pid": expected_pid})
    return dest


class Heartbeat:
    """Thread that rewrites .locks/X/heartbeat every every_s seconds; it never calls TensorFlow."""

    def __init__(self, lock_dir, job_id, every_s=None):
        self.lock_dir = Path(lock_dir)
        self.job_id = job_id
        self.every_s = float(every_s or os.environ.get("JISA_HEARTBEAT_S", 60))
        self.phase = "inicio"
        self.row = 0
        self.n = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def text(self):
        return (f"pid={os.getpid()} host={hostname()} ts={now_iso()} phase={self.phase} "
                f"row={self.row}/{self.n} job={self.job_id}\n")

    def beat(self):
        try:
            if self.lock_dir.exists():
                write_text_atomic(self.lock_dir / "heartbeat", self.text())
        except OSError:
            pass

    def set_phase(self, phase, n=None):
        self.phase = phase
        if n is not None:
            self.n = n
        self.beat()

    def _run(self):
        while not self._stop.wait(self.every_s):
            self.beat()

    def stop(self, join_timeout_s=10.0):
        """Stop the heartbeat thread and wait until it has finished."""
        self._stop.set()
        if self._t.is_alive() and self._t is not threading.current_thread():
            self._t.join(timeout=join_timeout_s)


def live_locks_on_host(results_root, host=None, exclude_job=None):
    host = host or hostname()
    res = []
    d = Path(results_root) / ".locks"
    if not d.exists():
        return res
    for L in d.iterdir():
        if not L.is_dir() or L.name == exclude_job:
            continue
        st = lock_status(L, host)
        if st["same_host"] and st["alive"]:
            res.append((L.name, st["info"]))
        elif not st["info"] and st["alive"]:
            res.append((L.name, {}))
    return res


@contextlib.contextmanager
def state_flock(results_root, job_id):
    P = JobPaths(results_root, job_id)
    P.state.mkdir(parents=True, exist_ok=True)
    with open(P.state / ".flock", "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield P
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_fail_count(results_root, job_id):
    try:
        return int((Path(results_root) / ".state" / job_id / "fail_count").read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _next_index(state_dir, prefix):
    n = 0
    for p in state_dir.glob(f"{prefix}_*"):
        m = re.match(rf"^{re.escape(prefix)}_(\d+)", p.name)
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def record_failure(results_root, job_id, text):
    """error_N.txt (N = fail_count + 1) y fail_count++ (atomico bajo flock). Devuelve N."""
    with state_flock(results_root, job_id) as P:
        n = read_fail_count(results_root, job_id) + 1
        write_text_atomic(P.state / f"error_{n}.txt", f"[{now_iso()}] host={hostname()} pid={os.getpid()}\n{text}\n")
        write_text_atomic(P.state / "fail_count", f"{n}\n")
        return n


def record_event(results_root, job_id, prefix, content, ext="txt"):
    """halt_N.txt, interrupted_N.txt, oom_N.json, dirty_N.txt... (no suman fallo)."""
    with state_flock(results_root, job_id) as P:
        n = _next_index(P.state, prefix)
        path = P.state / f"{prefix}_{n}.{ext}"
        if ext == "json":
            write_json_atomic(path, content)
        else:
            write_text_atomic(path, f"[{now_iso()}] host={hostname()} pid={os.getpid()}\n{content}\n")
        return path


def account_lock_move(results_root, job_id, moved_dir, text):
    """Account once for a lock moved to .state: adds one failure; returns N, or None if it was already accounted."""
    try:
        os.mkdir(Path(moved_dir) / "accounted")
    except FileExistsError:
        return None
    except OSError:
        return None
    info = read_lock_info(moved_dir) or {}
    n = record_failure(results_root, job_id, f"{text}\nlock movido: {moved_dir}\ninfo.json: {json.dumps(info)}")
    write_json_atomic(Path(moved_dir) / "accounted" / "by.json", {"pid": os.getpid(), "host": hostname(),
                                                                  "ts": now_iso(), "error_n": n})
    return n


def account_orphan_moves(results_root, job_id, log=None):
    """Account for locks moved to .state/X/lock_obsoleto_* that were never counted: one failure each. Idempotent."""
    st = Path(results_root) / ".state" / job_id
    if not st.exists():
        return 0
    k = 0
    for d in sorted(st.glob("lock_obsoleto_*")):
        if d.is_dir() and not (d / "accounted").exists():
            n = account_lock_move(results_root, job_id, d,
                                  "proceso muerto sin registrar (posible Killed/OOM del sistema); lock movido a "
                                  ".state sin contabilizar (recuperacion manual o por otro worker)")
            if n is not None:
                k += 1
                if log:
                    log(f"{job_id}: lock movido sin contabilizar ({d.name}) -> error_{n}.txt, fail_count={n}")
    return k


def adopt_stale_tmps(P, host, mypid, log=None):
    """Adopt abandoned attempts (dead PID, same host): with scores.npy they are moved to .state, without scores they are removed."""
    actions = []
    if not P.tmp_root.exists():
        return actions
    prefix = P.job_id + "."
    for t in sorted(P.tmp_root.iterdir()):
        if not t.is_dir() or not t.name.startswith(prefix):
            continue
        rest = t.name[len(prefix):]
        if "." not in rest:
            continue
        h, pid = rest.rsplit(".", 1)
        if h != host or not pid.isdigit() or int(pid) == mypid:
            continue
        if pid_alive(pid, None, needle="run_job.py"):
            continue
        if (t / "scores.npy").exists():
            with state_flock(P.root, P.job_id):
                k = _next_index(P.state, "attempt")
                dest = P.state / f"attempt_{k}_{h}_{pid}"
                os.rename(t, dest)
            write_json_atomic(dest / "adopted.json", {"from": str(t), "by_pid": mypid, "ts": now_iso(),
                                                      "reason": "intento abandonado con scores.npy (se conserva)"})
            actions.append({"action": "moved", "from": str(t), "to": str(dest)})
        else:
            shutil.rmtree(t, ignore_errors=True)
            actions.append({"action": "deleted_no_scores", "path": str(t)})
        if log:
            log(f"{P.job_id}: intento abandonado {t.name}: {actions[-1]['action']}")
    return actions


def preserve_tmp(P, tmp, reason):
    """Tras un fallo con scores.npy ya escrito: mueve el .tmp a .state/X/attempt_<k>_<host>_<pid> (M5)."""
    tmp = Path(tmp)
    if not tmp.exists() or not (tmp / "scores.npy").exists():
        return None
    with state_flock(P.root, P.job_id):
        k = _next_index(P.state, "attempt")
        dest = P.state / f"attempt_{k}_{hostname()}_{os.getpid()}"
        os.rename(tmp, dest)
    write_json_atomic(dest / "preserved.json", {"ts": now_iso(), "reason": reason[-2000:]})
    return dest


def list_previous_attempts(P):
    res = []
    if not P.state.exists():
        return res
    for d in sorted(P.state.iterdir()):
        if d.is_dir() and re.match(r"^(attempt|duplicate)_\d+", d.name):
            s = d / "scores.npy"
            res.append({"dir": str(d), "has_scores": s.exists(),
                        "scores_sha256": sha256_file(s) if s.exists() else None,
                        "mtime": dt.datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")})
    return res


def commit(tmp, P):
    """Write DONE inside .tmp (fsync) and os.rename(tmp, final) as the single commit point; returns OK or DUPLICATE."""
    tmp = Path(tmp)
    fsync_tree(tmp)
    write_text_atomic(tmp / "DONE", now_iso() + "\n")
    fsync_dir(tmp)
    try:
        os.rename(tmp, P.final)
    except OSError:
        if P.done.exists():
            with state_flock(P.root, P.job_id):
                k = _next_index(P.state, "duplicate")
                dest = P.state / f"duplicate_{k}_{hostname()}_{os.getpid()}"
                os.rename(tmp, dest)
            (dest / "DONE").rename(dest / "DONE.duplicate")
            return DUPLICATE
        raise
    fsync_dir(P.root)
    return OK


def git(root, *args, timeout=60):
    try:
        p = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def worktree_dirty(code_root):
    rc, out, err = git(code_root, "status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        return f"git status fallo: {err[:200]}"
    return out


def head_commit(root):
    rc, out, _ = git(root, "rev-parse", "HEAD")
    return out if rc == 0 else None


def tags_at_head(root):
    rc, out, _ = git(root, "tag", "--points-at", "HEAD")
    return out.split() if rc == 0 else []


def runner_tag_at_head(root):
    tags = [t for t in tags_at_head(root) if re.fullmatch(r"runner-v\d+", t)]
    return sorted(tags, key=lambda t: int(t.split("v")[-1]))[-1] if tags else None


def tag_commit(root, tag):
    rc, out, _ = git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    return out if rc == 0 and out else None


def is_registered_worktree(code_root, main_repo=MAIN_REPO):
    rc, out, _ = git(main_repo, "worktree", "list", "--porcelain")
    if rc != 0:
        return False
    return any(realpath(ln[len("worktree "):]) == realpath(code_root)
               for ln in out.splitlines() if ln.startswith("worktree "))


def src_identical(code_root, ref="submitted-v1", paths=("src", "experiments/frames")):
    """Whether the original code of the working tree (tracked, and untracked but not ignored) equals `ref`."""
    if tag_commit(code_root, ref) is None:
        return False, f"no existe el tag {ref} en {code_root}"
    rc, out, err = git(code_root, "diff", "--name-only", f"{ref}", "--", *paths)
    if rc != 0:
        return False, f"git diff fallo: {err[:200]}"
    if out:
        return False, f"difiere de {ref}: {out.splitlines()[:10]}"
    rc, out2, _ = git(code_root, "ls-files", "--others", "--exclude-standard", "--", *paths)
    if out2:
        return False, f"ficheros no versionados en {paths}: {out2.splitlines()[:10]}"
    return True, f"identico a {ref} ({tag_commit(code_root, ref)[:10]}) en {list(paths)}"


CKPT_PROBE_PATHS = ["ckpt/model.pkl", "ckpt/output.pkl", "ckpt/kitnet_model.pkl", "ckpt/ae0/class.pkl",
                    "ckpt/ae0/model/saved_model.pb", "ckpt/ae0/model/keras_metadata.pb",
                    "ckpt/ae0/model/fingerprint.pb", "ckpt/ae0/model/variables/variables.index",
                    "ckpt/ae0/model/variables/variables.data-00000-of-00001", "ae_errors.npy", "scores.npy",
                    "labels.npy", "model_stats.npz", "DONE"]


def ckpt_paths_ignored(results_root, job_id, main_repo=MAIN_REPO):
    """Whether git ignores the heavy files of a job; returns (ok|None, detail), None if results_root is outside the repository."""
    if not is_under(results_root, main_repo):
        return None, "results_root fuera del repositorio: no versionable"
    rel_root = os.path.relpath(realpath(results_root), realpath(main_repo))
    paths = [f"{rel_root}/{job_id}/{p}" for p in CKPT_PROBE_PATHS]
    rc, out, err = git(main_repo, "check-ignore", "--no-index", "--", *paths)
    if rc not in (0, 1):
        return False, f"git check-ignore fallo: {err[:200]}"
    ignored = set(out.splitlines())
    missing = [p for p in paths if p not in ignored]
    if missing:
        return False, ("NO ignorados por .gitignore (anadir p. ej. 'results/revision/**/ckpt/' y "
                       f"'results/revision/**/*.pkl'): {missing}")
    return True, f"{len(paths)} rutas de prueba ignoradas"


TIMING_EXCLUDED_FAMILIES = ("S",)
TBAR_LEVELS = ["familia+variante+dataset", "familia+variante", "variante (otra familia; NO fiable)"]


def collect_timings(results_root, rows):
    """wall_seconds de los DONE, EXCLUYENDO la familia S (sondas de minutos; su estimacion esta en probe.json)."""
    res = []
    R = Path(results_root)
    for r in rows:
        if block_family(r.get("block")) in TIMING_EXCLUDED_FAMILIES:
            continue
        t = R / r["job_id"] / "timing.json"
        if (R / r["job_id"] / "DONE").exists() and t.exists():
            d = read_json(t, {}) or {}
            w = d.get("wall_seconds")
            if isinstance(w, (int, float)):
                res.append((r, float(w)))
    return res


TBAR_FACTOR_COLS = ("window", "k_clusters", "grouping", "contamination")


def _tbar_factors(r):
    return tuple(str(r.get(c, "")).strip() for c in TBAR_FACTOR_COLS)


def tbar_info(timings, row):
    """Median wall time of comparable finished jobs as (median, level, reliable); (None, None, False) without data."""
    fam, var, ds, fac = block_family(row["block"]), row["variant"], row["dataset"], _tbar_factors(row)
    preds = (lambda r: block_family(r["block"]) == fam and r["variant"] == var and r["dataset"] == ds
             and _tbar_factors(r) == fac,
             lambda r: block_family(r["block"]) == fam and r["variant"] == var and _tbar_factors(r) == fac,
             lambda r: r["variant"] == var)
    for lvl, pred in enumerate(preds):
        vals = [w for r, w in timings if pred(r)]
        if vals:
            return statistics.median(vals), TBAR_LEVELS[lvl], lvl < 2
    return None, None, False


def tbar_seconds(timings, row):
    """Compatibilidad: solo el valor de tbar_info (incluye el nivel no fiable)."""
    return tbar_info(timings, row)[0]


def fits_before_freeze(row, results_root, rows, at=None):
    """Fail-closed gate of the results freeze: a job is claimed only with a mean time from its own family."""
    at = at or now()
    tb, lvl, reliable = tbar_info(collect_timings(results_root, rows), row)
    if tb is None:
        return False, "t_medio desconocido (fail-closed desde el 30 sep)"
    if not reliable:
        return False, f"t_medio solo de otra familia ({lvl}): no fiable, fail-closed desde el 30 sep"
    end = at + dt.timedelta(seconds=tb * FREEZE_MARGIN)
    return end <= FREEZE, f"t_medio={tb / 3600:.2f} h ({lvl}) x {FREEZE_MARGIN} -> fin {end:%Y-%m-%d %H:%M}"


class ProtocolRegionError(ValueError):
    """A (calibration, evaluation) pair that would overlap calibration with evaluation, or evaluation with training."""


def _interval(v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(i, int) for i in v) and 0 <= v[0] <= v[1]):
        raise ProtocolRegionError(f"{what} no es un intervalo semiabierto [a, b) de enteros: {v!r}")
    return int(v[0]), int(v[1])


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def _merge_intervals(ivs):
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in ivs if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract_intervals(iv, excl):
    """[a, b) menos la union de `excl` -> lista de intervalos disjuntos, ordenados y no vacios."""
    segs, cur = [], iv[0]
    for a, b in _merge_intervals(excl):
        a, b = max(a, iv[0]), min(b, iv[1])
        if b <= a:
            continue
        if a > cur:
            segs.append([cur, a])
        cur = max(cur, b)
    if cur < iv[1]:
        segs.append([cur, iv[1]])
    return segs


def eval_exclusions(regions, candidate):
    """Intervals that the post-processing must remove from the evaluation of `candidate`, clipped to that evaluation."""
    c = ((regions or {}).get("detector_protocol_candidates") or {}).get(candidate) or {}
    ev = _interval(c.get("eval"), "eval")
    excl = [_interval(v, "eval_exclude") for v in (c.get("eval_exclude") or [])]
    t4 = (regions or {}).get("t4e_training_contamination")
    if t4:
        ex = _interval(t4.get("eval_exclude_with_guards"), "t4e_training_contamination.eval_exclude_with_guards")
        pool = _interval(t4.get("donor_pool"), "t4e_training_contamination.donor_pool")
        if not (ex[0] <= pool[0] < pool[1] <= ex[1]):
            raise ProtocolRegionError(f"T4-E: pool de donantes {list(pool)} fuera de la exclusion {list(ex)}")
        dr = t4.get("donor_rows") or {}
        if dr and not (pool[0] <= int(dr.get("first", pool[0])) and int(dr.get("last", pool[0])) < pool[1]):
            raise ProtocolRegionError(f"T4-E: filas donantes [{dr.get('first')}, {dr.get('last')}] fuera del pool {list(pool)}")
        excl.append(ex)
    return [iv for iv in _merge_intervals([[max(a, ev[0]), min(b, ev[1])] for a, b in excl]) if iv[1] > iv[0]]


def label_counts_key(dataset):
    """Clave de plan_support/label_counts.json (knad_<ataque>, cic2017, cic2018, aci)."""
    ds = str(dataset)
    if ds.upper().startswith("KNAD:"):
        return "knad_" + ds.split(":", 1)[1].lower()
    return {"CIC2017": "cic2017", "CIC2018": "cic2018", "ACI-IOT-2023": "aci"}.get(ds)


def t4e_eval_exclusion(dataset, code_root=None):
    """Exclusion de E_T4 (analysis_plan.md §8.3 y §8.5) leida del insumo congelado plan_support/label_counts.json: pool
    de donantes con 800 filas de guarda a cada lado. Es la MISMA exclusion que T4-E registra en su config y la que debe
    aplicarse a su referencia c = 0 (celdas T1-A semilla A del dataset), cuyos jobs no la llevan: p. ej.
    check_protocol_regions(regions_T1A, 'calib_first_10pct_post_train',
                           extra_eval_exclude=[t4e_eval_exclusion('KNAD:Active_Wiretap')['excluded_with_guards']]).
    Fail-closed: ProtocolRegionError si falta la entrada o es incoherente."""
    path = Path(code_root or MAIN_REPO) / LABEL_COUNTS_REL
    lc = read_json(path)
    key = label_counts_key(dataset)
    ent = ((lc or {}).get(key) or {}).get("t4p_donor_pool") if key else None
    if not ent:
        raise ProtocolRegionError(f"{path}: sin t4p_donor_pool para {dataset!r} (clave {key!r}); insumo del plan congelado")
    ex = _interval(ent.get("excluded_with_guards"), "t4p_donor_pool.excluded_with_guards")
    pool = _interval(ent.get("pool"), "t4p_donor_pool.pool")
    if not (ex[0] <= pool[0] < pool[1] <= ex[1]):
        raise ProtocolRegionError(f"{path} [{key}]: pool {list(pool)} fuera de la exclusion {list(ex)}")
    return {"dataset": dataset, "key": key, "pool": list(pool), "excluded_with_guards": list(ex),
            "attack_rows_needed": ent.get("attack_rows_needed"), "source": str(path), "sha256": sha256_file(path)}


def check_protocol_regions(regions, candidate, allow_regression_only=False, with_exclusions=False,
                           extra_eval_exclude=None):
    """Return the checked (calibration, evaluation) regions of a candidate of config.json['regions']; raises ProtocolRegionError."""
    cands = (regions or {}).get("detector_protocol_candidates") or {}
    if candidate not in cands:
        raise ProtocolRegionError(f"candidata {candidate!r} no existe; disponibles: {sorted(k for k in cands if not k.startswith('_'))}")
    c = cands[candidate]
    n = regions.get("n_rows")
    calib, ev = _interval(c.get("calib"), "calib"), _interval(c.get("eval"), "eval")
    if c.get("regression_only") and not allow_regression_only:
        raise ProtocolRegionError(f"{candidate}: solo regresion del envio (referencia dentro de la evaluacion); "
                                  "prohibida para resultados nuevos")
    if not c.get("regression_only"):
        if _overlap(calib, ev):
            raise ProtocolRegionError(f"{candidate}: calib {list(calib)} solapa eval {list(ev)} (fuga critica)")
        if calib[1] > ev[0]:
            raise ProtocolRegionError(f"{candidate}: calib {list(calib)} no termina antes de eval {list(ev)}.")
        if "train" in regions:
            tr = _interval(regions["train"], "train")
            if _overlap(ev, tr):
                raise ProtocolRegionError(f"{candidate}: eval {list(ev)} solapa train {list(tr)} (fuga critica)")
            if _overlap(calib, tr) and not c.get("calib_in_sample"):
                raise ProtocolRegionError(f"{candidate}: calib {list(calib)} solapa train {list(tr)} sin la marca "
                                          f"calib_in_sample (declaracion inconsistente)")
    if ev[1] <= ev[0]:
        raise ProtocolRegionError(f"{candidate}: eval vacia {list(ev)}")
    if calib[1] <= calib[0]:
        raise ProtocolRegionError(f"{candidate}: calib vacia {list(calib)}")
    if isinstance(n, int) and (calib[1] > n or ev[1] > n):
        raise ProtocolRegionError(f"{candidate}: intervalos fuera de [0, {n})")
    excl = eval_exclusions(regions, candidate)
    extra = [_interval(v, "extra_eval_exclude") for v in (extra_eval_exclude or [])]
    if excl and not (with_exclusions or extra_eval_exclude is not None):
        raise ProtocolRegionError(f"{candidate}: la evaluacion {list(ev)} contiene filas que no se pueden evaluar "
                                  f"{excl} (T4-E: pool de donantes con los que se entreno el modelo, "
                                  "analysis_plan.md §8.5); pide with_exclusions=True y usa los segmentos devueltos "
                                  "(fuga critica si no)")
    if not (with_exclusions or extra_eval_exclude is not None):
        return list(calib), list(ev)
    for iv in excl + extra:
        if _overlap(iv, calib):
            raise ProtocolRegionError(f"{candidate}: exclusion {list(iv)} solapa la calibracion {list(calib)}")
    segs = _subtract_intervals(ev, excl + [list(iv) for iv in extra])
    if not segs:
        raise ProtocolRegionError(f"{candidate}: la evaluacion queda vacia tras las exclusiones {excl + extra}")
    return list(calib), segs


@contextlib.contextmanager
def host_claim(results_root, host=None, timeout_s=600.0, should_stop=None, poll_s=0.2):
    """Cede True si obtuvo el flock de reclamacion del host; False si se agoto timeout_s o should_stop()."""
    host = host or hostname()
    d = Path(results_root) / ".tmp"
    d.mkdir(parents=True, exist_ok=True)
    fh = open(d / f".claim-{host}.flock", "a")
    got = False
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except BlockingIOError:
                if (should_stop and should_stop()) or time.time() - t0 > timeout_s:
                    break
                time.sleep(poll_s)
        yield got
    finally:
        if got:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


def signal_name(rc):
    try:
        return signal.Signals(-rc).name if rc < 0 else ""
    except ValueError:
        return f"SIG{-rc}"
