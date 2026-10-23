#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e1_metricas_t10.py - metrics and statistics of the modified Conv1D/Conv2D encoders against the control."""
import sys

sys.dont_write_bytecode = True

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        if not (1 <= int(os.environ.get(_v, "")) <= 4):
            os.environ[_v] = "4"
    except ValueError:
        os.environ[_v] = "4"

import numpy as np

R = Path("/home/acid137/Projects/review-zd")
sys.path.insert(0, str(R / "revision" / "experiments"))

import common as C
import metrics_rev as MR
import stats as ST


def _cargar_modulo(nombre, ruta):
    """Importa un guion por ruta, sin anadir su carpeta a sys.path y sin escribir bytecode."""
    spec = importlib.util.spec_from_file_location(nombre, str(ruta))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MSB_PY = R / "revision" / "analysis" / "mlp_sin_barajar_metricas.py"
PUERTA_PY = R / "revision" / "analysis" / "reparacion" / "puerta_fase_b.py"
COMUN_PY = Path(__file__).resolve().parent / "comun_fase_b.py"
MSB = _cargar_modulo("_e1_mlp_sin_barajar_metricas", MSB_PY)
PUERTA = _cargar_modulo("_e1_puerta_fase_b", PUERTA_PY)
COMUN = _cargar_modulo("_e1_comun_fase_b", COMUN_PY)
SCRIPT_SHA256 = C.sha256_file(__file__)
MSB_SHA256 = C.sha256_file(MSB_PY)
PUERTA_PY_SHA256 = C.sha256_file(PUERTA_PY)
COMUN_SHA256 = C.sha256_file(COMUN_PY)

RESULTS = R / "results" / "revision"
AGG_DIR = RESULTS / "final-20261007-1206"
CLIP = R / "revision" / "analysis" / "clip_y_score.json"
AQUI = R / "revision" / "analysis" / "fase_b"
OUT_REGISTRO = AQUI / "e1_ejecuciones.jsonl"
PREFIJO_SALIDA = "e1_metricas_t10"
PREFIJO_CONTROL = "e1_control_positivo"
LOG_COLA_B = R / "logs" / "fase-b-sB.log"
PUERTA_REL = "revision/analysis/reparacion/puerta_fase_b.json"
DECISIONS_REL = "revision/decisions.md"
ENM_REL = "revision/analysis/analysis_plan_enmienda_1.md"
PUERTA_PY_REL = "revision/analysis/reparacion/puerta_fase_b.py"
MSB_REL = "revision/analysis/mlp_sin_barajar_metricas.py"
SCRIPT_REL = "revision/analysis/fase_b/e1_metricas_t10.py"
COMUN_REL = "revision/analysis/fase_b/comun_fase_b.py"
TAG_ENM = "analysis-plan-enmienda-1"
ARNES = {"A": {"tag": "brazo-rep-v1", "rel": "revision/analysis/reparacion/brazo_rep.py", "decision": "D-59"},
         "B": {"tag": "brazo-rep-v2", "rel": "revision/analysis/reparacion_semilla_b/brazo_rep.py",
               "decision": "D-76"}}
TAG_PUERTA_PY = ARNES["A"]["tag"]
TAG_GUION = "e1-t10-v1"
SESION_COLA = "fase-b"
PROCESO_ARNES = "revision/analysis/reparacion/brazo_rep.py"
PROCESOS_COLA_B = ("revision/analysis/reparacion_semilla_b/brazo_rep.py", "cola_fase_b_semilla_b.sh")
CIERRE_B = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) fase B semilla B termina \(problemas: (\d+)\)$")
FECHA_MIN_CIERRE_B = "2026-10-19"
ULTIMA_DECISION_PREVIA = 77
_PRINCIPIO_DE_FRASE = r"(?:^|(?<=[.!?] )|(?<=[.!?][»\"'”’)\]] ))"
MARCADOR_CONGELAR_B = re.compile(_PRINCIPIO_DE_FRASE + r"CONGELAR\s+SEMILLA\s+B\s*:", re.IGNORECASE)
MARCADOR_SOLO_A = re.compile(_PRINCIPIO_DE_FRASE + r"E1\s+PARCIAL\s+SOLO\s+A\s*:", re.IGNORECASE)
MARCADOR_PENDIENTES = re.compile(_PRINCIPIO_DE_FRASE + r"PENDIENTES\s+COMO\s+PERDIDAS\s*:", re.IGNORECASE)
NOMBRE_MARCADOR = {MARCADOR_CONGELAR_B.pattern: "CONGELAR SEMILLA B:", MARCADOR_SOLO_A.pattern: "E1 PARCIAL SOLO A:",
                   MARCADOR_PENDIENTES.pattern: "PENDIENTES COMO PERDIDAS:"}
MARCADOR_AUDITOR = "leakage-auditor"
PROPUESTAS_ABIERTAS = ("regla del verbo (Holm <= 0,05 y signo del HL; si no, 'did not differ in')",
                       "imputacion de las ejecuciones divergidas a 0,5 (plan §12.2)",
                       "control recortado de S-CLIP = clip_y_score.json hypothesis_sensitivity.cells.C10.alternative",
                       "nombre de la familia: 'modified X' frente al nombre de la plantilla de la enmienda",
                       "descriptivo sin p fuera de los 3 streams del prevuelo y valores de los DONE fuera de S*_d")
SUMS_CLAVES = {"aggregate": "input:aggregate_json_sha256", "stats": "stats.json"}
CLIP_REL = "revision/analysis/clip_y_score.json"
E1_VERSION = "2.2.0"
SCHEMA = "jisa-t10-rep-e1/2"
SCHEMA_CONTROL = "jisa-t10-rep-e1-control/2"
SCHEMA_PUERTA = "jisa-t10-rep-puerta/1"

TAG = COMUN.ETIQUETA
ROTULO_POST_HOC = COMUN.ROTULO_POST_HOC
ROTULO_PARCIAL = "[PARCIAL: solo semilla A (--solo-semilla-a); no es el alcance de la enmienda, que pide las semillas A y B]"
ETIQUETA_FUERA = ("[EXPLORATORIO] fuera de S*_d: se reporta (plan §4.6); no entra en ninguna media por stream ni en "
                  "ningun test")
ALCANCE = ("post hoc; fuera de S*_d del plan, de H1-H4 y del agregado FINAL; semillas A y B con la regla del prefijo "
           "del plan §4.5 por stream; control original = su unica ejecucion T1-A (AG-12); la semilla C fuera (post "
           "hoc, sin T1-C)")
TAG_E1 = TAG + " E1 (enmienda §5): Holm con m = familias que entran; sin 'significant' (enmienda §6)"
PROHIBIDAS = COMUN.PROHIBIDAS
PLANTILLA = ("In a post hoc comparison specified after the pre-specified results were known (Holm-adjusted within "
             "this post hoc family, m = {m}), {nombre} {verbo} ROC-AUC {prep} the KitNET-style control (HL = {hl}, "
             "unadjusted {cob}% CI [{lo}, {hi}]; exact Wilcoxon p = {p}, "
             "Holm-adjusted p = {ph}; {n_txt}; higher in {k}/{n}; {semillas}).")
ATRIBUCION = ("This difference is not attributed to context or to the window: the {nombre} models also differ from the "
              "control in how the input is clipped, and both are scored with RMSE. No single comparison isolates the "
              "context: the per-record MLP shares the input clipping but is scored with MSE, so that comparison is "
              "given with both ways of scoring the MLP.")
ATRIBUCION_SIN_DIFERENCIA = ("The {nombre} models differ from the control in more than the context: they also differ "
                             "in how the input is clipped, and both are scored with RMSE. No single comparison "
                             "isolates the context: the per-record MLP shares the input clipping but is scored with "
                             "MSE, so that comparison is given with both ways of scoring the MLP.")
DECLARACION_DISENO = ("The list of changes and its order were designed with all detection results of the "
                      "pre-specified analysis known, and two of the changes (R and I) bring the architecture closer "
                      "to a per-record model.")
DECLARACION_TRAMO = ("Each modified autoencoder is fitted on the first 90% of the rows it receives (the last 10% is "
                     "held out to check convergence), and its min-max normalisation comes from those rows only, so the "
                     "clipping range used at scoring time comes from fewer rows than in the {nombre} models without "
                     "the changes. No comparison here separates the changes {factores} from this difference. The "
                     "control does not clip its input at scoring time.")
DECLARACION_TRAMO_BENIGNO = ("As in the pre-specified analysis, the benign segment on which these models and the "
                             "control are fitted was delimited with the documentation and the attack labels of each "
                             "corpus and checked with those labels to contain no attack, so this step of the pipeline "
                             "is not free of labels.")
DECLARACION_RECUENTO = ("The entry criterion of the {nombre} models, applied with seed A only and without labels, "
                        "requires a validation ratio below 0.90 in at least two thirds of the evaluable TSR "
                        "autoencoders and at least 7 of the 9 streams with at least one autoencoder below 0.90. "
                        "{partes}")
DECLARACION_SEMILLA_C = ("A third seed (C) was added post hoc, after the count of the modified Conv1D models with "
                         "seed B was known; its count with the same rule is given in the convergence table, and it "
                         "does not enter this comparison.")
DECLARACION_PREVUELO = ("The changes {factores} were chosen on {k} of the 9 streams (the pre-flight streams), without "
                        "attack labels; the same difference without those {k} streams is reported separately, as a "
                        "descriptive result without a test.")
NOTA_S_DUPCAL = ("S-DUPCAL (metrics_rev.py:1482-1501, [EXPLORATORIO] en el analisis principal: E sin las filas con "
                 "un gemelo exacto en F_cal) NO se calcula en E1: necesita el x.npy de la cache de cada stream "
                 "(metrics_rev.calib_eval_twins) y E1 solo lee scores, etiquetas y config; el primario no cambia "
                 "(F_cal no se toca).")
ROTULO_GUION_DISTINTO = ("[APERTURAS PREVIAS CON OTRO GUION: T10 ya se abrio con una version distinta de "
                         "e1_metricas_t10.py (ver ejecuciones_previas[].script_sha256)]")

FAMILIAS = {"conv1d": {"nombre": "Conv1D", "factores": "P,I,R", "C_ar": "C1", "C_tsr": "C2"},
            "conv2d": {"nombre": "Conv2D", "factores": "P,I,R,A", "C_ar": "C3", "C_tsr": "C4"}}
ARS = ("T", "F")
PARADIGMA = {"T": "AR", "F": "TSR"}
SEMILLAS = ("A", "B")
SEMILLA_ORIGINAL = "A"
TOL = 1e-12
MARGEN = ST.FAILURE_MARGIN
N_AE = 4
N_NONEXEC = 9

_PERMISO = {"concedido": False, "root": None, "familias": frozenset(), "semillas": frozenset(),
            "motivo": "sin guarda de la puerta"}


class GuardaError(RuntimeError):
    """Guarda incumplida: salida 3 sin abrir nada."""


class Parada(RuntimeError):
    """El calculo no es comparable con el de la rejilla: salida 1 sin escribir cifras de T10."""


class PermisoDenegado(RuntimeError):
    """Se intento abrir un job T10 sin la guarda de la puerta superada."""


@dataclass
class Rutas:
    """Donde se lee y se escribe. En el modo oficial todo apunta al arbol principal; en seco, al mundo sintetico."""
    repo_puerta: Path
    results: Path
    results_esperado: str
    agg_dir: Path
    clip: Path
    dir_salida: Path
    log_cola_b: Path
    en_seco: bool = False
    registro: Path = None
    oficial: bool = False
    salida_control: Path = None
    salida: Path = None
    salida_md: Path = None


def rutas_oficiales():
    return Rutas(repo_puerta=R, results=RESULTS, results_esperado=str(RESULTS), agg_dir=AGG_DIR, clip=CLIP,
                 dir_salida=AQUI, log_cola_b=LOG_COLA_B, registro=OUT_REGISTRO, oficial=True)


def log(msg):
    print("%s [E1] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def nombre_ds(did):
    return ST.DS_NAME[did]


def slug(did):
    return C.ds_slug(ST.DS_NAME[did])


def t1_job_id(did, variante, ar, semilla="A"):
    """Job T1 de la rejilla (corpus de mlp_sin_barajar_metricas.corpus)."""
    return "T1-%s-%s__%s__%s__ar%s__corrected.v1__L800__k4__s%s__r1" % (
        semilla, MSB.corpus(nombre_ds(did)), slug(did), variante, ar, semilla)


def t10_job_id(did, fam, ar, semilla="A"):
    """Job identifier of a modified-encoder run: the seed A identifier with the suffix of the requested seed."""
    jid = PUERTA.job_id(slug(did), fam, ar)
    if "__sA__" not in jid:
        raise Parada("job_id de la semilla A sin '__sA__': %s" % jid)
    return jid if semilla == "A" else jid.replace("__sA__", "__s%s__" % semilla)


def jobs_control(semillas=SEMILLAS):
    """Jobs T1 del control positivo, por stream: original arT (solo A), y por semilla Conv1D y Conv2D {AR, TSR} y las
    dos replicas del MLP (arF = TSR, arT = AR; C9 = su media). La mascara KNAD se calcula una vez por stream."""
    out = []
    for did in ST.DS:
        for s in semillas:
            if s == SEMILLA_ORIGINAL:
                out.append((did, "original", "T", s))
            for fam in FAMILIAS:
                for ar in ARS:
                    out.append((did, fam, ar, s))
            out.append((did, "mlp", "F", s))
            out.append((did, "mlp", "T", s))
    return out


def _git_bytes(repo, *args):
    p = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, timeout=60)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")


def guarda_entorno():
    """RF-2 y R-7 de metrics_rev (plan congelado y lock de envs/postproc)."""
    try:
        return {"env": MR.env_guard(R), "plan": MR.plan_guard(R)}
    except (MR.EnvGuardError, MR.PlanGuardError) as e:
        raise GuardaError(str(e))


def guarda_enmienda(repo=R):
    """Tag analysis-plan-enmienda-1 presente y la enmienda de la copia de trabajo identica a la del tag."""
    commit = C.tag_commit(repo, TAG_ENM)
    if not commit:
        raise GuardaError("no existe el tag %s" % TAG_ENM)
    rc, blob, err = _git_bytes(repo, "show", "%s:%s" % (TAG_ENM, ENM_REL))
    if rc != 0:
        raise GuardaError("git show %s:%s fallo: %s" % (TAG_ENM, ENM_REL, err[:200]))
    sha_tag = hashlib.sha256(blob).hexdigest()
    sha_wt = C.sha256_file(Path(repo) / ENM_REL)
    if sha_tag != sha_wt:
        raise GuardaError("la enmienda de la copia de trabajo (%s) no es la del tag (%s)" % (sha_wt[:16], sha_tag[:16]))
    return {"tag": TAG_ENM, "commit": commit, "sha256": sha_tag, "check": "copia de trabajo = tag"}


def _sha_en(repo, ref, rel):
    """sha256 de `git show <ref>:<rel>` (None si no existe)."""
    rc, blob, _err = _git_bytes(repo, "show", "%s:%s" % (ref, rel))
    return hashlib.sha256(blob).hexdigest() if rc == 0 else None


def guarda_arnes(repo=R, semillas=SEMILLAS):
    """Check that each seed's harness and the entry-rule script match their tags, and the metrics module matches HEAD."""
    out = {"semillas": {}}
    for s in semillas:
        a = ARNES[s]
        commit = C.tag_commit(repo, a["tag"])
        if not commit:
            raise GuardaError("no existe el tag %s (arnes de la semilla %s)" % (a["tag"], s))
        sha_tag = _sha_en(repo, a["tag"], a["rel"])
        if sha_tag is None:
            raise GuardaError("git show %s:%s fallo" % (a["tag"], a["rel"]))
        p = Path(repo) / a["rel"]
        sha_wt = C.sha256_file(p) if p.exists() else None
        out["semillas"][s] = {"tag": a["tag"], "commit": commit, "fichero": a["rel"], "sha256_tag": sha_tag,
                              "sha256_copia_de_trabajo": sha_wt, "copia_de_trabajo_igual": sha_wt == sha_tag,
                              "decision_en_los_jobs": a["decision"]}
        if sha_wt != sha_tag:
            raise GuardaError("%s de la copia de trabajo no es el del tag %s (semilla %s)" % (a["rel"], a["tag"], s))
    puerta_tag = _sha_en(repo, TAG_PUERTA_PY, PUERTA_PY_REL)
    msb_head = _sha_en(repo, "HEAD", MSB_REL)
    out["puerta_fase_b_py"] = {"sha256_importado": PUERTA_PY_SHA256, "sha256_tag": puerta_tag, "tag": TAG_PUERTA_PY,
                               "igual": PUERTA_PY_SHA256 == puerta_tag}
    out["mlp_sin_barajar_metricas_py"] = {"sha256_importado": MSB_SHA256, "sha256_head": msb_head,
                                          "igual": MSB_SHA256 == msb_head}
    if not out["puerta_fase_b_py"]["igual"]:
        raise GuardaError("puerta_fase_b.py importado (%s) no es el del tag %s (%s): el paso f depende de el"
                          % (PUERTA_PY_SHA256[:16], TAG_PUERTA_PY, str(puerta_tag)[:16]))
    if not out["mlp_sin_barajar_metricas_py"]["igual"]:
        raise GuardaError("mlp_sin_barajar_metricas.py importado (%s) no es el de HEAD (%s)"
                          % (MSB_SHA256[:16], str(msb_head)[:16]))
    return out


def guarda_guion(repo=R, tag=TAG_GUION):
    """Official mode: check that this script and its helpers match the freezing tag and that a decision freezes it."""
    commit = C.tag_commit(repo, tag)
    if not commit:
        raise GuardaError("no existe el tag %s: el guion se congela con ese tag antes de abrir T10" % tag)
    out = {"tag": tag, "commit": commit, "ficheros": {}}
    for rel, sha in ((SCRIPT_REL, SCRIPT_SHA256), (COMUN_REL, COMUN_SHA256), (MSB_REL, MSB_SHA256)):
        st = _sha_en(repo, tag, rel)
        out["ficheros"][rel] = {"sha256_importado": sha, "sha256_tag": st, "igual": sha == st}
        if sha != st:
            raise GuardaError("%s importado (%s) no es el del tag %s (%s)" % (rel, sha[:16], tag, str(st)[:16]))
    out["decision"] = decision_guion(repo, SCRIPT_SHA256, tag)
    return out


def decision_guion(repo, sha_guion, tag=TAG_GUION):
    """Find the committed decision that freezes this script, citing the tag, the script hash and the leakage check."""
    rc, blob, err = _git_bytes(repo, "show", "HEAD:%s" % DECISIONS_REL)
    if rc != 0:
        raise GuardaError("git show HEAD:%s fallo: %s" % (DECISIONS_REL, err[:120]))
    for p in parrafos_md(blob.decode("utf-8", "replace")):
        mm = re.match(r"^\s*-\s*\*\*(D-(\d+)) \(", p)
        if (mm and int(mm.group(2)) > ULTIMA_DECISION_PREVIA and tag in p and sha_guion in p
                and MARCADOR_AUDITOR in p):
            return {"decision": mm.group(1), "parrafo": p}
    raise GuardaError("el fichero de decisiones de HEAD no tiene la decision que congela el guion: hace falta una '- "
                      "**D-NN (' con NN > %d que cite '%s', el sha256 de e1_metricas_t10.py (%s) y '%s', y que "
                      "decida: %s" % (ULTIMA_DECISION_PREVIA, tag, sha_guion, MARCADOR_AUDITOR,
                                      "; ".join(PROPUESTAS_ABIERTAS)))


def limpiar_parrafo(parrafo):
    """Parrafo sin "*" ni "`" y con los espacios (saltos de linea y sangrias incluidos) reducidos a uno."""
    return re.sub(r"\s+", " ", (parrafo or "").replace("*", "").replace("`", "")).strip()


def marcador_en(parrafo, marcador):
    """True si el marcador esta en el parrafo con dos puntos y al principio de su frase (MARCADOR_*, m1)."""
    return bool(marcador.search(limpiar_parrafo(parrafo)))


def texto_tras_marcador(parrafo, marcador):
    """Lo que sigue al marcador en el parrafo limpio ("" si no esta)."""
    mm = marcador.search(limpiar_parrafo(parrafo))
    return limpiar_parrafo(parrafo)[mm.end():] if mm else ""


def parrafo_dnn(texto, decision):
    """Paragraph of a decision entry in the decisions file; None if absent, an error if it appears more than once."""
    pat = re.compile(r"^- \*\*%s \(" % re.escape(decision))
    ps = [p for p in parrafos_md(texto) if pat.match(p)]
    if len(ps) > 1:
        raise GuardaError("%s aparece %d veces en el fichero de decisiones: no se sabe cual vale" % (decision, len(ps)))
    return ps[0] if ps else None


def decision_posterior(repo, decision, minimo, marcador=None, opcion="decision"):
    """Return a committed decision numbered above a minimum and, if asked, carrying a marker at a sentence start."""
    mm = re.fullmatch(r"D-(\d+)", (decision or "").strip())
    if not mm:
        raise GuardaError("%s: %r no tiene la forma D-NN" % (opcion, decision))
    if int(mm.group(1)) <= minimo:
        raise GuardaError("%s: %s no vale; hace falta una decision posterior a D-%d" % (opcion, decision,
                                                                                                   minimo))
    rc, blob, err = _git_bytes(repo, "show", "HEAD:%s" % DECISIONS_REL)
    if rc != 0:
        raise GuardaError("git show HEAD:%s fallo: %s" % (DECISIONS_REL, err[:120]))
    parrafo = parrafo_dnn(blob.decode("utf-8", "replace"), decision.strip())
    if parrafo is None:
        raise GuardaError("%s: %s no esta en git show HEAD:%s (sin commit no cuenta; formato '- **D-NN (')"
                          % (opcion, decision, DECISIONS_REL))
    if marcador is not None and not marcador_en(parrafo, marcador):
        raise GuardaError("%s: %s no lleva el marcador %r (con dos puntos, al principio de su frase)"
                          % (opcion, decision, NOMBRE_MARCADOR.get(marcador.pattern, marcador.pattern)))
    return {"decision": decision.strip(), "parrafo": parrafo}


def leer_registro(path):
    """Rows of the append-only run log (empty if there is none); a line that is not a JSON object stops the run."""
    p = Path(path)
    if not p.exists() and not p.is_symlink():
        return []
    try:
        texto = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise GuardaError("no se puede leer el registro %s: %s" % (p, e))
    filas = []
    for i, x in enumerate(texto.splitlines(), start=1):
        if not x.strip():
            continue
        try:
            f = json.loads(x)
        except ValueError:
            raise GuardaError("registro %s, linea %d ilegible (no es JSON; truncada?): no se puede saber que "
                              "aperturas de T10 hubo; no se abre nada" % (p, i))
        if not isinstance(f, dict):
            raise GuardaError("registro %s, linea %d: no es un objeto JSON" % (p, i))
        filas.append(f)
    return filas


def primera_apertura(registro):
    """Marca local (datetime sin zona) del primer evento "inicio" del registro: la primera vez que se abrio T10."""
    ts = []
    for f in leer_registro(registro):
        if f.get("evento") == "inicio" and f.get("cuando"):
            try:
                ts.append(_hora_local(str(f["cuando"])))
            except ValueError:
                ts.append(dt.datetime.min)
    return min(ts) if ts else None


def decision_antes_de_abrir(repo, decision, registro, opcion, marcador=None):
    """Check that a decision was already committed, unchanged, when the modified runs were first opened."""
    filas = leer_registro(registro)
    inicios = [f for f in filas if f.get("evento") == "inicio"]
    primera = primera_apertura(registro)
    out = {"primera_apertura_t10": primera.isoformat() if primera else None, "registro": str(registro),
           "regla": "el parrafo de la D-NN (con su marcador) ya estaba en el fichero de decisiones del git_head del "
                    "primer 'inicio' del registro e igual al de HEAD"}
    if not inicios:
        out["comprobacion"] = "sin aperturas de T10 en el registro: basta con que este en HEAD"
        return out
    head = str(inicios[0].get("git_head") or "")
    out["git_head_primera_apertura"] = head or None
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise GuardaError("%s: el primer evento 'inicio' del registro (%s) no tiene un git_head valido (%r): no se "
                          "puede comprobar que %s ya estaba comprometida antes de la primera apertura de T10"
                          % (opcion, inicios[0].get("cuando"), head or None, decision))
    if C.git(repo, "merge-base", "--is-ancestor", head, "HEAD")[0] != 0:
        raise GuardaError("%s: el git_head de la primera apertura (%s) no es antecesor de HEAD en %s: no se puede "
                          "comprobar %s antes de la primera apertura" % (opcion, head[:12], repo, decision))
    rc, blob, err = _git_bytes(repo, "show", "%s:%s" % (head, DECISIONS_REL))
    if rc != 0:
        raise GuardaError("%s: git show %s:%s fallo (%s): no se puede comprobar %s antes de la primera apertura"
                          % (opcion, head[:12], DECISIONS_REL, err[:120], decision))
    antes = parrafo_dnn(blob.decode("utf-8", "replace"), decision)
    rc, blob, err = _git_bytes(repo, "show", "HEAD:%s" % DECISIONS_REL)
    if rc != 0:
        raise GuardaError("git show HEAD:%s fallo: %s" % (DECISIONS_REL, err[:120]))
    ahora = parrafo_dnn(blob.decode("utf-8", "replace"), decision)
    pie = ("una decision tomada o retocada con cifras de T10 a la vista no vale (cambiarlo exige un tag nuevo del "
           "guion y declararlo)")
    if antes is None:
        raise GuardaError("%s: %s no estaba en el fichero de decisiones del commit de la primera apertura de T10 (%s, "
                          "%s): se comprometio despues de la primera apertura; %s" % (opcion, decision, head[:12],
                                                                              out["primera_apertura_t10"], pie))
    if marcador is not None and not marcador_en(antes, marcador):
        raise GuardaError("%s: en el commit de la primera apertura de T10 (%s), %s no llevaba el marcador %r: se "
                          "anadio despues de la primera apertura; %s"
                          % (opcion, head[:12], decision, NOMBRE_MARCADOR.get(marcador.pattern, marcador.pattern),
                             pie))
    if antes != ahora:
        raise GuardaError("%s: el parrafo de %s cambio despues de la primera apertura de T10 (commit %s): %s"
                          % (opcion, decision, head[:12], pie))
    out.update(parrafo_igual_al_de_la_primera_apertura=True,
               comprobacion="parrafo de %s en %s igual al de HEAD" % (decision, head[:12]))
    return out


def ultima_linea(path):
    """Ultima linea no vacia de un fichero de texto (None si no se puede leer o esta vacio)."""
    try:
        lineas = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
                  if ln.strip()]
    except OSError:
        return None
    return lineas[-1] if lineas else None


def _argv(pid):
    """Linea de ordenes de un proceso (lista vacia si ya no existe)."""
    try:
        return [x for x in Path("/proc/%s/cmdline" % pid).read_bytes().decode("utf-8", "replace").split("\0") if x]
    except OSError:
        return []


INTERPRETES = re.compile(r"^(python[0-9.]*|bash|sh|dash)$")


def ancestros(pid=None):
    """pids del proceso y de sus ancestros (PPid de /proc/<pid>/status, sin listar /proc)."""
    out, p = set(), str(pid or os.getpid())
    while p and p not in out and p != "0":
        out.add(p)
        try:
            st = Path("/proc/%s/status" % p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            break
        mm = re.search(r"^PPid:\s*(\d+)", st, re.MULTILINE)
        p = mm.group(1) if mm else None
    return out


def ejecuta_guion(argv, patron):
    """True si argv es un interprete (INTERPRETES) que ejecuta `patron` como argumento propio: un token sin espacios
    igual a patron o terminado en "/" + patron, que no sea la orden de un "-c". Un "bash -c '... patron ...'" que
    solo lo nombra (p. ej. el envoltorio de una herramienta o un pgrep dentro de una orden) no cuenta."""
    if not argv or not INTERPRETES.match(Path(argv[0]).name):
        return False
    for i, tok in enumerate(argv[1:], start=1):
        if argv[i - 1] == "-c" or any(ch.isspace() for ch in tok):
            continue
        if tok == patron or tok.endswith("/" + patron):
            return True
    return False


def procesos_vivos(patrones):
    """pids de los procesos que ejecutan alguno de los guiones (pgrep -f da candidatos; ejecuta_guion decide). Se
    ignoran: el servidor de tmux (su argv conserva la orden con la que se creo la primera sesion, "tmux new-session
    ... cola_fase_b_semilla_b.sh", y sigue vivo mientras exista cualquier otra sesion), este proceso y sus ancestros
    (el shell que lo lanzo puede llevar el patron en su orden) y los procesos que solo nombran el guion."""
    try:
        vivos, ignorados, otros = [], [], []
        propios = ancestros()
        for patron in patrones:
            p = subprocess.run(["pgrep", "-f", patron], capture_output=True, text=True, timeout=30)
            for x in p.stdout.split():
                x = x.strip()
                if not x or x in propios or x in vivos or x in ignorados or x in otros:
                    continue
                argv = _argv(x)
                if not argv:
                    continue
                if Path(argv[0]).name.startswith("tmux"):
                    ignorados.append(x)
                elif ejecuta_guion(argv, patron):
                    vivos.append(x)
                else:
                    otros.append(x)
        return {"ok": not vivos, "pids": vivos, "ignorados_tmux": ignorados, "ignorados_no_ejecutan": otros,
                "patrones": list(patrones)}
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "pids": None, "patrones": list(patrones), "motivo": "no se puede comprobar: %s" % e}


def _registro_de(rutas):
    return Path(rutas.registro) if rutas.registro else Path(rutas.dir_salida) / "e1_ejecuciones.jsonl"


def guarda_cola_b(rutas, solo_a=None, congelar_b=None):
    """Check that the seed B queue ended cleanly, or that a committed decision allows seed A only or freezes B."""
    registro = _registro_de(rutas)
    out = {"log": str(rutas.log_cola_b), "fecha_minima": FECHA_MIN_CIERRE_B, "parcial": bool(solo_a)}
    out["procesos"] = (procesos_vivos(PROCESOS_COLA_B) if not rutas.en_seco else
                       {"ok": True, "nota": "en seco: no se miran los procesos reales"})
    if solo_a:
        if not isinstance(solo_a, str):
            raise GuardaError("--solo-semilla-a necesita una decision: --solo-semilla-a "
                              "D-NN (posterior a D-%d, en HEAD, con 'E1 PARCIAL SOLO A:')" % ULTIMA_DECISION_PREVIA)
        dec = decision_posterior(rutas.repo_puerta, solo_a, ULTIMA_DECISION_PREVIA, MARCADOR_SOLO_A,
                                 "--solo-semilla-a")
        dec["momento"] = decision_antes_de_abrir(rutas.repo_puerta, dec["decision"], registro, "--solo-semilla-a",
                                                 MARCADOR_SOLO_A)
        if not out["procesos"]["ok"]:
            raise GuardaError("--solo-semilla-a: la cola de la semilla B sigue viva o no se puede comprobar (%s); con "
                              "la B en marcha no se abre la A" % out["procesos"])
        out.update(ok=True, semillas=["A"], rotulo=ROTULO_PARCIAL, decidida_por=dec,
                   motivo="--solo-semilla-a %s: la B no entra; la salida va rotulada PARCIAL" % dec["decision"])
        return out
    ult = ultima_linea(rutas.log_cola_b)
    m = CIERRE_B.match(ult or "")
    out["ultima_linea"] = ult
    out["cierre"] = {"fecha": m.group(1), "problemas": int(m.group(2))} if m else None
    out["cierre_limpio"] = bool(m and int(m.group(2)) == 0 and m.group(1)[:10] >= FECHA_MIN_CIERRE_B)
    esperados = [t10_job_id(did, fam, ar, "B") for fam in FAMILIAS for did in ST.DS for ar in ARS]
    con_done = [j for j in esperados if (Path(rutas.results) / j / "DONE").exists()]
    out["jobs_sB_con_done"], out["jobs_sB_esperados"] = len(con_done), len(esperados)
    if not out["procesos"]["ok"]:
        raise GuardaError("la cola de la semilla B sigue viva o no se puede comprobar (%s); no se abre nada"
                          % out["procesos"])
    if out["cierre_limpio"] or len(con_done) == len(esperados):
        out.update(ok=True, semillas=["A", "B"],
                   motivo=("cola B cerrada limpia (%s)" % ult if out["cierre_limpio"] else
                           "los %d jobs sB tienen DONE" % len(esperados)))
        return out
    if congelar_b:
        dec = decision_posterior(rutas.repo_puerta, congelar_b, ULTIMA_DECISION_PREVIA, MARCADOR_CONGELAR_B,
                                 "--congelar-semilla-b")
        dec["momento"] = decision_antes_de_abrir(rutas.repo_puerta, dec["decision"], registro, "--congelar-semilla-b",
                                                 MARCADOR_CONGELAR_B)
        out["congelada_por"] = dec
        out.update(ok=True, semillas=["A", "B"],
                   motivo="semilla B congelada por %s con %d/%d jobs sB con DONE: lo que no tiene DONE acorta S*_d "
                          "(plan §4.5)" % (congelar_b, len(con_done), len(esperados)))
        return out
    raise GuardaError("la cola de la semilla B no ha terminado limpia: ultima linea de %s = %r; %d/%d jobs sB con "
                      "DONE. Opciones: esperar al cierre 'fase B semilla B termina (problemas: 0)' (fecha >= %s); "
                      "--solo-semilla-a D-NN (salida PARCIAL; 'E1 PARCIAL SOLO A:'); o --congelar-semilla-b D-NN "
                      "('CONGELAR SEMILLA B:'), decisiones posteriores a D-%d, en HEAD y anteriores a la primera "
                      "apertura de T10" % (rutas.log_cola_b, ult, len(con_done), len(esperados),
                                                   FECHA_MIN_CIERRE_B, ULTIMA_DECISION_PREVIA))


def guarda_registro(repo, registro):
    """Official mode: the run log is committed and unchanged, every committed version of it is a prefix of the current one, and the outputs of earlier finished runs are committed with the hash the log recorded."""
    p = Path(registro)
    try:
        rel = str(p.resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        raise GuardaError("el registro %s no esta dentro del repositorio %s" % (p, repo))
    rc, st, err = C.git(repo, "status", "--porcelain", "--", rel)
    if rc != 0 or st:
        raise GuardaError("%s tiene ejecuciones sin commit (%r): commit del registro y de sus salidas antes de volver "
                          "a ejecutar" % (rel, (st or err)[:200]))
    if p.exists() and C.git(repo, "ls-files", "--error-unmatch", rel)[0] != 0:
        raise GuardaError("%s no esta versionado" % rel)
    rc, log_, err = C.git(repo, "log", "--format=%H", "--full-history", "--", rel)
    if rc != 0:
        raise GuardaError("git log -- %s fallo: %s" % (rel, err[:120]))
    commits = log_.split()
    if commits and not p.exists():
        raise GuardaError("%s tiene historia en git (%d commits) y no existe: borrar el registro no anula las "
                          "aperturas de T10" % (rel, len(commits)))
    actual = p.read_bytes() if p.exists() else b""
    for c in commits:
        rc, blob, err = _git_bytes(repo, "show", "%s:%s" % (c, rel))
        if rc != 0:
            raise GuardaError("en el commit %s no existe %s (se borro): la historia del registro no es append-only"
                              % (c[:12], rel))
        if not actual.startswith(blob):
            raise GuardaError("la version de %s del commit %s no es prefijo de la actual: el registro se reescribio "
                              "(es append-only)" % (rel, c[:12]))
    filas = leer_registro(p)
    return {"registro": rel, "existe": p.exists(), "sha256": C.sha256_file(p) if p.exists() else None,
            "commits": len(commits), "historia_append_only": True,
            "aperturas_previas": sum(1 for f in filas if f.get("evento") == "inicio"),
            "salidas_previas": salidas_previas(repo, filas)}


def salidas_previas(repo, filas):
    """m3: cada salida de un evento "fin" anterior (salida JSON y salida_md) esta dentro del repositorio, versionada,
    sin cambios sin commit y, si el registro anoto su sha256, con ese sha256."""
    out = []
    for f in filas:
        if f.get("evento") != "fin":
            continue
        for clave, clave_sha in (("salida", "salida_sha256"), ("salida_md", "salida_md_sha256")):
            s = f.get(clave)
            if not s:
                continue
            sp = Path(s)
            try:
                rel = str(sp.resolve().relative_to(Path(repo).resolve()))
            except ValueError:
                raise GuardaError("la salida previa %s (registro, %s) no esta dentro del repositorio %s" % (s, clave,
                                                                                                          repo))
            if C.git(repo, "ls-files", "--error-unmatch", rel)[0] != 0:
                raise GuardaError("la salida previa %s no esta versionada: commit de las salidas antes de volver a "
                                  "ejecutar" % rel)
            rc, st, err = C.git(repo, "status", "--porcelain", "--", rel)
            if rc != 0 or st or not sp.exists():
                raise GuardaError("la salida previa %s tiene cambios sin commit o no existe (%r)" % (rel,
                                                                                                  (st or err)[:200]))
            sha = C.sha256_file(sp)
            if f.get(clave_sha) and sha != f[clave_sha]:
                raise GuardaError("la salida previa %s no es la que se escribio (sha256 %s, el registro dice %s)"
                                  % (rel, sha[:16], str(f[clave_sha])[:16]))
            out.append({"salida": rel, "sha256": sha, "sha256_del_registro": f.get(clave_sha)})
    return out


def guarda_insumos(rutas):
    """sha256 de aggregate.json, stats.json y clip_y_score.json. Si existe <agg_dir>/SHA256SUMS (siempre en el modo
    oficial), los dos primeros tienen que ser los suyos; en el modo oficial, clip_y_score.json tiene que ser el de
    HEAD y sin cambios. Asi la cifra del MLP en RMSE (que E1 toma de clip_y_score.json) es la versionada."""
    agg, sts, clip = (Path(rutas.agg_dir) / "aggregate" / "aggregate.json", Path(rutas.agg_dir) / "stats.json",
                      Path(rutas.clip))
    out = {"aggregate_json": {"ruta": str(agg)}, "stats_json": {"ruta": str(sts)},
           "clip_y_score_json": {"ruta": str(clip)}}
    for k, p in (("aggregate_json", agg), ("stats_json", sts), ("clip_y_score_json", clip)):
        if not p.exists():
            raise GuardaError("no existe el insumo %s" % p)
        out[k]["sha256"] = C.sha256_file(p)
    sums = Path(rutas.agg_dir) / "SHA256SUMS"
    if sums.exists() or rutas.oficial:
        try:
            tabla = {ln.split()[1]: ln.split()[0] for ln in sums.read_text(encoding="utf-8").splitlines()
                     if len(ln.split()) == 2}
        except OSError as e:
            raise GuardaError("no se puede leer %s: %s" % (sums, e))
        for k, clave in (("aggregate_json", SUMS_CLAVES["aggregate"]), ("stats_json", SUMS_CLAVES["stats"])):
            out[k]["sha256_SHA256SUMS"] = tabla.get(clave)
            if tabla.get(clave) != out[k]["sha256"]:
                raise GuardaError("%s (%s) no es el de %s:%s (%s)" % (out[k]["ruta"], out[k]["sha256"][:16], sums,
                                                                       clave, str(tabla.get(clave))[:16]))
    if rutas.oficial:
        sha_head = _sha_en(R, "HEAD", CLIP_REL)
        rc, st, _err = C.git(R, "status", "--porcelain", "--", CLIP_REL)
        out["clip_y_score_json"].update(sha256_head=sha_head, sin_cambios=(rc == 0 and not st))
        if sha_head != out["clip_y_score_json"]["sha256"] or rc != 0 or st:
            raise GuardaError("clip_y_score.json no es el de HEAD o tiene cambios (%s / %s)"
                              % (out["clip_y_score_json"]["sha256"][:16], str(sha_head)[:16]))
    return out


def estado_cola():
    """La cola de la semilla A ha terminado: no queda la sesion tmux SESION_COLA ni ningun proceso del arnes de la A.
    Si no se puede comprobar, no se da por terminada."""
    out = {"sesion_tmux": SESION_COLA, "proceso": PROCESO_ARNES}
    try:
        t = subprocess.run(["tmux", "has-session", "-t", "=" + SESION_COLA], capture_output=True, timeout=30)
        out["tmux_viva"] = t.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        return dict(out, ok=False, motivo="no se puede comprobar la cola: %s" % e)
    pv = procesos_vivos([PROCESO_ARNES])
    if pv["pids"] is None:
        return dict(out, ok=False, motivo="no se puede comprobar la cola: %s" % pv.get("motivo"))
    out["procesos_arnes"] = pv["pids"]
    out["ok"] = not out["tmux_viva"] and not out["procesos_arnes"]
    if not out["ok"]:
        out["motivo"] = "la cola de la fase B (semilla A) sigue viva (tmux %s: %s; procesos del arnes: %s)" % (
            SESION_COLA, out["tmux_viva"], out["procesos_arnes"])
    return out


def _json_ida_y_vuelta(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def _hora_local(iso):
    """Hora local sin zona de una marca ISO (con o sin zona)."""
    t = dt.datetime.fromisoformat(iso)
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def parrafos_md(texto):
    """Entradas de un .md: cada linea que empieza por "- " o "#" abre una; las lineas de continuacion (sangradas)
    se le unen; una linea en blanco la cierra."""
    out, cur = [], []
    for ln in texto.splitlines():
        if not ln.strip() or re.match(r"^(- |#)", ln):
            if cur:
                out.append("\n".join(cur))
            cur = [ln] if ln.strip() else []
        else:
            cur.append(ln)
    if cur:
        out.append("\n".join(cur))
    return out


def guarda_puerta(repo, results_esperado, puerta_mod=PUERTA, familias_esperadas=tuple(FAMILIAS)):
    """Check that the entry-rule result is committed, recorded in the decisions file and still recomputes the same."""
    repo = Path(repo)
    g = {"ok": False, "repo": str(repo), "fichero": PUERTA_REL, "pasos": {}, "motivo": None}

    def falla(paso, motivo):
        g["pasos"][paso] = {"ok": False, "motivo": motivo}
        g["motivo"] = "paso %s: %s" % (paso, motivo)
        return g

    pj = repo / PUERTA_REL
    if not pj.exists():
        return falla("a", "no existe %s" % PUERTA_REL)
    J = C.read_json(pj)
    if not isinstance(J, dict):
        return falla("a", "puerta_fase_b.json ilegible")
    if J.get("schema") != SCHEMA_PUERTA or J.get("enmienda") != TAG_ENM:
        return falla("a", "schema %r / enmienda %r inesperados" % (J.get("schema"), J.get("enmienda")))
    if J.get("results_root") != str(results_esperado):
        return falla("a", "results_root %r != %r" % (J.get("results_root"), str(results_esperado)))
    fams = J.get("familias") or {}
    if set(fams) != set(familias_esperadas):
        return falla("a", "familias %s != %s (una ejecucion con una sola familia pisa el fichero)"
                     % (sorted(fams), sorted(familias_esperadas)))
    for f, v in sorted(fams.items()):
        if not str(v.get("estado") or "").startswith("DECIDIDA"):
            return falla("a", "%s: estado %r (no DECIDIDA)" % (f, v.get("estado")))
        if not isinstance(v.get("entra"), bool):
            return falla("a", "%s: entra = %r no es booleano" % (f, v.get("entra")))
    entran = sorted(f for f, v in fams.items() if v["entra"])
    m = len(entran)
    g["pasos"]["a"] = {"ok": True, "familias": {f: {"entra": v["entra"], "estado": v["estado"],
                                                    "factores": v.get("factores")} for f, v in sorted(fams.items())},
                       "escrito": J.get("escrito")}
    rc, _, err = C.git(repo, "ls-files", "--error-unmatch", PUERTA_REL)
    if rc != 0:
        return falla("b", "puerta_fase_b.json no esta versionado (%s)" % err[:120])
    rc, st, err = C.git(repo, "status", "--porcelain", "--", PUERTA_REL, DECISIONS_REL)
    if rc != 0 or st:
        return falla("b", "cambios sin comprometer en puerta_fase_b.json o en el fichero de decisiones: %r" % (st or err)[:200])
    g["pasos"]["b"] = {"ok": True}
    sha_wt = C.sha256_file(pj)
    rc, blob, err = _git_bytes(repo, "show", "HEAD:%s" % PUERTA_REL)
    if rc != 0:
        return falla("c", "git show HEAD:%s fallo: %s" % (PUERTA_REL, err[:120]))
    sha_head = hashlib.sha256(blob).hexdigest()
    if sha_head != sha_wt:
        return falla("c", "sha256 de la copia de trabajo %s != HEAD %s" % (sha_wt[:16], sha_head[:16]))
    g["pasos"]["c"] = {"ok": True, "sha256": sha_wt}
    rc, blob, err = _git_bytes(repo, "show", "HEAD:%s" % DECISIONS_REL)
    if rc != 0:
        return falla("d", "git show HEAD:%s fallo: %s" % (DECISIONS_REL, err[:120]))
    parrafos = [p for p in parrafos_md(blob.decode("utf-8", "replace"))
                if "puerta_fase_b.json" in p and re.search(r"(?<![0-9a-f])%s(?![0-9a-f])" % sha_wt, p)]
    if not parrafos:
        return falla("d", "el fichero de decisiones de HEAD no tiene ningun parrafo con puerta_fase_b.json y su sha256 %s"
                     % sha_wt[:16])
    for p in parrafos:
        limpio = p.replace("*", "").replace("`", "")
        for f, v in fams.items():
            mt = re.search(r"\b%s\b(?:\s*\[[^\]\n]*\])?\W+(entra|no entra)\b" % re.escape(f), limpio, re.IGNORECASE)
            if not mt:
                return falla("d", "el parrafo no dice '%s: entra|no entra'" % f)
            if (mt.group(1).lower() == "entra") != v["entra"]:
                return falla("d", "el fichero de decisiones dice '%s: %s' y el JSON entra = %s" % (f, mt.group(1), v["entra"]))
        mm = (re.search(r"\bm\s*=\s*(\d+)\b", limpio)
              or re.search(r"\bcorrige\s+(\d+)\s+comparaci", limpio, re.IGNORECASE))
        if not mm or int(mm.group(1)) != m:
            return falla("d", "comparaciones declaradas %s != familias que entran %d" % (mm.group(1) if mm else None,
                                                                                         m))
    g["pasos"]["d"] = {"ok": True, "parrafos": parrafos,
                       "es_decision_del_usuario": [bool(re.match(r"^\s*-\s*\*\*D-\d+", p)) for p in parrafos]}
    rc, out, err = C.git(repo, "log", "--format=%H %cI", "-S%s" % sha_wt, "--", DECISIONS_REL)
    if rc != 0 or not out:
        return falla("e", "git log -S no encuentra el commit que anota el sha256 (%s)" % err[:120])
    commit, cfecha = out.splitlines()[-1].split()
    if C.git(repo, "merge-base", "--is-ancestor", commit, "HEAD")[0] != 0:
        return falla("e", "el commit %s no es antecesor de HEAD" % commit[:12])
    try:
        escrito = dt.datetime.fromisoformat(str(J.get("escrito")))
    except ValueError:
        return falla("e", "escrito = %r no es una fecha" % J.get("escrito"))
    if _hora_local(cfecha) < escrito:
        return falla("e", "el commit %s (%s) es anterior a escrito %s" % (commit[:12], cfecha, J.get("escrito")))
    g["pasos"]["e"] = {"ok": True, "commit": commit, "fecha": cfecha}
    for f in sorted(fams):
        nuevo = _json_ida_y_vuelta(puerta_mod.familia(str(results_esperado), f))
        if nuevo != fams[f]:
            return falla("f", "puerta.familia(%s) recalculada != la comprometida: los informes cambiaron" % f)
    g["pasos"]["f"] = {"ok": True}
    g.update(ok=True, entran=entran, m=m, familias={f: dict(entra=v["entra"], estado=v["estado"],
                                                          factores=v.get("factores"),
                                                          texto_si_no_entra=v.get("texto_si_no_entra"))
                                                  for f, v in sorted(fams.items())},
             sha256=sha_wt, commit_decision=commit, fecha_decision=cfecha, escrito=J.get("escrito"))
    return g


def conceder_permiso(g, results, familias, semillas=SEMILLAS):
    """Habilita el unico punto de apertura de T10 para estas familias, semillas y raiz."""
    if not (isinstance(g, dict) and g.get("ok")):
        raise PermisoDenegado("la guarda de la puerta no esta superada")
    _PERMISO.update(concedido=True, root=Path(results).resolve(), familias=frozenset(familias),
                    semillas=frozenset(semillas), motivo="guarda OK")


def retirar_permiso():
    _PERMISO.update(concedido=False, root=None, familias=frozenset(), semillas=frozenset(), motivo="retirado")


def _con_permiso(results, familias, semillas):
    return (_PERMISO["concedido"] and _PERMISO["root"] == Path(results).resolve()
            and set(familias) <= set(_PERMISO["familias"]) and set(semillas) <= set(_PERMISO["semillas"]))


def metricas_job(job_dir):
    """mlp_sin_barajar_metricas.metricas_job (camino principal de metrics_rev, MSB:96-155) mas el sha256 de la
    mascara de duplicados y el de labels.npy."""
    J = Path(job_dir)
    met = MSB.metricas_job(J)
    cfg = C.read_json(J / "config.json") or {}
    N, T = int(met["N"]), int(met["T"])
    dup, fsinfo = MR.fit_set_mask(cfg, N, T, log=lambda _m: None)
    c0, c1 = met["calib"]
    if int(dup[c0:c1].sum()) != met["n_dup_excluded"]:
        raise Parada("%s: la mascara de duplicados no da n_dup_excluded = %d" % (J.name, met["n_dup_excluded"]))
    met["fit_set_sha256"] = fsinfo.get("sha256")
    met["fit_set_mask_source"] = fsinfo.get("mask_source")
    met["labels_sha256"] = C.sha256_file(J / "labels.npy")
    return met


def _region(met):
    """Lo que tiene que coincidir entre un T10 y su T1: region, mascara, etiquetas, prevalencia y AUC_t."""
    return [met["calib"], met["eval_segments"], met["n_fit"], met["n_E"], met["fit_set_sha256"], met["labels_sha256"],
            met["prevalence"], met["auc_t"]]


def _d(a, b):
    return None if (a is None or b is None) else abs(float(a) - float(b))


def control_positivo(rutas, dids=tuple(ST.DS), semillas=SEMILLAS):
    """Recalcula los T1 de referencia (semillas pedidas) y los compara con su metrics.json, con el agregado y con
    clip_y_score.json.

    :return: (dict para el JSON, refs {(did, variante, ar, semilla): {...}}, contexto {V, cells, clip, alt, mlp_clip}).
    """
    agg = C.read_json(rutas.agg_dir / "aggregate" / "aggregate.json")
    stats_final = C.read_json(rutas.agg_dir / "stats.json")
    clip = C.read_json(rutas.clip)
    if not agg or not stats_final or not clip:
        raise Parada("no se puede leer el agregado (%s) o clip_y_score.json (%s)" % (rutas.agg_dir, rutas.clip))
    V, cells = agg["value_sets"]["main"], agg["cells"]
    helpers = MR.helper_hashes()
    alt = clip["hypothesis_sensitivity"]["cells"]["C10"]
    mlp_clip = {}
    for j in clip["mlp_score_convention"]["jobs"]:
        if j.get("seed") in semillas:
            mlp_clip.setdefault((j["dataset_id"], j["seed"]), {})[j["replica"]] = j
    filas, refs = [], {}
    lista = [x for x in jobs_control(semillas) if x[0] in dids]
    for did, var, ar, s in lista:
        d = rutas.results / t1_job_id(did, var, ar, s)
        t0 = time.time()
        fila = {"job_id": d.name, "dataset_id": did, "variante": var, "ar": ar, "semilla": s, "problemas": []}
        mj = MSB.metricas_de_metrics_json(d) if (d / "DONE").exists() else None
        mfull = C.read_json(d / "metrics.json") if mj else None
        if not mj or not mfull:
            fila["problemas"].append("sin DONE o sin metrics.json")
            filas.append(fila)
            continue
        met = metricas_job(d)
        dif = {k: _d(met.get(k), mj.get(k)) for k in MSB.CHECK_KEYS}
        dif = {k: v for k, v in dif.items() if v is not None}
        fila["dif_metrics_json"] = dif
        fila["max_abs_diff"] = max(dif.values()) if dif else None
        if not dif or len(dif) != len(MSB.CHECK_KEYS) or fila["max_abs_diff"] > TOL:
            fila["problemas"].append("no reproduce metrics.json en %s" % {k: v for k, v in dif.items() if v > TOL})
        fila["helpers_iguales"] = mj.get("helpers_sha256") == helpers
        if not fila["helpers_iguales"]:
            fila["problemas"].append("helpers de metrics_rev distintos de los del metrics.json")
        reg = mfull.get("regions") or {}
        fs = reg.get("fit_set") or {}
        arr = C.read_json(d / "arrays.json") or {}
        fila["region"] = {"calib": met["calib"] == reg.get("calib"),
                          "E_segments": met["eval_segments"] == reg.get("E_segments"),
                          "n_fit": met["n_fit"] == fs.get("n_fit"), "E_rows": met["n_E"] == reg.get("E_rows"),
                          "T": met["T"] == reg.get("T"), "N": met["N"] == reg.get("N"),
                          "mascara_sha256": met["fit_set_sha256"] == fs.get("sha256"),
                          "labels_sha256_arrays_json": met["labels_sha256"] == (arr.get("labels.npy") or {}).get(
                              "sha256")}
        if not all(fila["region"].values()):
            fila["problemas"].append("region distinta: %s" % [k for k, v in fila["region"].items() if not v])
        sn = {}
        if var in FAMILIAS:
            cid = FAMILIAS[var]["C_ar" if ar == "T" else "C_tsr"]
            ce = ((cells.get(did) or {}).get(cid) or {}).get("seeds", {}).get(s) or {}
            sn["cells_%s_seeds_%s_auroc_FUS" % (cid, s)] = _d(met["auroc_FUS"], ce.get("auroc_FUS"))
            sn["cells_%s_seeds_%s_auc_loc_RAW" % (cid, s)] = _d(met["auc_loc_RAW"], ce.get("auc_loc_RAW"))
        elif var == "original":
            vc = (V.get(did) or {}).get("C10") or {}
            sn["value_sets_main_C10_auroc_FUS"] = _d(met["auroc_FUS"], vc.get("auroc_FUS"))
            sn["value_sets_main_C10_auc_loc_RAW"] = _d(met["auc_loc_RAW"], vc.get("auc_loc_RAW"))
            sn["clip_C10_as_run"] = _d(met["auroc_FUS"], (alt.get("as_run") or {}).get(did))
            for s2 in ("A", "B"):
                c10 = ((cells.get(did) or {}).get("C10") or {}).get("seeds", {}).get(s2)
                if c10 is not None:
                    sn["cells_C10_seeds_%s_auroc_FUS" % s2] = _d(met["auroc_FUS"], c10.get("auroc_FUS"))
        elif var == "mlp":
            rep = "TSR" if ar == "F" else "AR"
            mc = mlp_clip.get((did, s)) or {}
            sn["clip_mlp_%s_%s_auroc_mse_as_run_FUS" % (s, rep)] = _d(met["auroc_FUS"], ((mc.get(rep) or {}).get(
                "auroc_mse_as_run") or {}).get("FUS"))
            fila_clip = mc.get(rep) or {}
            fila["clip_mlp"] = {"job_id_igual": fila_clip.get("job_id") == d.name,
                                "score_rebuilt_from_ae_errors_byte_equal": fila_clip.get(
                                    "score_rebuilt_from_ae_errors_byte_equal"),
                                "max_abs_diff_vs_metrics_json": fila_clip.get("max_abs_diff_vs_metrics_json"),
                                "auroc_rmse_FUS": (fila_clip.get("auroc_rmse") or {}).get("FUS")}
            fc = fila["clip_mlp"]
            if not (fc["job_id_igual"] and fc["score_rebuilt_from_ae_errors_byte_equal"] is True
                    and fc["max_abs_diff_vs_metrics_json"] is not None and fc["max_abs_diff_vs_metrics_json"] <= TOL
                    and isinstance(fc["auroc_rmse_FUS"], float)):
                fila["problemas"].append("fila del MLP en clip_y_score.json sin reconstruccion verificada: %s" % fc)
            otra = refs.get((did, "mlp", "T" if ar == "F" else "F", s))
            if otra is not None:
                c9 = ((cells.get(did) or {}).get("C9") or {}).get("seeds", {}).get(s) or {}
                sn["cells_C9_seeds_%s_auroc_FUS_media_replicas" % s] = _d(
                    (met["auroc_FUS"] + otra["metricas"]["auroc_FUS"]) / 2.0, c9.get("auroc_FUS"))
        fila["segundo_nivel"] = sn
        malos_sn = {k: v for k, v in sn.items() if v is None or v > TOL}
        if malos_sn:
            fila["problemas"].append("segundo nivel distinto o ausente: %s" % malos_sn)
        fila["divergente"] = met["divergent"]
        fila["segundos"] = round(time.time() - t0, 1)
        refs[(did, var, ar, s)] = {"job_id": d.name, "metricas": met,
                                   "scores_sha256": (arr.get("scores.npy") or {}).get("sha256")}
        filas.append(fila)
        estado = "OK" if not fila["problemas"] else "PROBLEMAS: %s" % fila["problemas"]
        log("control %-72s |dif| max %.1e  %s" % (d.name, fila["max_abs_diff"] or 0.0, estado))
    falta_c9 = [(did, s) for did in dids for s in semillas
                if not any(("cells_C9_seeds_%s_auroc_FUS_media_replicas" % s) in (f.get("segundo_nivel") or {})
                           for f in filas if f["dataset_id"] == did and f["variante"] == "mlp" and f["semilla"] == s)]
    h1 = ST.h1(copy.deepcopy(V))
    ok_h1, dif_h1 = MSB.reproduce(h1, stats_final["H1"])
    sha_stats = C.sha256_file(ST.__file__)
    stats_sha_ok = sha_stats == (stats_final.get("provenance") or {}).get("stats_sha256")
    rcs = (clip.get("range_and_clip") or {}).get("streams") or {}
    dif_alt = [_d((alt.get("alternative") or {}).get(x),
                  (((rcs.get(x) or {}).get("dA") or {}).get("auroc_with_clip") or {}).get("FUS")) for x in dids]
    alt_ok = all(v is not None and v == 0.0 for v in dif_alt)
    difs = [f["max_abs_diff"] for f in filas if f.get("max_abs_diff") is not None]
    difs_sn = [v for f in filas for v in (f.get("segundo_nivel") or {}).values() if v is not None]
    n_esperados = len(lista)
    c10_div = [f["dataset_id"] for f in filas if f["variante"] == "original" and f.get("divergente")]
    reg = (regresion_h1(agg, stats_final) if list(dids) == list(ST.DS) else
           {"ok": None, "nota": "no aplicable: control parcial (--streams)"})
    todo_ok = (len(filas) == n_esperados and all(not f["problemas"] for f in filas) and ok_h1 and stats_sha_ok
               and alt_ok and not falta_c9 and reg["ok"] is not False and not c10_div)
    out = {"jobs": filas, "n_jobs": len(filas), "n_esperados": n_esperados, "datasets": list(dids),
           "semillas": list(semillas), "completo": list(dids) == list(ST.DS),
           "n_jobs_por_semilla": {s: sum(1 for f in filas if f["semilla"] == s) for s in semillas},
           "max_abs_diff_metrics_json": max(difs) if difs else None,
           "max_abs_diff_segundo_nivel": max(difs_sn) if difs_sn else None,
           "tolerancia": TOL, "claves": list(MSB.CHECK_KEYS), "c9_sin_comprobar": falta_c9,
           "stats_h1_reproduce_stats_json": bool(ok_h1), "stats_h1_diferencias": dif_h1,
           "stats_py_sha256": sha_stats, "stats_py_sha256_igual_stats_json": bool(stats_sha_ok),
           "clip_C10_alternative_igual_range_and_clip": bool(alt_ok),
           "regresion_h1_camino_por_stream": reg,
           "c10_divergido": c10_div,
           "all_ok": bool(todo_ok),
           "regla": "el mismo calculo por job (mlp_sin_barajar_metricas.metricas_job) reproduce el metrics.json de "
                    "metrics_rev (|dif| <= 1e-12, mismos helpers, misma region y mascara) y las cifras del agregado "
                    "FINAL (cells por semilla, C10 de la unica ejecucion A, C9 = media de replicas) y de "
                    "clip_y_score.json (filas del MLP reconstruidas byte a byte); stats.h1 reproduce stats.json; el "
                    "camino por stream de E1 sobre los T1 sin modificar reproduce stats.json H1 Conv1D/Conv2D; el "
                    "control original (C10) no esta divergido en ningun stream (si lo estuviera, PARADA: m5)"}
    return out, refs, {"V": V, "cells": cells, "clip": clip, "alt": alt, "mlp_clip": mlp_clip}


def regresion_h1(agg, stats_final, tol=TOL):
    """Regresion del camino propio de E1 (valor_stream -> comparaciones_e1) con datos reales: aplicado a los T1 SIN
    modificar (cells.C1-C4.seeds por semilla, S*_d del agregado FINAL, divergidas segun divergent_seeds) tiene que
    reproducir stats.json H1 Conv1D/Conv2D_vs_original: d por stream, HL, IC, cobertura, p exacta, n, k y las reglas
    de deriva y dependencia (no Holm: H1 tiene m = 5). Orden de las medias: E1 hace AR/TSR por semilla y despues la
    media de semillas; aggregate hace semillas por celda y despues paradigmas; la diferencia es de ~1 ulp."""
    cells, V = agg["cells"], agg["value_sets"]["main"]
    dsets = agg.get("datasets") or {}
    vals, loc = {}, {}
    for fam in FAMILIAS:
        vals[fam], loc[fam] = {}, {}
        for x in ST.DS:
            S = list((dsets.get(x) or {}).get("S_star") or [])

            def rec(ar, s, fam=fam, x=x):
                ce = (cells.get(x) or {}).get(FAMILIAS[fam]["C_ar" if ar == "T" else "C_tsr"]) or {}
                v = (ce.get("seeds") or {}).get(s)
                if ce.get("lost") or v is None:
                    return None
                return {"metricas": {"auroc_FUS": v.get("auroc_FUS"), "auc_loc_RAW": v.get("auc_loc_RAW"),
                                     "divergent": s in (ce.get("divergent_seeds") or [])}}
            recs = {s: (rec("T", s), rec("F", s)) for s in S}
            vals[fam][x] = valor_stream(recs, S)["media"]
            loc[fam][x] = valor_stream(recs, S, "auc_loc_RAW")["media"]
    c10 = {x: ((V.get(x) or {}).get("C10") or {}).get("auroc_FUS") for x in ST.DS}
    lc10 = {x: ((V.get(x) or {}).get("C10") or {}).get("auc_loc_RAW") for x in ST.DS}
    fam_out = comparaciones_e1(vals, loc, c10, lc10, dict(c10), len(FAMILIAS), list(FAMILIAS))
    peor, distintos = 0.0, []
    for fam in FAMILIAS:
        nm = FAMILIAS[fam]["nombre"]
        a = fam_out["comparisons"]["%smod_vs_original" % nm]
        b = ((stats_final.get("H1") or {}).get("comparisons") or {}).get("%s_vs_original" % nm) or {}
        ra, rb = a.get("robustness") or {}, b.get("robustness") or {}
        pares = {"hl": (a.get("hl"), b.get("hl")),
                 "ci_lower": ((a.get("hl_ci") or {}).get("lower"), (b.get("hl_ci") or {}).get("lower")),
                 "ci_upper": ((a.get("hl_ci") or {}).get("upper"), (b.get("hl_ci") or {}).get("upper")),
                 "ci_coverage": ((a.get("hl_ci") or {}).get("coverage"), (b.get("hl_ci") or {}).get("coverage")),
                 "p_exacta": ((a.get("wilcoxon") or {}).get("p"), (b.get("wilcoxon") or {}).get("p")),
                 "n": (a.get("n"), b.get("n")), "k_pos": (a.get("k_pos"), b.get("k_pos")),
                 "k_zero": (a.get("k_zero"), b.get("k_zero")),
                 "deriva_hl": ((ra.get("drift") or {}).get("hl_auc_loc_raw"),
                               (rb.get("drift") or {}).get("hl_auc_loc_raw")),
                 "deriva_robusta": ((ra.get("drift") or {}).get("robust"), (rb.get("drift") or {}).get("robust")),
                 "dependencia_ok": (((ra.get("dependence") or {}).get("rule") or {}).get("dependence_ok"),
                                    ((rb.get("dependence") or {}).get("rule") or {}).get("dependence_ok"))}
        for x in ST.DS:
            pares["d_%s" % x] = ((a.get("d") or {}).get(x), (b.get("d") or {}).get(x))
        for k, (u, v) in pares.items():
            if isinstance(u, bool) or isinstance(v, bool) or u is None or v is None:
                if u != v:
                    distintos.append("%s.%s: %r != %r" % (nm, k, u, v))
            else:
                dd = abs(float(u) - float(v))
                peor = max(peor, dd)
                if dd > tol:
                    distintos.append("%s.%s: |dif| %.3g" % (nm, k, dd))
    return {"ok": not distintos, "max_abs_diff": peor, "distintos": distintos, "tolerancia": tol,
            "comparaciones": ["%s_vs_original" % FAMILIAS[f]["nombre"] for f in FAMILIAS],
            "nota": "solo diferencias frente a stats.json (ya publicado); no imprime cifras nuevas"}


def validar_job_t10(nombre_dir, cfg, arrays, sha_scores_fichero, did, fam, ar, semilla, ref, sha_arnes,
                    factores_puerta):
    """Punto 4 de la especificacion (modelo: mlp_sin_barajar_metricas.validar_job_t9). Solo diccionarios.

    :return: (lista de problemas, lista de motivos de PARADA inmediata).
    """
    malos, parada = [], []
    row = cfg.get("manifest_row") or {}
    t10 = cfg.get("t10_rep") or {}
    vs = t10.get("vs_reference") or {}
    harness = t10.get("harness") or {}
    if cfg.get("job_id") != nombre_dir:
        malos.append("config.job_id %r != directorio %r" % (cfg.get("job_id"), nombre_dir))
    esperado = {"block": "T10-REP", "seed": semilla, "rep": "1", "pipeline": "corrected", "variant": fam + "_rep",
                "dataset": nombre_ds(did)}
    for k, v in esperado.items():
        if row.get(k) != v:
            malos.append("manifest_row.%s = %r, se esperaba %r" % (k, row.get(k), v))
    if row.get("ar") is None or C.parse_bool(row.get("ar")) != (ar == "T"):
        malos.append("manifest_row.ar = %r no es %s" % (row.get("ar"), PARADIGMA[ar]))
    if t10.get("control") is not False or t10.get("ensayo_cpu") is not False or str(cfg.get("gpu_arg")) == "cpu":
        malos.append("control/ensayo_cpu/gpu_arg: %r/%r/%r" % (t10.get("control"), t10.get("ensayo_cpu"),
                                                              cfg.get("gpu_arg")))
    if t10.get("familia") != fam:
        malos.append("t10_rep.familia = %r" % t10.get("familia"))
    if t10.get("decision") != ARNES[semilla]["decision"]:
        malos.append("t10_rep.decision = %r, se esperaba %r (semilla %s)" % (t10.get("decision"),
                                                                            ARNES[semilla]["decision"], semilla))
    if t10.get("factores") != FAMILIAS[fam]["factores"] or t10.get("factores") != factores_puerta:
        malos.append("factores %r != %r (esperados) / %r (regla de entrada)" % (t10.get("factores"), FAMILIAS[fam]["factores"],
                                                                factores_puerta))
    fits = t10.get("fits") or []
    if t10.get("all_fits_assert_ok") is not True or len(fits) != N_AE or t10.get("requirements_failed") != []:
        malos.append("fits: assert %r, %d autoencoders, requisitos %r" % (t10.get("all_fits_assert_ok"), len(fits),
                                                                         t10.get("requirements_failed")))
    if (harness.get("puerta_git") != "OK" or harness.get("sha256") != sha_arnes
            or harness.get("tag") != ARNES[semilla]["tag"]):
        malos.append("arnes: puerta_git %r, tag %r, sha256 %s != %s (%s)" % (
            harness.get("puerta_git"), harness.get("tag"), str(harness.get("sha256"))[:16], sha_arnes[:16],
            ARNES[semilla]["tag"]))
    for etiqueta, v in (("labels.npy.identical", (vs.get("labels.npy") or {}).get("identical")),
                        ("feature_map.identical", (vs.get("feature_map") or {}).get("identical")),
                        ("regions_identical", vs.get("regions_identical")),
                        ("hyperparameters_frozen_identical", vs.get("hyperparameters_frozen_identical")),
                        ("desde_cache.todo_igual", (vs.get("desde_cache") or {}).get("todo_igual"))):
        if v is not True:
            malos.append("vs_reference.%s = %r" % (etiqueta, v))
    sc_ident = (vs.get("scores") or {}).get("identical")
    if sc_ident is True:
        parada.append("scores IDENTICOS a los del job sin modificar: la modificacion no ha actuado")
    elif sc_ident is not False:
        malos.append("vs_reference.scores.identical = %r" % sc_ident)
    rf = cfg.get("reference") or {}
    if rf.get("job_id") != ref["job_id"]:
        malos.append("reference.job_id %r != %r" % (rf.get("job_id"), ref["job_id"]))
    if not ref.get("scores_sha256") or rf.get("scores_sha256") != ref.get("scores_sha256"):
        malos.append("reference.scores_sha256 no es el del arrays.json del T1")
    sha_arr = (arrays.get("scores.npy") or {}).get("sha256")
    if not sha_arr or sha_arr != sha_scores_fichero:
        malos.append("sha256 de scores.npy %s != arrays.json %s" % (str(sha_scores_fichero)[:16], str(sha_arr)[:16]))
    return malos, parada


def abrir_job_t10(results, did, fam, ar, semilla, ref, sha_arnes, factores_puerta):
    """UNICO punto que lee un directorio T10. Exige el permiso de la guarda de la puerta.

    :return: registro del job, o None si no tiene DONE.
    :raises PermisoDenegado: sin guarda superada, otra raiz, otra familia u otra semilla.
    :raises Parada: el job no es lo que dice ser o no se evalua en la region de su T1.
    """
    if not _con_permiso(results, [fam], [semilla]):
        raise PermisoDenegado("abrir %s %s %s s%s sin permiso (%s)" % (did, fam, ar, semilla, _PERMISO["motivo"]))
    d = Path(results) / t10_job_id(did, fam, ar, semilla)
    if not (d / "DONE").exists():
        return None
    cfg = C.read_json(d / "config.json") or {}
    arrays = C.read_json(d / "arrays.json") or {}
    sha_fich = C.sha256_file(d / "scores.npy") if (d / "scores.npy").exists() else None
    malos, parada = validar_job_t10(d.name, cfg, arrays, sha_fich, did, fam, ar, semilla, ref, sha_arnes,
                                    factores_puerta)
    if parada or malos:
        raise Parada("%s: %s" % (d.name, "; ".join(parada + malos)))
    met = metricas_job(d)
    if _region(met) != _region(ref["metricas"]):
        nombres = ("calib", "eval_segments", "n_fit", "n_E", "fit_set_sha256", "labels_sha256", "prevalence", "auc_t")
        distintos = [n for n, a, b in zip(nombres, _region(met), _region(ref["metricas"])) if a != b]
        raise Parada("%s no se evalua en la misma region, con las mismas etiquetas, que %s: %s"
                     % (d.name, ref["job_id"], distintos))
    an = list(cfg.get("anomalies") or [])
    if (d / "anomalies.json").exists():
        an += [a for a in ((C.read_json(d / "anomalies.json") or {}).get("anomalies") or []) if a not in an]
    vs = (cfg.get("t10_rep") or {}).get("vs_reference") or {}
    sc = vs.get("scores") or {}
    claves = ("auroc_FUS", "auc_loc_RAW", "auroc_KMD_f", "auroc_MVD_f", "auroc_RAW", "auprc_FUS", "f1_FUS",
              "fpr_FUS", "recall_FUS", "eer_FUS", "auc_t", "prevalence", "n_E", "n_fit", "calib", "eval_segments",
              "frac_nonfinite_exec", "divergent", "nonfinite_replaced", "fit_set_sha256")
    return {"tag": TAG, "job_id": d.name, "familia": fam, "ar": PARADIGMA[ar], "dataset_id": did,
            "dataset": nombre_ds(did), "semilla": semilla,
            "reference_job": ref["job_id"], "metricas": {k: met.get(k) for k in claves},
            "misma_region_que_T1": True, "mismas_etiquetas_que_T1": True, "labels_sha256": met["labels_sha256"],
            "procedencia": {"harness_sha256": ((cfg.get("t10_rep") or {}).get("harness") or {}).get("sha256"),
                            "harness_tag": ((cfg.get("t10_rep") or {}).get("harness") or {}).get("tag"),
                            "runner_tag": (cfg.get("code") or {}).get("runner_tag"), "scores_sha256": sha_fich,
                            "scores_identical_to_reference": sc.get("identical"),
                            "scores_rows_different": sc.get("rows_different"),
                            "scores_max_abs_diff_vs_reference": sc.get("max_abs_diff"),
                            "factores": (cfg.get("t10_rep") or {}).get("factores"),
                            "done": (d / "DONE").read_text().strip(),
                            "wall_seconds": (C.read_json(d / "timing.json") or {}).get("wall_seconds")},
            "anomalias": an}


def estado_job(results, jid):
    """State of a run from its marker files: DONE, lost (not executable or failed at least 3 times) or not finished."""
    if (Path(results) / jid / "DONE").exists():
        return "DONE", None, 0
    fc = C.read_fail_count(results, jid)
    nonexec = [k for k in range(1, N_NONEXEC + 1)
               if (Path(results) / ".state" / jid / ("nonexec_%d.json" % k)).exists()]
    if nonexec:
        return "PERDIDA", "no ejecutable (nonexec_%d.json)" % nonexec[0], fc
    if fc >= C.MAX_FAILS:
        return "PERDIDA", "FAILED/BLOCKED (fail_count %d >= %d)" % (fc, C.MAX_FAILS), fc
    return "PENDIENTE", "sin DONE y con fail_count %d < %d: sin terminar, no perdida" % (fc, C.MAX_FAILS), fc


def censo_t10(results, familias, semillas):
    """Antes de abrir ninguna metrica: estado de cada job T10 (semillas x familias x streams x {AR, TSR}). Solo mira
    la existencia de DONE, .state/<job>/fail_count y .state/<job>/nonexec_k.json (rutas construidas, sin listar
    nada). Cuenta tambien los DONE de la semilla C, que no se abre. Exige el mismo permiso que abrir_job_t10.

    :return: dict JSON con listas; la clave "_est" lleva {(semilla, fam, did, ar): estado} (no va al JSON).
    """
    if not _con_permiso(results, familias, semillas):
        raise PermisoDenegado("censo de T10 sin permiso (%s)" % _PERMISO["motivo"])
    out = {"semillas": list(semillas), "familias": list(familias), "con_done": [], "perdidas": [], "pendientes": [],
           "semilla_c": {"jobs_con_done": [], "uso": "fuera de E1: post hoc, sin T1-C; no se abre ningun job sC"},
           "_est": {}}
    for s in semillas:
        for fam in familias:
            for did in ST.DS:
                for ar in ARS:
                    jid = t10_job_id(did, fam, ar, s)
                    est, motivo, fc = estado_job(results, jid)
                    out["_est"][(s, fam, did, ar)] = est
                    if est == "DONE":
                        out["con_done"].append(jid)
                        continue
                    fila = {"semilla": s, "familia": fam, "dataset_id": did, "paradigma": PARADIGMA[ar],
                            "job_id": jid, "fail_count": fc, "estado": est, "motivo": motivo}
                    out["perdidas" if est == "PERDIDA" else "pendientes"].append(fila)
    for fam in familias:
        for did in ST.DS:
            for ar in ARS:
                jc = t10_job_id(did, fam, ar, "C")
                if (Path(results) / jc / "DONE").exists():
                    out["semilla_c"]["jobs_con_done"].append(jc)
    return out


def prefijo_stream(est, familias, did, semillas=SEMILLAS):
    """Longest prefix of the seeds in which every cell of a stream that is not lost has finished (plan §4.5)."""
    celdas = [(f, ar) for f in familias for ar in ARS]
    for k in range(len(semillas), -1, -1):
        P = list(semillas[:k])
        perd = {}
        for (f, ar) in celdas:
            for s in P:
                if est.get((s, f, did, ar)) == "PERDIDA":
                    perd[(f, ar)] = s
                    break
        if all(est.get((s, f, did, ar)) == "DONE" for (f, ar) in celdas if (f, ar) not in perd for s in P):
            return P, perd
    return [], {}


def recuento_convergencia(results, fam, semilla):
    """Count of the entry rule with the convergence reports of one seed (A or B), without labels."""
    if not _con_permiso(results, [fam], [semilla]):
        raise PermisoDenegado("recuento de convergencia %s s%s sin permiso (%s)" % (fam, semilla, _PERMISO["motivo"]))
    orig = PUERTA.job_id
    if semilla != "A":
        PUERTA.job_id = lambda stream, familia, ar: orig(stream, familia, ar).replace("__sA__", "__s%s__" % semilla)
    try:
        v = PUERTA.familia(str(results), fam)
    finally:
        PUERTA.job_id = orig
    t = v["tsr"]
    return {"semilla": semilla, "estado": v["estado"], "cumple": v["entra"], "evaluables": t["evaluables"],
            "entrenados": t["entrenados"], "streams_con_algun_entrenado": t["streams_con_algun_entrenado"],
            "jobs_tsr_leidos": t["jobs_leidos"], "jobs_tsr_que_faltan": t["jobs_que_faltan"],
            "fuera_de_los_streams_del_prevuelo": v["fuera_de_los_streams_del_prevuelo"],
            "ar_descriptivo": v["ar_descriptivo"],
            "fuente": "puerta_fase_b.familia sobre los convergence_report.json de la semilla %s (sin etiquetas)"
                      % semilla,
            "oficial": semilla == "A"}


def _media(a, b):
    return None if (a is None or b is None) else (a + b) / 2.0


def _media_lista(vals):
    vals = list(vals)
    return None if (not vals or any(v is None for v in vals)) else float(sum(vals) / len(vals))


def valor_familia(rec_ar, rec_tsr, clave="auroc_FUS"):
    """Value of a family in one stream and one seed: mean of AR and TSR, with diverged runs imputed to 0.5."""
    def uno(rec):
        if rec is None:
            return {"perdida": True, "divergente": False, "crudo": None, "main": None,
                    "itt": 0.5 if clave == "auroc_FUS" else None, "noimp": None}
        v = rec["metricas"][clave]
        dv = bool(rec["metricas"]["divergent"])
        return {"perdida": False, "divergente": dv, "crudo": v, "main": 0.5 if dv else v, "itt": 0.5 if dv else v,
                "noimp": None if dv else v}
    a, t = uno(rec_ar), uno(rec_tsr)
    return {"AR": a["crudo"], "TSR": t["crudo"], "AR_perdida": a["perdida"], "TSR_perdida": t["perdida"],
            "AR_divergente": a["divergente"], "TSR_divergente": t["divergente"],
            "AR_main": a["main"], "TSR_main": t["main"],
            "media": _media(a["main"], t["main"]), "media_s_itt": _media(a["itt"], t["itt"]),
            "media_sin_imputar": _media(a["noimp"], t["noimp"])}


def valor_stream(recs, S, clave="auroc_FUS"):
    """Valor por stream con las semillas S de S*_d: media de semillas de la media AR/TSR (enmienda §5). Una celda
    divergida en cualquier semilla sale de la sensibilidad sin imputar (aggregate.build_value_sets, s_div_noimp).

    :param recs: {semilla: (registro AR o None, registro TSR o None)}.
    """
    por = {s: valor_familia(recs[s][0], recs[s][1], clave) for s in S}

    def m(k):
        return _media_lista(por[s][k] for s in S)
    out = {"semillas": list(S), "n_semillas": len(S), "por_semilla": por,
           "AR": m("AR"), "TSR": m("TSR"), "AR_main": m("AR_main"), "TSR_main": m("TSR_main"),
           "AR_perdida": bool(S) and any(por[s]["AR_perdida"] for s in S),
           "TSR_perdida": bool(S) and any(por[s]["TSR_perdida"] for s in S),
           "AR_divergente_en": [s for s in S if por[s]["AR_divergente"]],
           "TSR_divergente_en": [s for s in S if por[s]["TSR_divergente"]],
           "media": m("media"), "media_s_itt": m("media_s_itt"), "media_sin_imputar": m("media_sin_imputar")}
    medias = [por[s]["media"] for s in S]
    ok = len(S) >= 2 and all(v is not None for v in medias)
    out["min_entre_semillas"], out["max_entre_semillas"] = (min(medias), max(medias)) if ok else (None, None)
    return out


def _signo(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def s_clip(val_fam, alt, datasets, hl):
    """S-CLIP check: Hodges-Lehmann shift against the clipped control; not robust if zero or of the opposite sign."""
    vals = {d: (None if (val_fam.get(d) is None or alt.get(d) is None) else val_fam[d] - alt[d]) for d in datasets}
    if not datasets or hl is None or any(v is None for v in vals.values()):
        return {"evaluable": False, "robust": False, "hl_clip": None, "n": len(datasets), "d_clip": vals,
                "note": "no evaluable: no robusta"}
    h = ST.hodges_lehmann([vals[d] for d in datasets])
    robust = not (_signo(h) * _signo(hl) < 0 or h == 0)
    return {"evaluable": True, "robust": bool(robust), "hl_clip": h, "hl_primary": hl, "n": len(datasets),
            "d_clip": vals, "if_not_robust": "not robust to input clipping of the control",
            "rule": "enmienda §5.3: signo opuesto al de la primaria o exactamente 0 -> no robusta"}


def comparaciones_e1(vals, loc, c10, loc_c10, alt, m, entran):
    """Familia E1 (enmienda §5): cada familia que entra frente a C10, Wilcoxon exacto + Holm con m fijo, y las tres
    reglas de robustez (deriva, dependencia, S-CLIP). Devuelve la salida de stats.family con las anotaciones."""
    comps = {}
    for fam in entran:
        nombre = FAMILIAS[fam]["nombre"]
        d = {x: (None if (vals[fam].get(x) is None or c10.get(x) is None) else vals[fam][x] - c10[x]) for x in ST.DS}
        c = ST.compare(d)
        c["contrast"] = "AUC(%s modified) - AUC(original)" % nombre
        dl = {x: (None if (loc[fam].get(x) is None or loc_c10.get(x) is None) else loc[fam][x] - loc_c10[x])
              for x in ST.DS}
        c["robustness"] = {"drift": ST.drift(dl, c["datasets"], c["hl"]), "dependence": ST.dependence(d),
                           "s_clip": s_clip(vals[fam], alt, c["datasets"], c["hl"])}
        comps["%smod_vs_original" % nombre] = c
    fam_out = ST.family(comps, m, TAG_E1)
    fam_out["rotulo_post_hoc"] = ROTULO_POST_HOC
    for name, c in comps.items():
        ST.annotate(c, confirmatory=False, robust_rules=True)
        c["stats_annotate"] = {k: c.pop(k) for k in ("robust", "abstract_eligible", "x13_if_claimed_first_higher",
                                                     "x13_if_claimed_first_lower") if k in c}
        c["stats_annotate"]["nota"] = ("regla de stats.annotate sin S-CLIP; la regla de E1 esta en robust_e1 y "
                                       "resumen_permitido")
        c.pop("wording", None)
        rob = c["robustness"]
        c["robust_e1"] = bool(rob["drift"].get("robust") and (rob["dependence"].get("rule") or {}).get("dependence_ok")
                              and rob["s_clip"].get("robust"))
        c["resumen_permitido"] = c["robust_e1"]
        c["frases_de_robustez"] = [t for ok, t in (
            (rob["drift"].get("robust"), "not robust to within-block evaluation"),
            (rob["s_clip"].get("robust"), "not robust to input clipping of the control")) if not ok]
        dep = frase_dependencia(rob["dependence"].get("rule") or {})
        if dep:
            c["frases_de_robustez"].append(dep)
    return fam_out


def frase_dependencia(regla):
    """English sentence for a failed dependence check, naming the branch that failed (KNAD collapsed, without KNAD)."""
    if regla.get("dependence_ok"):
        return None
    ramas = []
    if regla.get("direction_preserved_KNAD_collapsed") is not True:
        ramas.append("when the six KNAD captures were collapsed into one unit")
    if regla.get("direction_preserved_without_KNAD") is not True:
        ramas.append("outside the KNAD testbed")
    if not ramas:
        ramas = ["when the six KNAD captures were collapsed into one unit", "outside the KNAD testbed"]
    return "the direction was not preserved %s" % ", nor ".join(ramas)


def sensibilidad(vals_s, c10, m, entran, etiqueta, semillas_por_stream=None):
    """Sensitivity analysis (S-ITT, diverged runs not imputed, or S-SEED): same family, no robustness checks."""
    comps = {}
    for fam in entran:
        d = {x: (None if (vals_s[fam].get(x) is None or c10.get(x) is None) else vals_s[fam][x] - c10[x])
             for x in ST.DS}
        c = ST.compare(d)
        c["contrast"] = "AUC(%s modified) - AUC(original) %s" % (FAMILIAS[fam]["nombre"], etiqueta)
        comps["%smod_vs_original" % FAMILIAS[fam]["nombre"]] = c
    f = ST.family(comps, m, "%s %s" % (TAG, etiqueta))
    for c in comps.values():
        ST.annotate(c, confirmatory=False, robust_rules=False)
    return {"tag": f["tag"], "rotulo_post_hoc": ROTULO_POST_HOC, "semillas_por_stream": semillas_por_stream,
            "m": f["m"], "status": f["status"], "comparisons": {k: resumen(c) for k, c in f["comparisons"].items()}}


def resumen(c):
    """MSB.resumen_comparacion sin "robust_rule_5.8" (la regla de stats.py sin S-CLIP) y con robust_e1 al lado."""
    r = MSB.resumen_comparacion(c)
    r.pop("robust_rule_5.8", None)
    r.pop("wording_rule", None)
    if "robust_e1" in c:
        r["robust_e1"] = c["robust_e1"]
        r["s_clip_robust"] = ((c.get("robustness") or {}).get("s_clip") or {}).get("robust")
    return r


def descriptivo(d, contraste):
    """Sin p (enmienda §5, descriptivo): HL, IC, k/n y el recuento con margen de empate 0,01 (plan §10)."""
    c = ST.compare(d, test=False)
    v = [c["d"][k] for k in c["datasets"]]
    ci = c.get("hl_ci") or {}
    return {"contrast": contraste, "n": c["n"], "datasets": c["datasets"], "d": c["d"], "hl": c["hl"],
            "ci_lower": ci.get("lower"), "ci_upper": ci.get("upper"), "ci_coverage": ci.get("coverage"),
            "k_pos": c["k_pos"], "k_neg": c["k_neg"], "k_zero": c["k_zero"],
            "margen_0_01": {"mayor": sum(1 for x in v if x > MARGEN), "menor": sum(1 for x in v if x < -MARGEN),
                            "empate": sum(1 for x in v if abs(x) <= MARGEN)},
            "equivalent_permitido": bool(c.get("equivalent_rope")),
            "regla_equivalent": "solo si ci_within_rope y reaches_95 (stats.describe; enmienda §6)", "tag": TAG}


def nombre_familia(fam, estilo):
    """Name of the family in the generated text; the default is 'modified X'."""
    return ("repaired-%s" if estilo == "repaired" else "modified %s") % FAMILIAS[fam]["nombre"]


def f3(x):
    """Cifra para el texto: tres decimales con signo, sin "-0.000". Un valor no nulo que redondea a 0 se da con dos
    cifras significativas; el 0 exacto, "0.000"."""
    if x == 0:
        return "0.000"
    if abs(x) < 0.0005:
        return "%+.2g" % x
    return "%+.3f" % x


def factores_en_texto(fam):
    """"P, I and R" / "P, I, R and A"."""
    f = FAMILIAS[fam]["factores"].split(",")
    return ", ".join(f[:-1]) + " and " + f[-1]


def declaraciones(fam, estilo="modified", recuento=None):
    """Statements that must accompany every figure of the comparison, in a fixed order."""
    out = [DECLARACION_DISENO, DECLARACION_TRAMO.format(nombre=FAMILIAS[fam]["nombre"],
                                                        factores=factores_en_texto(fam)),
           DECLARACION_TRAMO_BENIGNO,
           DECLARACION_PREVUELO.format(factores=factores_en_texto(fam), k=len(PUERTA.PREVUELO))]
    if recuento:
        out.append(frase_recuento(fam, recuento))
    return out


def frase_recuento(fam, recuento):
    """Declaracion del recuento de la puerta por semilla (orden A, B) a partir de recuento_convergencia, con la
    remision a la tabla de convergencia por la semilla C al final (DECLARACION_SEMILLA_C, sin cifras; m6)."""
    partes = []
    for s in SEMILLAS:
        r = recuento.get(s)
        if not r:
            continue
        t = ("With seed %s, %d of %d evaluable TSR autoencoders had a validation ratio below 0.90, in %d of the 9 "
             "streams" % (s, r["entrenados"], r["evaluables"], r["streams_con_algun_entrenado"]))
        if r["jobs_tsr_leidos"] < 9:
            t += " (from %d of the 9 TSR runs)" % r["jobs_tsr_leidos"]
        if r["cumple"] is False:
            t += ", which does not meet the criterion"
        partes.append(t + ".")
    return (DECLARACION_RECUENTO.format(nombre=nombre_familia(fam, "modified"), partes=" ".join(partes)).strip()
            + " " + DECLARACION_SEMILLA_C)


def texto_semillas(semillas_por_stream, datasets):
    """Lo que la frase dice de las semillas de los streams de la comparacion (regla del prefijo)."""
    ns = [len(semillas_por_stream.get(x) or []) for x in datasets]
    if ns and all(n == 2 for n in ns):
        return "means over two seeds (A, B) in every stream"
    if ns and all(n == 1 for n in ns):
        return "one seed (A)"
    k = sum(1 for n in ns if n == 2)
    return "means over two seeds (A, B) in %d of the %d streams and seed A alone in the others" % (k, len(ns))


def rellenar_plantilla(c, fam, m, estilo="modified", semillas_por_stream=None, recuento=None):
    """Fill the sentence template of the amendment for one comparison, with the verb set by Holm and the HL sign."""
    w = c.get("wording_e1") or {}
    status = str(c.get("status") or "")
    if status != "tested":
        return {"texto": None, "motivo": "sin plantilla: %s" % (status or "sin test"), "regla_verbo": w}
    if c.get("reject") and (c.get("hl") or 0) > 0:
        verbo, prep = "had a higher", "than"
    elif c.get("reject") and (c.get("hl") or 0) < 0:
        verbo, prep = "had a lower", "than"
    else:
        verbo, prep = "did not differ in", "from"
    ci = c.get("hl_ci") or {}
    wil = c.get("wilcoxon") or {}
    sps = semillas_por_stream or {x: ["A"] for x in ST.DS}
    n_eff = wil.get("n_eff")
    n_txt = ("n = %d (n_eff = %d non-zero differences)" % (c["n"], n_eff) if (n_eff is not None and n_eff < c["n"])
             else "n = %d" % c["n"])
    texto = PLANTILLA.format(m=m, nombre=nombre_familia(fam, estilo), verbo=verbo, prep=prep, hl=f3(c["hl"]),
                             cob="%.1f" % (100.0 * ci["coverage"]), lo=f3(ci["lower"]), hi=f3(ci["upper"]),
                             p="%.4f" % wil["p"], ph="%.4f" % c["p_holm"], n_txt=n_txt, n=c["n"], k=c["k_pos"],
                             semillas=texto_semillas(sps, c.get("datasets") or []))
    return {"texto": texto, "verbo": verbo, "n": c["n"], "n_eff": n_eff,
            "nota_n": "si n_eff < n el texto da los dos (plan §5.2)",
            "regla_verbo": "Holm <= 0,05 y signo del HL; si no, 'did not differ in'",
            "rotulo_obligatorio": ROTULO_POST_HOC,
            "declaraciones_obligatorias": declaraciones(fam, estilo, recuento),
            "nota": "el rotulo y las declaraciones van en la misma frase, tabla o pie que la cifra (enmienda §6 y "
                    "§0.2)"}


def revisar_palabras(textos):
    """Stop the run if any generated English text contains a forbidden word."""
    for i, t in enumerate(textos):
        malas = COMUN.palabras_prohibidas(t)
        if malas:
            raise Parada("el texto %d de textos_del_json lleva una palabra prohibida (%r)" % (i, malas[0]))
    return True


def textos_del_json(out):
    """Todo texto en ingles que el guion genera (lo que puede acabar en el articulo)."""
    t = [out.get("rotulo_post_hoc")]
    for c in ((out.get("e1") or {}).get("comparisons") or {}).values():
        t.append((c.get("plantilla") or {}).get("texto"))
        t += (c.get("plantilla") or {}).get("declaraciones_obligatorias") or []
        t.append(c.get("atribucion"))
        t += c.get("frases_de_robustez") or []
    for fam in (out.get("descriptivos") or {}).values():
        t += fam.get("declaraciones") or []
    return t


def id_semillas(out):
    usadas = sorted({s for S in (out.get("semillas_por_stream") or {}).values() for s in S})
    sid = "s" + "".join(usadas) if usadas else "s-"
    return sid + ("-PARCIAL" if out.get("parcial") else "")


def numeros_propuestos(out, rel_json):
    filas = []
    head = (out.get("provenance") or {}).get("git_head")
    comando = (out.get("provenance") or {}).get("command")
    sid = id_semillas(out)
    sps = out.get("semillas_por_stream") or {}
    desc_sem = "semillas por stream (regla del prefijo, plan §4.5): %s" % ", ".join(
        "%s %s" % (x, "+".join(sps.get(x) or []) or "-") for x in ST.DS)
    for name, c in ((out.get("e1") or {}).get("comparisons") or {}).items():
        base = "e1.comparisons.%s" % name
        ci = c.get("hl_ci") or {}
        for clave, valor in (("hl", c.get("hl")), ("hl_ci.lower", ci.get("lower")), ("hl_ci.upper", ci.get("upper")),
                             ("hl_ci.coverage", ci.get("coverage")), ("wilcoxon.p", (c.get("wilcoxon") or {}).get("p")),
                             ("p_holm", c.get("p_holm")), ("n", c.get("n")), ("k_pos", c.get("k_pos"))):
            filas.append({"ID": "E1-%s-%s-%s" % (sid, name, clave), "Valor": valor, "Ubicacion": "",
                          "Fuente": "%s %s:%s.%s (%s)%s%s" % (TAG, rel_json, base, clave, desc_sem,
                                                             (" " + ROTULO_PARCIAL) if out.get("parcial") else "",
                                                             (" " + ROTULO_GUION_DISTINTO)
                                                             if out.get("rotulo_guion_distinto") else ""),
                          "Script + commit": "revision/analysis/fase_b/e1_metricas_t10.py @ %s" % head,
                          "Comando para regenerar": comando})
    return filas


def marca_tiempo():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def stem_libre(directorio, prefijo, sufijo="", exts=(".json", ".md")):
    """<prefijo>_<fecha-hora><sufijo>, con _2, _3... si ya existe algun fichero con ese nombre."""
    directorio = Path(directorio)
    base = "%s_%s%s" % (prefijo, marca_tiempo(), sufijo)
    stem, k = base, 2
    while any((directorio / (stem + e)).exists() for e in exts):
        stem, k = "%s_%d" % (base, k), k + 1
    return directorio / stem


def escribir_sin_sobrescribir(path, texto):
    """Escribe un temporal y lo enlaza con os.link, que falla si el destino existe: nunca se pisa un fichero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    umask = os.umask(0)
    os.umask(umask)
    os.fchmod(fd, 0o666 & ~umask)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(texto)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, str(path))
        except FileExistsError:
            raise Parada("%s ya existe: no se sobrescribe" % path)
    finally:
        os.unlink(tmp)
    return path


def json_texto(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=C._json_default) + "\n"


def _p(x):
    return "—" if x is None else "%.4f" % x


def _si(x):
    return "sí" if x is True else ("no" if x is False else "—")


def resumen_md(out):
    """Resumen corto en espanol llano (el JSON es la fuente; este .md solo lo lee)."""
    L = ["# E1 de la fase B: modelos modificados frente al control %s" % TAG, ""]
    if out.get("en_seco"):
        L += ["**EN SECO: mundo sintético con valores inventados. Nada de esto es un resultado.**", ""]
    if out.get("parcial"):
        L += ["**%s** La enmienda pide las semillas A y B con la regla del prefijo; esta salida no la sustituye." %
              ROTULO_PARCIAL, ""]
    if out.get("rotulo_guion_distinto"):
        L += ["**%s** Aperturas anteriores con otro guion: %s." % (ROTULO_GUION_DISTINTO, "; ".join(
            "%s (sha256 %s)" % (e.get("inicio"), str(e.get("script_sha256"))[:16])
            for e in out.get("aperturas_con_otro_guion") or [])), ""]
    L += ["Rótulo obligatorio junto a cada cifra: *%s*." % ROTULO_POST_HOC, "",
          "- Fichero de datos: `%s` (creado %s; guion sha256 %s)." % (
              Path(out.get("_salida_json") or "").name, out.get("created"),
              ((out.get("provenance") or {}).get("script_sha256") or "")[:16]),
          "- Familias que entraron por la puerta (sin etiquetas, semilla A): %s. Holm corrige m = %d comparaciones."
          % (", ".join(FAMILIAS[f]["nombre"] for f in out.get("familias_que_entran") or []) or "ninguna",
             out.get("m_holm") or 0)]
    sps = out.get("semillas_por_stream") or {}
    L.append("- Semillas por stream (regla del prefijo, plan §4.5): " + "; ".join(
        "%s %s" % (x, "+".join(sps.get(x) or []) or "ninguna") for x in ST.DS) + ".")
    cb = out.get("cola_b") or {}
    L.append("- Cola de la semilla B: %s." % (cb.get("motivo") or "—"))
    sc = (out.get("censo_t10") or {}).get("semilla_c") or {}
    L.append("- Semilla C: %d jobs con DONE; no entran en E1 (post hoc, sin T1-C) y no se han abierto." % len(sc.get("jobs_con_done") or []))
    perd = out.get("celdas_perdidas") or []
    L.append("- Celdas perdidas (fuera de su comparación; en S-ITT valen 0,5): %s." % (
        "; ".join("%s %s %s (%s)" % (FAMILIAS[p["familia"]]["nombre"], p["dataset_id"], p["paradigma"], p["motivo"])
                  for p in perd) or "ninguna"))
    pf = out.get("perdidas_fuera_de_S_star") or []
    if pf:
        L.append("- Celdas perdidas en una semilla que no entra en su stream (no cuentan en E1): %s." % "; ".join(
            "%s %s %s semilla %s" % (FAMILIAS[p["familia"]]["nombre"], p["dataset_id"], p["paradigma"], p["semilla"])
            for p in pf))
    fu = out.get("fuera_de_S_star") or []
    L.append("- Ejecuciones con DONE fuera de S*_d: %s." % (
        ("%d; se calculan aparte en `fuera_de_S_star` (plan §4.6) y no entran en ninguna media ni test" % len(fu))
        if fu else "ninguna"))
    for fam, por in (out.get("recuento_convergencia_por_semilla") or {}).items():
        L.append("- Recuento de la puerta (sin etiquetas; la oficial es la de la A), %s: %s." % (
            FAMILIAS[fam]["nombre"], "; ".join(
                "semilla %s, %d de %d AE TSR bajo 0,90 en %d de 9 streams%s (%s)" % (
                    s_, r["entrenados"], r["evaluables"], r["streams_con_algun_entrenado"],
                    "" if r["jobs_tsr_leidos"] == 9 else ", con %d de 9 jobs TSR" % r["jobs_tsr_leidos"],
                    "cumple la regla" if r["cumple"] is True else "NO cumple la regla")
                for s_, r in por.items())))
    L.append("- La semilla C no se cuenta aquí (E1 no lee nada de la C, que se añadió post hoc): su recuento va en "
             "la tabla de convergencia, rotulado [POST HOC].")
    prev = out.get("streams_del_prevuelo") or []
    L.append("- Streams del prevuelo (la fase A eligió los factores con ellos, sin etiquetas): %s. Marcados en "
             "`per_stream` y con un descriptivo sin p que los excluye (`e_fuera_del_prevuelo`, [EXPLORATORIO])."
             % (", ".join(prev) or "ninguno"))
    clip_head = ((out.get("provenance") or {}).get("insumos") or {}).get("clip_y_score_json", {}).get("sha256_head")
    L.append("- La comparación con el MLP en RMSE toma la cifra de clip_y_score.json (no la recalcula E1; sus filas "
             "están reconstruidas byte a byte%s)." % (
                 " y el fichero es el de HEAD, sha256 %s" % clip_head[:16] if clip_head else
                 "; en seco no se compara con HEAD"))
    prev_ej = out.get("ejecuciones_previas") or []
    L.append("- Aperturas anteriores de T10 en el registro: %s." % (
        "; ".join("%s (rc %s%s)" % (e.get("inicio"), e.get("rc"), ", PARCIAL" if e.get("parcial") else "")
                  for e in prev_ej) or "ninguna"))
    L.append("- Sensibilidades del plan que E1 no calcula: %s" % "; ".join(
        "%s: %s" % (k, v) for k, v in (out.get("sensibilidades_no_calculadas") or {}).items()))
    div = [(j["familia"], j["dataset_id"], j["ar"], j["semilla"]) for j in out.get("per_job") or []
           if j["metricas"].get("divergent")]
    L.append("- Ejecuciones divergidas (> 1 %% de scores no finitos; ROC-AUC imputado a 0,5): %s." % (
        "; ".join("%s %s %s semilla %s" % (FAMILIAS[f]["nombre"], x, a, s) for f, x, a, s in div) or "ninguna"))
    L += ["", "| Familia | Diferencia (HL) | IC sin ajustar | p exacta | p de Holm | Streams a favor | "
          "1. Dentro de bloque | 2. Dependencia | 3. Recorte del control | ¿Puede ir al resumen? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in ((out.get("e1") or {}).get("comparisons") or {}).items():
        ci = c.get("hl_ci") or {}
        rob = c.get("robustness") or {}
        ic = ("[%s, %s] (%.1f %%)" % (f3(ci["lower"]), f3(ci["upper"]), 100.0 * ci["coverage"])
              if ci.get("lower") is not None else "—")
        L.append("| %s modificado | %s | %s | %s | %s | %s de %s | %s | %s | %s | %s |" % (
            name.replace("mod_vs_original", ""), f3(c["hl"]) if c.get("hl") is not None else "—", ic,
            _p((c.get("wilcoxon") or {}).get("p")), _p(c.get("p_holm")), c.get("k_pos"), c.get("n"),
            _si((rob.get("drift") or {}).get("robust")),
            _si(((rob.get("dependence") or {}).get("rule") or {}).get("dependence_ok")),
            _si((rob.get("s_clip") or {}).get("robust")), _si(c.get("robust_e1"))))
    if not ((out.get("e1") or {}).get("comparisons")):
        L.append("| (ninguna familia entra: sin test) | | | | | | | | | |")
    L += ["", "Cómo leer la tabla. Diferencia = mediana de Hodges–Lehmann de (ROC-AUC del modelo modificado − ROC-AUC "
          "del control) sobre los streams; positiva quiere decir que el modificado detecta mejor. El intervalo no "
          "está ajustado por Holm. La p exacta es la del Wilcoxon de rangos con signo; la de Holm es la que decide "
          "(≤ 0,05). Streams a favor = en cuántos el modificado queda por encima. Las tres reglas de robustez de la "
          "enmienda (1: la misma comparación con el score crudo dentro de bloque; 2: con las seis capturas KNAD como "
          "una sola y sin KNAD; 3: frente al control con la entrada recortada) no cambian la p: deciden cómo se puede "
          "escribir y si la frase puede ir al resumen (solo si pasa las tres).", ""]
    for name, c in ((out.get("e1") or {}).get("comparisons") or {}).items():
        pl = c.get("plantilla") or {}
        L.append("**%s.** " % name.replace("mod_vs_original", "") + (
            "Frase en inglés, con su rótulo y sus declaraciones al lado:" if pl.get("texto") else
            "Sin frase: %s." % pl.get("motivo")))
        if pl.get("texto"):
            L += ["", "> %s %s" % (pl["texto"], c.get("atribucion") or "")]
            for fr in c.get("frases_de_robustez") or []:
                L.append("> (%s)" % fr)
            for dcl in pl.get("declaraciones_obligatorias") or []:
                L.append("> %s" % dcl)
        L.append("")
    av = out.get("avisos") or []
    if av:
        L += ["Avisos:", ""] + ["- %s" % a for a in av] + [""]
    L.append("[ENMIENDA-1] [EXPLORATORIO]")
    return "\n".join(L) + "\n"


def _provenance(t0, base):
    return {"script": str(Path(__file__).resolve()), "script_sha256": SCRIPT_SHA256,
            "e1_version": E1_VERSION, "command": "%s %s" % (sys.executable, " ".join(sys.argv)),
            "msb_sha256": MSB_SHA256, "puerta_py_sha256": PUERTA_PY_SHA256, "comun_sha256": COMUN_SHA256,
            "guion_tag": base.get("guion"),
            "metrics_rev_helpers_sha256": MR.helper_hashes(), "metrics_rev_version": MR.METRICS_REV_VERSION,
            "stats_py_sha256": C.sha256_file(ST.__file__), "stats_version": ST.STATS_VERSION,
            "versions": MR.versions(), "env": base.get("env"), "plan": base.get("plan"),
            "enmienda": base.get("enmienda"), "arnes": base.get("arnes"), "insumos": base.get("insumos"),
            "registro": base.get("registro"), "git_head": C.head_commit(R),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
            "seconds": round(time.time() - t0, 1)}


def guardas_base(semillas=SEMILLAS):
    base = guarda_entorno()
    base["enmienda"] = guarda_enmienda(R)
    base["arnes"] = guarda_arnes(R, semillas)
    return base


def ejecutar_control(rutas, dids=tuple(ST.DS), semillas=SEMILLAS):
    """--control-positivo: guardas de entorno, plan, enmienda y arneses; despues el control (ninguna ruta T10)."""
    t0 = time.time()
    try:
        base = guardas_base(semillas)
        base["insumos"] = guarda_insumos(rutas)
    except GuardaError as e:
        log("GUARDA: %s" % e)
        return 3
    try:
        ctl, _refs, _ctx = control_positivo(rutas, dids, semillas)
    except Parada as e:
        log("PARADA: %s" % e)
        return 1
    out = {"schema": SCHEMA_CONTROL, "tag": TAG, "rotulo_post_hoc": ROTULO_POST_HOC, "created": C.now_iso(),
           "alcance": "control positivo del calculo de E1 sobre los jobs T1 de referencia (semillas %s); no abre "
                      "nada de T10" % "+".join(semillas),
           "en_seco": rutas.en_seco, "control_positivo": ctl, "provenance": _provenance(t0, base)}
    destino = (Path(rutas.salida_control) if rutas.salida_control else
               stem_libre(rutas.dir_salida, PREFIJO_CONTROL, exts=(".json",)).with_suffix(".json"))
    try:
        escribir_sin_sobrescribir(destino, json_texto(out))
    except Parada as e:
        log("PARADA: %s" % e)
        return 1
    rutas.salida_control = destino
    log("control positivo: %d/%d jobs (%s), |dif| max metrics.json %s, segundo nivel %s, stats.h1 %s, stats.py %s, "
        "camino por stream frente a H1 %s (|dif| %s) -> %s"
        % (ctl["n_jobs"], ctl["n_esperados"], ctl["n_jobs_por_semilla"], ctl["max_abs_diff_metrics_json"],
           ctl["max_abs_diff_segundo_nivel"], ctl["stats_h1_reproduce_stats_json"],
           ctl["stats_py_sha256_igual_stats_json"], ctl["regresion_h1_camino_por_stream"]["ok"],
           ctl["regresion_h1_camino_por_stream"].get("max_abs_diff"), "OK" if ctl["all_ok"] else "FALLA"))
    log("JSON -> %s (%.1f s)" % (destino, time.time() - t0))
    return 0 if ctl["all_ok"] else 1


def registrar(path, entrada):
    """Una linea JSON mas en el registro append-only de ejecuciones (nunca se reescribe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(entrada, cuando=C.now_iso()), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def ejecuciones_previas(path):
    """TODAS las aperturas anteriores de T10 (cada evento "inicio" del registro, con el "fin" que le siga), terminaran
    con 0, con 1 (PARADA) o sin fin (excepcion o corte): se citan en la salida, no se ocultan."""
    filas = leer_registro(path)
    out = []
    for i, f in enumerate(filas):
        if f.get("evento") != "inicio":
            continue
        fin = next((g for g in filas[i + 1:] if g.get("evento") in ("fin", "inicio")), None)
        fin = fin if (fin and fin.get("evento") == "fin") else {}
        out.append({"inicio": f.get("cuando"), "fin": fin.get("cuando"), "rc": fin.get("rc"),
                    "oficial": f.get("oficial"), "parcial": f.get("parcial"), "semillas": f.get("semillas"),
                    "script_sha256": f.get("script_sha256"), "command": f.get("command"),
                    "git_head": f.get("git_head"), "salida": fin.get("salida"),
                    "salida_sha256": fin.get("salida_sha256"), "salida_md": fin.get("salida_md")})
    return out


def ejecutar_completo(rutas, descriptivos_no_entran=False, estilo="modified", solo_a=None, congelar_b=None,
                      pendientes_como_perdidas=None):
    """Guardas (entorno, arneses, insumos, guion, registro, cola B, puerta), control positivo, censo de T10 y, solo
    entonces, T10 y la familia E1. solo_a: D-NN de --solo-semilla-a (True sin decision: guarda). Codigos: 0 OK;
    1 PARADA; 3 guarda (sin abrir nada de T10)."""
    t0 = time.time()
    retirar_permiso()
    rutas.salida = rutas.salida_md = None
    registro = _registro_de(rutas)
    semillas = ("A",) if solo_a else SEMILLAS
    try:
        base = guardas_base(semillas)
        base["insumos"] = guarda_insumos(rutas)
        if rutas.oficial:
            base["guion"] = guarda_guion(R)
            base["registro"] = guarda_registro(R, registro)
        if Path(rutas.results).resolve() != Path(rutas.results_esperado).resolve():
            raise GuardaError("la raiz de resultados %s no es la de la puerta %s" % (rutas.results,
                                                                                     rutas.results_esperado))
        base["cola_b"] = guarda_cola_b(rutas, solo_a, congelar_b)
        if pendientes_como_perdidas:
            dec = decision_posterior(rutas.repo_puerta, pendientes_como_perdidas, ULTIMA_DECISION_PREVIA,
                                     MARCADOR_PENDIENTES, "--pendientes-como-perdidas")
            dec["momento"] = decision_antes_de_abrir(rutas.repo_puerta, dec["decision"], registro,
                                                     "--pendientes-como-perdidas", MARCADOR_PENDIENTES)
            base["pendientes_como_perdidas"] = dec
        previas = ejecuciones_previas(registro)
    except GuardaError as e:
        log("GUARDA: %s" % e)
        return 3
    otro_guion = [e for e in previas if e.get("script_sha256") != SCRIPT_SHA256]
    if otro_guion:
        log("AVISO: %d aperturas anteriores de T10 con otro guion (%s): la salida va rotulada"
            % (len(otro_guion), sorted({str(e.get("script_sha256"))[:16] for e in otro_guion})))
    g = guarda_puerta(rutas.repo_puerta, rutas.results_esperado)
    if not g["ok"]:
        log("GUARDA DE LA PUERTA: %s (no se abre nada de T10)" % g["motivo"])
        return 3
    head_decisiones = C.head_commit(rutas.repo_puerta)
    if not head_decisiones:
        log("GUARDA: no se puede leer el HEAD de %s: sin el, la apertura no queda fechada en el registro (no se abre "
            "nada de T10)" % rutas.repo_puerta)
        return 3
    log("puerta: entran %s (m = %d); decision en %s (%s); semillas %s (%s)"
        % (g["entran"] or "ninguna", g["m"], g["commit_decision"][:12], g["fecha_decision"], "+".join(semillas),
           base["cola_b"]["motivo"]))
    try:
        ctl, refs, ctx = control_positivo(rutas, tuple(ST.DS), semillas)
    except Parada as e:
        log("PARADA en el control positivo: %s" % e)
        return 1
    if ctl.get("c10_divergido"):
        log("PARADA: el control original (C10) esta divergido en %s: E1 no imputa el control; "
            "hace falta una decision de los autores. No se abre T10" % ctl["c10_divergido"])
        return 1
    if not ctl["all_ok"]:
        log("PARADA: el control positivo no reproduce los T1 (|dif| max %s); no se abre T10"
            % ctl["max_abs_diff_metrics_json"])
        return 1
    abrir = list(g["entran"]) + ([f for f in FAMILIAS if f not in g["entran"]] if descriptivos_no_entran else [])
    conceder_permiso(g, rutas.results, abrir, semillas)
    try:
        cola = (estado_cola() if not rutas.en_seco else
                {"ok": True, "nota": "en seco: no se mira la cola real (el mundo sintetico no tiene cola)"})
        if not cola["ok"]:
            log("GUARDA: %s (no se abre nada de T10)" % cola.get("motivo"))
            return 3
        censo = censo_t10(rutas.results, abrir, semillas)
        pend_a = [p for p in censo["pendientes"] if p["semilla"] == "A"]
        if pend_a and not pendientes_como_perdidas:
            log("GUARDA: %d jobs T10 de la semilla A sin DONE que no estan FAILED/BLOCKED ni son no ejecutables (sin "
                "terminar; con la cola A cerrada no deberian existir): %s. Tratarlos como celda perdida exige una "
                "decision (--pendientes-como-perdidas D-NN). No se abre nada de T10."
                % (len(pend_a), [p["job_id"] for p in pend_a]))
            return 3
        if pendientes_como_perdidas:
            lista = texto_tras_marcador(base["pendientes_como_perdidas"]["parrafo"], MARCADOR_PENDIENTES)
            sin_listar = [p["job_id"] for p in pend_a if p["job_id"] not in lista]
            if sin_listar:
                log("GUARDA: %s no lista (tras 'PENDIENTES COMO PERDIDAS:') los jobs sin terminar de la A %s; no se "
                    "abre nada de T10" % (pendientes_como_perdidas, sin_listar))
                return 3
        entrada = {"evento": "inicio", "script_sha256": SCRIPT_SHA256, "e1_version": E1_VERSION,
                   "command": "%s %s" % (sys.executable, " ".join(sys.argv)), "dir_salida": str(rutas.dir_salida),
                   "oficial": rutas.oficial, "en_seco": rutas.en_seco, "git_head": head_decisiones,
                   "repo_decisiones": str(rutas.repo_puerta), "git_head_proyecto": C.head_commit(R),
                   "semillas": list(semillas), "parcial": bool(solo_a), "cola_b": base["cola_b"].get("motivo"),
                   "congelar_semilla_b": congelar_b, "puerta_sha256": g["sha256"],
                   "familias_abiertas": abrir, "estilo": estilo,
                   "pendientes_como_perdidas": pendientes_como_perdidas,
                   "jobs_con_done": len(censo["con_done"]), "perdidas": len(censo["perdidas"]),
                   "pendientes": len(censo["pendientes"])}
        registrar(registro, entrada)
        rc = 1
        try:
            rc = _abrir_y_calcular(rutas, base, g, ctl, refs, ctx, abrir, censo, cola, estilo, t0, semillas,
                                   bool(solo_a), previas, otro_guion)
        finally:
            registrar(registro, {"evento": "fin", "rc": rc, "oficial": rutas.oficial, "parcial": bool(solo_a),
                                 "salida": str(rutas.salida) if rutas.salida else None,
                                 "salida_sha256": (C.sha256_file(rutas.salida)
                                                   if rutas.salida and Path(rutas.salida).exists() else None),
                                 "salida_md": str(rutas.salida_md) if rutas.salida_md else None,
                                 "salida_md_sha256": (C.sha256_file(rutas.salida_md)
                                                      if rutas.salida_md and Path(rutas.salida_md).exists() else None),
                                 "segundos": round(time.time() - t0, 1)})
        return rc
    finally:
        retirar_permiso()


def _abrir_y_calcular(rutas, base, g, ctl, refs, ctx, abrir, censo, cola, estilo, t0, semillas, parcial, previas,
                      otro_guion=()):
    """Abre los jobs T10 de S*_d (con el permiso ya concedido), calcula E1 y escribe la salida (JSON y .md). Los DONE
    de las semillas consideradas que quedan fuera de S*_d se abren y se calculan APARTE (plan §4.6: todas las
    ejecuciones se reportan, con la etiqueta de su uso); no entran en ninguna media ni en ningun test."""
    est = dict(censo.pop("_est"))
    pend_como_perd = base.get("pendientes_como_perdidas")
    if pend_como_perd:
        for p in censo["pendientes"]:
            if p["semilla"] == "A":
                est[("A", p["familia"], p["dataset_id"], "T" if p["paradigma"] == "AR" else "F")] = "PERDIDA"
                p["motivo"] += "; tratada como celda perdida por %s" % pend_como_perd["decision"]
    entran = list(g["entran"])
    motivo_de = {(p["semilla"], p["familia"], p["dataset_id"], p["paradigma"]): p for p in
                 censo["perdidas"] + censo["pendientes"]}
    grupos = [entran] + [[f] for f in abrir if f not in entran]
    prefijo = {}
    for grupo in grupos:
        if not grupo:
            continue
        for did in ST.DS:
            S, perd = prefijo_stream(est, grupo, did, semillas)
            for f in grupo:
                prefijo.setdefault(f, {})[did] = (S, {k: v for k, v in perd.items() if k[0] == f})
    semillas_por_stream = {did: (prefijo[entran[0]][did][0] if entran else []) for did in ST.DS}
    per_job, perdidas, fuera, pend_b, avisos, perdidas_fuera = [], [], [], [], [], []
    recs = {}
    try:
        for fam in abrir:
            for did in ST.DS:
                S, perd = prefijo[fam][did]
                for (f, ar), s_perd in sorted(perd.items()):
                    p = dict(motivo_de[(s_perd, fam, did, PARADIGMA[ar])])
                    p["regla"] = ("celda perdida en la semilla %s: perdida en todas las semillas de S*_d (%s) y fuera "
                                  "de su comparacion (plan §4.5, §12.2)" % (s_perd, "+".join(S)))
                    perdidas.append(p)
                for s in semillas:
                    for ar in ARS:
                        jid = t10_job_id(did, fam, ar, s)
                        if s not in S:
                            motivo = ("fuera de S*_d = %s (la regla del prefijo corta en este stream)"
                                      % ("+".join(S) or "vacio"))
                            if est[(s, fam, did, ar)] == "DONE":
                                rec = abrir_job_t10(rutas.results, did, fam, ar, s, refs[(did, fam, ar, s)],
                                                    base["arnes"]["semillas"][s]["sha256_tag"],
                                                    g["familias"][fam]["factores"])
                                if rec is None:
                                    raise Parada("%s: tenia DONE en el censo y ya no lo tiene" % jid)
                                fuera.append(dict(rec, uso=ETIQUETA_FUERA, motivo=motivo))
                            elif est[(s, fam, did, ar)] == "PERDIDA":
                                p = dict(motivo_de[(s, fam, did, PARADIGMA[ar])])
                                p["regla"] = motivo + "; celda perdida en una semilla que no entra en este stream"
                                perdidas_fuera.append(p)
                            elif s == "B" and est[(s, fam, did, ar)] == "PENDIENTE":
                                pend_b.append(jid)
                            continue
                        if (fam, ar) in perd:
                            recs[(fam, did, ar, s)] = None
                            continue
                        ref = refs[(did, fam, ar, s)]
                        rec = abrir_job_t10(rutas.results, did, fam, ar, s, ref,
                                            base["arnes"]["semillas"][s]["sha256_tag"],
                                            g["familias"][fam]["factores"])
                        if rec is None:
                            raise Parada("%s: tenia DONE en el censo y ya no lo tiene" % jid)
                        recs[(fam, did, ar, s)] = rec
                        per_job.append(rec)
                        if rec["metricas"]["divergent"]:
                            avisos.append("%s: > 1 %% de scores no finitos en ejecucion; ROC-AUC y AUC_loc imputados "
                                          "a 0,5 (plan §12.2)" % rec["job_id"])
                        if rec["anomalias"]:
                            avisos.append("%s: anomalias %s" % (rec["job_id"], rec["anomalias"]))
    except PermisoDenegado as e:
        log("GUARDA: %s" % e)
        return 3
    except Parada as e:
        log("PARADA: %s" % e)
        return 1
    if perdidas:
        avisos.append("%d celdas perdidas (plan §12.2): %s" % (len(perdidas), ["%s (%s)" % (p["job_id"], p["motivo"])
                                                                              for p in perdidas]))
    if pend_b:
        avisos.append("semilla B: %d jobs sin DONE (sin terminar) acortan S*_d a la semilla "
                      "A en sus streams (plan §4.5): %s" % (len(pend_b), pend_b))
    if fuera:
        avisos.append("%d jobs con DONE fuera de S*_d: se calculan aparte (plan §4.6) y no entran en ninguna media "
                      "ni test: %s" % (len(fuera), [x["job_id"] for x in fuera]))
    if perdidas_fuera:
        avisos.append("%d celdas perdidas en semillas fuera de S*_d (no cuentan en E1): %s"
                      % (len(perdidas_fuera), [p["job_id"] for p in perdidas_fuera]))
    if censo["semilla_c"]["jobs_con_done"]:
        avisos.append("semilla C: %d jobs sC con DONE que E1 no usa ni abre (post hoc, sin T1-C)"
                      % len(censo["semilla_c"]["jobs_con_done"]))
    if parcial:
        avisos.append(ROTULO_PARCIAL)
    if otro_guion:
        avisos.append("%s: %d de %d aperturas anteriores con otro sha256 del guion (%s)" % (
            ROTULO_GUION_DISTINTO, len(otro_guion), len(previas),
            "; ".join("%s: %s" % (e.get("inicio"), str(e.get("script_sha256"))[:16]) for e in otro_guion)))
    try:
        recuento = {fam: {s: recuento_convergencia(rutas.results, fam, s) for s in semillas} for fam in abrir}
    except PermisoDenegado as e:
        log("GUARDA: %s" % e)
        return 3
    for fam in abrir:
        for s, r in recuento[fam].items():
            if r["cumple"] is not True:
                avisos.append("%s, semilla %s: el recuento de la puerta no cumple la regla (%d de %d AE TSR bajo 0,90, "
                              "%d de 9 streams, %d de 9 jobs TSR leidos): se declara junto a cada cifra"
                              % (FAMILIAS[fam]["nombre"], s, r["entrenados"], r["evaluables"],
                                 r["streams_con_algun_entrenado"], r["jobs_tsr_leidos"]))
    prevuelo = [x for x in ST.DS if slug(x) in PUERTA.PREVUELO]
    c10 = {x: refs[(x, "original", "T", SEMILLA_ORIGINAL)]["metricas"]["auroc_FUS"] for x in ST.DS}
    loc_c10 = {x: refs[(x, "original", "T", SEMILLA_ORIGINAL)]["metricas"]["auc_loc_RAW"] for x in ST.DS}
    alt = {x: (ctx["alt"].get("alternative") or {}).get(x) for x in ST.DS}

    def mlp_de(x, S):
        por = {}
        for s in S:
            mc = ctx["mlp_clip"].get((x, s)) or {}
            por[s] = {"MSE": _media_lista(((mc.get(r) or {}).get("auroc_mse_as_run") or {}).get("FUS")
                                          for r in ("TSR", "AR")),
                      "RMSE": _media_lista(((mc.get(r) or {}).get("auroc_rmse") or {}).get("FUS")
                                           for r in ("TSR", "AR"))}
        return {"MSE": _media_lista(por[s]["MSE"] for s in S), "RMSE": _media_lista(por[s]["RMSE"] for s in S),
                "por_semilla": por}

    def recs_de(fam, x, S, desde_refs=False):
        if desde_refs:
            return {s: tuple(refs[(x, fam, ar, s)] for ar in ARS) for s in S}
        return {s: tuple(recs.get((fam, x, ar, s)) for ar in ARS) for s in S}

    per_stream, vals, loc, vals_itt, vals_noimp, sin_mod = {}, {}, {}, {}, {}, {}
    for fam in abrir:
        per_stream[fam], vals[fam], loc[fam], vals_itt[fam], vals_noimp[fam], sin_mod[fam] = {}, {}, {}, {}, {}, {}
        for x in ST.DS:
            S = prefijo[fam][x][0]
            va = valor_stream(recs_de(fam, x, S), S)
            vl = valor_stream(recs_de(fam, x, S), S, "auc_loc_RAW")
            un = valor_stream(recs_de(fam, x, S, desde_refs=True), S)
            mlp = mlp_de(x, S)
            vals[fam][x], loc[fam][x] = va["media"], vl["media"]
            vals_itt[fam][x], vals_noimp[fam][x] = va["media_s_itt"], va["media_sin_imputar"]
            sin_mod[fam][x] = un
            per_stream[fam][x] = dict(va, tag=TAG, loc=vl["media"], loc_AR=vl["AR"], loc_TSR=vl["TSR"],
                                      stream_del_prevuelo=x in prevuelo,
                                      C10=c10[x], C10_loc=loc_c10[x], C10_recortado=alt[x],
                                      C10_nota="original: una ejecucion (T1-A) para todas las semillas (AG-12)",
                                      MLP_MSE=mlp["MSE"], MLP_RMSE=mlp["RMSE"], MLP_por_semilla=mlp["por_semilla"],
                                      sin_modificar={"AR": un["AR_main"], "TSR": un["TSR_main"], "media": un["media"],
                                                     "por_semilla": {s: {"AR": un["por_semilla"][s]["AR_main"],
                                                                         "TSR": un["por_semilla"][s]["TSR_main"]}
                                                                     for s in S}})
            for s in S:
                for ar in ARS:
                    cid = FAMILIAS[fam]["C_ar" if ar == "T" else "C_tsr"]
                    ce = (((ctx["cells"].get(x) or {}).get(cid) or {}).get("seeds") or {}).get(s) or {}
                    v = un["por_semilla"][s]["%s_main" % PARADIGMA[ar]]
                    if _d(v, ce.get("auroc_FUS")) is None or _d(v, ce.get("auroc_FUS")) > TOL:
                        avisos.append("%s %s %s semilla %s: la version sin modificar (%r) no es cells.%s.seeds.%s (%r)"
                                      % (x, fam, PARADIGMA[ar], s, v, cid, s, ce.get("auroc_FUS")))
    m = g["m"]
    e1 = comparaciones_e1(vals, loc, c10, loc_c10, alt, m, entran) if entran else {
        "m": 0, "status": "ninguna familia entra: sin test (enmienda §4)", "comparisons": {}, "tag": TAG_E1,
        "rotulo_post_hoc": ROTULO_POST_HOC}
    palabra = "repaired" if estilo == "repaired" else "modified"
    for name, c in (e1.get("comparisons") or {}).items():
        fam = [f for f in entran if name.startswith(FAMILIAS[f]["nombre"])][0]
        c["wording_e1"] = {"nota": "la frase de stats.wording no se guarda: su verbo no es el de E1; la unica frase "
                                   "citable es plantilla.texto (enmienda §6)"}
        c["semillas_por_stream"] = {x: prefijo[fam][x][0] for x in ST.DS}
        c["plantilla"] = rellenar_plantilla(c, fam, m, estilo, c["semillas_por_stream"], recuento[fam])
        c["atribucion"] = (ATRIBUCION if (c.get("status") == "tested" and c.get("reject")) else
                           ATRIBUCION_SIN_DIFERENCIA).format(nombre=palabra)
        c["resumen"] = resumen(c)
    sens = {"s_itt": None, "s_div_noimp": None, "s_seed": None}
    if entran:
        sens["s_itt"] = sensibilidad(vals_itt, c10, m, entran, "[SENSIBILIDAD plan §12.2 S-ITT]", semillas_por_stream)
        sens["s_div_noimp"] = sensibilidad(vals_noimp, c10, m, entran,
                                           "[SENSIBILIDAD plan §12.2 divergidas sin imputar]", semillas_por_stream)
        Ls = [len(semillas_por_stream[x]) for x in ST.DS]
        L = min(Ls)
        if L == max(Ls):
            sens["s_seed"] = {"tag": "%s [SENSIBILIDAD plan §4.5 S-SEED]" % TAG, "prefijo_comun": L,
                              "nota": "todos los streams tienen el mismo |S*_d|: S-SEED coincide con la principal"}
        elif L == 0:
            sens["s_seed"] = {"tag": "%s [SENSIBILIDAD plan §4.5 S-SEED]" % TAG, "prefijo_comun": 0,
                              "nota": "algun stream tiene |S*_d| = 0: S-SEED no es calculable (AG-7)"}
        else:
            v_seed = {f: {x: valor_stream(recs_de(f, x, semillas_por_stream[x][:L]),
                                          semillas_por_stream[x][:L])["media"] for x in ST.DS} for f in entran}
            sp = {x: semillas_por_stream[x][:L] for x in ST.DS}
            sens["s_seed"] = sensibilidad(v_seed, c10, m, entran, "[SENSIBILIDAD plan §4.5 S-SEED]", sp)
            sens["s_seed"]["prefijo_comun"] = L
    desc = {}
    for fam in abrir:
        nm = FAMILIAS[fam]["nombre"]
        ps = per_stream[fam]
        mods = {ar: {x: ps[x]["%s_main" % PARADIGMA[ar]] for x in ST.DS} for ar in ARS}
        desc[fam] = {
            "tag": TAG, "semillas_por_stream": {x: prefijo[fam][x][0] for x in ST.DS},
            "declaraciones": declaraciones(fam, estilo, recuento[fam]),
            "nota_mlp_rmse": "MLP_RMSE sale de clip_y_score.json (auroc_rmse.FUS, media de las dos replicas): E1 no lo "
                             "recalcula; exige que esa fila reconstruyera el score byte a byte desde ae_errors.npy y "
                             "reprodujera metrics.json (control positivo) y que el fichero sea el de HEAD (insumos)",
            "nota_c": "frente a la version sin modificar de la MISMA semilla (T1-A / T1-B) cambian a la vez los "
                      "factores %s y el tramo del 90 %% con su min-max (DECLARACION_TRAMO): no aisla los factores"
                      % FAMILIAS[fam]["factores"],
            "a_vs_mlp_mse": descriptivo({x: _dif(vals[fam][x], ps[x]["MLP_MSE"]) for x in ST.DS},
                                        "AUC(%s modified) - AUC(MLP, MSE as run, same seeds)" % nm),
            "a_vs_mlp_rmse": descriptivo({x: _dif(vals[fam][x], ps[x]["MLP_RMSE"]) for x in ST.DS},
                                         "AUC(%s modified) - AUC(MLP rescored with RMSE, same seeds)" % nm),
            "b_primaria_solo_tsr": descriptivo({x: _dif(mods["F"][x], c10[x]) for x in ST.DS},
                                               "AUC(%s modified, TSR) - AUC(original)" % nm),
            "c_vs_sin_modificar": descriptivo({x: _dif(vals[fam][x], sin_mod[fam][x]["media"]) for x in ST.DS},
                                              "AUC(%s modified) - AUC(%s unmodified), same seeds" % (nm, nm)),
            "c_vs_sin_modificar_AR": descriptivo({x: _dif(mods["T"][x], sin_mod[fam][x]["AR_main"]) for x in ST.DS},
                                                 "AR: modified - unmodified, same seeds"),
            "c_vs_sin_modificar_TSR": descriptivo({x: _dif(mods["F"][x], sin_mod[fam][x]["TSR_main"])
                                                   for x in ST.DS}, "TSR: modified - unmodified, same seeds"),
            "d_ar_menos_tsr": descriptivo({x: _dif(mods["T"][x], mods["F"][x]) for x in ST.DS},
                                          "AUC(%s modified AR) - AUC(%s modified TSR)" % (nm, nm)),
            "e_fuera_del_prevuelo": dict(descriptivo({x: _dif(vals[fam][x], c10[x]) for x in ST.DS
                                                      if x not in prevuelo},
                                                     "AUC(%s modified) - AUC(original), %d streams outside the "
                                                     "pre-flight, no test" % (nm, len(ST.DS) - len(prevuelo))),
                                         tag="%s [EXPLORATORIO] sin los streams del prevuelo %s" % (TAG, prevuelo))}
        if fam not in entran:
            desc[fam]["frente_a_C10_sin_test"] = descriptivo({x: _dif(vals[fam][x], c10[x]) for x in ST.DS},
                                                             "AUC(%s modified) - AUC(original), sin test" % nm)
    no_entran = {f: {"estado": g["familias"][f]["estado"], "texto": g["familias"][f]["texto_si_no_entra"],
                     "valores_descriptivos": ("calculados (--descriptivos-no-entran)" if f in abrir else
                                          "no calculados (sin --descriptivos-no-entran)")}
                 for f in FAMILIAS if f not in entran}
    out = {"schema": SCHEMA, "tag": TAG, "rotulo_post_hoc": ROTULO_POST_HOC, "alcance": ALCANCE,
           "parcial": parcial, "rotulo_parcial": ROTULO_PARCIAL if parcial else None,
           "rotulo_guion_distinto": ROTULO_GUION_DISTINTO if otro_guion else None,
           "aperturas_con_otro_guion": [{"inicio": e.get("inicio"), "script_sha256": e.get("script_sha256"),
                                         "rc": e.get("rc")} for e in otro_guion],
           "semillas_consideradas": list(semillas), "semillas_por_stream": semillas_por_stream,
           "n_semillas_por_stream": {x: len(semillas_por_stream[x]) for x in ST.DS},
           "created": C.now_iso(), "en_seco": rutas.en_seco, "oficial": rutas.oficial,
           "cola_b": base["cola_b"], "pendientes_como_perdidas": base.get("pendientes_como_perdidas"),
           "guarda_puerta": g, "cola": cola, "censo_t10": censo, "m_holm": m, "familias_que_entran": entran,
           "familias_abiertas": abrir,
           "control_positivo": {k: v for k, v in ctl.items() if k != "jobs"},
           "control_positivo_jobs": [{k: f.get(k) for k in ("job_id", "semilla", "max_abs_diff", "problemas")}
                                     for f in ctl["jobs"]],
           "per_job": per_job, "per_stream": per_stream, "e1": e1, "s_itt": sens["s_itt"],
           "s_div_noimp": sens["s_div_noimp"], "s_seed": sens["s_seed"],
           "descriptivos": desc, "margen_0_01": MARGEN, "no_entran": no_entran, "celdas_perdidas": perdidas,
           "fuera_de_S_star": fuera, "etiqueta_fuera_de_S_star": ETIQUETA_FUERA,
           "perdidas_fuera_de_S_star": perdidas_fuera, "pendientes_semilla_b": pend_b,
           "recuento_convergencia_por_semilla": recuento, "streams_del_prevuelo": prevuelo,
           "ejecuciones_previas": previas,
           "sensibilidades_no_calculadas": {"S-DUPCAL": NOTA_S_DUPCAL}, "avisos": avisos}
    out["provenance"] = _provenance(t0, base)
    stem = stem_libre(rutas.dir_salida, PREFIJO_SALIDA, "_PARCIAL-solo-A" if parcial else "")
    salida, salida_md = stem.with_suffix(".json"), stem.with_suffix(".md")
    try:
        rel = str(salida.resolve().relative_to(R))
    except ValueError:
        rel = str(salida)
    out["numbers_propuestos"] = numeros_propuestos(out, rel)
    try:
        revisar_palabras(textos_del_json(out))
    except Parada as e:
        log("PARADA: %s" % e)
        return 1
    out["provenance"]["seconds"] = round(time.time() - t0, 1)
    out["_salida_json"] = str(salida)
    md = resumen_md(out)
    out.pop("_salida_json")
    out["resumen_md"] = str(salida_md)
    try:
        escribir_sin_sobrescribir(salida, json_texto(out))
        rutas.salida = salida
        escribir_sin_sobrescribir(salida_md, md)
        rutas.salida_md = salida_md
    except Parada as e:
        log("PARADA: %s" % e)
        return 1
    for name, c in (e1.get("comparisons") or {}).items():
        r = c["resumen"]
        log("%s n %s HL %s p %s Holm %s robusta_e1 %s" % (name, r["n"], r["hl"], r["p_exact"], r["p_holm"],
                                                          c["robust_e1"]))
    log("JSON -> %s; resumen -> %s (%.1f s)" % (salida, salida_md, time.time() - t0))
    return 0


def _dif(a, b):
    return None if (a is None or b is None) else a - b


SECO_N, SECO_T, SECO_FM, SECO_CAL = 60_000, 20_000, 5_000, 4_000
SECO_FUERZA = {"original": 0.30, "conv1d": 0.10, "conv2d": 0.12, "mlp": 0.40, "conv1d_rep": 3.00, "conv2d_rep": 0.30}
SECO_DECISION_CONGELAR = "D-78"
SECO_LINEA_CONGELAR = ("- **D-78 (en seco, 2026-10-19).** Decision de prueba del mundo sintetico. **CONGELAR SEMILLA "
                       "B:** lo que no tenga DONE acorta S*_d.")
SECO_DECISION_SOLO_A = "D-79"
SECO_LINEA_SOLO_A = ("- **D-79 (en seco, 2026-10-19).** Decision de prueba del mundo sintetico. **E1 PARCIAL SOLO A:** "
                     "salida rotulada PARCIAL.")
SECO_ESCENARIOS = ("ab", "ab-limpio", "m1")


def _escribe_npy(path, arr):
    np.save(path, arr)
    return C.sha256_file(path)


def _sintetico_job(d, scores, labels, cfg):
    d.mkdir(parents=True, exist_ok=True)
    sha_s = _escribe_npy(d / "scores.npy", scores)
    sha_l = _escribe_npy(d / "labels.npy", labels)
    C.write_json_atomic(d / "arrays.json", {"scores.npy": {"sha256": sha_s, "shape": list(scores.shape),
                                                           "dtype": str(scores.dtype)},
                                            "labels.npy": {"sha256": sha_l, "shape": list(labels.shape)}})
    C.write_json_atomic(d / "config.json", dict(cfg, job_id=d.name))
    C.write_json_atomic(d / "timing.json", {"wall_seconds": 1.0})
    return sha_s


def parrafo_puerta(sha, entra):
    """Paragraph with the entry-rule result in the format of the decisions file, for the synthetic world."""
    fams = " ".join("**%s [%s]: %s**." % (FAMILIAS[f]["nombre"], FAMILIAS[f]["factores"],
                                          "ENTRA" if entra[f] else "NO ENTRA") for f in FAMILIAS)
    k = sum(1 for f in FAMILIAS if entra[f])
    return ("- **Resultado mecanico de la puerta de la fase B (enmienda §4), en seco. No es una decision del "
            "usuario:**\n  es la regla de la enmienda aplicada por `puerta_fase_b.py`. Fichero\n"
            "  `revision/analysis/reparacion/puerta_fase_b.json`, sha256 `%s`.\n  %s Holm de E1 corrige **%d "
            "comparaciones** (enmienda §4).") % (sha, fams, k)


def construir_mundo_seco(base, escenario="ab", sha_arnes=None):
    """Build a synthetic world (runs, aggregate, queue log and a temporary git repository) for the dry run."""
    if escenario not in SECO_ESCENARIOS:
        raise GuardaError("escenario %r desconocido (%s)" % (escenario, ", ".join(SECO_ESCENARIOS)))
    base = Path(base).resolve()
    if C.is_under(base, R):
        raise GuardaError("el mundo en seco no puede vivir dentro del repositorio (%s)" % base)
    if base.exists() or base.is_symlink():
        raise GuardaError("el mundo en seco necesita un directorio que todavia no exista (%s): nunca se escribe encima "
                          "de nada" % base)
    base.mkdir(parents=True)
    root = base / "results" / "revision"
    agg = base / "agg"
    repo = base / "repo"
    for p in (root, agg / "aggregate", repo / "revision" / "analysis" / "reparacion", base / "cache", base / "logs",
              base / "salidas"):
        p.mkdir(parents=True, exist_ok=True)
        if C.is_under(p, R):
            raise GuardaError("%s cae dentro del repositorio (enlace?)" % p)
    sha_arnes = sha_arnes or {s: v["sha256_tag"] for s, v in guarda_arnes(R)["semillas"].items()}
    rng = np.random.default_rng(20261019)
    N, T, FM, CAL = SECO_N, SECO_T, SECO_FM, SECO_CAL
    cells, V, alt, as_run, rc, mlp_jobs, datasets = {}, {}, {}, {}, {}, [], {}
    for k, did in enumerate(ST.DS):
        ds = nombre_ds(did)
        y = np.zeros(N, np.int64)
        for a in rng.integers(T + CAL + 1000, N - 6000, size=3):
            y[int(a):int(a) + 5000] = 1
        aux = np.zeros(N, bool)
        aux[rng.choice(np.arange(T, T + CAL), size=40, replace=False)] = True
        cache = base / "cache" / slug(did)
        cache.mkdir(parents=True, exist_ok=True)
        sha_aux = _escribe_npy(cache / "aux_dup_train.npy", aux)
        regions = {"units": "indices de fila (EN SECO)", "n_rows": N, "fm": [0, FM], "ae_train": [FM, T],
                   "train": [0, T], "execution": [T, N], "post_train": [T, N],
                   "fit_set_exclusion": {"path": str(cache / "aux_dup_train.npy"), "sha256": sha_aux},
                   "detector_protocol_candidates": {MR.PRIMARY_CANDIDATE: {
                       "calib": [T, T + CAL], "eval": [T + CAL, N], "calib_in_sample": False,
                       "calib_uses_labels": False, "regression_only": False}}}
        fondo = rng.gamma(2.0, 0.05, N)

        def scores_de(var, ar, s):
            r = np.random.default_rng(int(hashlib.sha256(("%s|%s|%s|%s" % (did, var, ar, s)).encode()).hexdigest()[:8],
                                          16))
            efecto = SECO_FUERZA[var] * (1.0 + 0.4 * np.sin(k + (0.3 if ar == "T" else 0.0)))
            return (fondo + r.normal(0.0, 0.03, N) + efecto * y * r.uniform(0.2, 1.0, N)).astype(np.float64)

        refs_seco = {}
        for s in SEMILLAS:
            lista = (([("original", "T")] if s == SEMILLA_ORIGINAL else [])
                     + [(f, a) for f in FAMILIAS for a in ARS] + [("mlp", "F"), ("mlp", "T")])
            for var, ar in lista:
                d = root / t1_job_id(did, var, ar, s)
                row = {"block": "T1-%s-%s" % (s, MSB.corpus(ds)), "dataset": ds, "variant": var,
                       "ar": str(ar == "T"), "paradigm": PARADIGMA[ar], "pipeline": "corrected",
                       "pipeline_version": "v1", "seed": s, "rep": "1"}
                sha = _sintetico_job(d, scores_de(var, ar, s), y, {"manifest_row": row, "regions": regions,
                                                                    "gpu_arg": 0, "en_seco": True})
                met = metricas_job(d)
                C.write_json_atomic(d / "metrics.json", {
                    "schema": MR.SCHEMA, "status": "ok", "job_id": d.name, "en_seco": True,
                    "primary": {"value": met["auroc_FUS"], "auc_loc_raw": met["auc_loc_RAW"]},
                    "E": {"KMD_f": {"auroc": met["auroc_KMD_f"]}, "MVD_f": {"auroc": met["auroc_MVD_f"]},
                          "RAW": {"auroc": met["auroc_RAW"]},
                          "FUS": {"auroc": met["auroc_FUS"], "auprc": met["auprc_FUS"], "f1": met["f1_FUS"],
                                  "fpr": met["fpr_FUS"]}, "trivial": {"auc_t": met["auc_t"]}},
                    "regions": {"calib": met["calib"], "E_segments": met["eval_segments"], "E_rows": met["n_E"],
                                "T": met["T"], "N": met["N"],
                                "fit_set": {"n_fit": met["n_fit"], "sha256": met["fit_set_sha256"]}},
                    "provenance": {"helpers_sha256": MR.helper_hashes()}})
                (d / "DONE").write_text("OK en seco\n")
                refs_seco[(var, ar, s)] = (d.name, sha, row, met)
        un = {key: v[3] for key, v in refs_seco.items()}
        cells[did] = {}
        for f in FAMILIAS:
            for a in ARS:
                cells[did][FAMILIAS[f]["C_ar" if a == "T" else "C_tsr"]] = {"seeds": {s: {
                    "auroc_FUS": un[(f, a, s)]["auroc_FUS"], "auc_loc_RAW": un[(f, a, s)]["auc_loc_RAW"]}
                    for s in SEMILLAS}}
        cells[did]["C9"] = {"seeds": {s: {"auroc_FUS": (un[("mlp", "F", s)]["auroc_FUS"]
                                                        + un[("mlp", "T", s)]["auroc_FUS"]) / 2.0} for s in SEMILLAS}}
        cells[did]["C10"] = {"seeds": {s: {"auroc_FUS": un[("original", "T", "A")]["auroc_FUS"],
                                           "auc_loc_RAW": un[("original", "T", "A")]["auc_loc_RAW"]}
                                       for s in SEMILLAS}}
        datasets[did] = {"S_star": list(SEMILLAS)}
        V[did] = {c: {"auroc_FUS": float(rng.uniform(0.4, 0.95)), "auc_loc_RAW": float(rng.uniform(0.45, 0.6))}
                  for c in ST.CONFIGS}
        for c, (v, a) in (("C1", ("conv1d", "T")), ("C2", ("conv1d", "F")), ("C3", ("conv2d", "T")),
                          ("C4", ("conv2d", "F"))):
            V[did][c] = {kk: float(np.mean([un[(v, a, s)][kk2] for s in SEMILLAS]))
                         for kk, kk2 in (("auroc_FUS", "auroc_FUS"), ("auc_loc_RAW", "auc_loc_RAW"))}
        V[did]["C9"] = {"auroc_FUS": float(np.mean([cells[did]["C9"]["seeds"][s]["auroc_FUS"] for s in SEMILLAS])),
                        "auc_loc_RAW": float(rng.uniform(0.45, 0.6))}
        V[did]["C10"] = {"auroc_FUS": un[("original", "T", "A")]["auroc_FUS"],
                         "auc_loc_RAW": un[("original", "T", "A")]["auc_loc_RAW"]}
        as_run[did] = un[("original", "T", "A")]["auroc_FUS"]
        alt[did] = float(min(1.0, as_run[did] + rng.uniform(-0.02, 0.08)))
        rc[did] = {"dA": {"auroc_with_clip": {"FUS": alt[did]}}}
        for s in SEMILLAS:
            for rep, ar in (("TSR", "F"), ("AR", "T")):
                mm = un[("mlp", ar, s)]["auroc_FUS"]
                mlp_jobs.append({"job_id": refs_seco[("mlp", ar, s)][0], "dataset_id": did, "seed": s,
                                 "replica": rep, "auroc_mse_as_run": {"FUS": mm},
                                 "score_rebuilt_from_ae_errors_byte_equal": True,
                                 "max_abs_diff_vs_metrics_json": 0.0,
                                 "auroc_rmse": {"FUS": float(min(1.0, mm + 0.03))}})
        for s in SEMILLAS:
            for fam in FAMILIAS:
                for ar in ARS:
                    jid = t10_job_id(did, fam, ar, s)
                    d = root / jid
                    especial = None
                    if escenario == "ab" and s == "B" and fam == "conv2d" and did == "D3" and ar == "F":
                        especial = 1
                    if escenario == "ab" and s == "B" and fam == "conv2d" and did == "D2" and ar == "T":
                        especial = C.MAX_FAILS
                    if especial is not None:
                        d.mkdir(parents=True, exist_ok=True)
                        st = root / ".state" / jid
                        st.mkdir(parents=True, exist_ok=True)
                        (st / "fail_count").write_text("%d\n" % especial)
                        continue
                    sc = scores_de(fam + "_rep", ar, s)
                    if escenario == "ab" and s == "A" and fam == "conv1d" and ar == "F" and did == "D5":
                        sc[T + rng.choice(N - T, size=int(0.02 * (N - T)), replace=False)] = np.nan
                    rid, sha_ref, row, _met = refs_seco[(fam, ar, s)]
                    cfg = {"manifest_row": dict(row, job_id=jid, block="T10-REP", variant=fam + "_rep", rep="1",
                                                priority="9100"),
                           "regions": regions, "gpu_arg": "0", "code": {"runner_tag": "runner-v2"},
                           "reference": {"job_id": rid, "scores_sha256": sha_ref},
                           "t10_rep": {"decision": ARNES[s]["decision"], "tag": TAG, "control": False, "replica": 1,
                                       "ensayo_cpu": False, "familia": fam, "factores": FAMILIAS[fam]["factores"],
                                       "harness": {"sha256": sha_arnes[s], "puerta_git": "OK",
                                                   "tag": ARNES[s]["tag"]},
                                       "fits": [{"ae_index": i, "assert_ok": True} for i in range(N_AE)],
                                       "all_fits_assert_ok": True, "requirements_failed": [],
                                       "vs_reference": {"scores": {"identical": False, "rows_different": N - FM,
                                                                   "max_abs_diff": 1.0},
                                                        "labels.npy": {"identical": True},
                                                        "feature_map": {"identical": True}, "regions_identical": True,
                                                        "hyperparameters_frozen_identical": True,
                                                        "desde_cache": {"todo_igual": True}}},
                           "anomalies": [], "en_seco": True}
                    _sintetico_job(d, sc, y, cfg)
                    no_entra = escenario == "m1" and fam == "conv2d"
                    razones = [1.02] * 4 if (no_entra and ar == "F") else [0.5, 0.6, 0.7, 1.02]
                    C.write_json_atomic(d / "convergence_report.json", {
                        "schema": "jisa-t10-rep-convergencia/1", "en_seco": True, "familia": fam,
                        "paradigma": PARADIGMA[ar], "factores": FAMILIAS[fam]["factores"],
                        "autoencoders": [{"ae_index": i, "n_visible": 8, "suelo": 0.01, "razon": r,
                                          "cociente_contexto_cambiado": 1.2,
                                          "razon_persistencia": 0.9 if ar == "T" else None, "muertas": 0.0,
                                          "filas_train": 13_500, "filas_val": 1_500} for i, r in enumerate(razones)]})
                    (d / "DONE").write_text("OK en seco\n")
    C.write_json_atomic(agg / "aggregate" / "aggregate.json", {"schema": "EN SECO", "datasets": datasets,
                                                               "cells": cells, "value_sets": {"main": V}})
    h1 = json.loads(json.dumps(ST.h1(copy.deepcopy(V)), default=C._json_default))
    C.write_json_atomic(agg / "stats.json", {"run_type": "EN SECO", "H1": h1,
                                             "provenance": {"stats_sha256": C.sha256_file(ST.__file__)}})
    (agg / "SHA256SUMS").write_text("%s  %s\n%s  %s\n" % (
        C.sha256_file(agg / "aggregate" / "aggregate.json"), SUMS_CLAVES["aggregate"],
        C.sha256_file(agg / "stats.json"), SUMS_CLAVES["stats"]), encoding="utf-8")
    clip = base / "clip_y_score.json"
    C.write_json_atomic(clip, {"schema": "EN SECO", "hypothesis_sensitivity": {"cells": {"C10": {
        "as_run": as_run, "alternative": alt}}}, "range_and_clip": {"streams": rc},
        "mlp_score_convention": {"jobs": mlp_jobs}})
    log_b = base / "logs" / "fase-b-sB.log"
    log_b.write_text("2026-10-16T23:47:00 fase B semilla B termina (problemas: 1)\n\n===== 2026-10-19T08:48:28 "
                     "relanzamiento (en seco) =====\n2026-10-19T20:55:00 fase B semilla B termina (problemas: %d)\n"
                     % (1 if escenario == "ab" else 0), encoding="utf-8")
    pj = repo / PUERTA_REL
    r = subprocess.run([sys.executable, "-B", str(PUERTA_PY), "--familias", ",".join(FAMILIAS), "--results-root",
                        str(root), "--salida", str(pj)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("puerta_fase_b.py en seco salio con %d: %s" % (r.returncode, r.stdout[-400:]))
    J = C.read_json(pj)
    parrafo = parrafo_puerta(C.sha256_file(pj), {f: J["familias"][f]["entra"] for f in FAMILIAS})
    (repo / "revision").mkdir(exist_ok=True)
    (repo / DECISIONS_REL).write_text("# Decisiones (EN SECO)\n\n%s\n%s\n%s\n" % (parrafo, SECO_LINEA_CONGELAR,
                                                                                SECO_LINEA_SOLO_A), encoding="utf-8")
    git_commit_temporal(repo, [PUERTA_REL, DECISIONS_REL], "puerta en seco")
    return Rutas(repo_puerta=repo, results=root, results_esperado=str(root), agg_dir=agg, clip=clip,
                 dir_salida=base / "salidas", log_cola_b=log_b, en_seco=True,
                 registro=base / "salidas" / "e1_ejecuciones.jsonl")


def git_commit_temporal(repo, rutas_rel, mensaje):
    """git init (si hace falta), add y commit en un repositorio TEMPORAL (nunca el del proyecto)."""
    repo = Path(repo).resolve()
    if C.is_under(repo, R):
        raise GuardaError("no se hace commit dentro del repositorio del proyecto")
    env = dict(os.environ, GIT_AUTHOR_NAME="e1-en-seco", GIT_AUTHOR_EMAIL="e1@en-seco.invalid",
               GIT_COMMITTER_NAME="e1-en-seco", GIT_COMMITTER_EMAIL="e1@en-seco.invalid")
    sin_ganchos = ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env, capture_output=True)
    subprocess.run(["git", "-C", str(repo), *sin_ganchos, "add", "--", *rutas_rel], check=True, env=env,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), *sin_ganchos, "commit", "-q", "-m", mensaje], check=True, env=env,
                   capture_output=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Metricas y estadistica de E1 (bloque T10-REP, Enmienda 1). %s" % TAG)
    ap.add_argument("--control-positivo", "--solo-control-positivo", dest="control", action="store_true",
                    help="solo el control positivo sobre los T1 (no construye ninguna ruta T10)")
    ap.add_argument("--en-seco", action="store_true", help="flujo completo sobre un mundo sintetico temporal")
    ap.add_argument("--dir-seco", default=None, help="directorio del mundo en seco (nuevo, fuera del repositorio)")
    ap.add_argument("--escenario", choices=SECO_ESCENARIOS, default="ab")
    ap.add_argument("--streams", default=None, help="solo con --control-positivo: p. ej. D7,D8,D9 (parcial)")
    ap.add_argument("--salida", default=None,
                    help="solo con --control-positivo: fichero JSON (no puede existir; por defecto "
                         "fase_b/e1_control_positivo_<fecha-hora>.json)")
    ap.add_argument("--solo-semilla-a", nargs="?", const=True, default=None, metavar="D-NN",
                    help="solo la semilla A. Con --control-positivo, sin valor. En el modo completo exige una "
                         "decision D-NN (> 77, en HEAD, con 'E1 PARCIAL SOLO A:', anterior a la primera apertura de "
                         "T10) y la cola B parada; la salida va rotulada PARCIAL (la enmienda pide las semillas A y "
                         "B)")
    ap.add_argument("--congelar-semilla-b", default=None,
                    help="D-NN (> 77, en HEAD, con 'CONGELAR SEMILLA B:', anterior a la primera apertura de T10): "
                         "decision que da la B por cerrada aunque su cola no haya terminado limpia; lo que no tenga "
                         "DONE acorta S*_d")
    ap.add_argument("--pendientes-como-perdidas", default=None,
                    help="D-NN (> 77, en HEAD, con 'PENDIENTES COMO PERDIDAS:' y la lista de los job_id, anterior a "
                         "la primera apertura de T10) con la que se decide tratar como celda perdida los jobs T10 de "
                         "la semilla A sin DONE que no estan FAILED/BLOCKED")
    ap.add_argument("--descriptivos-no-entran", action="store_true",
                    help="ROC-AUC sin test (descriptivo) de las familias que no entran")
    ap.add_argument("--nombre", choices=("modified",), default="modified",
                    help="nombre de la familia en la plantilla; solo 'modified X' (el nombre de la plantilla de la "
                         "enmienda queda vetado)")
    a = ap.parse_args(argv)
    if a.streams and not a.control:
        ap.error("--streams solo vale con --control-positivo")
    if a.salida and not a.control:
        ap.error("--salida solo vale con --control-positivo (el modo completo pone fecha y hora al nombre)")
    if a.control and a.en_seco:
        ap.error("--control-positivo y --en-seco son excluyentes")
    if a.control and (a.congelar_semilla_b or a.pendientes_como_perdidas):
        ap.error("--congelar-semilla-b y --pendientes-como-perdidas no se pueden usar con --control-positivo")
    if a.control and isinstance(a.solo_semilla_a, str):
        ap.error("con --control-positivo, --solo-semilla-a va sin decision (el control no abre T10)")
    if a.solo_semilla_a and a.congelar_semilla_b:
        ap.error("--solo-semilla-a y --congelar-semilla-b son excluyentes")
    if a.salida:
        sal = Path(a.salida).resolve()
        if C.is_under(sal, R) and not C.is_under(sal, AQUI):
            ap.error("dentro del repositorio solo se escribe en revision/analysis/fase_b/ (nunca en results/ ni en "
                     "las carpetas de los arneses)")
        if sal.exists():
            ap.error("%s ya existe: no se sobrescribe" % sal)
    if a.en_seco:
        base = Path(a.dir_seco) if a.dir_seco else Path(tempfile.mkdtemp(prefix="e1_en_seco_")) / "mundo"
        log("en seco: construyendo el mundo sintetico en %s (escenario %s)" % (base, a.escenario))
        try:
            rutas = construir_mundo_seco(base, a.escenario)
        except GuardaError as e:
            log("GUARDA: %s" % e)
            return 3
        return ejecutar_completo(rutas, a.descriptivos_no_entran, a.nombre, a.solo_semilla_a, a.congelar_semilla_b,
                                 a.pendientes_como_perdidas)
    rutas = rutas_oficiales()
    if a.control:
        dids = tuple(ST.DS)
        if a.streams:
            dids = tuple(x.strip() for x in a.streams.split(",") if x.strip())
            if not dids or any(x not in ST.DS for x in dids):
                ap.error("--streams: D1..D9")
        if a.salida:
            rutas.salida_control = Path(a.salida)
        return ejecutar_control(rutas, dids, ("A",) if a.solo_semilla_a else SEMILLAS)
    return ejecutar_completo(rutas, a.descriptivos_no_entran, a.nombre, a.solo_semilla_a, a.congelar_semilla_b,
                             a.pendientes_como_perdidas)


if __name__ == "__main__":
    sys.exit(main())
