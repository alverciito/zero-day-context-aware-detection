"""supl_fase_b.py - values of Tables S71 and S72 of the supplement (modified encoders), standard library only: reads the comparison file and the convergence_report.json of the 108 runs, counts as puerta_fase_b.py does and stops if a total differs from the expected one."""
import glob
import hashlib
import json
import math
import os
import statistics
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AQUI = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(AQUI, "e1_metricas_t10_20261019-222526.json")
RESULTADOS = os.path.join(RAIZ, "results", "revision")
STREAMS = [("D1", "knad-active_wiretap", "Active Wiretap"), ("D2", "knad-fuzzing", "Fuzzing"),
           ("D3", "knad-os_scan", "OS Scan"), ("D4", "knad-ssdp_flood", "SSDP Flood"),
           ("D5", "knad-ssl_renegotiation", "SSL Renegotiation"), ("D6", "knad-video_injection", "Video Injection"),
           ("D7", "aci-iot-2023", "ACI-IoT-2023"), ("D8", "cic2017", "CIC-IDS2017"), ("D9", "cic2018", "CSE-CIC-IDS2018")]
PREVUELO = {"D1", "D7", "D9"}
FAMILIAS = ["conv1d", "conv2d"]
SEMILLAS = "ABC"
ESPERADO = {("conv1d", "A"): (21, 29, 9, 9), ("conv1d", "B"): (19, 29, 11, 12), ("conv1d", "C"): (23, 29, 6, 9),
            ("conv2d", "A"): (29, 29, 2, 19), ("conv2d", "B"): (28, 29, 8, 19), ("conv2d", "C"): (29, 29, 5, 19)}


def finito(x):
    """Devuelve True si x es un numero finito."""
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def contar(familia, semilla, paradigma, stream):
    """Cuenta los autoencoders de un trabajo T10-REP como puerta_fase_b.py.

    :param familia: conv1d o conv2d.
    :param semilla: A, B o C.
    :param paradigma: arF (TSR) o arT (AR).
    :param stream: nombre del stream en el job_id.
    :return: dict con evaluables, bajo, contexto y las razones evaluables.
    """
    patron = os.path.join(RESULTADOS, f"T10-REP__{stream}__{familia}_rep__{paradigma}__*__s{semilla}__r1")
    dirs = glob.glob(patron)
    if len(dirs) != 1 or not os.path.exists(os.path.join(dirs[0], "DONE")):
        sys.exit(f"PARADA: {patron} -> {dirs} sin un unico trabajo con DONE")
    with open(os.path.join(dirs[0], "convergence_report.json")) as f:
        informe = json.load(f)
    c = {"evaluables": 0, "bajo": 0, "contexto": 0, "razones": []}
    for ae in informe["autoencoders"]:
        if not (finito(ae["suelo"]) and ae["suelo"] >= 1e-4):
            continue
        c["evaluables"] += 1
        if finito(ae["razon"]):
            c["razones"].append(ae["razon"])
            if ae["razon"] < 0.90:
                c["bajo"] += 1
                q = ae.get("cociente_contexto_cambiado")
                if finito(q) and q >= 1.10:
                    c["contexto"] += 1
    return c


def walsh_ci(values):
    """Hodges-Lehmann estimate and exact 96.1 % interval (n = 9: order statistics 6 and 40 of 45)."""
    w = sorted((values[i] + values[j]) / 2 for i in range(len(values)) for j in range(i, len(values)))
    if len(values) != 9:
        sys.exit("PARADA: el intervalo exacto solo esta fijado para n = 9")
    return statistics.median(w), w[5], w[39]


def margen(values, m=0.01):
    """Recuento con margen de empate (plan, seccion 10): mayor, menor y dentro del margen."""
    return [sum(1 for v in values if v > m), sum(1 for v in values if v < -m),
            sum(1 for v in values if abs(v) <= m)]


def main():
    """Escribe supl_fase_b.json y supl_fase_b_tablas.tex e imprime el resumen."""
    with open(E1, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(E1) as f:
        e1 = json.load(f)
    out = {"tag": "[ENMIENDA-1] [EXPLORATORIO]", "e1": os.path.relpath(E1, RAIZ), "e1_sha256": sha,
           "convergencia": {}, "auc": {}, "comparaciones": {}}
    for fam in FAMILIAS:
        for s in SEMILLAS:
            tsr = {d: contar(fam, s, "arF", st) for d, st, _ in STREAMS}
            ar = {d: contar(fam, s, "arT", st) for d, st, _ in STREAMS}
            tot = (sum(c["bajo"] for c in tsr.values()), sum(c["evaluables"] for c in tsr.values()),
                   sum(c["contexto"] for c in tsr.values()), sum(c["bajo"] for c in ar.values()))
            if tot != ESPERADO[(fam, s)]:
                sys.exit(f"PARADA: {fam} {s} da {tot}, se esperaba {ESPERADO[(fam, s)]}")
            out["convergencia"][f"{fam}|{s}"] = {
                "tsr_por_stream": {d: [c["bajo"], c["evaluables"]] for d, c in tsr.items()},
                "tsr_bajo": tot[0], "tsr_evaluables": tot[1], "contexto": tot[2], "ar_bajo": tot[3],
                "contexto_ar": sum(c["contexto"] for c in ar.values()),
                "ar_evaluables": sum(c["evaluables"] for c in ar.values()),
                "streams_con_alguno": sum(1 for c in tsr.values() if c["bajo"] > 0),
                "mediana_razon_tsr": statistics.median([r for c in tsr.values() for r in c["razones"]])}
    for fam in FAMILIAS:
        ps = e1["per_stream"][fam]
        filas = {}
        for d, _, _ in STREAMS:
            v = ps[d]
            filas[d] = {"control": v["C10"], "sin_modificar": v["sin_modificar"]["media"], "modificado": v["media"],
                        "A": v["por_semilla"]["A"]["media"], "B": v["por_semilla"]["B"]["media"],
                        "bloque_control": v["C10_loc"], "bloque_modificado": v["loc"]}
        out["auc"][fam] = filas
        clave = ("Conv1Dmod" if fam == "conv1d" else "Conv2Dmod") + "_vs_original"
        comp = e1["e1"]["comparisons"][clave]
        rob = comp["robustness"]
        des = e1["descriptivos"][fam]
        dep = rob["dependence"]
        out["comparaciones"][fam] = {
            "control": {"hl": comp["hl"], "ci": [comp["hl_ci"]["lower"], comp["hl_ci"]["upper"]],
                        "p": comp["wilcoxon"]["p"], "p_holm": comp["p_holm"], "k_pos": comp["k_pos"]},
            "bloque_hl": rob["drift"]["hl_auc_loc_raw"], "control_recortado_hl": rob["s_clip"]["hl_clip"],
            "colapsado": [dep["KNAD_collapsed"]["hl"], dep["KNAD_collapsed"]["k_pos"], dep["KNAD_collapsed"]["n"]],
            "sin_knad": [dep["without_KNAD"]["hl"], dep["without_KNAD"]["k_pos"], dep["without_KNAD"]["n"]],
            **{k: {"hl": des[k]["hl"], "ci": [des[k]["ci_lower"], des[k]["ci_upper"]], "k_pos": des[k]["k_pos"],
                   "n": des[k]["n"]}
               for k in ["c_vs_sin_modificar", "e_fuera_del_prevuelo", "a_vs_mlp_mse", "a_vs_mlp_rmse"]},
            "max_dif_semillas": max((abs(f["A"] - f["B"]), d) for d, f in filas.items()),
            "solo_tsr": {"hl": des["b_primaria_solo_tsr"]["hl"],
                         "ci": [des["b_primaria_solo_tsr"]["ci_lower"], des["b_primaria_solo_tsr"]["ci_upper"]],
                         "k_pos": des["b_primaria_solo_tsr"]["k_pos"]},
            "ar_menos_tsr": {"hl": des["d_ar_menos_tsr"]["hl"],
                             "ci": [des["d_ar_menos_tsr"]["ci_lower"], des["d_ar_menos_tsr"]["ci_upper"]],
                             "k_pos": des["d_ar_menos_tsr"]["k_pos"]},
            "bloque_ci": list(walsh_ci([rob["drift"]["d_auc_loc_raw"][d] for d, _, _ in STREAMS])),
            "bloque_k_pos": sum(1 for d, _, _ in STREAMS if rob["drift"]["d_auc_loc_raw"][d] > 0),
            "margen_0_01_control": margen([comp["d"][d] for d, _, _ in STREAMS]),
            "margen_0_01_sin_modificar": des["c_vs_sin_modificar"].get("margen_0_01"),
            "auc_t": {d: sorted({round(j["metricas"]["auc_t"], 12) for j in e1["per_job"]
                                 if j["dataset_id"] == d}) for d, _, _ in STREAMS}}
        hl, lo, hi = walsh_ci([comp["d"][d] for d, _, _ in STREAMS])
        if abs(hl - comp["hl"]) > 1e-12 or abs(lo - comp["hl_ci"]["lower"]) > 1e-12 or abs(hi - comp["hl_ci"]["upper"]) > 1e-12:
            sys.exit(f"PARADA: el intervalo de Walsh de {fam} no reproduce E1")
    out["contexto_ab"] = {fam: [sum(out["convergencia"][f"{fam}|{s}"]["contexto"] for s in "AB"),
                                sum(out["convergencia"][f"{fam}|{s}"]["tsr_bajo"] for s in "AB")]
                          for fam in FAMILIAS}
    tex = ["% Tabla de ROC-AUC (supl_fase_b.py)"]
    for d, _, nombre in STREAMS:
        a, b = out["auc"]["conv1d"][d], out["auc"]["conv2d"][d]
        if abs(a["control"] - b["control"]) > 1e-12 or abs(a["bloque_control"] - b["bloque_control"]) > 1e-12:
            sys.exit(f"PARADA: el control de {d} no coincide entre familias")
        marca = "$^{\\dagger}$" if d in PREVUELO else ""
        auc_t = out["comparaciones"]["conv1d"]["auc_t"][d]
        if len(auc_t) != 1:
            sys.exit(f"PARADA: AUC_t de {d} no es unico")
        vals = [auc_t[0], a["control"], a["sin_modificar"], a["modificado"], b["sin_modificar"], b["modificado"],
                a["bloque_control"], a["bloque_modificado"], b["bloque_modificado"]]
        tex.append(f"{nombre}{marca} & " + " & ".join(f"{x:.3f}" for x in vals) + " \\\\")
    med = [statistics.median(out["comparaciones"]["conv1d"]["auc_t"][d][0] for d, _, _ in STREAMS)] + [
           statistics.median(out["auc"][f][d][k] for d, _, _ in STREAMS)
           for f, k in [("conv1d", "control"), ("conv1d", "sin_modificar"), ("conv1d", "modificado"),
                        ("conv2d", "sin_modificar"), ("conv2d", "modificado"), ("conv1d", "bloque_control"),
                        ("conv1d", "bloque_modificado"), ("conv2d", "bloque_modificado")]]
    out["medianas_tabla_auc"] = med
    tex.append("\\hline")
    tex.append("Median & " + " & ".join(f"{x:.3f}" for x in med) + " \\\\")
    tex.append("% Tabla de recuentos (supl_fase_b.py)")
    for d, _, nombre in STREAMS:
        marca = "$^{\\dagger}$" if d in PREVUELO else ""
        celdas = [out["convergencia"][f"{f}|{s}"]["tsr_por_stream"][d] for f in FAMILIAS for s in SEMILLAS]
        tex.append(f"{nombre}{marca} & " + " & ".join(f"{k}/{n}" for k, n in celdas) + " \\\\")
    cv = [out["convergencia"][f"{f}|{s}"] for f in FAMILIAS for s in SEMILLAS]
    tex.append("\\hline")
    tex.append("All streams & " + " & ".join(f"{c['tsr_bajo']}/{c['tsr_evaluables']}" for c in cv) + " \\\\")
    tex.append("Streams with at least one & " + " & ".join(f"{c['streams_con_alguno']}/9" for c in cv) + " \\\\")
    tex.append("Median ratio & " + " & ".join(f"{c['mediana_razon_tsr']:.3f}" for c in cv) + " \\\\")
    tex.append("Window replaced & " + " & ".join(f"{c['contexto']}/{c['tsr_bajo']}" for c in cv)
               + " \\\\")
    tex.append("Autoregressive & " + " & ".join(f"{c['ar_bajo']}/{c['ar_evaluables']}" for c in cv) + " \\\\")
    with open(os.path.join(AQUI, "supl_fase_b.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(AQUI, "supl_fase_b_tablas.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(tex))
    print(json.dumps({"comparaciones": out["comparaciones"], "medianas": med}, indent=1))


if __name__ == "__main__":
    main()
