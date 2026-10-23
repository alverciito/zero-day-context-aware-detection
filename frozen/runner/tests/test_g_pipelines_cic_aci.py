#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(g) Corrected pipelines of CIC2017, CIC2018 and ACI-IoT-2023 on small fixtures built from real lines of the datasets (CPU)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

sys.path.insert(0, str(U.CODE_ROOT))
import numpy as np
import pandas as pd
import pipelines as P

T0_CACHE = C.MAIN_REPO / "data" / "cache" / "t0" / "full"
T0_DIR = C.MAIN_REPO / "revision" / "analysis" / "t0"


def spec_for(dataset, pipeline="corrected"):
    return C.validate_row(U.make_row(dataset=dataset, pipeline=pipeline, variant="stat"), test_mode=True)


def load(spec, root, data_root, cache="cache", **kw):
    return P.load_stream(spec, cache_root=str(root / cache), data_root=str(data_root), code_root=str(U.CODE_ROOT),
                         test_mode=True, log=lambda m: None, **kw)


def raises(fn, exc):
    try:
        fn()
        return None
    except exc as e:
        return str(e)


def head_lines(src, n):
    out = []
    with open(src, "rb") as fh:
        header = fh.readline()
        for ln in fh:
            if len(out) >= n:
                break
            out.append(ln)
    return header, out


def write_lines(dst, header, lines):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(header)
        for ln in lines:
            fh.write(ln if ln.endswith(b"\n") else ln + b"\n")


def set_field(line, idx, value):
    parts = line.rstrip(b"\r\n").split(b",")
    parts[idx] = value if isinstance(value, bytes) else str(value).encode()
    return b",".join(parts) + b"\n"


def col_index(header, name):
    return [h.decode() for h in header.rstrip(b"\r\n").split(b",")].index(name)


def expected_x(df_stream, feats, cats, T):
    """Reimplementacion independiente del preprocesado C5 (sin llamar a pipelines)."""
    num = [c for c in feats if c not in cats]
    xn = df_stream[num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64).copy()
    xn[np.isnan(xn)] = 0.0
    for j in range(xn.shape[1]):
        tr = xn[:T, j]
        fill = tr[np.isfinite(tr)].max() if np.isfinite(tr).any() else 0.0
        xn[np.isinf(xn[:, j]), j] = fill
    mu, sd = xn[:T].mean(axis=0), xn[:T].std(axis=0)
    xn = (xn - mu) / (sd + 1e-6)
    if not cats:
        return xn
    codes = []
    for c in cats:
        s = df_stream[c].fillna("<NA>").astype(str).to_numpy()
        vocab = sorted(set(s[:T].tolist()))
        codes.append(np.array([vocab.index(v) if v in vocab else len(vocab) for v in s], np.uint8))
    return np.concatenate([xn, np.unpackbits(np.stack(codes, axis=1), axis=1)], axis=1)


def const_train_exec_counts(df_stream, feats, cats, T):
    """Columnas numericas constantes en [0, T) tras fillna(0) e inf -> max[0, T): filas de [T, N) con otro valor."""
    out = {}
    for c in feats:
        if c in cats:
            continue
        v = pd.to_numeric(df_stream[c], errors="coerce").to_numpy(np.float64).copy()
        v[np.isnan(v)] = 0.0
        fin = np.isfinite(v[:T])
        v[np.isinf(v)] = v[:T][fin].max() if fin.any() else 0.0
        if np.all(v[:T] == v[0]):
            out[c] = int((v[T:] != v[0]).sum())
    return out


def brute_dup_mask(df_stream, feats, T):
    """Duplicados exactos por fuerza bruta: tuplas de valores crudos (-0 = +0, NaN = NaN)."""
    def key(row):
        out = []
        for v in row:
            if isinstance(v, float):
                out.append("nan" if np.isnan(v) else v + 0.0)
            else:
                out.append(v)
        return tuple(out)
    vals = df_stream[feats].to_numpy(dtype=object)
    train = {key(r) for r in vals[:T]}
    m = np.zeros(len(vals), bool)
    for i in range(T, len(vals)):
        m[i] = key(vals[i]) in train
    return m


def numeric_frame(df, cats):
    out = df.copy()
    for c in out.columns:
        if c not in cats:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
    return out


def t0_fresh(data_root):
    """Copia privada NUEVA de warmup_audit.py con DS = fixture (referencia para la equivalencia; sin cache T0)."""
    import importlib.util
    mspec = importlib.util.spec_from_file_location("_t0_ref_for_tests", U.CODE_ROOT / P.T0_SCRIPT)
    wa = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(wa)
    wa.cached_npz = lambda _p, _d, compute: compute()
    wa.DS = Path(data_root)
    return wa


class _Args:
    sample_rows = 0
    workers = 1


def t0_reference_pairs(wa, ds, variant, train_rows):
    S = wa.assemble_csv(ds, _Args(), ".", need_orig_hash=False, need_corr_hash=False)
    consts = wa.Consts(1)
    if wa.corrected_variants(ds)[variant]["train_rule"] != "days":
        consts.TRAIN = train_rows
    st, _ = wa.build_corrected(S, variant, wa.corrected_variants(ds)[variant], consts)
    fidx = np.searchsorted(S.offsets, st.gid, side="right") - 1
    return [S.files[int(f)] for f in fidx], (st.gid - S.offsets[fidx]).astype(np.int64), int(st.train_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-full", action="store_true")
    a = ap.parse_args()
    ck = U.Checker("test_g_pipelines_cic_aci")
    root = U.make_root("g")
    try:
        ok = all(raises(lambda d=d: C.check_supported(spec_for(d)), NotImplementedError) is None
                 for d in C.OTHER_DATASETS)
        fake = spec_for("CIC2017")
        fake.kind = "otro"
        msg = raises(lambda: C.check_supported(fake), NotImplementedError)
        ck.check("g0.supported", ok and msg is not None and "D-7" in msg, f"CIC/ACI corrected admitidos; otro -> {msg and msg[:80]}")
        ck.check("g0.train_rows", [C.corrected_train_rows(spec_for(d)) for d in ("KNAD:OS_Scan", "CIC2017", "CIC2018", "ACI-IOT-2023")]
                 == [200_000, 200_000, 200_000, 72_264] and P.default_train_rows(spec_for("KNAD:OS_Scan")) == P.TRAIN_ROWS
                 and P.builder_for(spec_for("ACI-IOT-2023"), None) is P.build_cic_aci_corrected,
                 "T: KNAD/CIC 200.000, ACI 72.264")
        prov = P.build_cache_provenance_problems(C.MAIN_REPO)
        ck.check("g0.build_cache_provenance", any("arbol principal" in m for m in prov),
                 f"--build-cache desde el arbol principal rechazado: {prov[:1]}")
        fps = {}
        for key, builder in (("knad-active_wiretap__corrected__v1", P.build_knad_corrected),
                             ("knad-ssdp_flood__corrected__v1", P.build_knad_corrected),
                             ("knad-os_scan__orig__v1", P.build_knad_orig_class)):
            mp = C.CACHE_ROOT / key / "meta.json"
            if mp.exists():
                fps[key] = U.load_json(mp)["builder"]["fingerprint_sha256"] == P.builder_fingerprint(builder, U.CODE_ROOT)
        if fps:
            ck.check("g0.knad_fingerprints", all(fps.values()), f"huellas de las caches KNAD de produccion (solo lectura): {fps}")
        else:
            ck.warn("g0.knad_fingerprints", "sin caches KNAD de produccion para comparar")

        wa = P.t0_module(U.CODE_ROOT)
        fx = root / "data"
        d17 = fx / "CIC2017"
        src17 = C.DATA_ROOT / "CIC2017"
        files17 = list(wa.CIC2017_CHRONO)
        n_mon, n_other, T17 = 1200, 300, 1000
        lines17 = {}
        for f in files17:
            hdr, ls = head_lines(src17 / f, n_mon if f.startswith("Monday") else n_other)
            lines17[f] = ls
        h17 = hdr
        i_fbs, i_fps, i_port = col_index(h17, "Flow Bytes/s"), col_index(h17, " Flow Packets/s"), col_index(h17, " Destination Port")
        mon, tue, wed = files17[0], files17[1], files17[2]
        lines17[mon][5] = set_field(lines17[mon][5], i_fbs, "Infinity")
        lines17[mon][6] = set_field(lines17[mon][6], i_fps, "NaN")
        lines17[tue][10] = lines17[mon][20]
        parts = lines17[mon][21].rstrip(b"\n").split(b",")
        zero = next(k for k, v in enumerate(parts[:-1]) if v.strip() == b"0")
        lines17[tue][11] = set_field(lines17[mon][21], zero, "-0.0")
        lines17[wed][3] = set_field(lines17[wed][3], i_fbs, "-Infinity")
        i_bpsh = col_index(h17, " Bwd PSH Flags")
        lines17[wed][4] = set_field(lines17[wed][4], i_bpsh, "1")
        for f in files17:
            write_lines(d17 / f, h17, lines17[f])
        s17 = spec_for("CIC2017")
        x, y, extra = P.build_cic_aci_corrected(U.CODE_ROOT, fx, s17, train_rows=T17)
        aux = extra["_aux_arrays"]
        N17 = n_mon + 7 * n_other
        exp_file = np.repeat(np.arange(8), [n_mon] + [n_other] * 7)
        exp_row = np.concatenate([np.arange(n_mon)] + [np.arange(n_other)] * 7)
        ck.check("g1.order_chrono", x.shape == (N17, 78) and x.dtype == np.float64 and
                 np.array_equal(aux["src_file"], exp_file) and np.array_equal(aux["src_row"], exp_row) and
                 extra["t0"]["files"][1].startswith("Tuesday") and sorted(files17)[1] != files17[1],
                 f"x {x.shape}; lunes -> martes -> ... (no alfabetico: {sorted(files17)[1][:28]})")
        df17 = pd.concat([pd.read_csv(d17 / f) for f in files17], ignore_index=True)
        feats17 = [c for c in df17.columns if c != " Label"]
        ck.check("g1.labels", y.dtype == np.int64 and np.array_equal(y, (df17[" Label"] != "BENIGN").astype(np.int64).to_numpy())
                 and extra["regions"] == {**extra["regions"], "train": [0, T17], "post_train": [T17, N17]},
                 f"y alineada fila a fila; ataques={int(y.sum())}")
        xe = expected_x(df17, feats17, [], T17)
        ck.check("g1.preprocess_train_only", np.array_equal(x, xe) and np.isfinite(x).all() and
                 extra["preprocessing"]["inf_to_train_max"]["exec"] >= 1 and extra["preprocessing"]["nan_to_zero"]["train"] >= 1,
                 f"x == reimplementacion independiente (fillna 0, inf -> max[0,T), z-score[0,T)); "
                 f"inf={extra['preprocessing']['inf_to_train_max']}")
        z17 = extra["preprocessing"]["zscore"]
        i_seed = n_mon + n_other + 4
        ck.check("g1.constant_in_train", z17["constant_in_train_exec_rows_differing"] == const_train_exec_counts(df17, feats17, [], T17)
                 and z17["constant_in_train_exec_rows_differing"].get(" Bwd PSH Flags", 0) >= 1 and
                 abs(x[i_seed, feats17.index(" Bwd PSH Flags")] - 1e6) < 1e-6 and
                 sorted(z17["constant_in_train"]) == sorted(z17["constant_in_train_exec_rows_differing"]),
                 f"constantes en [0, T) con otro valor en la ejecucion (x1e6, declarado): "
                 f"{ {k: v for k, v in z17['constant_in_train_exec_rows_differing'].items() if v} }")
        bm = brute_dup_mask(df17, feats17, T17)
        i_dup1, i_dup2 = n_mon + 10, n_mon + 11
        ck.check("g1.duplicates", np.array_equal(aux["dup_train"], bm) and aux["dup_train"][i_dup1] and aux["dup_train"][i_dup2]
                 and not aux["dup_train"][:T17].any() and extra["duplicates_train_exec"]["n"] == int(bm.sum()),
                 f"mascara == fuerza bruta ({int(bm.sum())}; sembrados {i_dup1} y {i_dup2} con -0)")
        ddos = files17[-1]
        lines_alt = list(lines17[ddos])
        lines_alt[0] = set_field(lines_alt[0], i_port, "999999999999")
        fx_alt = root / "data_alt"
        for f in files17:
            write_lines(fx_alt / "CIC2017" / f, h17, lines_alt if f == ddos else lines17[f])
        x2, _y2, extra2 = P.build_cic_aci_corrected(U.CODE_ROOT, fx_alt, s17, train_rows=T17)
        diff_rows = np.flatnonzero((x2 != x).any(axis=1))
        ck.check("g1.no_leak_c5", np.array_equal(x2[:T17], x[:T17]) and diff_rows.tolist() == [N17 - n_other] and
                 extra2["preprocessing"]["zscore"] == extra["preprocessing"]["zscore"] and
                 extra2["preprocessing"]["inf_to_train_max"]["fill"] == extra["preprocessing"]["inf_to_train_max"]["fill"],
                 f"filas distintas {diff_rows.tolist()} (solo la modificada); estadisticos identicos")
        fx_bad = root / "data_bad"
        lines_bad = list(lines17[mon])
        lines_bad[500] = set_field(lines_bad[500], -1, "DDoS")
        for f in files17:
            write_lines(fx_bad / "CIC2017" / f, h17, lines_bad if f == mon else lines17[f])
        msg = raises(lambda: P.build_cic_aci_corrected(U.CODE_ROOT, fx_bad, s17, train_rows=T17), C.PipelineError)
        ck.check("g1.benign_guard", msg is not None and "guarda benigna" in msg and "500" in msg and "D-7" in msg, (msg or "")[:160])
        wa_h = P.t0_module(U.CODE_ROOT)

        def reader_shifted(pth):
            arrs, info = wa_h.labels_csv("cic2017", pth, 0, ".")
            if Path(pth).name != wed:
                return arrs, info
            codes = arrs["codes"].copy()
            codes[3] = len(info["vocab"])
            return dict(arrs, codes=codes), dict(info, vocab=list(info["vocab"]) + ["ZZZ_desalineada"])
        msg = raises(lambda: P.build_cic_aci_corrected(U.CODE_ROOT, fx, s17, train_rows=T17, labels_reader=reader_shifted),
                     C.PipelineError)
        ck.check("g1.labels_recheck_with_features", msg is not None and "desalineados" in msg and
                 extra["features"]["labels_rechecked_with_features"]["mismatches"] == 0,
                 f"etapa 1 alterada en {wed} fila 3 -> {(msg or 'sin PipelineError')[:140]}")
        import run_job as RJ
        reg = RJ.regions_for(s17, N17, {"fm": 400, "ad": 600}, {"extra": extra})
        pairs = {k: C.check_protocol_regions(reg, k) for k in reg["detector_protocol_candidates"] if not k.startswith("_")}
        byf = extra["duplicates_train_exec"]["by_calib_fraction"]
        ok_frac = all(pairs[f"calib_first_{k}pct_post_train"][0] == v["calib"] and
                      v["n_dup"] == int(aux["dup_train"][v["calib"][0]:v["calib"][1]].sum()) for k, v in byf.items())
        ck.check("g1.regions", reg["train"] == [0, T17] and reg["post_train"] == [T17, N17] and ok_frac and
                 reg["detector_protocol_candidates"]["calib_in_sample_train_tail"]["calib_in_sample"] is True,
                 f"candidatas {pairs}")
        old_T = dict(C.CORRECTED_TRAIN_ROWS)
        C.CORRECTED_TRAIN_ROWS["CIC2017"] = T17
        try:
            xc, yc, mc = load(s17, root, fx)
            xc2, yc2, mc2 = load(s17, root, fx)
            key_dir = root / "cache" / mc["key"]
            aux_ok = all(np.array_equal(np.load(mc2["aux"][k]["path"]), aux[k]) for k in aux)
            ck.check("g1.cache_aux", mc["cached"] and mc.get("built_now") and np.array_equal(xc, x) and np.array_equal(yc, y)
                     and np.array_equal(xc2, x) and aux_ok and sorted(mc2["aux"]) == sorted(aux) and
                     mc2["verified_on_load"].get("aux_sha256_recomputed") == sorted(aux) and
                     mc2["builder"]["function"] == "build_cic_aci_corrected" and len(mc2["sources"]) == 8 and
                     P.expected_rows(s17, fx) is None and "_aux_arrays" not in mc2["extra"],
                     f"cache {key_dir.name}: aux {sorted(mc2['aux'])} verificados; x/y identicos")
            f_aux = key_dir / "aux_dup_train.npy"
            os.chmod(key_dir, 0o755)
            os.chmod(f_aux, 0o644)
            b = bytearray(f_aux.read_bytes())
            b[-1] ^= 1
            f_aux.write_bytes(bytes(b))
            msg = raises(lambda: load(s17, root, fx), C.CacheIntegrityError)
            ck.check("g1.cache_aux_tamper", msg is not None and "aux_dup_train" in msg, (msg or "")[:120])
        finally:
            C.CORRECTED_TRAIN_ROWS.clear()
            C.CORRECTED_TRAIN_ROWS.update(old_T)

        d18 = fx / "CIC2018"
        src18 = C.DATA_ROOT / "CIC2018"
        T18 = 1000
        lines18 = {}
        with open(src18 / "02-14-2018.csv", "rb") as fh:
            h18 = fh.readline()
            sel = []
            for ln in fh:
                hh = ln.split(b",", 3)[2][11:13]
                if hh in (b"08", b"09"):
                    sel.append(ln)
                if len(sel) >= 1500:
                    break
        lines18["02-14-2018.csv"] = sel
        for f in wa.CIC2018_DAYS7[1:]:
            lines18[f] = head_lines(src18 / f, 300)[1]
        lines18["02-14-2018.csv"][7] = set_field(lines18["02-14-2018.csv"][7], 2, "01/01/1970 00:00:00")
        lines18["02-21-2018.csv"][5] = set_field(lines18["02-21-2018.csv"][5], 2, "21/02/2018 01:10:00")
        lines18["02-16-2018.csv"].insert(50, h18)
        for f in wa.CIC2018_DAYS7:
            write_lines(d18 / f, h18, lines18[f])
        for f in wa.CIC2018_EXCLUDED_ORIG:
            hh, ll = head_lines(src18 / f, 20)
            write_lines(d18 / f, hh, ll)
        s18 = spec_for("CIC2018")
        x, y, extra = P.build_cic_aci_corrected(U.CODE_ROOT, fx, s18, train_rows=T18)
        aux = extra["_aux_arrays"]
        excl = {k: v["n"] for k, v in extra["t0"]["excluded_before_sort"].items()}
        ts = aux["ts"]
        guard = extra["t0"]["cuts"]["guard_ties_dropped"]["n"]
        n_total = 1500 + 6 * 300
        i_shift = int(np.flatnonzero((np.array(extra["t0"]["files"])[aux["src_file"]] == "02-21-2018.csv") & (aux["src_row"] == 5))[0])
        ck.check("g2.clock_exclusions_order", excl == {"repeated header row": 1, "timestamp outside the capture day": 1} and
                 x.shape == (n_total - 1 - guard, 78) and np.all(np.diff(ts) >= 0) and ts[T18] > ts[:T18].max() and
                 (ts[i_shift] % 86400) // 3600 == 13 and extra["features"]["repeated_header_rows_skipped"] == {"02-16-2018.csv": [50]}
                 and "Timestamp" not in extra["features"]["raw_columns"],
                 f"exclusiones {excl}; empates fuera {guard}; ts monotono; 01:10 -> 13 h; x {x.shape}")
        wa_ref = t0_fresh(fx)
        rf, rr, rT = t0_reference_pairs(wa_ref, "cic2018", "days7", T18)
        mine_f = [extra["t0"]["files"][int(i)] for i in aux["src_file"]]
        ck.check("g2.t0_equivalence", rT == T18 and mine_f == rf and np.array_equal(aux["src_row"].astype(np.int64), rr),
                 f"orden fila a fila == warmup_audit.assemble_csv + build_corrected ({len(rf)} filas)")
        frames18 = [pd.read_csv(d18 / f, dtype=str, skiprows=([51] if f == "02-16-2018.csv" else None))
                    for f in wa.CIC2018_DAYS7]
        rows = []
        for fi, r in zip(aux["src_file"], aux["src_row"]):
            f = wa.CIC2018_DAYS7[int(fi)]
            rr_ = int(r) - (1 if (f == "02-16-2018.csv" and int(r) > 50) else 0)
            rows.append(frames18[int(fi)].iloc[rr_])
        df18 = pd.DataFrame(rows).reset_index(drop=True)
        feats18 = extra["features"]["raw_columns"]
        ck.check("g2.labels_preprocess", np.array_equal(y, (df18["Label"] != "Benign").astype(np.int64).to_numpy()) and
                 np.array_equal(x, expected_x(numeric_frame(df18[feats18], []), feats18, [], T18)) and
                 np.array_equal(aux["dup_train"], brute_dup_mask(numeric_frame(df18[feats18], []), feats18, T18)),
                 f"y, x y duplicados ({int(aux['dup_train'].sum())}) == reimplementacion independiente")

        daci = fx / "ACI-IOT-2023"
        stride = {b"2023-10-30": 400, b"31/10/2023": 40, b"2023-01-11": 300, b"2023-02-11": 100, b"03/11/2023": 100}
        seen = {k: 0 for k in stride}
        la = []
        with open(C.DATA_ROOT / "ACI-IOT-2023" / "ACI-IoT-2023.csv", "rb") as fh:
            hA = fh.readline()
            for ln in fh:
                day = ln.split(b",", 7)[6][:10]
                if day in seen:
                    if seen[day] % stride[day] == 0:
                        la.append(ln)
                    seen[day] += 1
        k31 = sum(1 for ln in la if ln.split(b",", 7)[6][:10] == b"31/10/2023")
        i_ct = col_index(hA, "Connection Type")
        first31 = next(ln for ln in la if ln.split(b",", 7)[6][:10] == b"31/10/2023")
        dup = first31
        for name, val in (("Flow ID", "10.0.0.1-10.0.0.2-1-2-6"), ("Src IP", "10.0.0.1"), ("Src Port", "65000"),
                          ("Dst IP", "10.0.0.2"), ("Timestamp", "2023-02-11 12:00:00")):
            dup = set_field(dup, col_index(hA, name), val)
        la.append(dup)
        i_unseen = next(k for k, ln in enumerate(la) if ln.split(b",", 7)[6][:10] == b"2023-01-11")
        la[i_unseen] = set_field(la[i_unseen], i_ct, "fiber")
        write_lines(daci / "ACI-IoT-2023.csv", hA, la)
        sA = spec_for("ACI-IOT-2023")
        x, y, extra = P.build_cic_aci_corrected(U.CODE_ROOT, fx, sA, train_rows=k31)
        aux = extra["_aux_arrays"]
        ts = aux["ts"]
        dfA = pd.read_csv(daci / "ACI-IoT-2023.csv", dtype=str).iloc[aux["src_row"]].reset_index(drop=True)
        featsA = extra["features"]["raw_columns"]
        catA = extra["features"]["categorical"]
        dfA_num = numeric_frame(dfA[featsA], catA)
        i_dup = int(np.flatnonzero(aux["src_row"] == len(la) - 1)[0])
        i_uns = int(np.flatnonzero(aux["src_row"] == i_unseen)[0])
        day_of = lambda t: str(np.datetime64(int(t), "s"))[:10]
        ck.check("g3.days_columns", x.shape[1] == 86 and len(featsA) == 79 and
                 not {"Src Port", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"} & set(featsA) and
                 {day_of(t) for t in ts[:k31]} == {"2023-10-31"} and ts[k31] > ts[:k31].max() and
                 "2023-10-30" not in {day_of(t) for t in ts} and extra["train_benign_check"]["rows"] == [0, k31],
                 f"dim {x.shape[1]} (78 + 8 bits); T = {k31} filas del 31-oct; sin 30-oct; sin Src Port")
        ck.check("g3.labels_preprocess_dups", np.array_equal(y, (dfA["Label"] != "Benign").astype(np.int64).to_numpy()) and
                 np.array_equal(x, expected_x(dfA_num, featsA, catA, k31)) and
                 np.array_equal(aux["dup_train"], brute_dup_mask(dfA_num, featsA, k31)) and aux["dup_train"][i_dup] and
                 extra["preprocessing"]["categorical"]["Connection Type"]["unseen_values_exec"] == ["fiber"] and
                 x[i_uns, -8:].tolist() == [0, 0, 0, 0, 0, 0, 1, 0],
                 f"y, x y duplicados == independiente; duplicado sin identificadores marcado; 'fiber' -> codigo 2")
        msg = raises(lambda: P.build_cic_aci_corrected(U.CODE_ROOT, fx, sA, train_rows=k31 + 1), C.PipelineError)
        ck.check("g3.train_rows_guard", msg is not None and "DEV-006" in msg, (msg or "")[:140])
        rf, rr, rT = t0_reference_pairs(t0_fresh(fx), "aci", "oct31_train_chrono", k31)
        ck.check("g3.t0_equivalence", rT == k31 and np.array_equal(aux["src_row"].astype(np.int64), rr),
                 f"orden fila a fila == warmup_audit ({len(rr)} filas)")

        cuts = json.loads((T0_DIR / "stream_cuts.json").read_text(encoding="utf-8"))
        for ds, key in (("CIC2017", "cic2017"), ("CIC2018", "cic2018"), ("ACI-IOT-2023", "aci")):
            sp = spec_for(ds)
            variant = C.CORRECTED_VARIANT[ds]

            def reader(path, key=key):
                stem = Path(path).stem.replace(" ", "_")
                base = T0_CACHE / "labels" / key
                meta = json.loads((base / f"{stem}.json").read_text())
                st = os.stat(path)
                src = meta["deps"]["src"]
                if meta["deps"].get("nrows") not in (0, None) or st.st_size != src["size"] or st.st_mtime_ns != src["mtime_ns"]:
                    raise FileNotFoundError(f"cache T0 de etiquetas no valida para {path}")
                with np.load(base / f"{stem}.npz", allow_pickle=False) as z:
                    return {k: z[k] for k in z.files}, meta["info"]
            idx = T0_CACHE / "streams" / f"{key}__{variant}.npz"
            try:
                plan = P.corrected_plan(sp, C.DATA_ROOT, U.CODE_ROOT, C.corrected_train_rows(sp), labels_reader=reader)
            except FileNotFoundError as e:
                ck.warn(f"g4.{key}", f"sin cache T0 de etiquetas: {e}")
                continue
            if not idx.exists():
                ck.warn(f"g4.{key}", f"sin {idx}")
                continue
            with np.load(idx) as z:
                gid_t0, off_t0 = z["gid"], z["file_offsets"]
            files_t0 = {"cic2017": wa.CIC2017_CHRONO, "cic2018": wa.CIC2018_ALL, "aci": [wa.ACI_FILE]}[key]
            mine_gid = off_t0[[files_t0.index(plan["vspec"]["files"][int(i)]) for i in range(len(plan["vspec"]["files"]))]][
                plan["src_file"].astype(np.int64)] + plan["src_row"].astype(np.int64)
            v = cuts[key]["variantes"][variant]
            exp = C.CORRECTED_EXPECTED[ds]
            got = {"n_rows": plan["n"], "train_end": plan["T"], "first_attack_index": plan["first_attack_index"],
                   "n_attack_train": plan["n_attack_train"], "discarded_label_filter": plan["discarded_label_filter"],
                   "guard_ties_dropped": int(plan["cuts"]["guard_ties_dropped"]["n"]),
                   "excluded_before_sort": int(sum(e["n"] for e in plan["excluded_before_sort"].values())),
                   "n_attack": plan["n_attack"]}
            exp = {k: v for k, v in exp.items() if k != "dim"}
            ck.check(f"g4.{key}", got == exp and np.array_equal(mine_gid, gid_t0) and
                     v["evaluacion"]["fin_modelo"] == plan["n"] and v["evaluacion"]["inicio_modelo"] == plan["T"] and
                     v["guarda_empates_timestamp"]["n"] == got["guard_ties_dropped"] and
                     int(plan["y"][plan["T"]:].sum()) == round(v["evaluacion"]["n_filas"] * v["evaluacion"]["pct_malicioso"] / 100),
                     f"{ds} {variant}: {got}; orden == t0/streams ({len(gid_t0)} filas)")

        import make_manifest as MM
        g_ok = MM.benign_guard_t0(C.OTHER_DATASETS)
        bad = json.loads((T0_DIR / "stream_cuts.json").read_text(encoding="utf-8"))
        bad["aci"]["variantes"]["oct31_train_chrono"]["entrenamiento"]["ataques_en_entrenamiento"] = 5
        bad["cic2018"]["variantes"]["days7"]["descartadas_por_filtro_de_etiqueta"]["n"] = 7
        bp = root / "stream_cuts_bad.json"
        bp.write_text(json.dumps(bad), encoding="utf-8")
        g_bad = MM.benign_guard_t0(C.OTHER_DATASETS, cuts_path=bp)
        other_script = root / "warmup_audit_otro.py"
        other_script.write_text("# otra version\n", encoding="utf-8")
        g_sha = MM.benign_guard_t0(["CIC2017"], t0_script=other_script)
        ck.check("g5.manifest_guard_t0", all(not g["problems"] for g in g_ok.values()) and g_bad["ACI-IOT-2023"]["problems"]
                 and g_bad["CIC2018"]["problems"] and not g_bad["CIC2017"]["problems"] and
                 any("warmup_audit.py actual" in m for m in g_sha["CIC2017"]["problems"]),
                 f"T0: { {k: (g['rows_checked'], g['n_attack']) for k, g in g_ok.items()} }; alterado -> "
                 f"{g_bad['ACI-IOT-2023']['problems'][:1]}")

        r = U.make_row(dataset="CIC2017", pipeline="corrected", variant="stat", priority=1)
        man = root / "m_g6.csv"
        U.write_manifest(man, [r])
        rc, out = U.run_job(root, man, r["job_id"], ["--data-root", str(fx), "--test-override",
                                                     "fm_grace=400,ad_grace=600,execution_window=1000,train_rows=1000"],
                            timeout=600)
        J = root / "results" / r["job_id"]
        if rc == 0 and (J / "DONE").exists():
            cfg = U.load_json(J / "config.json")
            reg = cfg["regions"]
            pairs = {k: C.check_protocol_regions(reg, k) for k in reg["detector_protocol_candidates"] if not k.startswith("_")}
            lab = np.load(J / "labels.npy")
            sc = np.load(J / "scores.npy")
            ck.check("g6.run_job", reg["train"] == [0, 1000] and reg["post_train"] == [1000, N17] and len(pairs) == 4 and
                     lab.shape == (N17,) and np.array_equal(lab, (df17[" Label"] != "BENIGN").to_numpy().astype(np.int8)) and
                     sc.shape == (N17,) and cfg["cache"]["extra"]["variant"] == "mon_first",
                     f"DONE; regiones {pairs['calib_first_10pct_post_train']}; labels.npy alineadas")
        else:
            ck.check("g6.run_job", False, f"rc={rc}\n{out[-2500:]}")

        if a.with_full:
            for ds, exp_dim, exp_dup in (("ACI-IOT-2023", 86, (0, 0, 0)), ("CIC2017", 78, (7678, 351, 4693))):
                p = subprocess.run([U.PY, str(U.RUNNER_DIR / "pipelines.py"), "--summary", ds], capture_output=True,
                                   text=True, timeout=600, env=U.base_env())
                s = json.loads(p.stdout)
                dd = s["extra"]["duplicates_train_exec"]
                okd = exp_dup is None or (dd["n"], dd["n_attack"], dd["by_calib_fraction"]["10"]["n_dup"]) == exp_dup
                ck.check(f"gfull.{ds}", p.returncode == 0 and s["x_shape"][1] == exp_dim and okd and
                         s["extra"]["expected_production"]["checked"] and dd["reference"]["matches"] and
                         s["extra"]["t0"]["script_matches_stream_cuts"],
                         f"{ds}: x {s['x_shape']}, duplicados {dd['n']} ({dd['n_attack']} ataque), 10 %: {dd['by_calib_fraction']['10']}")
    except Exception:
        import traceback
        ck.check("g.exception", False, traceback.format_exc()[-3000:])
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
