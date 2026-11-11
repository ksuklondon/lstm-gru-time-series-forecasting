"""Projekt 2A - analiza jednorazowej oceny testowej do Raportu 4.

Modul CZYTA wyniki_test.csv i prognozy_test/*.npz zapisane przez tm.evaluate_test (oraz wyniki
walidacyjne z Raportu 3) i tworzy tabele i wykresy Raportu 4. Niczego nie trenuje i nie wybiera
konfiguracji: konfiguracje sa zamrozone w final_configs.json.

Do przygotowania raportu przed ocena testowa sluzy build_standin(): tworzy pliki w identycznym formacie
z prognoz WALIDACYJNYCH (zbior testowy nie jest wtedy w ogole czytany). Wyniki zastepcze sa oznaczone
w kolumnie "zrodlo" i nie moga trafic do raportu jako wyniki testowe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ts_data as td
import ts_models as tm
import ts_raport3 as r3

NAMES = r3.NAMES
ND = r3.ND
MID_H = r3.MID_H
BASELINES = ("persistence", "seasonal_naive")


def _save_tab(df, out_dir, name):
    return r3._save_tab(df, out_dir, name)


# ---------------------------------------------------------------------------
# Wyniki zastepcze (z walidacji) do przygotowania raportu przed ocena testowa
# ---------------------------------------------------------------------------


def build_standin(out=tm.OUT, dest="outputs/zastepcze_z_walidacji") -> Path:
    """Tworzy wyniki_test.csv i prognozy_test/ w formacie tm.evaluate_test, ale z prognoz WALIDACYJNYCH.
    Zbior testowy nie jest czytany. Sluzy tylko do sprawdzenia kodu, tabel i wykresow Raportu 4."""
    out, dest = Path(out), Path(dest)
    pdir = dest / "prognozy_test"
    pdir.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(out / tm.CONFIG_FILE))
    common = {}
    for c in cfg["komorki"]:
        for rid in c["run_ids"]:
            t = np.load(out / "prognozy" / f"{rid}_val.npz")["tgt"]
            k = (c["key"], c["h"])
            common[k] = np.intersect1d(common.get(k, t), t)
    rows, done = [], set()
    for c in cfg["komorki"]:
        key, h = c["key"], c["h"]
        tgt = common[(key, h)]
        y = tm.prepare(key)["y"][tgt]
        bp = tm.baseline_predictions(key, h, tgt)
        mae_p = float(np.mean(np.abs(y - bp["persistence"])))
        if (key, h) not in done:
            for name, pred in bp.items():
                rows.append({"grupa": "bazowe", "key": key, "model": name, "h": h, "L": np.nan, "features": "",
                             "variant": "", "seed": np.nan, "run_id": f"{key}_{name}_h{h}", "n_test": len(tgt),
                             **{f"test_{k}": v for k, v in tm.metrics(y, pred, key, mae_p).items()}})
                np.savez_compressed(pdir / f"{key}_{name}_h{h}_test.npz", tgt=tgt, yhat=pred.astype("float32"))
            done.add((key, h))
        for rid in c["run_ids"]:
            z = np.load(out / "prognozy" / f"{rid}_val.npz")
            m = np.isin(z["tgt"], tgt)
            yhat = z["yhat"][m]
            np.savez_compressed(pdir / f"{rid}_test.npz", tgt=z["tgt"][m], yhat=yhat)
            seed = np.nan if c["model"] == "linreg" else int(rid.rsplit("_s", 1)[1])
            rows.append({"grupa": c["grupa"], "key": key, "model": c["model"], "h": h, "L": c["L"],
                         "features": c.get("features", "uni"), "variant": c["variant"], "seed": seed, "run_id": rid,
                         "n_test": len(tgt), **{f"test_{k}": v for k, v in tm.metrics(y, yhat, key, mae_p).items()}})
    res = pd.DataFrame(rows)
    res["zrodlo"] = "ZASTEPCZE: walidacja"
    res.to_csv(dest / "wyniki_test.csv", index=False)
    return dest


def load_test(path=tm.OUT) -> pd.DataFrame:
    res = pd.read_csv(Path(path) / "wyniki_test.csv")
    if "zrodlo" not in res.columns:
        res["zrodlo"] = "test"
    return res


def is_standin(res) -> bool:
    return bool((res.zrodlo != "test").any())


def pred(run_id: str, path=tm.OUT):
    z = np.load(Path(path) / "prognozy_test" / f"{run_id}_test.npz")
    return z["tgt"], z["yhat"].astype(float)


# ---------------------------------------------------------------------------
# Tabela koncowa
# ---------------------------------------------------------------------------


def _agg(g: pd.DataFrame) -> dict:
    return {"MAE": g.test_MAE.mean(), "MAE_sd": g.test_MAE.std() if len(g) > 1 else 0.0,
            "MSE": g.test_MSE.mean(), "RMSE": g.test_RMSE.mean(), "MAPE": g.test_MAPE.mean(),
            "sMAPE": g.test_sMAPE.mean(), "skill": g.test_skill.mean(),
            "skill_sd": g.test_skill.std() if len(g) > 1 else 0.0, "n": len(g), "run_ids": list(g.run_id)}


def final_table(res: pd.DataFrame, group: str = "A", df_val: pd.DataFrame | None = None,
                bl_val: pd.DataFrame | None = None, out_dir=None) -> pd.DataFrame:
    """Koncowa tabela porownawcza dla grupy konfiguracji ('A' albo 'najlepsza') wraz z modelami naiwnymi,
    walidacyjnym MAE tych samych konfiguracji oraz kosztem (czas uczenia i prognozy z treningow)."""
    rows = []
    for (key, h), gk in res.groupby(["key", "h"]):
        for m in BASELINES:
            g = gk[(gk.grupa == "bazowe") & (gk.model == m)]
            rows.append({"key": key, "h": h, "model": m, "konfiguracja": "", **_agg(g)})
        for m in ("linreg", "Dense", "LSTM", "GRU", "TCN"):
            g = gk[(gk.grupa == group) & (gk.model == m)]
            if g.empty:
                continue
            r0 = g.iloc[0]
            conf = f"L = {int(r0.L)}" + (", + zmienne" if r0.features == "multi" else "") + \
                   (f", {r0.variant}" if key == "air" else "")
            rows.append({"key": key, "h": h, "model": m, "konfiguracja": conf, **_agg(g)})
    t = pd.DataFrame(rows)
    if df_val is not None:
        v = df_val.set_index("run_id")
        t["val_MAE"] = [np.mean([v.val_MAE.get(r, np.nan) for r in ids]) if m not in BASELINES + ("linreg",) else np.nan
                        for m, ids in zip(t.model, t.run_ids)]
        t["czas_uczenia_s"] = [np.mean([v.train_time_s.get(r, np.nan) for r in ids]) if m not in BASELINES + ("linreg",)
                               else np.nan for m, ids in zip(t.model, t.run_ids)]
        t["inferencja_ms"] = [np.median([v.infer_1000_ms.get(r, np.nan) for r in ids]) if m not in BASELINES + ("linreg",)
                              else np.nan for m, ids in zip(t.model, t.run_ids)]
    if bl_val is not None:
        b = bl_val.set_index("run_id")
        for i, (m, ids) in enumerate(zip(t.model, t.run_ids)):
            if m in BASELINES:
                rid = f"{t.key.iloc[i]}_{m}_h{t.h.iloc[i]}"
                t.loc[t.index[i], "val_MAE"] = b.val_MAE.get(rid, np.nan) if not isinstance(b.val_MAE.get(rid), pd.Series) \
                    else b.val_MAE.get(rid).iloc[0]
            elif m == "linreg":
                t.loc[t.index[i], "val_MAE"] = b.val_MAE.get(ids[0], np.nan)
    best = t.groupby(["key", "h"]).MAE.transform("min")
    t["najlepszy"] = t.MAE == best
    t["key"] = pd.Categorical(t.key, tm.KEYS)
    t = t.sort_values(["key", "h"], kind="stable").reset_index(drop=True)
    _save_tab(t.drop(columns="run_ids"), out_dir, f"test_koncowa_{group}")
    return t


def best_simple(t: pd.DataFrame, key: str, h: int, by="test") -> pd.Series:
    """Najlepszy prosty model bazowy (persistence, seasonal naive, regresja liniowa). by='val': wybrany na
    walidacji (zgodnie z kryterium H3 z Raportu 1), by='test': najmniejsze MAE testowe."""
    s = t[(t.key == key) & (t.h == h) & t.model.isin(BASELINES + ("linreg",))]
    col = "val_MAE" if by == "val" else "MAE"
    return s.loc[s[col].idxmin()]


def improvement_table(t: pd.DataFrame, out_dir=None, tag="A") -> pd.DataFrame:
    """Czy LSTM/GRU dają rzeczywistą poprawę: różnica MAE względem najlepszego prostego modelu
    (wybranego na walidacji) i względem Dense, w odniesieniu do rozrzutu między ziarnami."""
    rows = []
    for (key, h), g in t.groupby(["key", "h"], observed=True):
        bs = best_simple(t, key, h, by="val")
        dense = g[g.model == "Dense"]
        for m in ("LSTM", "GRU", "TCN"):
            r = g[g.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            d = bs.MAE - r.MAE
            rows.append({"key": key, "h": h, "model": m, "MAE": r.MAE, "MAE_sd": r.MAE_sd,
                         "najlepszy_prosty": NAMES[bs.model], "MAE_prosty": bs.MAE,
                         "poprawa_%": 100 * d / bs.MAE,
                         "wyraźna": bool(d > r.MAE_sd),
                         "vs_Dense_%": 100 * (dense.MAE.iloc[0] - r.MAE) / dense.MAE.iloc[0] if len(dense) else np.nan})
    out = pd.DataFrame(rows)
    _save_tab(out, out_dir, f"test_poprawa_{tag}")
    return out


# ---------------------------------------------------------------------------
# Hipotezy (kryteria z Raportu 1, punkt 12.5)
# ---------------------------------------------------------------------------


def hypotheses(res: pd.DataFrame, df_val: pd.DataFrame, bl_val: pd.DataFrame, out_dir=None) -> dict:
    tA = final_table(res, "A", df_val, bl_val)
    out = {}
    # H1: skill(h=6) > skill(h=1) i skill(h=24) > skill(h=1), obie roznice > rozrzut ziaren; dla obu sieci
    rows = []
    for key in ("jena", "household", "metro"):
        for m in ("LSTM", "GRU"):
            g = tA[(tA.key == key) & (tA.model == m)].set_index("h")
            s1, s6, s24 = g.skill[1], g.skill[6], g.skill[24]
            sd = g.skill_sd
            ok6 = (s6 - s1) > max(sd[1], sd[6])
            ok24 = (s24 - s1) > max(sd[1], sd[24])
            rows.append({"key": key, "model": m, "skill_h1": s1, "skill_h6": s6, "skill_h24": s24,
                         "h6>h1": ok6, "h24>h1": ok24, "warunek": ok6 and ok24})
    h1 = pd.DataFrame(rows)
    per_set = h1.groupby("key").warunek.all()
    out["H1"] = {"tabela": h1, "potwierdzona": int(per_set.sum()) >= 2,
                 "opis": f"warunek spełniony dla obu sieci w {int(per_set.sum())} z 3 zbiorów godzinowych"}
    # H3: Air - ani LSTM, ani GRU nie ma MAE nizszego od najlepszego prostego (wybranego na walidacji) o wiecej niz sd
    rows = []
    for h in td.SPECS["air"].horizons:
        bs = best_simple(tA, "air", h, by="val")
        for m in ("LSTM", "GRU"):
            r = tA[(tA.key == "air") & (tA.h == h) & (tA.model == m)].iloc[0]
            rows.append({"h": h, "model": m, "MAE": r.MAE, "MAE_sd": r.MAE_sd, "prosty": NAMES[bs.model],
                         "MAE_prosty": bs.MAE, "wyraźnie_lepsza": bool(bs.MAE - r.MAE > r.MAE_sd)})
    h3 = pd.DataFrame(rows)
    out["H3"] = {"tabela": h3, "potwierdzona_dla_h": {h: not g.wyraźnie_lepsza.any() for h, g in h3.groupby("h")}}
    # H5: w wiekszosci komorek bloku A |MAE_GRU - MAE_LSTM| <= max(sd) oraz sredni czas uczenia GRU < LSTM
    rows = []
    for (key, h), g in tA.groupby(["key", "h"], observed=True):
        l, r = g[g.model == "LSTM"].iloc[0], g[g.model == "GRU"].iloc[0]
        rows.append({"key": key, "h": h, "MAE_LSTM": l.MAE, "MAE_GRU": r.MAE, "różnica": r.MAE - l.MAE,
                     "rozrzut": max(l.MAE_sd, r.MAE_sd), "porównywalne": abs(r.MAE - l.MAE) <= max(l.MAE_sd, r.MAE_sd),
                     "czas_LSTM_s": l.get("czas_uczenia_s", np.nan), "czas_GRU_s": r.get("czas_uczenia_s", np.nan)})
    h5 = pd.DataFrame(rows)
    a = df_val[(df_val.block == "A") & df_val.model.isin(["LSTM", "GRU"])]
    times = a.groupby("model").train_time_s.mean()
    out["H5"] = {"tabela": h5, "komorki_porownywalne": int(h5["porównywalne"].sum()), "komorki": len(h5),
                 "czas_LSTM_s": float(times["LSTM"]), "czas_GRU_s": float(times["GRU"]),
                 "potwierdzona": bool(h5["porównywalne"].sum() > len(h5) / 2 and times["GRU"] < times["LSTM"])}
    # H2 i H4: kryteria z Raportu 1 dotycza walidacji (bloki B i C) - liczymy je z wynikow walidacyjnych
    tb = r3.block_b_table(df_val, bl_val)
    rows = []
    short = {"jena": 12, "metro": 72, "air": 6}
    for key, Ls in short.items():
        for (h, m), g in tb[(tb.key == key) & tb.model.isin(["LSTM", "GRU"])].groupby(["h", "model"]):
            g = g[~((g.key == "jena") & (g.L == 30))].set_index("L")
            s = g.loc[Ls]
            longer = g[g.index > Ls]
            ok = bool(((s.MAE - longer.MAE) > np.maximum(s.MAE_sd, longer.MAE_sd)).all())
            rows.append({"key": key, "h": h, "model": m, "MAE_krótkie": s.MAE,
                         "MAE_dłuższe": ", ".join(f"{int(L)}: {v:.3f}" for L, v in longer.MAE.items()), "warunek": ok})
    h2 = pd.DataFrame(rows)
    out["H2"] = {"tabela": h2, "spełnione": int(h2.warunek.sum()), "komorki": len(h2)}
    tc = r3.block_c_table(df_val, bl_val)
    out["H4"] = {"tabela": tc[["key", "h", "LSTM_uni", "LSTM_multi", "zmiana_LSTM_%", "wszystkie_ziarna_lepsze"]],
                 "potwierdzona_dla": {k: bool(g.wszystkie_ziarna_lepsze.all()) for k, g in tc.groupby("key")}}
    for k, v in out.items():
        _save_tab(v["tabela"], out_dir, f"hipoteza_{k}")
    return out


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------


def _models_for_plot(res, key, h, group="A"):
    """run_id (ziarno 42 albo jedyny) dla LSTM, GRU i regresji liniowej w danej grupie."""
    out = {}
    for m in ("LSTM", "GRU", "linreg", "TCN"):
        g = res[(res.grupa == group) & (res.key == key) & (res.h == h) & (res.model == m)]
        if g.empty:
            continue
        g42 = g[g.seed == 42]
        out[m] = (g42 if len(g42) else g).run_id.iloc[0]
    return out


def representative_start(res, key, h, path, n, group="A"):
    """Poczatek fragmentu o dlugosci n, w ktorym MAE LSTM jest najblizsze medianie MAE wszystkich fragmentow
    (fragment reprezentatywny, a nie wybrany pod najlepszy wynik)."""
    rid = _models_for_plot(res, key, h, group)["LSTM"]
    tgt, yhat = pred(rid, path)
    y = tm.prepare(key)["y"][tgt]
    e = np.abs(y - yhat)
    if len(e) <= n:
        return 0
    starts = np.arange(0, len(e) - n, max(1, n // 4))
    maes = np.array([e[s:s + n].mean() for s in starts])
    return int(starts[np.argmin(np.abs(maes - np.median(maes)))])


def plot_test_forecast(res, key, h, path=tm.OUT, group="A", out_dir=None, n=None, start=None, title_note=""):
    import matplotlib.pyplot as plt
    P = tm.prepare(key)
    ids = _models_for_plot(res, key, h, group)
    n = n or (10_000 if key == "air" else 24 * 14)
    st = start if start is not None else (0 if key == "air" else representative_start(res, key, h, path, n, group))
    t0, _ = pred(ids["LSTM"], path)
    tgt0 = t0[st:st + n]
    idx = P["index"][tgt0]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(idx, P["y"][tgt0], color="#222222", lw=1.4, label="wartość rzeczywista")
    bp = tm.baseline_predictions(key, h, tgt0)
    ax.plot(idx, bp["persistence"], color=tm.MODEL_COLORS["persistence"], lw=0.9, ls=":", label="persistence")
    ax.plot(idx, bp["seasonal_naive"], color=tm.MODEL_COLORS["seasonal_naive"], lw=0.9, ls=":", label="seasonal naive")
    for m, rid in ids.items():
        t, yh = pred(rid, path)
        mk = np.isin(t, tgt0)
        ax.plot(P["index"][t[mk]], yh[mk], color=tm.MODEL_COLORS[m], lw=1.0 if m != "linreg" else 0.9,
                ls="--" if m == "linreg" else "-", label=NAMES[m])
    ax.set_ylabel(td.SPECS[key].unit)
    ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}: prognozy na fragmencie zbioru testowego{title_note}")
    ax.legend(ncol=7, loc="upper left", fontsize=8)
    fig.tight_layout()
    td._save(fig, out_dir, f"test_prognoza_{key}_h{h}_{group}")
    return fig, (idx[0], idx[-1])


def errors_frame(res, key, h, path=tm.OUT, group="A") -> pd.DataFrame:
    """Bledy (prognoza - rzeczywistosc) w czasie dla LSTM, GRU, regresji liniowej i modeli naiwnych."""
    P = tm.prepare(key)
    ids = _models_for_plot(res, key, h, group)
    t0, _ = pred(ids["LSTM"], path)
    d = pd.DataFrame(index=P["index"][t0])
    d["y"] = P["y"][t0]
    bp = tm.baseline_predictions(key, h, t0)
    for m in BASELINES:
        d[m] = bp[m] - d.y.values
    for m, rid in ids.items():
        t, yh = pred(rid, path)
        s = pd.Series(yh, index=P["index"][t]).reindex(d.index)
        d[m] = s.values - d.y.values
    return d


def plot_error_time(res, key, h, path=tm.OUT, group="A", out_dir=None):
    """Bezwzgledny blad wygladzony srednia kroczaca (7 dni; Air: bez wygladzania) w calym okresie testowym."""
    import matplotlib.pyplot as plt
    d = errors_frame(res, key, h, path, group)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    # persistence pomijamy: przy h > 1 jej blad jest wielokrotnie wiekszy i splaszczalby pozostale krzywe
    cols = [c for c in ("seasonal_naive", "linreg", "LSTM", "GRU") if c in d]
    for c in cols:
        e = d[c].abs()
        if key != "air":
            e = e.rolling("7D", min_periods=24).mean()
        ax.plot(e.index, e.values, lw=1.0 if c in ("LSTM", "GRU") else 0.8, color=tm.MODEL_COLORS[c],
                ls=":" if c in BASELINES else ("--" if c == "linreg" else "-"), label=NAMES[c])
    ax.set_ylabel(f"|błąd| [{td.SPECS[key].unit}]")
    ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}: błąd bezwzględny w okresie testowym"
                 + (" (średnia krocząca 7 dni)" if key != "air" else ""))
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    fig.tight_layout()
    td._save(fig, out_dir, f"test_blad_czas_{key}_h{h}")
    return fig


def plot_residuals(res, path=tm.OUT, group="A", out_dir=None):
    """Rozklad reszt (prognoza - rzeczywistosc) LSTM, GRU i regresji liniowej dla horyzontu srodkowego."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    stats = []
    for ax, key in zip(axes, tm.KEYS):
        h = MID_H[key]
        d = errors_frame(res, key, h, path, group)
        for c in ("linreg", "LSTM", "GRU"):
            if c not in d:
                continue
            e = d[c].dropna()
            bins = 10 if key == "air" else 80
            ax.hist(e, bins=bins, histtype="step", lw=1.2, color=tm.MODEL_COLORS[c], label=NAMES[c], density=True)
            stats.append({"key": key, "h": h, "model": c, "średnia reszta": e.mean(), "mediana": e.median(),
                          "odch. std.": e.std(), "skośność": e.skew(), "P5": e.quantile(0.05), "P95": e.quantile(0.95)})
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}", fontsize=10)
        ax.set_xlabel(f"reszta [{td.SPECS[key].unit}]")
    axes[0].set_ylabel("gęstość")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Rozkład reszt (prognoza − wartość rzeczywista) na zbiorze testowym", fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "test_reszty")
    st = pd.DataFrame(stats)
    _save_tab(st, out_dir, "test_reszty_statystyki")
    return fig, st


def plot_error_profile(res, path=tm.OUT, group="A", out_dir=None):
    """MAE w zaleznosci od godziny doby (zbiory godzinowe) - gdzie modele myla sie najbardziej."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    for ax, key in zip(axes, ("jena", "household", "metro")):
        h = MID_H[key]
        d = errors_frame(res, key, h, path, group)
        for c in ("seasonal_naive", "linreg", "LSTM", "GRU"):
            if c not in d:
                continue
            p = d[c].abs().groupby(d.index.hour).mean()
            ax.plot(p.index, p.values, marker="o", ms=2.5, color=tm.MODEL_COLORS[c],
                    ls=":" if c in BASELINES else ("--" if c == "linreg" else "-"), label=NAMES[c])
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("godzina doby (chwila prognozowana)")
        ax.set_title(f"{td.SPECS[key].name}, h = {h} h", fontsize=10)
        ax.set_ylabel(f"MAE [{td.SPECS[key].unit}]")
    axes[-1].legend(fontsize=8)
    fig.suptitle("MAE na zbiorze testowym według godziny doby", fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "test_blad_godzina")
    return fig


def largest_errors(res, key, h, path=tm.OUT, group="A", model="LSTM", top=10, out_dir=None) -> pd.DataFrame:
    """Dni (Air: miesiace) z najwiekszym MAE wybranego modelu wraz z bledem modeli bazowych tego dnia
    i opisem kalendarza (dzien tygodnia, swieto w Metro)."""
    d = errors_frame(res, key, h, path, group)
    P = tm.prepare(key)
    if key == "air":
        g = d.abs()
        g["data"] = d.index.strftime("%Y-%m")
        day = g.set_index("data")
        day["y"] = d.y.values
    else:
        a = d.drop(columns="y").abs()
        a["y"] = d.y
        day = a.groupby(d.index.normalize()).mean()
        day["n"] = a.groupby(d.index.normalize()).size()
        day = day[day.n >= 12]
        day.index = day.index.strftime("%Y-%m-%d")
    day = day.sort_values(model, ascending=False).head(top)
    out = pd.DataFrame({"okres": day.index, f"MAE {model}": day[model].values, "średnia y": day.y.values})
    for c in ("GRU", "linreg", "seasonal_naive", "persistence"):
        if c in day and c != model:
            out[f"MAE {NAMES[c]}"] = day[c].values
    if key != "air":
        dts = pd.to_datetime(out.okres)
        out.insert(1, "dzień", [["pn", "wt", "śr", "cz", "pt", "so", "nd"][x.dayofweek] for x in dts])
        if key == "metro":
            hol = P["h_f"]["holiday"].groupby(P["h_f"].index.normalize()).max()
            out.insert(2, "święto", ["tak" if hol.get(x, 0) > 0 else "" for x in dts])
    _save_tab(out, out_dir, f"test_najwieksze_bledy_{key}_h{h}")
    return out


def val_vs_test(t: pd.DataFrame, out_dir=None, tag="A") -> pd.DataFrame:
    """Uogolnienie: MAE walidacyjne i testowe tych samych zamrozonych konfiguracji."""
    s = t[["key", "h", "model", "val_MAE", "MAE", "skill"]].copy()
    s["zmiana_%"] = 100 * (s.MAE / s.val_MAE - 1)
    _save_tab(s, out_dir, f"walidacja_vs_test_{tag}")
    return s


def plot_test_skill(t: pd.DataFrame, out_dir=None, tag="A"):
    """Skill score wzgledem persistence na zbiorze testowym w funkcji horyzontu (grupa konfiguracji tag)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, key in zip(axes, tm.KEYS):
        hs = list(td.SPECS[key].horizons)
        x = np.arange(len(hs))
        for m in ("seasonal_naive", "linreg", "Dense", "LSTM", "GRU", "TCN"):
            g = t[(t.key == key) & (t.model == m)].set_index("h").reindex(hs)
            if g.MAE.isna().all():
                continue
            y = g.skill.values
            if m in ("seasonal_naive", "linreg"):
                ax.plot(x, np.maximum(y, -0.5), marker="s", ms=3.5, ls=":", color=tm.MODEL_COLORS[m], label=NAMES[m])
                for xi, yi in zip(x, y):
                    if yi < -0.5:
                        ax.annotate(f"{yi:.2f}".replace(".", ","), (xi, -0.5), textcoords="offset points",
                                    xytext=(4, 4), fontsize=7.5, color=tm.MODEL_COLORS[m])
            else:
                ax.errorbar(x, y, yerr=g.skill_sd.values, marker="o", ms=4, capsize=3, color=tm.MODEL_COLORS[m], label=m)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_ylim(bottom=-0.55)
        ax.set_xticks(x, [f"{h} {td.SPECS[key].step_word}" for h in hs])
        ax.set_title(td.SPECS[key].name, fontsize=10)
    axes[0].set_ylabel("skill score względem persistence")
    hd, lb = [], []
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in lb:
                hd.append(h_); lb.append(l_)
    fig.legend(hd, lb, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Skill score na zbiorze testowym (0 = persistence, 1 = prognoza idealna)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    td._save(fig, out_dir, f"test_skill_{tag}")
    return fig
