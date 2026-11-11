"""Projekt 2A - analiza wynikow uczenia i walidacji do Raportu 3 (bloki A-E).

Modul tylko CZYTA zapisane wyniki (wyniki.csv, modele_bazowe_val.csv, historie, prognozy walidacyjne,
final_configs.json, testy_programu.csv) i tworzy tabele oraz wykresy. Niczego nie trenuje i nie korzysta
ze zbioru testowego. Kazda funkcja zwraca DataFrame albo rysunek; z argumentem out_dir zapisuje go
w <out_dir>/tables/*.csv lub <out_dir>/figures/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ts_data as td
import ts_models as tm

NAMES = {"persistence": "persistence", "seasonal_naive": "seasonal naive", "linreg": "regresja liniowa",
         "Dense": "Dense", "LSTM": "LSTM", "GRU": "GRU", "TCN": "TCN"}
ORDER = ["persistence", "seasonal_naive", "linreg", "Dense", "LSTM", "GRU", "TCN"]
MID_H = {"jena": 6, "household": 6, "metro": 6, "air": 3}


def _save_tab(df: pd.DataFrame, out_dir, name: str):
    if out_dir is not None:
        p = Path(out_dir) / "tables"
        p.mkdir(parents=True, exist_ok=True)
        df.to_csv(p / f"{name}.csv", index=False)
    return df


def load(out=tm.OUT):
    """Wyniki sieci (bez przebiegow testowych kodu) i modeli bazowych."""
    df = tm.load_results(out)
    df = df[~df.quick.astype(bool)].copy()
    bl = tm.load_baselines(out)
    return df, bl


def pct_cols(key):
    return [f"val_{m}" for m in tm.PCT_METRICS[key]]


def fmt_pm(m, s, nd):
    return f"{m:.{nd}f}" if pd.isna(s) or s == 0 else f"{m:.{nd}f} ± {s:.{nd}f}"


ND = {"jena": 3, "household": 3, "metro": 0, "air": 1}

# ---------------------------------------------------------------------------
# Implementacja i ustawienia
# ---------------------------------------------------------------------------


def training_settings() -> pd.DataFrame:
    T = tm.TRAIN
    rows = [
        ("Funkcja straty", "MSE na standaryzowanym celu (Air: na przekształconym celu)"),
        ("Optymalizator", "Adam"),
        ("Współczynnik uczenia", f"{T['lr']:g} (blok D: także 0,0003)"),
        ("Rozmiar partii", f"{T['batch']} (Air Passengers: {T['batch_air']})"),
        ("Maksymalna liczba epok", str(T["epochs"])),
        ("Wczesne zatrzymanie", f"monitor val_loss, cierpliwość {T['patience']} epok, przywrócenie najlepszych wag"),
        ("Kolejność przykładów", "mieszanie okien zbioru uczącego co epokę (shuffle=True)"),
        ("Ziarna losowości", "42, 43, 44 (Air Passengers: 42–51)"),
        ("Ustawianie ziarna", "keras.utils.set_random_seed (Python, NumPy, TensorFlow) przed każdym treningiem"),
        ("Wybór konfiguracji", "najmniejsze średnie MAE z ziaren na walidacji"),
    ]
    return pd.DataFrame(rows, columns=["ustawienie", "wartość"])


def architecture_table(df: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    """Warstwy i liczba parametrow dla wszystkich uzytych kombinacji (model, okno, liczba cech)."""
    desc = {
        "Dense": "Flatten → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)",
        "LSTM": "LSTM(u) → Dropout(0,2) → Dense(1)",
        "GRU": "GRU(u) → Dropout(0,2) → Dense(1)",
        "TCN": "5 bloków resztkowych [2 × Conv1D przyczynowy(25, k=3, dylatacja 1–16) + Dropout] → ostatni krok → Dense(1)",
    }
    g = (df.groupby(["model", "units", "layers", "L", "features", "key"]).params.first().reset_index())
    rows = []
    for (m, u, l, L, f), gg in g.groupby(["model", "units", "layers", "L", "features"]):
        d = "wiele zmiennych" if f == "multi" else "1 zmienna"
        keys = ", ".join(td.SPECS[k].name for k in sorted(gg.key.unique()))
        p = sorted(gg.params.unique())
        rows.append({"model": m, "jednostki/filtry": int(u) if m != "Dense" else "–", "warstwy": int(l) if m != "Dense" else 2,
                     "okno L": int(L), "wejście": d, "zbiory": keys,
                     "parametry": ", ".join(f"{x:,}".replace(",", " ") for x in p)})
    out = pd.DataFrame(rows)
    out["opis"] = out.model.map(desc)
    out = out.sort_values(["model", "wejście", "okno L", "jednostki/filtry"], key=lambda s: s.astype(str))
    return _save_tab(out, out_dir, "architektury")


def model_summaries():
    """Wydruk model.summary() dla konfiguracji bazowych (okno 72, 1 zmienna wejsciowa)."""
    import keras
    for kind, kw in (("Dense", {}), ("LSTM", {}), ("GRU", {}), ("TCN", dict(units=tm.TCN_FILTERS, layers=tm.TCN_BLOCKS))):
        m = tm.build_model(kind, 72, 1, **kw)
        print(f"=== {kind} ===")
        m.summary()
        keras.backend.clear_session()


def tests_table(path) -> pd.DataFrame:
    t = pd.read_csv(path)
    cols = [c for c in ["nr", "obszar", "dane", "oczekiwany wynik", "wynik", "status"] if c in t.columns]
    return t[cols]


# ---------------------------------------------------------------------------
# Uczenie i przeuczenie
# ---------------------------------------------------------------------------


def overfit_summary(df: pd.DataFrame, out=tm.OUT, out_dir=None) -> pd.DataFrame:
    """Diagnostyka przeuczenia dla kazdego treningu i podsumowanie po (blok, zbior, model)."""
    rows = []
    for _, r in df.iterrows():
        H = tm.history(r.run_id, out)
        v, l = np.array(H["val_loss"]), np.array(H["loss"])
        b = int(np.argmin(v))
        rows.append({"block": r.block, "key": r.key, "model": r.model, "h": r.h, "run_id": r.run_id,
                     "epoki": len(l), "najlepsza": b + 1, "limit": len(l) >= tm.TRAIN["epochs"],
                     "val_train": v[b] / l[b], "wzrost_val": 100 * (v[-1] / v[b] - 1),
                     "MAE_val_train": r.val_MAE / r.train_MAE})
    d = pd.DataFrame(rows)
    s = (d.groupby(["key", "model"]).agg(treningi=("run_id", "size"), epoki=("epoki", "mean"),
                                         najlepsza=("najlepsza", "mean"), limit=("limit", "sum"),
                                         val_train=("val_train", "median"), wzrost_val=("wzrost_val", "median"),
                                         MAE_val_train=("MAE_val_train", "median")).reset_index())
    s["key"] = pd.Categorical(s.key, tm.KEYS)
    s["model"] = pd.Categorical(s.model, [m for m in ORDER if m in s.model.unique()])
    s = s.sort_values(["key", "model"])
    _save_tab(d, out_dir, "przeuczenie_treningi")
    return _save_tab(s, out_dir, "przeuczenie")


def plot_curves_grid(df: pd.DataFrame, out=tm.OUT, out_dir=None, block="A"):
    """Krzywe uczenia: wiersz = zbior (horyzont srodkowy), kolumna = model; wszystkie ziarna."""
    import matplotlib.pyplot as plt
    models = ("Dense", "LSTM", "GRU")
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for i, key in enumerate(tm.KEYS):
        h = MID_H[key]
        sel = df[(df.block == block) & (df.key == key) & (df.h == h)]
        if key == "air":
            sel = sel[sel.variant == "sdiff"]
        for j, m in enumerate(models):
            ax = axes[i, j]
            for _, r in sel[sel.model == m].iterrows():
                H = tm.history(r.run_id, out)
                ep = np.arange(1, len(H["loss"]) + 1)
                c = tm.MODEL_COLORS[m]
                ax.plot(ep, H["loss"], color=c, lw=0.9, alpha=0.75)
                ax.plot(ep, H["val_loss"], color=c, lw=0.9, ls="--", alpha=0.75)
                b = int(np.argmin(H["val_loss"]))
                ax.plot(b + 1, H["val_loss"][b], "o", color=c, ms=3.5)
            ax.set_yscale("log")
            ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}: {m}", fontsize=10)
            if i == 3:
                ax.set_xlabel("epoka")
            if j == 0:
                ax.set_ylabel("strata MSE")
    fig.suptitle("Strata ucząca (linia ciągła) i walidacyjna (linia przerywana); kropka = epoka najlepszej straty walidacyjnej",
                 fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "krzywe_uczenia_A")
    return fig


def plot_curves_jena_all(df: pd.DataFrame, out=tm.OUT, out_dir=None):
    """Krzywe uczenia LSTM, GRU i TCN dla Jena Climate (okno 72 h) na trzech horyzontach."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    sel = df[(df.key == "jena") & (df.L == 72) & (df.features == "uni") & (df.units.isin([64, tm.TCN_FILTERS]))
             & (df.block.isin(["A", "E"])) & df.model.isin(["LSTM", "GRU", "TCN"])]
    for ax, h in zip(axes, td.SPECS["jena"].horizons):
        for _, r in sel[sel.h == h].iterrows():
            H = tm.history(r.run_id, out)
            ep = np.arange(1, len(H["loss"]) + 1)
            c = tm.MODEL_COLORS[r.model]
            ax.plot(ep, H["val_loss"], color=c, lw=1, label=r.model if r.seed == 42 else None)
        ax.set_yscale("log")
        ax.set_title(f"h = {h} h", fontsize=10)
        ax.set_xlabel("epoka")
    axes[0].set_ylabel("strata walidacyjna MSE")
    axes[-1].legend()
    fig.suptitle("Jena Climate, okno 72 h: strata walidacyjna LSTM, GRU i TCN (po 3 ziarna)", fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "krzywe_jena_tcn")
    return fig


# ---------------------------------------------------------------------------
# Porownanie z modelami bazowymi (blok A)
# ---------------------------------------------------------------------------


def block_a_table(df: pd.DataFrame, bl: pd.DataFrame, key: str, out_dir=None) -> pd.DataFrame:
    """Metryki walidacyjne bloku A dla jednego zbioru: modele bazowe i sieci (srednia ± odch. std. z ziaren).
    Dla Air Passengers sieci i regresja w wariancie wybranym na walidacji (najmniejsze srednie MAE)."""
    nd = ND[key]
    rows = []
    for h in td.SPECS[key].horizons:
        b = bl[(bl.key == key) & (bl.h == h)]
        for m in ("persistence", "seasonal_naive"):
            r = b[b.model == m].iloc[0]
            rows.append((h, m, r.val_MAE, 0, r.val_RMSE, r.get("val_sMAPE", np.nan), r.get("val_MAPE", np.nan), r.val_skill, 0, ""))
        lin = b[(b.model == "linreg") & (b.L == tm.WINDOW_A[key]) & (b.features == "uni")]
        r = lin.loc[lin.val_MAE.idxmin()]
        rows.append((h, "linreg", r.val_MAE, 0, r.val_RMSE, r.val_sMAPE, r.val_MAPE, r.val_skill, 0,
                     r.variant if key == "air" else ""))
        a = df[(df.block == "A") & (df.key == key) & (df.h == h)]
        for m in ("Dense", "LSTM", "GRU"):
            g = a[a.model == m]
            v = g.groupby("variant").val_MAE.mean().idxmin()
            g = g[g.variant == v]
            rows.append((h, m, g.val_MAE.mean(), g.val_MAE.std(), g.val_RMSE.mean(), g.val_sMAPE.mean(),
                         g.val_MAPE.mean(), g.val_skill.mean(), g.val_skill.std(), v if key == "air" else ""))
    t = pd.DataFrame(rows, columns=["h", "model", "MAE", "MAE_sd", "RMSE", "sMAPE", "MAPE", "skill", "skill_sd", "wariant"])
    best = t.groupby("h").MAE.transform("min")
    t["najlepszy"] = t.MAE == best
    _save_tab(t, out_dir, f"walidacja_A_{key}")
    return t


def block_a_display(t: pd.DataFrame, key: str) -> pd.DataFrame:
    nd = ND[key]
    out = pd.DataFrame({
        "h": t.h, "model": t.model.map(NAMES) + np.where(t.wariant != "", " (" + t.wariant + ")", ""),
        "MAE": [fmt_pm(m, s, nd) for m, s in zip(t.MAE, t.MAE_sd)],
        "RMSE": t.RMSE.round(nd)})
    for c in tm.PCT_METRICS[key]:
        out[f"{c} [%]"] = t[c].round(1)
    out["skill"] = [fmt_pm(m, s, 3) for m, s in zip(t.skill, t.skill_sd)]
    return out


def plot_skill(df: pd.DataFrame, bl: pd.DataFrame, out_dir=None):
    """Skill score wzgledem persistence (walidacja, blok A) w funkcji horyzontu; slupki = odch. std. ziaren.
    Dla Air Passengers wariant wybrany na walidacji. Os pionowa przycieta od dolu do -0,5; wartosci nizsze
    (seasonal naive dla Jena Climate) zaznaczone strzalka z podpisem."""
    import matplotlib.pyplot as plt
    s = tm.summarize(df[df.block == "A"], by=("key", "model", "h", "variant"))
    s = s.loc[s.groupby(["key", "model", "h"]).val_MAE.idxmin()]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, key in zip(axes, tm.KEYS):
        hs = list(td.SPECS[key].horizons)
        x = np.arange(len(hs))
        for m in ("seasonal_naive", "linreg"):
            b = bl[(bl.key == key) & (bl.model == m)]
            if m == "linreg":
                b = b[(b.features == "uni") & (b.L == tm.WINDOW_A[key])]
                b = b.loc[b.groupby("h").val_MAE.idxmin()]
            y = b.set_index("h").val_skill.reindex(hs).values
            ax.plot(x, np.maximum(y, -0.5), marker="s", ms=3.5, ls=":", color=tm.MODEL_COLORS[m], label=NAMES[m])
            for xi, yi in zip(x, y):
                if yi < -0.5:
                    ax.annotate(f"{yi:.2f}".replace(".", ","), (xi, -0.5), textcoords="offset points", xytext=(4, 4),
                                fontsize=7.5, color=tm.MODEL_COLORS[m])
        for m in ("Dense", "LSTM", "GRU"):
            g = s[(s.key == key) & (s.model == m)].set_index("h").reindex(hs)
            ax.errorbar(x, g.val_skill, yerr=g.val_skill_sd, marker="o", ms=4, capsize=3, color=tm.MODEL_COLORS[m], label=m)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_ylim(bottom=-0.55)
        ax.set_xticks(x, [f"{h} {td.SPECS[key].step_word}" for h in hs])
        ax.set_title(td.SPECS[key].name, fontsize=10)
    axes[0].set_ylabel("skill score względem persistence")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Blok A: skill score na walidacji (0 = persistence, 1 = prognoza idealna)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    td._save(fig, out_dir, "skill_horyzont_val")
    return fig


def val_forecast_figure(df: pd.DataFrame, key: str, h: int, out=tm.OUT, out_dir=None, n=None, start=None):
    """Fragment zbioru walidacyjnego: wartosci rzeczywiste, LSTM i GRU (ziarno 42, blok A, wariant wybrany
    na walidacji), uczony model bazowy (regresja liniowa z tym samym oknem), persistence i seasonal naive."""
    import matplotlib.pyplot as plt
    P = tm.prepare(key)
    a = df[(df.block == "A") & (df.key == key) & (df.h == h) & (df.seed == 42)]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    tgt0 = None
    for m in ("LSTM", "GRU"):
        g = df[(df.block == "A") & (df.key == key) & (df.h == h) & (df.model == m)]
        v = g.groupby("variant").val_MAE.mean().idxmin()
        rid = a[(a.model == m) & (a.variant == v)].run_id.iloc[0]
        z = np.load(Path(out) / "prognozy" / f"{rid}_val.npz")
        tgt, yhat = z["tgt"], z["yhat"]
        if tgt0 is None:
            n_ = n or (len(tgt) if key == "air" else 24 * 14)
            st = start if start is not None else (0 if key == "air" else len(tgt) // 2)
            tgt0 = tgt[st:st + n_]
            idx = P["index"][tgt0]
            ax.plot(idx, P["y"][tgt0], color="#222222", lw=1.4, label="wartość rzeczywista")
            bp = tm.baseline_predictions(key, h, tgt0)
            ax.plot(idx, bp["persistence"], color=tm.MODEL_COLORS["persistence"], lw=0.9, ls=":", label="persistence")
            ax.plot(idx, bp["seasonal_naive"], color=tm.MODEL_COLORS["seasonal_naive"], lw=0.9, ls=":", label="seasonal naive")
        mk = np.isin(tgt, tgt0)
        ax.plot(P["index"][tgt[mk]], yhat[mk], color=tm.MODEL_COLORS[m], lw=1.1, label=m)
    # uczony model bazowy: regresja liniowa z tym samym oknem (blok A), wariant Air wybrany na walidacji
    bl = tm.load_baselines(out)
    lin = bl[(bl.key == key) & (bl.h == h) & (bl.model == "linreg") & (bl.features == "uni") & (bl.L == tm.WINDOW_A[key])]
    rid = lin.loc[lin.val_MAE.idxmin()].run_id
    z = np.load(Path(out) / "prognozy" / f"{rid}_val.npz")
    mk = np.isin(z["tgt"], tgt0)
    ax.plot(P["index"][z["tgt"][mk]], z["yhat"][mk], color=tm.MODEL_COLORS["linreg"], lw=1.0, ls="--",
            label="regresja liniowa")
    ax.set_ylabel(td.SPECS[key].unit)
    ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}: prognozy na fragmencie zbioru walidacyjnego")
    ax.legend(ncol=6, loc="upper left", fontsize=8)
    fig.tight_layout()
    td._save(fig, out_dir, f"prognoza_val_{key}_h{h}")
    return fig


# ---------------------------------------------------------------------------
# Koszt obliczeniowy
# ---------------------------------------------------------------------------


def cost_table(df: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    a = df[((df.block == "A") | (df.block == "E")) & (df.features == "uni")]
    a = a[a.L == a.key.map(tm.WINDOW_A)]
    t = (a.groupby(["key", "model"]).agg(parametry=("params", "first"), epoki=("epochs", "mean"),
                                         czas_treningu_s=("train_time_s", "mean"), s_na_epoke=("s_per_epoch", "median"),
                                         inferencja_ms_1000=("infer_1000_ms", "median"), rozmiar_kb=("model_kb", "first"),
                                         pamiec_szczyt_mb=("mem_peak_mb", "median"), przyrost_pamieci_mb=("mem_delta_mb", "median"))
         .reset_index())
    t["key"] = pd.Categorical(t.key, tm.KEYS)
    t["model"] = pd.Categorical(t.model, ["Dense", "LSTM", "GRU", "TCN"])
    t = t.sort_values(["key", "model"])
    return _save_tab(t, out_dir, "koszt")


def block_time_table(df: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    t = df.groupby("block").agg(treningi=("run_id", "size"), czas_h=("train_time_s", lambda x: x.sum() / 3600)).reset_index()
    return _save_tab(t, out_dir, "czas_blokow")


# ---------------------------------------------------------------------------
# Bloki B-E
# ---------------------------------------------------------------------------


def block_b_table(df: pd.DataFrame, bl: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    """MAE (srednia ± sd) w funkcji okna dla LSTM i GRU oraz regresji liniowej z tym samym oknem.
    Okno 72 h / 24 mies. pochodzi z bloku A. Air: wariant wybrany na walidacji dla kazdego okna osobno."""
    rows = []
    for key in ("jena", "metro", "air"):
        sel = df[(df.key == key) & (df.features == "uni") & df.block.isin(["A", "B"]) & df.model.isin(["LSTM", "GRU"])
                 & (df.units == 64) & (df.layers == 1) & (df.dropout == 0.2) & (df.lr == 1e-3)]
        for (h, m, L), g in sel.groupby(["h", "model", "L"]):
            if len(g) < len(tm._seeds(key)) * (1 if key != "air" else 1):
                if not (key == "jena" and L == 30):
                    continue
            v = g.groupby("variant").val_MAE.mean().idxmin()
            gg = g[g.variant == v]
            rows.append({"key": key, "h": h, "model": m, "L": L, "variant": v, "n": len(gg),
                         "MAE": gg.val_MAE.mean(), "MAE_sd": gg.val_MAE.std(), "epoki": gg.epochs.mean(),
                         "czas_s": gg.train_time_s.mean()})
        lin = bl[(bl.key == key) & (bl.model == "linreg") & (bl.features == "uni")]
        for (h, L), g in lin.groupby(["h", "L"]):
            r = g.loc[g.val_MAE.idxmin()]
            rows.append({"key": key, "h": h, "model": "linreg", "L": int(L), "variant": r.variant, "n": 1,
                         "MAE": r.val_MAE, "MAE_sd": 0.0, "epoki": np.nan, "czas_s": np.nan})
    t = pd.DataFrame(rows).sort_values(["key", "h", "model", "L"])
    return _save_tab(t, out_dir, "blok_B")


def plot_block_b(t: pd.DataFrame, bl: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
    for i, key in enumerate(("jena", "metro", "air")):
        for j, h in enumerate(td.SPECS[key].horizons):
            ax = axes[i, j]
            for m in ("LSTM", "GRU", "linreg"):
                s = t[(t.key == key) & (t.h == h) & (t.model == m) & ~((t.key == "jena") & (t.L == 30))].sort_values("L")
                ax.errorbar(s.L, s.MAE, yerr=s.MAE_sd, marker="o" if m != "linreg" else "s", ms=4, capsize=3,
                            color=tm.MODEL_COLORS[m], ls="-" if m != "linreg" else ":", label=NAMES[m])
            b = bl[(bl.key == key) & (bl.h == h)]
            for m in ("persistence", "seasonal_naive"):
                ax.axhline(b[b.model == m].val_MAE.iloc[0], color=tm.MODEL_COLORS[m], lw=0.8, ls="--", label=NAMES[m])
            ax.set_xscale("log")
            Ls = sorted(t[(t.key == key) & ~((t.key == "jena") & (t.L == 30))].L.unique())
            ax.set_xticks(Ls, [str(int(x)) for x in Ls])
            ax.minorticks_off()
            ax.set_title(f"{td.SPECS[key].name}, h = {h} {td.SPECS[key].step_word}", fontsize=10)
            if i == 2:
                ax.set_xlabel("długość okna L")
            if j == 0:
                ax.set_ylabel(f"MAE [{td.SPECS[key].unit}]")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Blok B: MAE na walidacji w funkcji długości okna (średnia ± odchylenie standardowe z ziaren)", fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "blok_B_okno")
    return fig


def block_c_table(df: pd.DataFrame, bl: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    rows = []
    for key in ("jena", "household", "metro"):
        for h in td.SPECS[key].horizons:
            r = {"key": key, "h": h}
            for f in ("uni", "multi"):
                g = df[(df.key == key) & (df.h == h) & (df.model == "LSTM") & (df.L == 72) & (df.features == f)
                       & df.block.isin(["A", "C"])]
                r[f"LSTM_{f}"], r[f"LSTM_{f}_sd"] = g.val_MAE.mean(), g.val_MAE.std()
                r[f"LSTM_{f}_all"] = list(g.val_MAE)
                lb = bl[(bl.key == key) & (bl.h == h) & (bl.model == "linreg") & (bl.L == 72) & (bl.features == f)]
                r[f"linreg_{f}"] = lb.val_MAE.iloc[0]
            b = bl[(bl.key == key) & (bl.h == h)]
            r["persistence"] = b[b.model == "persistence"].val_MAE.iloc[0]
            r["seasonal_naive"] = b[b.model == "seasonal_naive"].val_MAE.iloc[0]
            r["zmiana_LSTM_%"] = 100 * (r["LSTM_multi"] / r["LSTM_uni"] - 1)
            r["zmiana_linreg_%"] = 100 * (r["linreg_multi"] / r["linreg_uni"] - 1)
            r["wszystkie_ziarna_lepsze"] = max(r["LSTM_multi_all"]) < min(r["LSTM_uni_all"])
            rows.append(r)
    t = pd.DataFrame(rows).drop(columns=["LSTM_uni_all", "LSTM_multi_all"])
    return _save_tab(t, out_dir, "blok_C")


def plot_block_c(t: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ("jena", "household", "metro")):
        s = t[t.key == key]
        x = np.arange(len(s))
        w = 0.2
        ax.bar(x - 1.5 * w, s.LSTM_uni, w, yerr=s.LSTM_uni_sd, color="#9cc3ee", label="LSTM, tylko target", capsize=2)
        ax.bar(x - 0.5 * w, s.LSTM_multi, w, yerr=s.LSTM_multi_sd, color=tm.MODEL_COLORS["LSTM"], label="LSTM, + zmienne", capsize=2)
        ax.bar(x + 0.5 * w, s.linreg_uni, w, color="#a8dcc6", label="regresja, tylko target")
        ax.bar(x + 1.5 * w, s.linreg_multi, w, color=tm.MODEL_COLORS["linreg"], label="regresja, + zmienne")
        ax.set_xticks(x, [f"h = {h} h" for h in s.h])
        ax.set_title(td.SPECS[key].name, fontsize=10)
        ax.set_ylabel(f"MAE [{td.SPECS[key].unit}]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Blok C: MAE na walidacji bez i ze zmiennymi pomocniczymi (okno 72 h)", fontsize=11)
    fig.tight_layout()
    td._save(fig, out_dir, "blok_C_zmienne")
    return fig


D_LABEL = {(64, 1, 0.2, 1e-3): "bazowa (64, 1 warstwa, dropout 0,2, lr 0,001)", (32, 1, 0.2, 1e-3): "32 jednostki",
           (128, 1, 0.2, 1e-3): "128 jednostek", (64, 2, 0.2, 1e-3): "2 warstwy", (64, 1, 0.0, 1e-3): "bez dropoutu",
           (64, 1, 0.2, 3e-4): "lr 0,0003"}


def block_d_table(df: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    sel = df[(df.key == "jena") & (df.h == 24) & (df.L == 72) & (df.features == "uni") & df.block.isin(["A", "D"])
             & df.model.isin(["LSTM", "GRU"])]
    rows = []
    for (m, u, l, d, lr), g in sel.groupby(["model", "units", "layers", "dropout", "lr"]):
        rows.append({"model": m, "wariant": D_LABEL[(int(u), int(l), round(float(d), 2), float(lr))],
                     "parametry": int(g.params.iloc[0]), "MAE": g.val_MAE.mean(), "MAE_sd": g.val_MAE.std(),
                     "train_MAE": g.train_MAE.mean(), "epoki": g.epochs.mean(), "czas_s": g.train_time_s.mean()})
    t = pd.DataFrame(rows)
    base = t[t.wariant.str.startswith("bazowa")].set_index("model")
    t["różnica_vs_bazowa"] = t.MAE - t.model.map(base.MAE)
    t["sd_bazowej"] = t.model.map(base.MAE_sd)
    t["lepszy_ponad_rozrzut"] = (t["różnica_vs_bazowa"] < 0) & (t["różnica_vs_bazowa"].abs() > t[["MAE_sd", "sd_bazowej"]].max(axis=1))
    return _save_tab(t.sort_values(["model", "parametry"]), out_dir, "blok_D")


def plot_block_d(t: pd.DataFrame, bl: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for m in ("LSTM", "GRU"):
        s = t[t.model == m].sort_values("MAE")
        ax.errorbar(s.parametry, s.MAE, yerr=s.MAE_sd, fmt="o", color=tm.MODEL_COLORS[m], capsize=3, label=m)
        seen = {}
        for _, r in s.iterrows():
            k = seen.get(r.parametry, 0)
            seen[r.parametry] = k + 1
            lab = r.wariant.split(" (")[0]
            ax.annotate(lab, (r.parametry, r.MAE), textcoords="offset points", fontsize=7.5, color=tm.MODEL_COLORS[m],
                        xytext=(7, -3 - 11 * k) if m == "GRU" else (7, 3 + 11 * k))
    b = bl[(bl.key == "jena") & (bl.h == 24)]
    lin = b[(b.model == "linreg") & (b.L == 72) & (b.features == "uni")].val_MAE.iloc[0]
    ax.axhline(lin, color=tm.MODEL_COLORS["linreg"], ls=":", lw=1, label="regresja liniowa (L = 72)")
    ax.axhline(b[b.model == "persistence"].val_MAE.iloc[0], color=tm.MODEL_COLORS["persistence"], ls="--", lw=0.8, label="persistence")
    ax.set_xscale("log")
    ax.set_xlabel("liczba parametrów (skala logarytmiczna)")
    ax.set_ylabel("MAE [°C]")
    ax.set_title("Blok D: Jena Climate, h = 24 h, okno 72 h – MAE na walidacji a rozmiar sieci")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    fig.tight_layout()
    td._save(fig, out_dir, "blok_D_architektura")
    return fig


def block_e_table(df: pd.DataFrame, bl: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    sel = df[(df.key == "jena") & (df.L == 72) & (df.features == "uni") & df.block.isin(["A", "E"])]
    rows = []
    for h in td.SPECS["jena"].horizons:
        b = bl[(bl.key == "jena") & (bl.h == h)]
        for m in ("persistence", "linreg"):
            r = b[(b.model == m)] if m == "persistence" else b[(b.model == m) & (b.L == 72) & (b.features == "uni")]
            rows.append({"h": h, "model": m, "parametry": np.nan, "MAE": r.val_MAE.iloc[0], "MAE_sd": 0.0,
                         "RMSE": r.val_RMSE.iloc[0], "epoki": np.nan, "s_na_epoke": np.nan, "inferencja_ms": np.nan})
        for m in ("Dense", "LSTM", "GRU", "TCN"):
            g = sel[(sel.h == h) & (sel.model == m)]
            rows.append({"h": h, "model": m, "parametry": int(g.params.iloc[0]), "MAE": g.val_MAE.mean(),
                         "MAE_sd": g.val_MAE.std(), "RMSE": g.val_RMSE.mean(), "epoki": g.epochs.mean(),
                         "s_na_epoke": g.s_per_epoch.median(), "inferencja_ms": g.infer_1000_ms.median()})
    return _save_tab(pd.DataFrame(rows), out_dir, "blok_E")


# ---------------------------------------------------------------------------
# Konfiguracje zamrozone
# ---------------------------------------------------------------------------


def frozen_table(out=tm.OUT, out_dir=None, group="najlepsza") -> pd.DataFrame:
    with open(Path(out) / tm.CONFIG_FILE) as f:
        cfg = json.load(f)
    d = pd.DataFrame(cfg["komorki"])
    d = d[d.grupa == group].copy()

    def opis(r):
        if r.model == "linreg":
            parts = [f"L = {r.L}"]
        else:
            parts = [f"blok {r.block}", f"L = {r.L}"]
            if r.model == "TCN":
                parts.append(f"{int(r.units)} filtrów, {int(r.layers)} bloków")
            elif r.model != "Dense" and (r.units != 64 or r.layers != 1 or r.dropout != 0.2 or r.lr != 1e-3):
                parts.append(f"u = {int(r.units)}, warstwy = {int(r.layers)}, dropout = {r.dropout:g}, lr = {r.lr:g}")
        if r.features == "multi":
            parts.append("+ zmienne pomocnicze")
        if r.key == "air":
            parts.append(r.variant)
        return ", ".join(parts)

    d["konfiguracja"] = d.apply(opis, axis=1)
    d["key"] = pd.Categorical(d.key, tm.KEYS)
    d["model"] = pd.Categorical(d.model, [m for m in ORDER if m in d.model.unique()])
    d = d.sort_values(["key", "h", "model"])
    cols = ["key", "h", "model", "konfiguracja", "val_MAE", "val_MAE_sd", "run_ids"]
    out_df = d[cols].copy()
    out_df["ziarna"] = out_df.run_ids.map(len)
    out_df = out_df.drop(columns="run_ids")
    _save_tab(out_df, out_dir, f"zamrozone_{group}")
    return out_df, cfg["utworzono"]
