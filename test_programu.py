"""Projekt 2A - testy poprawnosci programu (Etap 3).

Uruchomienie (w folderze projektu, w tym samym srodowisku co notatnik):
    python test_programu.py            # wszystkie testy; testy wymagajace TensorFlow sa pomijane, gdy go brak
    python test_programu.py --szybko   # bez testow na prawdziwych danych (tylko dane syntetyczne)

Wynik: tabela w konsoli oraz outputs/testy/testy_programu.csv i .md (do Raportu 3).
Kazdy test opisuje: obszar, dane, oczekiwany wynik, wynik uzyskany i status (OK / BLAD / POMINIETY).
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import ts_data as td
import ts_models as tm

RESULTS = []
HAS_TF = importlib.util.find_spec("tensorflow") is not None


def test(obszar, dane, oczekiwane, needs_data=False, needs_tf=False):
    def deco(fn):
        fn.meta = dict(obszar=obszar, dane=dane, oczekiwane=oczekiwane, needs_data=needs_data, needs_tf=needs_tf)
        RESULTS.append(fn)
        return fn
    return deco


# ---------------------------------------------------------------------------
# 1. Okna czasowe i podzial
# ---------------------------------------------------------------------------

@test("okna", "szereg 0, 1, ..., 199; (L, h) = (30, 1), (30, 6), (72, 24), (24, 12)",
      "cel = ostatni krok wejscia + h; cale wejscie przed celem; liczba okien = T - L - h + 1")
def t_windows_no_future():
    a = np.arange(200, dtype=float)
    for L, h in [(30, 1), (30, 6), (72, 24), (24, 12)]:
        X, y, tgt = td.make_windows(a, L, h)
        assert len(y) == 200 - L - h + 1, (L, h, len(y))
        assert np.all(y == X[:, -1, 0] + h)
        assert np.all(X.max(axis=(1, 2)) < y)
        assert np.all(tgt == y)                       # pozycja celu = wartosc w szeregu arytmetycznym
    return "wszystkie 4 pary (L, h) zgodne"


@test("okna", "szereg 0..99 z luka (NaN) w krokach 40-44; L = 10, h = 3",
      "pomijane sa dokladnie okna, ktorych wejscie lub cel obejmuje luke; pozostale bez zmian")
def t_windows_gaps():
    a = np.arange(100, dtype=float)
    a[40:45] = np.nan
    X, y, tgt = td.make_windows(a, 10, 3)
    assert not np.isnan(X).any() and not np.isnan(y).any()
    starts = tgt - 3 - 10 + 1
    expected = [s for s in range(100 - 10 - 3 + 1)
                if not np.isnan(a[s:s + 10]).any() and not np.isnan(a[s + 10 + 3 - 1])]
    assert list(starts) == expected
    return f"{len(y)} okien z {100 - 10 - 3 + 1}; pominieto {100 - 10 - 3 + 1 - len(y)} przecinajacych luke"


@test("okna", "szereg z etykietami train/val/test; L = 24, h = 6",
      "przyklad nalezy do zbioru wg czasu celu; wejscie moze siegac do konca poprzedniego zbioru")
def t_windows_split_by_target():
    T = 300
    idx = pd.date_range("2020-01-01", periods=T, freq="h")
    split = np.where(np.arange(T) < 180, "train", np.where(np.arange(T) < 240, "val", "test"))
    arr = tm.Arrays(np.arange(T, dtype=float)[:, None], np.arange(T, dtype=float), split, idx, lambda p, t: p)
    W = tm.windows(arr, 24, 6)
    for name in td.SPLITS:
        _, _, tgt = W[name]
        assert np.all(split[tgt] == name)
    Xv, _, tv = W["val"]
    assert Xv[0, 0, 0] < 180 <= tv[0]                 # pierwsze okno walidacyjne zaczyna sie w train
    return f"train/val/test: {len(W['train'][1])}/{len(W['val'][1])}/{len(W['test'][1])} okien, przypisanie zgodne"


@test("podzial", "4 zbiory (prawdziwe dane po przygotowaniu)",
      "zbiory rozlaczne i uporzadkowane w czasie; udzialy 60/20/20 (z dokladnoscia do zaokraglenia granic do polnocy); Air: 1949-56 / 1957-58 / 1959-60",
      needs_data=True)
def t_split_real():
    out = []
    for key in tm.KEYS:
        P = tm.prepare(key)
        idx, sp = P["index"], P["split"]
        tr, va, te = (idx[sp == s] for s in td.SPLITS)
        assert tr.max() < va.min() and va.max() < te.min(), key
        if key == "air":
            assert tr.min().year == 1949 and tr.max().year == 1956 and va.min().year == 1957 and te.min().year == 1959
        else:
            shares = [100 * len(x) / len(idx) for x in (tr, va, te)]
            tol = 100 * 24 / len(idx) + 0.05          # granice zaokraglone do polnocy (najwyzej 1 doba)
            assert abs(shares[0] - 60) < tol and abs(shares[1] - 20) < tol, (key, shares)
        out.append(f"{key}: {len(tr)}/{len(va)}/{len(te)}")
    return "; ".join(out)


@test("skalowanie", "szereg syntetyczny; wartosci val i test zmienione o +1000",
      "parametry skalera = srednia i odchylenie z train; zmiana val/test nie zmienia skalera")
def t_scaler_train_only():
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    rng = np.random.default_rng(0)
    h = pd.DataFrame({"y": rng.normal(5, 2, 100), "x": rng.normal(0, 1, 100)}, index=idx)
    split = pd.Series(np.where(np.arange(100) < 60, "train", np.where(np.arange(100) < 80, "val", "test")), index=idx)
    sc1, _ = td.fit_scaler(h, split, ["y", "x"])
    h2 = h.copy()
    h2.loc[split != "train", :] += 1000
    sc2, _ = td.fit_scaler(h2, split, ["y", "x"])
    tr = h[split == "train"]
    assert np.allclose(sc1.mean_, tr.mean().values) and np.allclose(sc1.scale_, tr.std(ddof=0).values)
    assert np.allclose(sc1.mean_, sc2.mean_) and np.allclose(sc1.scale_, sc2.scale_)
    return f"srednia y z train = {sc1.mean_[0]:.4f} w obu wariantach"


@test("skalowanie", "4 zbiory, wariant jednowymiarowy (prawdziwe dane)",
      "srednia standaryzowanego targetu na train = 0 i odchylenie = 1 (z dokladnoscia 1e-6)", needs_data=True)
def t_scaler_real():
    out = []
    for key in ("jena", "household", "metro"):
        arr = tm.build_arrays(key, "uni", "std", 1)
        a = arr.A[arr.split == "train", 0]
        a = a[~np.isnan(a)]
        assert abs(a.mean()) < 1e-6 and abs(a.std() - 1) < 1e-6, key
        out.append(f"{key}: {a.mean():+.1e} / {a.std():.6f}")
    return "; ".join(out)

# ---------------------------------------------------------------------------
# 2. Agregacja, etykiety godzin, braki
# ---------------------------------------------------------------------------

@test("agregacja", "syntetyczne pomiary minutowe Household 00:01-03:00, moc = numer minuty",
      "krok 01:00 = srednia z minut (00:00, 01:00], tzn. 00:01-01:00; etykieta = koniec przedzialu")
def t_hour_labels():
    t = pd.date_range("2020-01-01 00:01", "2020-01-01 03:00", freq="min")
    v = np.arange(1, len(t) + 1, dtype=float)
    df = pd.DataFrame({"time": t, **{c: v for c in td.HOUSEHOLD_COLS}})
    h = td.to_working_freq(df, td.SPECS["household"])
    got = h.loc["2020-01-01 01:00", "Global_active_power"]
    assert abs(got - np.mean(np.arange(1, 61))) < 1e-9, got
    got2 = h.loc["2020-01-01 02:00", "Global_active_power"]
    assert abs(got2 - np.mean(np.arange(61, 121))) < 1e-9
    return f"01:00 -> {got:.1f} (srednia minut 1-60), 02:00 -> {got2:.1f}"


@test("braki", "szereg z lukami 1, 3 i 4 kroki",
      "luki <= 3 uzupelnione ostatnia znana wartoscia; luka 4 krokow pozostaje w calosci; brak uzupelniania wstecz")
def t_fill_gaps():
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    y = pd.Series(np.arange(20, dtype=float), index=idx)
    y.iloc[[2]] = np.nan
    y.iloc[5:8] = np.nan
    y.iloc[11:15] = np.nan
    f, filled = td.fill_short_gaps(y.to_frame("y"), max_gap=3)
    f = f["y"]
    assert f.iloc[2] == 1 and list(f.iloc[5:8]) == [4, 4, 4]
    assert f.iloc[11:15].isna().all()
    return "luka 1 i 3 krokow: ostatnia wartosc; luka 4 krokow: brak"

# ---------------------------------------------------------------------------
# 3. Modele bazowe i metryki
# ---------------------------------------------------------------------------

@test("modele bazowe", "szereg y_t = t (0..199); h = 1, 6, 24; s = 24 i 168",
      "persistence: y(t+h) prognozowane jako y(t); seasonal naive: y(t+h-s(k+1)), k = floor((h-1)/s)")
def t_baselines_manual():
    y = pd.Series(np.arange(200, dtype=float))
    for h in (1, 6, 24):
        p = td.persistence_forecast(y, h)
        assert np.all(p.dropna().values == (y.index[h:] - h))
        for s in (24, 168):
            k = (h - 1) // s
            sn = td.seasonal_naive_forecast(y, h, s)
            assert np.all(sn.dropna().values == (y.index[s * (k + 1):] - s * (k + 1)))
    return "zgodne z recznym obliczeniem dla 3 horyzontow i 2 okresow"


@test("modele bazowe", "prawdziwe dane, wspolny zbior oceny walidacyjnej (Jena, h = 1)",
      "MAE persistence z funkcji = MAE policzone recznie z y(t+h) - y(t)", needs_data=True)
def t_persistence_real():
    tgt = tm.eval_targets("jena", 1)["val"]
    y = tm.prepare("jena")["y"]
    p = tm.baseline_predictions("jena", 1, tgt)["persistence"]
    mae_fun = tm.metrics(y[tgt], p, "jena")["MAE"]
    mae_man = float(np.mean(np.abs(y[tgt] - y[tgt - 1])))
    assert abs(mae_fun - mae_man) < 1e-12
    return f"MAE = {mae_fun:.4f} degC (n = {len(tgt)})"


@test("metryki", "y = [100, 200, 300, 400], y_hat = [110, 190, 330, 400]",
      "MAE 12,5; MSE 275; RMSE 16,583; MAPE 6,25%; sMAPE 6,044%; skill wzgl. MAE 25 = 0,5")
def t_metrics_manual():
    y = np.array([100, 200, 300, 400.])
    yh = np.array([110, 190, 330, 400.])
    m = tm.metrics(y, yh, "air", mae_persistence=25.0)
    smape = 100 * np.mean([10 / 105, 10 / 195, 30 / 315, 0])
    assert abs(m["MAE"] - 12.5) < 1e-12 and abs(m["MSE"] - 275) < 1e-12
    assert abs(m["RMSE"] - np.sqrt(275)) < 1e-12 and abs(m["MAPE"] - 6.25) < 1e-12
    assert abs(m["sMAPE"] - smape) < 1e-12 and abs(m["skill"] - 0.5) < 1e-12
    assert np.isnan(tm.metrics(y, yh, "jena")["MAPE"])       # Jena: bez metryk procentowych
    return f"MAE {m['MAE']}, MSE {m['MSE']}, RMSE {m['RMSE']:.3f}, MAPE {m['MAPE']}, sMAPE {m['sMAPE']:.3f}, skill {m['skill']}"


@test("metryki", "prawdziwe dane, 3 zbiory godzinowe: target standaryzowany i odwrocony",
      "inv(standaryzacja(y)) = y; MAE prognozy idealnej po odwroceniu = 0", needs_data=True)
def t_inverse_std():
    out = []
    for key in ("jena", "household", "metro"):
        arr = tm.build_arrays(key, "uni", "std", 1)
        ok = ~np.isnan(arr.y)
        pos = np.where(ok)[0]
        back = arr.inv(arr.A[pos, 0], pos)
        err = float(np.max(np.abs(back - arr.y[pos])))
        assert err < 1e-8, (key, err)
        out.append(f"{key}: max blad {err:.1e}")
    return "; ".join(out)


@test("metryki", "Air Passengers, warianty log i sdiff (lag 12), wszystkie miesiace z okna",
      "inv(transformacja(y)) = y w oryginalnych jednostkach dla obu wariantow; MAE po odwroceniu = 0",
      needs_data=True)
def t_inverse_air():
    out = []
    for v in ("log", "sdiff"):
        arr = tm.build_arrays("air", "uni", v, 1)
        pos = np.where(~np.isnan(arr.A[:, 0]))[0]
        back = arr.inv(arr.A[pos, 0], pos)
        err = float(np.max(np.abs(back - arr.y[pos])))
        assert err < 1e-6, (v, err)
        W = tm.windows(arr, 24, 12)
        X, ys, tg = W["val"]
        mae = tm.metrics(arr.y[tg], arr.inv(ys, tg), "air")["MAE"]
        assert mae < 1e-3
        out.append(f"{v}: max blad {err:.1e}, MAE prognozy idealnej {mae:.1e}")
    return "; ".join(out)

# ---------------------------------------------------------------------------
# 4. Wspolny zbior oceny i zmienne kalendarzowe Metro
# ---------------------------------------------------------------------------

@test("ocena", "prawdziwe dane; okna 12, 24, 72, 168 h (Air: 6, 12, 24 mies.)",
      "kazdy cel ze wspolnego zbioru oceny ma kompletne okno dla kazdej badanej dlugosci -> wszystkie modele "
      "oceniane na tych samych chwilach", needs_data=True)
def t_common_eval_set():
    out = []
    for key, Ls in (("jena", (12, 24, 72, 168)), ("metro", (72, 168)), ("air", (6, 12, 24))):
        for h in td.SPECS[key].horizons:
            common = tm.eval_targets(key, h)["val"]
            for L in Ls:
                variants = tm.AIR_VARIANTS if key == "air" else ("std",)
                for v in variants:
                    _, _, tgt = tm.windows(tm.build_arrays(key, "uni", v, h), L, h)["val"]
                    assert np.isin(common, tgt).all(), (key, h, L, v)
        out.append(f"{key}: {len(tm.eval_targets(key, 1)['val'])} celow walidacyjnych (h = 1)")
    return "; ".join(out)


@test("kalendarz Metro", "prawdziwe dane Metro, wariant wielowymiarowy, h = 1, 6, 24",
      "cechy kalendarzowe w wierszu t dotycza chwili t + h (godzina, dzien tygodnia, swieto); pogoda tylko z t",
      needs_data=True)
def t_metro_calendar():
    P = tm.prepare("metro")
    idx, hol = P["index"], P["h_f"]["holiday"].to_numpy()
    out = []
    for h in (1, 6, 24):
        arr = tm.build_arrays("metro", "multi", "std", h)
        c = arr.cols
        A = pd.DataFrame(arr.A, columns=c, index=idx)
        t = idx + pd.to_timedelta(h, unit="h")
        assert np.allclose(A["cel_godz_sin"], np.sin(2 * np.pi * t.hour / 24))
        assert np.allclose(A["cel_dzien_cos"], np.cos(2 * np.pi * t.day_of_week / 7))
        exp_hol = pd.Series(hol, index=idx).shift(-h)
        m = exp_hol.notna()
        assert np.array_equal(A["cel_swieto"][m].values, exp_hol[m].values)
        # zmienne obserwowane (pogoda) nie sa przesuniete w przod: kolumna temperatury = standaryzowana temp(t)
        temp = P["h_f"]["temp (degC)"]
        assert np.corrcoef(A["temp (degC)"].dropna(), temp[A["temp (degC)"].notna()])[0, 1] > 0.999999
        out.append(f"h = {h}: OK")
    return "; ".join(out)

# ---------------------------------------------------------------------------
# 5. Wznawianie eksperymentow i ochrona zbioru testowego
# ---------------------------------------------------------------------------

@test("wznawianie", "katalog tymczasowy; 3 przebiegi: zakonczony, przerwany (status 'started'), z inna konfiguracja",
      "za gotowy uznany tylko przebieg z wynikiem, plikami i ta sama konfiguracja (config_hash)")
def t_resume_logic():
    tmp = Path(tempfile.mkdtemp())
    try:
        r_done = tm.Run("A", "air", "LSTM", 1, 24, variant="log", seed=42)
        r_int = tm.Run("A", "air", "LSTM", 1, 24, variant="log", seed=43)
        r_old = tm.Run("A", "air", "GRU", 1, 24, variant="log", seed=42)
        for sub in ("modele", "historie", "prognozy"):
            (tmp / sub).mkdir()
        for r in (r_done, r_old):
            (tmp / "modele" / f"{r.run_id}.keras").write_text("x")
            (tmp / "historie" / f"{r.run_id}.json").write_text("{}")
            (tmp / "prognozy" / f"{r.run_id}_val.npz").write_text("x")
        rows = [{"run_id": r_done.run_id, "config_hash": tm.config_hash(r_done)},
                {"run_id": r_old.run_id, "config_hash": "stara_konf"}]
        pd.DataFrame(rows).to_csv(tmp / "wyniki.csv", index=False)
        tm.mark_status(tmp, r_int, "started")             # przerwany: brak wyniku i plikow
        done = tm.completed_runs(tmp)
        assert tm.is_done(r_done, done) and not tm.is_done(r_int, done) and not tm.is_done(r_old, done)
        (tmp / "modele" / f"{r_done.run_id}.keras").unlink()  # brak pliku modelu -> nie jest gotowy
        assert not tm.is_done(r_done, tm.completed_runs(tmp))
        return "zakonczony: gotowy; przerwany: do powtorzenia; inna konfiguracja: do powtorzenia; brak pliku: do powtorzenia"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test("zbior testowy", "funkcja oceny testowej wywolana bez potwierdzenia",
      "odmowa (RuntimeError); test uzywany dopiero w Raporcie 4 po jawnym potwierdzeniu")
def t_test_guard():
    try:
        tm.evaluate_test("nie_istnieje.json", Path(tempfile.mkdtemp()))
    except RuntimeError:
        return "odmowa bez confirm=True"
    raise AssertionError("ocena testowa wykonana bez potwierdzenia")


@test("zbior testowy", "krotki trening LSTM na Air Passengers (2 epoki) w katalogu tymczasowym",
      "trening i prognozy dotycza tylko train i val: brak plikow prognoz testowych, n_val = 24",
      needs_data=True, needs_tf=True)
def t_no_test_in_training():
    tmp = Path(tempfile.mkdtemp())
    try:
        row = tm.train_one(tm.Run("A", "air", "LSTM", 1, 24, variant="log", seed=42), tmp, quick=True)
        files = [p.name for p in tmp.rglob("*")]
        assert not any("test" in f for f in files), files
        assert row["n_val"] == 24
        return f"pliki: {sorted(p.parent.name for p in tmp.rglob('*.*'))}; n_val = {row['n_val']}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test("powtarzalnosc", "dwa treningi LSTM na Air Passengers z tym samym ziarnem (2 epoki)",
      "identyczna historia straty (roznica < 1e-5) przy tym samym ziarnie", needs_data=True, needs_tf=True)
def t_seed_reproducible():
    tmp1, tmp2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        r = tm.Run("A", "air", "LSTM", 1, 24, variant="log", seed=42)
        a = tm.train_one(r, tmp1, quick=True)
        b = tm.train_one(r, tmp2, quick=True)
        ha, hb = tm.history(r.run_id, tmp1), tm.history(r.run_id, tmp2)
        d = max(abs(x - y) for x, y in zip(ha["loss"] + ha["val_loss"], hb["loss"] + hb["val_loss"]))
        assert d < 1e-5, d
        return f"maks. roznica strat {d:.1e}; val MAE {a['val_MAE']:.3f} vs {b['val_MAE']:.3f}"
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


@test("architektura", "modele Dense, LSTM, GRU dla okna 72 i 1 cechy",
      "liczba parametrow: Dense 6785, LSTM 16 961, GRU 12 929 (wzory z Raportu 1)", needs_tf=True)
def t_param_counts():
    import keras
    got = {}
    for kind in ("Dense", "LSTM", "GRU"):
        got[kind] = tm.build_model(kind, 72, 1).count_params()
        keras.backend.clear_session()
    assert got == {"Dense": 6785, "LSTM": 16961, "GRU": 12929}, got
    return ", ".join(f"{k} {v}" for k, v in got.items())


@test("architektura TCN", "TCN (5 blokow, 25 filtrow, jadro 3) dla okna 72; losowe wejscie, zmiana jednego kroku",
      "17 276 parametrow; zmiana pierwszego kroku okna zmienia prognoze (pole widzenia >= 72); "
      "model zapisany do .keras i wczytany daje te same prognozy", needs_tf=True)
def t_tcn():
    import keras
    m = tm.build_model("TCN", 72, 1, units=tm.TCN_FILTERS, layers=tm.TCN_BLOCKS)
    n = m.count_params()
    assert n == 17276, n
    x = np.random.default_rng(0).normal(size=(4, 72, 1)).astype("float32")
    x2 = x.copy()
    x2[:, 0, 0] += 1.0                                 # najstarszy krok okna
    d_first = float(np.max(np.abs(m.predict(x, verbose=0) - m.predict(x2, verbose=0))))
    assert d_first > 0, "pierwszy krok okna nie wplywa na prognoze"
    tmp = Path(tempfile.mkdtemp())
    try:
        m.save(tmp / "tcn.keras")
        m2 = keras.models.load_model(tmp / "tcn.keras")
        d_load = float(np.max(np.abs(m.predict(x, verbose=0) - m2.predict(x, verbose=0))))
        assert d_load < 1e-6, d_load
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        keras.backend.clear_session()
    rf = 1 + 2 * (tm.TCN_KERNEL - 1) * (2 ** tm.TCN_BLOCKS - 1)
    return f"parametry {n}; pole widzenia {rf} krokow; wplyw 1. kroku {d_first:.1e}; roznica po wczytaniu {d_load:.1e}"



@test("zamrozenie konfiguracji", "katalog tymczasowy z istniejacym plikiem final_configs.json",
      "ponowne zamrozenie bez jawnej zgody jest odrzucane (FileExistsError), a plik pozostaje bez zmian")
def t_freeze_guard():
    tmp = Path(tempfile.mkdtemp())
    try:
        f = tmp / tm.CONFIG_FILE
        f.write_text('{"utworzono": "2026-11-01 19:53:08", "komorki": []}')
        before = f.read_bytes()
        try:
            tm.freeze_configs(tmp)
            raise AssertionError("plik zostal nadpisany")
        except FileExistsError:
            pass
        assert f.read_bytes() == before
        return "odmowa nadpisania (FileExistsError); plik bez zmian"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--szybko", action="store_true", help="bez testow na prawdziwych danych")
    ap.add_argument("--dane", default="data")
    args = ap.parse_args()
    tm.prepare.__defaults__ = (args.dane,)
    rows = []
    for i, fn in enumerate(RESULTS, 1):
        m = fn.meta
        t0 = time.time()
        if m["needs_tf"] and not HAS_TF:
            status, wynik = "POMINIĘTY", "brak TensorFlow (uruchom ponownie po instalacji)"
        elif m["needs_data"] and args.szybko:
            status, wynik = "POMINIĘTY", "tryb --szybko (bez prawdziwych danych)"
        else:
            try:
                wynik, status = fn(), "OK"
            except Exception as e:
                status = "BŁĄD"
                wynik = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        rows.append({"nr": i, "test": fn.__name__, "obszar": m["obszar"], "dane": m["dane"],
                     "oczekiwany wynik": m["oczekiwane"], "wynik": wynik, "status": status,
                     "czas [s]": round(time.time() - t0, 2)})
        print(f"[{status:9s}] {i:2d}. {m['obszar']}: {fn.__name__} ({rows[-1]['czas [s]']} s)", flush=True)
    df = pd.DataFrame(rows)
    out = Path("outputs/testy")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "testy_programu.csv", index=False)
    with open(out / "testy_programu.md", "w", encoding="utf-8") as f:     # tabela markdown bez dodatkowych pakietow
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |\n")
    print(f"\nOK: {(df.status == 'OK').sum()}, BŁĄD: {(df.status == 'BŁĄD').sum()}, "
          f"POMINIĘTE: {(df.status == 'POMINIĘTY').sum()} -> {out / 'testy_programu.csv'}")
    sys.exit(1 if (df.status == "BŁĄD").any() else 0)


if __name__ == "__main__":
    main()
