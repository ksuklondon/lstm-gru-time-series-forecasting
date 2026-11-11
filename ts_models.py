"""Projekt 2A - modele, trening, metryki i rejestr eksperymentow (bloki A-E).

Modul korzysta z ts_data.py (przygotowanie danych) i jest uzywany zarowno przez notatnik,
jak i przez skrypt run_experiments.py (dlugie treningi uruchamiane z terminala, z wznawianiem).

Zasady (zgodne z Raportem 1):
- podzial chronologiczny 60/20/20 (Air Passengers: pelne lata), skaler dopasowany tylko na train,
- przyklad nalezy do zbioru wg czasu CELU, okna z luka sa pomijane,
- wszystkie modele w danej kombinacji (zbior, horyzont) oceniamy na tych samych chwilach prognozy
  (wspolny zbior oceny), metryki liczymy w oryginalnych jednostkach targetu,
- test sluzy wylacznie do oceny konfiguracji zamrozonych na walidacji (final_configs.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import ts_data as td

# ---------------------------------------------------------------------------
# Ustawienia projektu
# ---------------------------------------------------------------------------

KEYS = ("jena", "household", "metro", "air")
SEEDS = (42, 43, 44)
AIR_SEEDS = tuple(range(42, 52))
SEASON = {"jena": 24, "household": 168, "metro": 168, "air": 12}   # okres s wybrany na train (R1, 13.6)
WINDOW_A = {"jena": 72, "household": 72, "metro": 72, "air": 24}
L_EVAL = {"jena": 168, "household": 168, "metro": 168, "air": 24}  # najdluzsze okno -> wspolny zbior oceny
AIR_VARIANTS = ("log", "sdiff")      # wariant podstawowy: logarytm; kontrolny: roznice sezonowe (lag 12)
NETS = ("Dense", "LSTM", "GRU")
TCN_KERNEL = 3                 # TCN: jadro splotu; 5 blokow -> pole widzenia 1 + 2*(3-1)*(2^5-1) = 125 krokow >= 72
TCN_BLOCKS = 5
TCN_FILTERS = 25               # 17 276 parametrow, najblizej LSTM(64) = 16 961
PCT_METRICS = {"jena": (), "household": ("MAPE", "sMAPE"), "metro": ("sMAPE",), "air": ("MAPE", "sMAPE")}

TRAIN = dict(epochs=100, patience=10, batch=256, batch_air=16, lr=1e-3, units=64, layers=1, dropout=0.2)
OUT = Path("outputs/eksperymenty")
CODE_VERSION = "R3-1"        # zmieniamy przy kazdej zmianie kodu wplywajacej na wyniki treningu


def config_hash(run) -> str:
    """Skrot konfiguracji przebiegu: parametry Run + ustawienia uczenia + wersja kodu. Wynik zapisany przy
    innej konfiguracji (np. po zmianie TRAIN) nie jest uznawany za gotowy i trening zostanie powtorzony."""
    payload = json.dumps({"run": asdict(run), "train": TRAIN, "code": CODE_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def sn_period(key: str, h: int) -> int:
    """Okres seasonal naive. Gdy h jest wielokrotnoscia s, SN pokrywa sie z persistence; dla Jena (s = 24)
    przy h = 24 uzywamy wtedy s = 168. Dla Air Passengers przy h = 12 SN = persistence (tylko raportujemy)."""
    s = SEASON[key]
    if key != "air" and h % s == 0:
        return 168
    return s

# ---------------------------------------------------------------------------
# Dane: przygotowanie (z cache), transformacje, okna
# ---------------------------------------------------------------------------

_PREP: dict[str, dict] = {}


def prepare(key: str, data_dir="data") -> dict:
    """Szereg roboczy po kontroli jakosci, agregacji i uzupelnieniu krotkich luk + etykiety podzialu.
    Wynik jest zapisywany w data/prepared_<key>.pkl, wiec kolejne uruchomienia nie parsuja danych surowych."""
    if key in _PREP:
        return _PREP[key]
    data_dir = Path(data_dir)
    cache = data_dir / f"prepared_{key}.pkl"
    spec = td.SPECS[key]
    if cache.exists():
        with open(cache, "rb") as f:
            h_f = pickle.load(f)
    else:
        td.download(key, data_dir)
        raw = td.load_raw(key, data_dir)
        clean, _ = td.apply_anomaly_rules(raw, key)
        h = td.to_working_freq(clean, spec)
        h_f, _ = td.fill_short_gaps(h, max_gap=3)
        with open(cache, "wb") as f:
            pickle.dump(h_f, f)
    split = td.chronological_split(h_f.index, spec)
    _PREP[key] = dict(spec=spec, h_f=h_f, split=split.to_numpy(), index=h_f.index,
                      y=h_f[spec.target].astype(float).to_numpy())
    return _PREP[key]


@dataclass
class Arrays:
    A: np.ndarray            # (T, d) wejscie po transformacji i skalowaniu; kolumna 0 = target
    y: np.ndarray            # (T,) target w oryginalnych jednostkach
    split: np.ndarray        # (T,) 'train' / 'val' / 'test'
    index: pd.DatetimeIndex
    inv: callable            # inv(pred_scaled, tgt_pos) -> prognoza w oryginalnych jednostkach
    cols: list = field(default_factory=list)


def _calendar_lead(index: pd.DatetimeIndex, h: int, holiday: np.ndarray) -> pd.DataFrame:
    """Zmienne kalendarzowe CHWILI PROGNOZOWANEJ t+h (znane z wyprzedzeniem): pora dnia, dzien tygodnia, swieto."""
    t = index + pd.to_timedelta(h, unit="h")
    hol = pd.Series(holiday, index=index).shift(-h).to_numpy()
    return pd.DataFrame({
        "cel_godz_sin": np.sin(2 * np.pi * t.hour / 24), "cel_godz_cos": np.cos(2 * np.pi * t.hour / 24),
        "cel_dzien_sin": np.sin(2 * np.pi * t.day_of_week / 7), "cel_dzien_cos": np.cos(2 * np.pi * t.day_of_week / 7),
        "cel_swieto": hol}, index=index)


def build_arrays(key: str, features: str = "uni", variant: str = "std", h: int = 1) -> Arrays:
    """Macierz wejscia po transformacji. features: 'uni' (tylko target) albo 'multi' (target + zmienne pomocnicze).
    variant: 'std' (standaryzacja, zbiory godzinowe), 'log' albo 'sdiff' (Air Passengers)."""
    P = prepare(key)
    spec, h_f, split, y = P["spec"], P["h_f"], P["split"], P["y"]
    train = split == "train"
    if key == "air":
        ly = np.log(y)
        if variant == "log":
            a = ly.copy()
        elif variant == "sdiff":
            a = ly - pd.Series(ly).shift(12).to_numpy()
        else:
            raise ValueError("Air Passengers: wariant 'log' albo 'sdiff'")
        mu, sd = np.nanmean(a[train]), np.nanstd(a[train])
        A = ((a - mu) / sd)[:, None]

        def inv(p, tgt, mu=mu, sd=sd, variant=variant):
            v = np.asarray(p) * sd + mu
            if variant == "sdiff":
                v = v + ly[np.asarray(tgt) - 12]
            return np.exp(v)
        return Arrays(A, y, split, P["index"], inv, [f"{variant}({spec.target})"])

    cols = [spec.target] if features == "uni" else td.feature_columns(h_f, spec)
    scaler, cont = td.fit_scaler(h_f, pd.Series(split, index=h_f.index), cols)
    Hs = td.apply_scaler(h_f, cols, scaler, cont)
    if features == "multi" and key == "metro":
        Hs = pd.concat([Hs, _calendar_lead(h_f.index, h, h_f["holiday"].to_numpy())], axis=1)
    i = cont.index(spec.target)
    mu, sd = float(scaler.mean_[i]), float(scaler.scale_[i])

    def inv(p, tgt, mu=mu, sd=sd):
        return np.asarray(p) * sd + mu
    return Arrays(Hs.to_numpy(dtype=float), y, split, P["index"], inv, list(Hs.columns))


def windows(arr: Arrays, L: int, h: int) -> dict:
    """Okna wg td.make_windows, przypisane do zbiorow wg czasu celu."""
    X, ys, tgt = td.make_windows(arr.A, L, h, target_col=0)
    lab = arr.split[tgt]
    return {s: (X[lab == s].astype("float32"), ys[lab == s].astype("float32"), tgt[lab == s]) for s in td.SPLITS}


_EVAL: dict = {}


def eval_targets(key: str, h: int) -> dict:
    """Wspolny zbior oceny dla (zbior, horyzont): pozycje celu, dla ktorych okno jednowymiarowe o dlugosci
    L_EVAL jest kompletne i istnieje prognoza seasonal naive. Kazde krotsze okno jest wtedy tez kompletne,
    wiec wszystkie modele (rozne okna, bloki A-D) sa oceniane na tych samych chwilach prognozy."""
    k = (key, h)
    if k in _EVAL:
        return _EVAL[k]
    P = prepare(key)
    if key == "air":
        arr = build_arrays(key, variant="sdiff")       # sdiff ma najkrotsza historie (pierwsze 12 mies. = NaN)
    else:
        arr = build_arrays(key, "uni", "std")
    _, _, tgt = td.make_windows(arr.A, L_EVAL[key], h)
    s = sn_period(key, h)
    back = tgt - s * ((h - 1) // s + 1)
    ok = (back >= 0)
    ok[ok] &= ~np.isnan(P["y"][back[ok]])
    tgt = tgt[ok]
    lab = P["split"][tgt]
    _EVAL[k] = {sp: tgt[lab == sp] for sp in td.SPLITS}
    return _EVAL[k]

# ---------------------------------------------------------------------------
# Metryki (oryginalne jednostki)
# ---------------------------------------------------------------------------


def metrics(y, yhat, key: str, mae_persistence: float | None = None) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = y - yhat
    out = {"MAE": float(np.mean(np.abs(e))), "MSE": float(np.mean(e ** 2))}
    out["RMSE"] = float(np.sqrt(out["MSE"]))
    out["MAPE"] = float(100 * np.mean(np.abs(e / y))) if "MAPE" in PCT_METRICS[key] else np.nan
    if "sMAPE" in PCT_METRICS[key]:
        den = (np.abs(y) + np.abs(yhat)) / 2
        out["sMAPE"] = float(100 * np.mean(np.where(den > 0, np.abs(e) / np.where(den > 0, den, 1), 0.0)))
    else:
        out["sMAPE"] = np.nan
    out["skill"] = float(1 - out["MAE"] / mae_persistence) if mae_persistence else np.nan
    return out


def baseline_predictions(key: str, h: int, tgt: np.ndarray) -> dict:
    """Persistence: y(t); seasonal naive: y(t+h - s(k+1)). tgt = pozycje celu t+h."""
    y = prepare(key)["y"]
    s = sn_period(key, h)
    return {"persistence": y[tgt - h], "seasonal_naive": y[tgt - s * ((h - 1) // s + 1)]}

# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------


def build_model(kind: str, L: int, d: int, units: int = 64, layers: int = 1, dropout: float = 0.2,
                lr: float = 1e-3):
    """Dense: Flatten -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1).
    LSTM/GRU: [warstwa rekurencyjna(units)] x layers -> Dropout -> Dense(1) liniowe.
    TCN: `layers` blokow resztkowych ze splotami przyczynowymi (units filtrow) -> ostatni krok -> Dense(1)."""
    import keras
    from keras import layers as kl
    inp = keras.Input(shape=(L, d))
    if kind == "Dense":
        x = kl.Flatten()(inp)
        x = kl.Dense(64, activation="relu")(x)
        x = kl.Dense(32, activation="relu")(x)
    elif kind in ("LSTM", "GRU"):
        Cell = kl.LSTM if kind == "LSTM" else kl.GRU
        x = inp
        for i in range(layers):
            x = Cell(units, return_sequences=i < layers - 1)(x)
        x = kl.Dropout(dropout)(x)
    elif kind == "TCN":
        # TCN (Bai i in., 2018): `layers` blokow resztkowych z dylatacja 1, 2, 4, ...; w bloku dwa sploty
        # przyczynowe (jadro TCN_KERNEL, `units` filtrow, ReLU, Dropout) i polaczenie resztkowe (splot 1x1,
        # gdy zmienia sie liczba kanalow). Zamiast normalizacji wag z oryginalu brak normalizacji (male sieci).
        # Prognoza z ostatniego kroku sekwencji (Cropping1D zamiast Lambda, aby model zapisywal sie bez kodu).
        x = inp
        for b in range(layers):
            dil = 2 ** b
            y = x
            for _ in range(2):
                y = kl.Conv1D(units, TCN_KERNEL, padding="causal", dilation_rate=dil, activation="relu")(y)
                y = kl.Dropout(dropout)(y)
            skip = x if x.shape[-1] == units else kl.Conv1D(units, 1)(x)
            x = kl.Activation("relu")(kl.Add()([y, skip]))
        x = kl.Flatten()(kl.Cropping1D((L - 1, 0))(x))
    else:
        raise ValueError(kind)
    out = kl.Dense(1)(x)
    model = keras.Model(inp, out, name=f"{kind}_{units}x{layers}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


class PeakMemory:
    """Szczyt pamieci RSS procesu (MB) w trakcie bloku with; probkowanie co 50 ms w osobnym watku."""

    def __init__(self, every=0.05):
        import psutil
        self.proc, self.every = psutil.Process(os.getpid()), every

    def __enter__(self):
        self.start = self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.every)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.peak_mb, self.delta_mb = self.peak / 2**20, (self.peak - self.start) / 2**20


def inference_time_ms(model, X: np.ndarray, n: int = 1000, reps: int = 5) -> float:
    """Czas prognozy dla n okien: mediana z reps powtorzen, po jednym wywolaniu rozgrzewajacym."""
    Xn = np.resize(X, (n,) + X.shape[1:]).astype("float32")
    model.predict(Xn, batch_size=n, verbose=0)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict(Xn, batch_size=n, verbose=0)
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))

# ---------------------------------------------------------------------------
# Rejestr eksperymentow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    block: str
    key: str
    model: str
    h: int
    L: int
    features: str = "uni"
    variant: str = "std"
    units: int = 64
    layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    seed: int = 42

    @property
    def run_id(self) -> str:
        arch = "" if self.model == "Dense" else f"_u{self.units}_l{self.layers}_d{self.dropout:g}"
        return (f"{self.key}_{self.model}_h{self.h}_L{self.L}_{self.features}_{self.variant}"
                f"{arch}_lr{self.lr:g}_s{self.seed}")


def _seeds(key):
    return AIR_SEEDS if key == "air" else SEEDS


def _variants(key):
    return AIR_VARIANTS if key == "air" else ("std",)


def registry(block: str) -> list[Run]:
    """Lista treningow sieci w bloku. Air Passengers: oba warianty przeksztalcenia (log i sdiff),
    wybor wariantu na walidacji; treningi Air trwaja sekundy, wiec nie zmieniaja budzetu czasu."""
    runs = []
    if block == "A":
        for key in KEYS:
            for v in _variants(key):
                for m in NETS:
                    for h in td.SPECS[key].horizons:
                        for s in _seeds(key):
                            runs.append(Run("A", key, m, h, WINDOW_A[key], "uni", v, seed=s))
    elif block == "B":
        for key, Ls in (("jena", (12, 24, 168)), ("metro", (168,)), ("air", (6, 12))):
            for v in _variants(key):
                for L in Ls:
                    for m in ("LSTM", "GRU"):
                        for h in td.SPECS[key].horizons:
                            for s in _seeds(key):
                                runs.append(Run("B", key, m, h, L, "uni", v, seed=s))
        runs.append(Run("B", "jena", "LSTM", 1, 30, seed=42))            # przebieg kontrolny, okno 30 h
    elif block == "C":
        for key in ("jena", "household", "metro"):
            for h in td.SPECS[key].horizons:
                for s in SEEDS:
                    runs.append(Run("C", key, "LSTM", h, 72, "multi", seed=s))
    elif block == "D":
        variants = [dict(units=32), dict(units=128), dict(layers=2), dict(dropout=0.0), dict(lr=3e-4)]
        for kw in variants:
            for m in ("LSTM", "GRU"):
                for s in SEEDS:
                    runs.append(Run("D", "jena", m, 24, 72, **kw, seed=s))
    elif block == "E":
        # rozszerzenie: TCN o liczbie parametrow zblizonej do LSTM, warunki jak w bloku A dla Jena Climate
        for h in td.SPECS["jena"].horizons:
            for s in SEEDS:
                runs.append(Run("E", "jena", "TCN", h, 72, units=TCN_FILTERS, layers=TCN_BLOCKS, seed=s))
    else:
        raise ValueError(block)
    return runs


def baseline_cells(block: str) -> list[tuple]:
    """(zbior, horyzont, okno, cechy, wariant) dla modeli bazowych liczonych raz (bez ziaren)."""
    cells = set()
    for r in registry(block):
        cells.add((r.key, r.h, r.L, r.features, r.variant))
    return sorted(cells)

# ---------------------------------------------------------------------------
# Trening jednej konfiguracji
# ---------------------------------------------------------------------------


def _atomic_json(obj, path: Path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Status przebiegow (bezpieczne wznawianie)
# ---------------------------------------------------------------------------

STATUS_COLS = ["run_id", "config_hash", "status", "time", "message"]


def mark_status(out: Path, run, status: str, message: str = ""):
    """Dziennik statusow (status.csv): started / done / failed / interrupted. Kazda zmiana to nowy wiersz,
    wiec historia przebiegow jest zachowana; obowiazuje ostatni wpis dla danego run_id."""
    path = Path(out) / "status.csv"
    row = pd.DataFrame([{"run_id": run.run_id, "config_hash": config_hash(run), "status": status,
                         "time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": str(message)[:300]}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_runs(out: Path) -> set:
    """Przebiegi uznane za zakonczone: wiersz w wyniki.csv z ta sama konfiguracja (config_hash) ORAZ istniejacy
    plik modelu, historia i prognozy walidacyjne. Przebieg przerwany (status 'started' bez wyniku) lub
    zakonczony bledem nie jest uznawany za gotowy."""
    out = Path(out)
    res = out / "wyniki.csv"
    if not res.exists():
        return set()
    df = pd.read_csv(res)
    if "config_hash" not in df.columns:
        return set()
    done = set()
    for rid, h in zip(df.run_id, df.config_hash):
        files = [out / "modele" / f"{rid}.keras", out / "historie" / f"{rid}.json", out / "prognozy" / f"{rid}_val.npz"]
        if all(f.exists() for f in files):
            done.add((rid, h))
    return done


def is_done(run, done: set) -> bool:
    return (run.run_id, config_hash(run)) in done


def _paths(out: Path):
    p = {k: out / k for k in ("historie", "modele", "prognozy")}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def _common_metrics(arr, run_tgt, pred_scaled, key, h, split_name):
    """Metryki na wspolnym zbiorze oceny (oryginalne jednostki) + skill wzgledem persistence."""
    common = eval_targets(key, h)[split_name]
    m = np.isin(run_tgt, common)
    tgt = run_tgt[m]
    yhat = arr.inv(pred_scaled[m], tgt)
    ytrue = arr.y[tgt]
    pers = baseline_predictions(key, h, tgt)["persistence"]
    mae_p = float(np.mean(np.abs(ytrue - pers)))
    return metrics(ytrue, yhat, key, mae_p), tgt, yhat


def train_one(run: Run, out: Path = OUT, quick: bool = False, verbose: int = 0) -> dict:
    """Trening jednej konfiguracji z wczesnym zatrzymaniem; zapisuje historie, model i prognozy walidacyjne.
    quick=True: krotki przebieg testowy (2 epoki, czesc danych) - tylko do sprawdzenia kodu."""
    import keras
    import tensorflow as tf
    paths = _paths(out)
    arr = build_arrays(run.key, run.features, run.variant, run.h)
    W = windows(arr, run.L, run.h)
    Xtr, ytr, ttr = W["train"]
    Xva, yva, tva = W["val"]
    if quick:
        Xtr, ytr, ttr = Xtr[-2000:], ytr[-2000:], ttr[-2000:]
    keras.utils.set_random_seed(run.seed)
    model = build_model(run.model, run.L, Xtr.shape[2], run.units, run.layers, run.dropout, run.lr)
    batch = TRAIN["batch_air"] if run.key == "air" else TRAIN["batch"]
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=TRAIN["patience"], restore_best_weights=True)
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=2 if quick else TRAIN["epochs"],
                         batch_size=batch, shuffle=True, callbacks=[es], verbose=verbose)
        train_time = time.perf_counter() - t0
    H = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best = int(np.argmin(H["val_loss"]))
    _atomic_json(H, paths["historie"] / f"{run.run_id}.json")
    mpath = paths["modele"] / f"{run.run_id}.keras"
    tmp = mpath.with_name(mpath.stem + ".tmp.keras")
    model.save(tmp)
    os.replace(tmp, mpath)                    # plik modelu pojawia sie dopiero w calosci
    infer = inference_time_ms(model, Xva)
    p_va = model.predict(Xva, batch_size=4096, verbose=0).ravel()
    p_tr = model.predict(Xtr, batch_size=4096, verbose=0).ravel()
    mv, tgt_v, yhat_v = _common_metrics(arr, tva, p_va, run.key, run.h, "val")
    np.savez_compressed(paths["prognozy"] / f"{run.run_id}_val.npz", tgt=tgt_v, yhat=yhat_v.astype("float32"))
    mae_tr = float(np.mean(np.abs(arr.y[ttr] - arr.inv(p_tr, ttr))))
    row = {"run_id": run.run_id, **asdict(run), "config_hash": config_hash(run), "code_version": CODE_VERSION,
           "params": int(model.count_params()),
           "epochs": len(H["loss"]), "best_epoch": best + 1,
           "train_loss_best": H["loss"][best], "val_loss_best": H["val_loss"][best],
           "train_time_s": round(train_time, 2), "s_per_epoch": round(train_time / len(H["loss"]), 3),
           "infer_1000_ms": round(infer, 2), "model_kb": round(mpath.stat().st_size / 1024, 1),
           "mem_peak_mb": round(mem.peak_mb, 1), "mem_delta_mb": round(mem.delta_mb, 1),
           "n_train": len(ytr), "n_val": len(tgt_v), "train_MAE": mae_tr,
           **{f"val_{k}": v for k, v in mv.items()},
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": quick}
    keras.backend.clear_session()
    del model
    return row


def baselines_one(key, h, L, features, variant, out: Path = OUT) -> list[dict]:
    """Persistence, seasonal naive (bez uczenia) i regresja liniowa (MNK na spłaszczonym oknie) na walidacji."""
    from sklearn.linear_model import LinearRegression
    paths = _paths(out)
    arr = build_arrays(key, features, variant, h)
    W = windows(arr, L, h)
    Xtr, ytr, _ = W["train"]
    Xva, _, tva = W["val"]
    t0 = time.perf_counter()
    lr = LinearRegression().fit(Xtr.reshape(len(Xtr), -1), ytr)
    fit_t = time.perf_counter() - t0
    p_va = lr.predict(Xva.reshape(len(Xva), -1))
    mv, tgt, yhat = _common_metrics(arr, tva, p_va, key, h, "val")
    rows = []
    base = dict(key=key, h=h, L=L, features=features, variant=variant, block="baseline")
    rid = f"{key}_linreg_h{h}_L{L}_{features}_{variant}"
    np.savez_compressed(paths["prognozy"] / f"{rid}_val.npz", tgt=tgt, yhat=yhat.astype("float32"))
    with open(paths["modele"] / f"{rid}.pkl", "wb") as f:
        pickle.dump(lr, f)
    rows.append({"run_id": rid, **base, "model": "linreg", "params": int(lr.coef_.size + 1),
                 "train_time_s": round(fit_t, 3), "n_train": len(ytr), "n_val": len(tgt),
                 **{f"val_{k}": v for k, v in mv.items()}})
    common = eval_targets(key, h)["val"]
    bp = baseline_predictions(key, h, common)
    y = prepare(key)["y"][common]
    mae_p = float(np.mean(np.abs(y - bp["persistence"])))
    for name, pred in bp.items():
        rid = f"{key}_{name}_h{h}"
        rows.append({"run_id": rid, **base, "L": np.nan, "features": "", "variant": "", "model": name,
                     "params": 0, "n_val": len(common), **{f"val_{k}": v for k, v in metrics(y, pred, key, mae_p).items()}})
    return rows

# ---------------------------------------------------------------------------
# Wyniki: wczytanie, zamrozenie konfiguracji, ocena testowa
# ---------------------------------------------------------------------------


def load_results(out: Path = OUT) -> pd.DataFrame:
    """Wyniki zakonczonych przebiegow; przy powtorzonym przebiegu obowiazuje ostatni wiersz danego run_id."""
    p = Path(out) / "wyniki.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p).drop_duplicates("run_id", keep="last").reset_index(drop=True)


def load_baselines(out: Path = OUT) -> pd.DataFrame:
    p = Path(out) / "modele_bazowe_val.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def summarize(df: pd.DataFrame, by=("key", "model", "h", "L", "features", "variant")) -> pd.DataFrame:
    """Srednia i odchylenie standardowe po ziarnach."""
    by = [c for c in by if c in df.columns]
    agg = df.groupby(list(by), dropna=False).agg(
        n=("seed", "size"), val_MAE=("val_MAE", "mean"), val_MAE_sd=("val_MAE", "std"),
        val_RMSE=("val_RMSE", "mean"), val_skill=("val_skill", "mean"), val_skill_sd=("val_skill", "std"),
        epochs=("epochs", "mean"), train_time_s=("train_time_s", "mean"), params=("params", "first"))
    return agg.reset_index()


CONFIG_FILE = "final_configs.json"
FAMILIES = ("Dense", "LSTM", "GRU", "TCN")


def _cell(key, model, h, g: pd.DataFrame, group: str, extra=None) -> dict:
    """Opis jednej zamrozonej konfiguracji: wszystkie ziarna wybranej konfiguracji sieci."""
    r = g.iloc[0]
    ids = sorted(g.run_id)
    c = {"grupa": group, "key": key, "model": model, "h": int(h), "block": r.block, "L": int(r.L),
         "features": r.features, "variant": r.variant, "units": int(r.units), "layers": int(r.layers),
         "dropout": float(r.dropout), "lr": float(r.lr), "val_MAE": round(float(g.val_MAE.mean()), 4),
         "val_MAE_sd": round(float(g.val_MAE.std()), 4), "run_ids": ids,
         "kompletne": len(ids) == len(_seeds(key))}
    c.update(extra or {})
    return c


def freeze_configs(out: Path = OUT, file: str = CONFIG_FILE, overwrite: bool = False) -> dict:
    """Zamrozenie konfiguracji na podstawie WYLACZNIE walidacji (zasada z Raportu 1, punkt 12.4):
    najmniejsze srednie MAE z ziaren. Dwie grupy dla kazdej pary (zbior, horyzont):
    - "A": porownanie w identycznych warunkach bloku A (okno 72 h / 24 mies., tylko historia targetu);
      na walidacji wybierany jest tylko wariant przeksztalcenia Air Passengers (log albo sdiff),
    - "najlepsza": najlepsza konfiguracja kazdej rodziny modeli ze wszystkich blokow A-E
      (okno, zmienne pomocnicze, architektura) oraz najlepsza regresja liniowa.
    Persistence i seasonal naive nie maja parametrow do wyboru; ocena testowa liczy je zawsze.
    Plik zapisywany jest PRZED jakimkolwiek uzyciem zbioru testowego i pozniej sie go nie zmienia:
    istniejacego pliku funkcja nie nadpisuje (FileExistsError), chyba ze overwrite=True."""
    target = Path(out) / file
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} juz istnieje (konfiguracje zamrozone); nie nadpisujemy go.")
    df = load_results(out)
    df = df[~df.quick.astype(bool)]
    bl = load_baselines(out)
    cfg = {"utworzono": time.strftime("%Y-%m-%d %H:%M:%S"),
           "zasada": "wybor wylacznie na walidacji: najmniejsze srednie MAE z ziaren (Air Passengers: 10 ziaren)",
           "komorki": []}
    arch = ["block", "L", "features", "variant", "units", "layers", "dropout", "lr"]
    for (key, h), gk in df.groupby(["key", "h"]):
        for model in FAMILIES:
            gm = gk[gk.model == model]
            if gm.empty:
                continue
            groups = {k: g for k, g in gm.groupby(arch) if len(g) == len(_seeds(key))}
            a = {k: g for k, g in groups.items()
                 if k[0] == "A" and k[1] == WINDOW_A[key] and k[2] == "uni"}
            if a:
                ka = min(a, key=lambda k: a[k].val_MAE.mean())
                cfg["komorki"].append(_cell(key, model, h, a[ka], "A", {"warianty_val_MAE": {
                    k[3]: round(float(g.val_MAE.mean()), 4) for k, g in a.items()}}))
            kb = min(groups, key=lambda k: groups[k].val_MAE.mean())
            cfg["komorki"].append(_cell(key, model, h, groups[kb], "najlepsza",
                                        {"liczba_kandydatow": len(groups)}))
        lin = bl[(bl.model == "linreg") & (bl.key == key) & (bl.h == h)]
        for group, sel in (("A", lin[(lin.L == WINDOW_A[key]) & (lin.features == "uni")]), ("najlepsza", lin)):
            r = sel.loc[sel.val_MAE.idxmin()]
            cfg["komorki"].append({"grupa": group, "key": key, "model": "linreg", "h": int(h), "block": "",
                                   "L": int(r.L), "features": r.features, "variant": r.variant,
                                   "val_MAE": round(float(r.val_MAE), 4), "val_MAE_sd": 0.0,
                                   "run_ids": [r.run_id], "kompletne": True})
    _atomic_json(cfg, Path(out) / file)
    return cfg


def evaluate_test(config_file: str = CONFIG_FILE, out: Path = OUT, confirm: bool = False,
                  overwrite: bool = False) -> pd.DataFrame:
    """Ocena testowa WYLACZNIE konfiguracji zapisanych w pliku konfiguracji zamrozonych (Raport 4).
    Wymaga jawnego potwierdzenia (confirm=True), aby zbior testowy nie zostal uzyty przypadkiem.
    Niczego nie trenuje: wczytuje zapisane modele (.keras, .pkl) i liczy prognozy, ktore dla wczytanego modelu
    sa deterministyczne (dropout nie dziala przy prognozie). Wszystkie modele danej pary (zbior, horyzont) oceniamy
    na tych samych chwilach testowych. Istniejacego wyniki_test.csv nie nadpisuje (chyba ze overwrite=True);
    wynik zapisywany jest dopiero po ocenie wszystkich konfiguracji, wiec przerwany przebieg niczego nie zostawia.
    Zapisuje wyniki_test.csv oraz prognozy testowe (npz) w prognozy_test/."""
    if not confirm:
        raise RuntimeError("Ocena na zbiorze testowym wymaga confirm=True (planowana dopiero w Raporcie 4).")
    import keras
    out = Path(out)
    if (out / "wyniki_test.csv").exists() and not overwrite:
        raise FileExistsError(f"{out / 'wyniki_test.csv'} juz istnieje: ocena testowa zostala wykonana.")
    with open(out / config_file) as f:        # plik konfiguracji jest tylko czytany, nigdy zapisywany
        cfg = json.load(f)
    pdir = out / "prognozy_test"
    pdir.mkdir(exist_ok=True)
    rows, cache, done_baselines = [], {}, set()
    # wspolny zbior oceny testowej: cele wspolnego zbioru oceny, dla ktorych okno istnieje w KAZDEJ zamrozonej
    # konfiguracji danej pary (zbior, horyzont) - wariant wielowymiarowy moze miec nieco mniej okien
    common = {}
    for c in cfg["komorki"]:
        arr = build_arrays(c["key"], c.get("features", "uni"), c["variant"], c["h"])
        tte = windows(arr, c["L"], c["h"])["test"][2]
        k = (c["key"], c["h"])
        base = common.get(k, eval_targets(c["key"], c["h"])["test"])
        common[k] = np.intersect1d(base, tte)
    for c in cfg["komorki"]:
        key, h, L, v = c["key"], c["h"], c["L"], c["variant"]
        features = c.get("features", "uni")
        arr = build_arrays(key, features, v, h)
        W = windows(arr, L, h)
        Xte, _, tte = W["test"]
        m = np.isin(tte, common[(key, h)])
        Xte, tte = Xte[m], tte[m]
        ytrue = arr.y[tte]
        bp = baseline_predictions(key, h, tte)
        mae_p = float(np.mean(np.abs(ytrue - bp["persistence"])))
        if (key, h) not in done_baselines:
            for name, pred in bp.items():
                rows.append({"grupa": "bazowe", "key": key, "model": name, "h": h, "L": np.nan, "features": "",
                             "variant": "", "seed": np.nan, "run_id": f"{key}_{name}_h{h}", "n_test": len(tte),
                             **{f"test_{k}": x for k, x in metrics(ytrue, pred, key, mae_p).items()}})
                np.savez_compressed(pdir / f"{key}_{name}_h{h}_test.npz", tgt=tte, yhat=pred.astype("float32"))
            done_baselines.add((key, h))
        for rid in c["run_ids"]:
            if rid not in cache:
                if c["model"] == "linreg":
                    with open(out / "modele" / f"{rid}.pkl", "rb") as f:
                        mdl = pickle.load(f)
                    p = mdl.predict(Xte.reshape(len(Xte), -1))
                else:
                    mdl = keras.models.load_model(out / "modele" / f"{rid}.keras", compile=False)
                    p = mdl.predict(Xte, batch_size=4096, verbose=0).ravel()
                    keras.backend.clear_session()
                yhat = arr.inv(p, tte)
                np.savez_compressed(pdir / f"{rid}_test.npz", tgt=tte, yhat=yhat.astype("float32"))
                cache[rid] = yhat
            yhat = cache[rid]
            seed = np.nan if c["model"] == "linreg" else int(rid.rsplit("_s", 1)[1])
            rows.append({"grupa": c["grupa"], "key": key, "model": c["model"], "h": h, "L": L,
                         "features": features, "variant": v, "seed": seed, "run_id": rid, "n_test": len(tte),
                         **{f"test_{k}": x for k, x in metrics(ytrue, yhat, key, mae_p).items()}})
    res = pd.DataFrame(rows)
    res.to_csv(out / "wyniki_test.csv", index=False)
    return res


def environment() -> dict:
    import tensorflow as tf
    import sklearn
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        ram = None
    return {"Python": platform.python_version(), "TensorFlow": tf.__version__,
            "Keras": __import__("keras").__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "system": platform.platform(),
            "procesor": platform.processor(), "rdzenie": os.cpu_count(), "RAM [GB]": ram,
            "GPU": [d.name for d in tf.config.list_physical_devices("GPU")]}

# ---------------------------------------------------------------------------
# Analiza wynikow: tabele i wykresy (notatnik, sekcje 4-5)
# ---------------------------------------------------------------------------

MODEL_COLORS = {"Dense": "#8a8a8a", "LSTM": "#2a78d6", "GRU": "#eb6834", "linreg": "#1baf7a",
                "persistence": "#555555", "seasonal_naive": "#b58a00", "TCN": "#8e44ad"}


def architecture_table() -> pd.DataFrame:
    """Warstwy i liczba parametrow modeli w konfiguracji bazowej (okno 72 h / 24 mies., 1 cecha)."""
    import keras
    rows = []
    for kind in NETS:
        for key, L in (("godzinowe", 72), ("Air Passengers", 24)):
            m = build_model(kind, L, 1)
            layers = " → ".join(f"{l.__class__.__name__}({getattr(l, 'units', '') or ''})".replace("()", "")
                                for l in m.layers[1:])
            rows.append({"model": kind, "zbiory": key, "okno": L, "warstwy": layers,
                         "parametry": int(m.count_params())})
            keras.backend.clear_session()
    return pd.DataFrame(rows)


def history(run_id: str, out: Path = OUT) -> dict:
    with open(Path(out) / "historie" / f"{run_id}.json") as f:
        return json.load(f)


def plot_learning_curves(df: pd.DataFrame, key: str, h: int, out: Path = OUT, models=("LSTM", "GRU"),
                         variant=None, fig_dir=None):
    """Strata uczaca (linia ciagla) i walidacyjna (przerywana) dla wszystkich ziaren; kropka = najlepsza epoka."""
    import matplotlib.pyplot as plt
    sel = df[(df.key == key) & (df.h == h) & df.model.isin(models)]
    if variant:
        sel = sel[sel.variant == variant]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for _, r in sel[sel.model == m].iterrows():
            H = history(r.run_id, out)
            ep = np.arange(1, len(H["loss"]) + 1)
            c = MODEL_COLORS[m]
            ax.plot(ep, H["loss"], color=c, lw=1, alpha=0.8)
            ax.plot(ep, H["val_loss"], color=c, lw=1, ls="--", alpha=0.8)
            b = int(np.argmin(H["val_loss"]))
            ax.plot(b + 1, H["val_loss"][b], "o", color=c, ms=4)
        ax.set_title(f"{m}")
        ax.set_xlabel("epoka")
        ax.set_yscale("log")
    axes[0].set_ylabel("strata MSE (skala standaryzowana)")
    fig.suptitle(f"{td.SPECS[key].name}, h = {h}: strata ucząca (—) i walidacyjna (- -)", fontsize=11)
    fig.tight_layout()
    if fig_dir:
        td._save(fig, fig_dir, f"krzywe_{key}_h{h}")
    return fig


def overfit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostyka przeuczenia: epoka najlepszej straty walidacyjnej, stosunek strat val/train w tej epoce
    i wzrost straty walidacyjnej od minimum do konca uczenia."""
    rows = []
    for _, r in df.iterrows():
        H = history(r.run_id)
        b = int(np.argmin(H["val_loss"]))
        rows.append({"run_id": r.run_id, "key": r.key, "model": r.model, "h": r.h, "variant": r.variant,
                     "seed": r.seed, "epoki": len(H["loss"]), "najlepsza epoka": b + 1,
                     "val/train w najlepszej epoce": H["val_loss"][b] / H["loss"][b],
                     "wzrost val od minimum [%]": 100 * (H["val_loss"][-1] / H["val_loss"][b] - 1),
                     "spadek train po minimum val [%]": 100 * (1 - H["loss"][-1] / H["loss"][b])})
    return pd.DataFrame(rows)


def plot_val_forecast(run_id: str, key: str, h: int, n: int = 24 * 14, out: Path = OUT, fig_dir=None,
                      start: int = 0, extra: dict | None = None):
    """Wartosci rzeczywiste i prognozy na fragmencie zbioru walidacyjnego (oryginalne jednostki)."""
    import matplotlib.pyplot as plt
    P = prepare(key)
    z = np.load(Path(out) / "prognozy" / f"{run_id}_val.npz")
    tgt, yhat = z["tgt"][start:start + n], z["yhat"][start:start + n]
    idx = P["index"][tgt]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(idx, P["y"][tgt], color="#222222", lw=1.2, label="wartość rzeczywista")
    ax.plot(idx, yhat, color=MODEL_COLORS.get(run_id.split("_")[1], "#2a78d6"), lw=1.1, label=run_id.split("_")[1])
    for name, (t2, y2) in (extra or {}).items():
        m = np.isin(t2, tgt)
        ax.plot(P["index"][t2[m]], y2[m], lw=0.9, ls="--", color=MODEL_COLORS.get(name, "#999999"), label=name)
    ax.set_ylabel(td.SPECS[key].unit)
    ax.set_title(f"{td.SPECS[key].name}, h = {h}: prognoza na zbiorze walidacyjnym")
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    if fig_dir:
        td._save(fig, fig_dir, f"prognoza_val_{run_id}")
    return fig


def plot_skill_vs_horizon(summary: pd.DataFrame, baselines: pd.DataFrame, fig_dir=None):
    """Skill score (walidacja) w funkcji horyzontu, panel na zbior, linia na model; slupki = odch. std. ziaren."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, key in zip(axes, KEYS):
        hs = td.SPECS[key].horizons
        for m in NETS:
            s = summary[(summary.key == key) & (summary.model == m)].sort_values("h")
            if len(s):
                ax.errorbar(range(len(s)), s.val_skill, yerr=s.val_skill_sd, marker="o", ms=4, capsize=3,
                            color=MODEL_COLORS[m], label=m)
        for m in ("linreg", "seasonal_naive"):
            b = baselines[(baselines.key == key) & (baselines.model == m)]
            if m == "linreg":
                b = b[(b.features == "uni") & (b.L == WINDOW_A[key])].groupby("h").val_skill.max()
            else:
                b = b.groupby("h").val_skill.first()
            b = b.reindex(hs)
            ax.plot(range(len(hs)), b.values, marker="s", ms=3, ls=":", color=MODEL_COLORS[m], label=m)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(range(len(hs)), [f"{h} {td.SPECS[key].step_word}" for h in hs])
        ax.set_title(td.SPECS[key].name, fontsize=10)
    axes[0].set_ylabel("skill score vs persistence (val)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if fig_dir:
        td._save(fig, fig_dir, "skill_horyzont_val")
    return fig
