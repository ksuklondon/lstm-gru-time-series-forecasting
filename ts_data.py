"""Projekt 2A - wczytywanie, kontrola jakosci, agregacja i podzial czterech szeregow.

Modul jest wspolny dla notatnika EDA (Raport 2) i pozniejszych eksperymentow
(Raport 3-4), zeby preprocessing byl zdefiniowany w jednym miejscu.

Zasady (zgodne z protokolem eksperymentalnym):
- EDA najpierw na danych surowych: braki, duplikaty, regularnosc znacznikow czasu,
  wartosci fizycznie niemozliwe - przed agregacja i uzupelnianiem luk.
- Agregacja do 1 h zdefiniowana osobno dla kazdej zmiennej; wartosc godzinowa
  oznaczona KONCEM przedzialu (closed='right', label='right').
- Uzupelnianie luk wylacznie przyczynowe (ffill) i tylko dla krotkich luk.
- Podzial chronologiczny; ACF, profile sezonowe i porownanie lagow
  liczone WYLACZNIE na czesci treningowej.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Specyfikacje zbiorow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    source: str
    url: str
    target: str
    unit: str
    freq: str                    # czestotliwosc robocza: 'h' albo 'MS'
    raw_step: str | None         # krok surowych danych (dla kontroli siatki)
    per_hour: int | None         # oczekiwana liczba surowych obserwacji na godzine
    horizons: tuple[int, ...]
    seasonal_candidates: tuple[int, ...]
    acf_nlags: int
    step_word: str               # 'h' albo 'mies.'


SPECS: dict[str, DatasetSpec] = {
    "jena": DatasetSpec(
        key="jena", name="Jena Climate",
        source="Max Planck Institute for Biogeochemistry (przez Keras)",
        url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        target="T (degC)", unit="°C", freq="h", raw_step="10min", per_hour=6,
        horizons=(1, 6, 24), seasonal_candidates=(24, 168), acf_nlags=400, step_word="h"),
    "household": DatasetSpec(
        key="household", name="Household Power Consumption",
        source="UCI ML Repository, doi:10.24432/C58K54",
        url="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        target="Global_active_power", unit="kW", freq="h", raw_step="1min", per_hour=60,
        horizons=(1, 6, 24), seasonal_candidates=(24, 168), acf_nlags=400, step_word="h"),
    "metro": DatasetSpec(
        key="metro", name="Metro Interstate Traffic Volume",
        source="UCI ML Repository, doi:10.24432/C5X60B",
        url="https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
        target="traffic_volume", unit="pojazdy/h", freq="h", raw_step="1h", per_hour=1,
        horizons=(1, 6, 24), seasonal_candidates=(24, 168), acf_nlags=400, step_word="h"),
    "air": DatasetSpec(
        key="air", name="Air Passengers",
        source="Box i Jenkins (1976), kopia z Rdatasets",
        url="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv",
        target="passengers", unit="tys. pasażerów", freq="MS", raw_step=None, per_hour=None,
        horizons=(1, 3, 12), seasonal_candidates=(12,), acf_nlags=36, step_word="mies."),
}

JENA_COLS = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
             "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
             "wv (m/s)", "max. wv (m/s)", "wd (deg)"]
HOUSEHOLD_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
                  "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
METRO_NUM = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]

# ---------------------------------------------------------------------------
# Pobieranie i wczytywanie surowych danych
# ---------------------------------------------------------------------------


def download(key: str, data_dir: str | Path = "data") -> Path:
    """Pobiera plik zrodlowy (jesli nie ma go w data_dir) i zwraca sciezke."""
    spec = SPECS[key]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / Path(spec.url).name
    if not path.exists():
        print(f"Pobieram {spec.name} ...")
        urllib.request.urlretrieve(spec.url, path)
    return path


def _zip_member(path: Path, suffixes: tuple[str, ...]) -> tuple[bytes, str]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(suffixes) and not n.startswith("__MACOSX")]
        if not names:
            raise FileNotFoundError(f"Brak pliku {suffixes} w {path}")
        return zf.read(names[0]), names[0]


def load_raw(key: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Surowe dane z kolumna 'time' (bez sortowania i bez usuwania duplikatow)."""
    path = download(key, data_dir)
    if key == "jena":
        raw, _ = _zip_member(path, (".csv",))
        df = pd.read_csv(io.BytesIO(raw))
        df["time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
        df = df.drop(columns="Date Time")
    elif key == "household":
        raw, _ = _zip_member(path, (".txt",))
        df = pd.read_csv(io.BytesIO(raw), sep=";", na_values=["?", ""], low_memory=False)
        df["time"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
        df = df.drop(columns=["Date", "Time"])
        df[HOUSEHOLD_COLS] = df[HOUSEHOLD_COLS].apply(pd.to_numeric, errors="coerce")
    elif key == "metro":
        raw, name = _zip_member(path, (".csv", ".csv.gz"))
        comp = "gzip" if name.endswith(".gz") else None
        df = pd.read_csv(io.BytesIO(raw), compression=comp)
        df["time"] = pd.to_datetime(df["date_time"], format="mixed")
        df = df.drop(columns="date_time")
        hol = df["holiday"].astype("string").str.strip()
        df["holiday"] = hol.where(~hol.isin(["None", "", "nan"]), pd.NA)
    elif key == "air":
        df = pd.read_csv(path)
        if len(df) != 144 or int(df["time"].iloc[0]) != 1949:
            raise ValueError("Nieoczekiwany format AirPassengers (oczekiwano 144 mies. od 1949)")
        df = pd.DataFrame({
            "time": pd.period_range("1949-01", periods=144, freq="M").to_timestamp(),
            "passengers": df["value"].astype(float).to_numpy(),
        })
    else:
        raise KeyError(key)
    cols = ["time"] + [c for c in df.columns if c != "time"]
    return df[cols]

# ---------------------------------------------------------------------------
# Kontrola jakosci danych surowych
# ---------------------------------------------------------------------------


def raw_regularity(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    """Regularnosc znacznikow czasu w danych surowych."""
    t = df["time"]
    uniq = t.drop_duplicates().sort_values()
    out = {
        "wiersze": len(df),
        "unikalne znaczniki czasu": t.nunique(),
        "wiersze ze zduplikowanym czasem": int(t.duplicated(keep=False).sum()),
        "nadmiarowe wiersze (duplikaty)": len(df) - t.nunique(),
        "posortowane rosnąco": bool(t.is_monotonic_increasing),
        "początek": t.min(),
        "koniec": t.max(),
    }
    if spec.raw_step is not None:
        step = pd.Timedelta(spec.raw_step)
        grid = pd.date_range(uniq.iloc[0].floor(step), uniq.iloc[-1], freq=step)
        on_grid = uniq.isin(grid)
        out["oczekiwane punkty siatki"] = len(grid)
        out["brakujące punkty siatki"] = int(len(grid) - on_grid.sum())
        out["znaczniki poza siatką"] = int((~on_grid).sum())
    else:
        grid = pd.date_range(uniq.iloc[0], uniq.iloc[-1], freq=spec.freq)
        out["oczekiwane punkty siatki"] = len(grid)
        out["brakujące punkty siatki"] = int(len(grid) - uniq.isin(grid).sum())
    diffs = uniq.diff().dropna()
    out["najczęstszy krok"] = diffs.mode().iloc[0] if len(diffs) else pd.NaT
    out["najdłuższa przerwa"] = diffs.max() if len(diffs) else pd.NaT
    return pd.Series(out, name=spec.name)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Braki w kazdej kolumnie surowych danych."""
    n = df.isna().sum()
    return pd.DataFrame({"braki": n, "braki %": (100 * n / len(df)).round(3)}).rename_axis("kolumna")


def duplicate_report(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Czy wiersze o tym samym czasie maja rozne wartosci (w szczegolnosci targetu)?"""
    dup = df[df["time"].duplicated(keep=False)]
    if dup.empty:
        return pd.DataFrame(columns=["kolumna", "grupy z różnymi wartościami"])
    rows = []
    for c in [c for c in df.columns if c != "time"]:
        n_diff = int((dup.groupby("time")[c].nunique(dropna=False) > 1).sum())
        rows.append({"kolumna": c, "grupy z różnymi wartościami": n_diff})
    out = pd.DataFrame(rows)
    out.attrs["grupy duplikatów"] = dup["time"].nunique()
    return out


# Reguly wartosci fizycznie niemozliwych (ustalone a priori, nie dopasowywane do danych).
ANOMALY_RULES = {
    "jena": [
        ("wv (m/s)", "prędkość wiatru < 0", lambda s: s < 0),
        ("max. wv (m/s)", "maks. prędkość wiatru < 0", lambda s: s < 0),
        ("rh (%)", "wilgotność poza [0, 100] %", lambda s: (s < 0) | (s > 100)),
        ("p (mbar)", "ciśnienie poza [800, 1100] mbar", lambda s: (s < 800) | (s > 1100)),
    ],
    "household": [
        ("Global_active_power", "moc czynna < 0", lambda s: s < 0),
        ("Global_reactive_power", "moc bierna < 0", lambda s: s < 0),
        ("Global_intensity", "natężenie < 0", lambda s: s < 0),
        ("Voltage", "napięcie poza [180, 280] V", lambda s: (s < 180) | (s > 280)),
    ],
    "metro": [
        ("temp", "temperatura ≤ 0 K", lambda s: s <= 0),
        ("rain_1h", "opad > 300 mm/h (powyżej rekordu świata)", lambda s: s > 300),
        ("snow_1h", "opad śniegu < 0", lambda s: s < 0),
        ("clouds_all", "zachmurzenie poza [0, 100] %", lambda s: (s < 0) | (s > 100)),
        ("traffic_volume", "natężenie ruchu < 0", lambda s: s < 0),
    ],
    "air": [
        ("passengers", "liczba pasażerów ≤ 0", lambda s: s <= 0),
    ],
}


def apply_anomaly_rules(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zamienia wartosci fizycznie niemozliwe na brak; zwraca tez tabele zliczen."""
    out = df.copy()
    rows = []
    for col, desc, rule in ANOMALY_RULES[key]:
        if col not in out.columns:
            continue
        mask = rule(out[col]).fillna(False).astype(bool)
        vals = out.loc[mask, col]
        rows.append({"kolumna": col, "reguła": desc, "liczba": int(mask.sum()),
                     "%": round(100 * mask.mean(), 4),
                     "przykładowe wartości": ", ".join(map(str, vals.unique()[:3]))})
        out.loc[mask, col] = np.nan
    return out, pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Agregacja do czestotliwosci roboczej
# ---------------------------------------------------------------------------


def _dedupe(d: pd.DataFrame, target: str) -> pd.DataFrame:
    """Laczy wiersze o tym samym czasie. Target: identyczne wartosci -> ta wartosc,
    sprzeczne wartosci -> brak (nie zgadujemy, ktory pomiar jest prawdziwy).
    Pozostale zmienne liczbowe: srednia."""
    g = d.groupby("time")
    out = g.mean(numeric_only=True)
    conflict = g[target].nunique() > 1
    out[target] = g[target].first().where(~conflict)
    return out.reset_index()


def _resample_hour(d: pd.DataFrame):
    return d.set_index("time").sort_index().resample("1h", closed="right", label="right")


def to_working_freq(df_clean: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Agregacja wg regul per zmienna. Zwraca ramke na regularnej siatce z kolumna 'coverage'.

    Godzina, w ktorej jest mniej niz 50% surowych pomiarow targetu, jest traktowana jako brak.
    """
    key = spec.key
    if key == "jena":
        d = _dedupe(df_clean, spec.target)
        rad = np.deg2rad(d["wd (deg)"])
        d["wx (m/s)"] = d["wv (m/s)"] * np.cos(rad)                          # kierunek wiatru jako wektor
        d["wy (m/s)"] = d["wv (m/s)"] * np.sin(rad)
        d = d.drop(columns="wd (deg)")
        r = _resample_hour(d)
        h = r.mean()
        h["max. wv (m/s)"] = r["max. wv (m/s)"].max()
        h["coverage"] = r[spec.target].count() / spec.per_hour
    elif key == "household":
        d = _dedupe(df_clean, spec.target)
        r = _resample_hour(d)
        h = r.mean()
        sub = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        # Energia w godzinie [Wh] OSZACOWANA z dostepnych minut: srednia * 60.
        # Przy pelnej godzinie rowna sumie; przy niepelnej (pokrycie 50-99%) to korekta na brakujace
        # minuty (zwykla suma zanizalaby energie). Liczbe takich godzin raportuje data_accounting.
        h[sub] = h[sub] * 60
        h["coverage"] = r[spec.target].count() / spec.per_hour
    elif key == "metro":
        d = df_clean.copy()
        d["temp"] = d["temp"] - 273.15           # K -> °C (po usunieciu 0 K)
        weather = pd.get_dummies(d["weather_main"], prefix="w").astype(float)
        d = pd.concat([d[["time"] + METRO_NUM], weather], axis=1)
        d["holiday"] = df_clean["holiday"].notna().astype(float)
        agg = {c: "mean" for c in METRO_NUM}
        agg[spec.target] = "first"
        agg.update({c: "max" for c in weather.columns})
        agg["holiday"] = "max"
        h = d.groupby("time").agg(agg)           # kilka raportow pogody na godzine -> jeden wiersz
        conflict = d.groupby("time")[spec.target].nunique() > 1
        h.loc[conflict[conflict].index, spec.target] = np.nan   # sprzeczne natezenie ruchu -> brak
        h = h.rename(columns={"temp": "temp (degC)"})
        # swieto jest znane z gory: oznaczamy cala dobe, nie tylko godzine z wpisem
        hol_days = h.index[h["holiday"] > 0].normalize().unique()
        full = pd.date_range(h.index.min().floor("h"), h.index.max(), freq="h")
        h = h.reindex(full)
        h["holiday"] = h.index.normalize().isin(hol_days).astype(float)
        h["coverage"] = h[spec.target].notna().astype(float)
    elif key == "air":
        h = df_clean.set_index("time").asfreq("MS")
        h["coverage"] = h[spec.target].notna().astype(float)
    else:
        raise KeyError(key)
    h.index.name = "time"
    if key != "air":
        h = h.asfreq("h")
    low = h["coverage"].fillna(0) < 0.5
    value_cols = [c for c in h.columns if c != "coverage"]
    h.loc[low, value_cols] = np.nan
    return h

# ---------------------------------------------------------------------------
# Luki
# ---------------------------------------------------------------------------


def nan_runs(s: pd.Series) -> pd.DataFrame:
    """Ciagi brakow: poczatek, koniec, dlugosc (w krokach)."""
    isna = s.isna().to_numpy()
    if not isna.any():
        return pd.DataFrame(columns=["początek", "koniec", "długość"])
    edges = np.diff(np.concatenate([[0], isna.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    return pd.DataFrame({"początek": s.index[starts], "koniec": s.index[ends], "długość": ends - starts + 1})


def gap_summary(s: pd.Series, max_fill: int, step_word: str) -> pd.DataFrame:
    runs = nan_runs(s)
    bins = [(1, 1), (2, max_fill), (max_fill + 1, 24), (25, 10**9)]
    labels = [f"1 {step_word}", f"2–{max_fill} {step_word}", f"{max_fill + 1}–24 {step_word}", f"> 24 {step_word}"]
    rows = []
    for (lo, hi), lab in zip(bins, labels):
        sel = runs[(runs["długość"] >= lo) & (runs["długość"] <= hi)] if len(runs) else runs
        rows.append({"długość luki": lab, "liczba luk": len(sel),
                     "brakujące kroki": int(sel["długość"].sum()) if len(sel) else 0,
                     "uzupełniana": "tak (ffill)" if hi <= max_fill else "nie"})
    return pd.DataFrame(rows)


def data_accounting(raw: pd.DataFrame, clean: pd.DataFrame, h: pd.DataFrame, h_f: pd.DataFrame,
                    spec: DatasetSpec) -> pd.DataFrame:
    """Bilans targetu od danych surowych do szeregu roboczego: ile polaczono, usunieto, uzupelniono."""
    y_raw, y_clean = raw[spec.target], clean[spec.target]
    dup = raw[raw["time"].duplicated(keep=False)]
    conflict = int((dup.groupby("time")[spec.target].nunique() > 1).sum()) if len(dup) else 0
    filled = h[spec.target].isna() & h_f[spec.target].notna()
    rows = [
        ("dane surowe", "wiersze", len(raw)),
        ("dane surowe", "nadmiarowe wiersze o powtórzonym czasie (połączone)", len(raw) - raw["time"].nunique()),
        ("dane surowe", "grupy duplikatów ze sprzecznym targetem (target → brak)", conflict),
        ("dane surowe", "braki targetu w pliku", int(y_raw.isna().sum())),
        ("dane surowe", "wartości targetu usunięte jako niemożliwe", int(y_clean.isna().sum() - y_raw.isna().sum())),
        ("szereg roboczy", f"kroki na regularnej siatce ({spec.freq})", len(h)),
        ("szereg roboczy", "kroki z niepełnym pokryciem 50–99% (wartość z dostępnych pomiarów)",
         int(((h["coverage"] >= 0.5) & (h["coverage"] < 1)).sum())),
        ("szereg roboczy", "kroki bez targetu po agregacji", int(h[spec.target].isna().sum())),
        ("szereg roboczy", "kroki targetu uzupełnione (luki ≤ 3)", int(filled.sum())),
        ("szereg roboczy", "kroki bez targetu po uzupełnieniu (długie luki)", int(h_f[spec.target].isna().sum())),
    ]
    return pd.DataFrame(rows, columns=["etap", "pozycja", "liczba"])


def fill_short_gaps(df: pd.DataFrame, max_gap: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Przyczynowe uzupelnienie luk <= max_gap krokow (ostatnia znana wartosc). Dluzsze luki zostaja."""
    out = df.copy()
    flag = pd.Series(False, index=df.index, name="filled")
    for c in out.columns:
        s = out[c]
        if c == "coverage" or not pd.api.types.is_numeric_dtype(s):
            continue
        isna = s.isna()
        if not isna.any():
            continue
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        short = isna & (run_len <= max_gap)
        filled = s.ffill()
        ok = short & filled.notna()
        out.loc[ok, c] = filled[ok]
        flag |= ok
    return out, flag

# ---------------------------------------------------------------------------
# Podzial chronologiczny
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def chronological_split(index: pd.DatetimeIndex, spec: DatasetSpec) -> pd.Series:
    """Etykieta zbioru dla kazdego kroku. Granice: 60/20/20 osi czasu, zaokraglone do polnocy;
    Air Passengers: pelne lata 1949-1956 / 1957-1958 / 1959-1960."""
    if spec.key == "air":
        b1, b2 = pd.Timestamp("1957-01-01"), pd.Timestamp("1959-01-01")
    else:
        n = len(index)
        b1 = index[int(0.6 * n)].normalize()
        b2 = index[int(0.8 * n)].normalize()
    lab = np.where(index < b1, "train", np.where(index < b2, "val", "test"))
    return pd.Series(lab, index=index, name="split")


def split_table(h: pd.DataFrame, split: pd.Series, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    for name in SPLITS:
        part = h.loc[split == name, spec.target]
        rows.append({"zbiór": name, "od": part.index.min(), "do": part.index.max(),
                     "kroki": len(part), "udział %": round(100 * len(part) / len(h), 1),
                     "braki targetu": int(part.isna().sum()),
                     "średnia": round(part.mean(), 3), "odch. std.": round(part.std(), 3)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Sezonowosc i autokorelacja (tylko train)
# ---------------------------------------------------------------------------


def acf_train(y_train: pd.Series, nlags: int, transform: str = "none") -> pd.Series:
    """ACF szeregu treningowego. transform: 'none', 'diff' (pierwsze roznice - usuwaja bezwladnosc
    i trend, wiec szczyty przy lagu s wskazuja na sezonowosc) albo 'dlog' (roznice logarytmu)."""
    from statsmodels.tsa.stattools import acf
    x = y_train.astype(float)
    if transform == "dlog":
        x = np.log(x).diff().iloc[1:]
    elif transform == "diff":
        x = x.diff().iloc[1:]
    nlags = min(nlags, len(x) // 2 - 1)
    vals = acf(x.to_numpy(), nlags=nlags, missing="conservative", fft=False)
    return pd.Series(vals, index=pd.RangeIndex(len(vals), name="lag"), name="ACF")


def seasonal_profiles(y_train: pd.Series, spec: DatasetSpec) -> dict[str, pd.DataFrame]:
    """Srednia i kwartyle targetu wg pory dnia, dnia tygodnia i miesiaca (train)."""
    idx = y_train.index
    groups = {"miesiąc": idx.month}
    if spec.freq == "h":
        groups = {"godzina": idx.hour, "dzień tygodnia": idx.dayofweek, **groups}
    out = {}
    for name, g in groups.items():
        out[name] = y_train.groupby(g).agg(
            średnia="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
    return out


def persistence_forecast(y: pd.Series, h: int) -> pd.Series:
    """ŷ(τ) = y(τ − h); indeks = czas celu τ."""
    return y.shift(h)


def seasonal_naive_forecast(y: pd.Series, h: int, s: int) -> pd.Series:
    """ŷ(t+h) = y(t+h − s(k+1)), k = ⌊(h−1)/s⌋; indeks = czas celu."""
    k = (h - 1) // s
    return y.shift(s * (k + 1))


def baseline_check_train(y_train: pd.Series, spec: DatasetSpec) -> pd.DataFrame:
    """MAE persistence i seasonal naive (kazdy kandydat s) na train; skill wzgledem persistence."""
    rows = []
    for h in spec.horizons:
        pers = persistence_forecast(y_train, h)
        for s in spec.seasonal_candidates:
            sn = seasonal_naive_forecast(y_train, h, s)
            m = y_train.notna() & pers.notna() & sn.notna()
            mae_p = (y_train[m] - pers[m]).abs().mean()
            mae_s = (y_train[m] - sn[m]).abs().mean()
            same = s * ((h - 1) // s + 1) == h
            rows.append({"horyzont": h, "s": s, "MAE persistence": round(mae_p, 4),
                         "MAE seasonal naive": round(mae_s, 4),
                         "skill SN": round(1 - mae_s / mae_p, 4),
                         "SN ≡ persistence": "tak" if same else "nie", "n": int(m.sum())})
    return pd.DataFrame(rows)


def propose_seasonal_period(lag_table: pd.DataFrame) -> int:
    """Okres s o najmniejszym srednim (po horyzontach) MAE seasonal naive na train."""
    return int(lag_table.groupby("s")["MAE seasonal naive"].mean().idxmin())


def feature_columns(h: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Kolumny liczbowe modelu: target na pierwszym miejscu, potem pozostale zmienne (bez 'coverage')."""
    cols = [c for c in h.columns if c not in ("coverage",) and pd.api.types.is_numeric_dtype(h[c])]
    return [spec.target] + [c for c in cols if c != spec.target]


def fit_scaler(h: pd.DataFrame, split: pd.Series, cols: list[str]):
    """StandardScaler dopasowany WYLACZNIE na zbiorze uczacym. Kolumny binarne (0/1) nie sa skalowane."""
    from sklearn.preprocessing import StandardScaler
    train = h.loc[split == "train", cols]
    binary = [c for c in cols if set(train[c].dropna().unique()) <= {0.0, 1.0}]
    cont = [c for c in cols if c not in binary]
    scaler = StandardScaler().fit(train[cont].dropna())
    return scaler, cont


def apply_scaler(h: pd.DataFrame, cols: list[str], scaler, cont: list[str]) -> pd.DataFrame:
    out = h[cols].astype(float).copy()
    out[cont] = scaler.transform(out[cont])
    return out


def scaling_table(h_scaled: pd.DataFrame, split: pd.Series, scaler, cont: list[str], spec: DatasetSpec) -> pd.DataFrame:
    """Parametry skalera (z train) i kontrola: srednia/odch. targetu po skalowaniu w kazdym zbiorze."""
    i = cont.index(spec.target)
    row = {"średnia z train (skaler)": round(float(scaler.mean_[i]), 4),
           "odch. std. z train (skaler)": round(float(scaler.scale_[i]), 4)}
    for name in SPLITS:
        y = h_scaled.loc[split == name, spec.target]
        row[f"{name}: średnia po skalowaniu"] = round(float(y.mean()), 3)
        row[f"{name}: odch. std. po skalowaniu"] = round(float(y.std()), 3)
    return pd.Series(row, name=spec.target).to_frame()


def make_windows(data, window_size: int = 30, horizon: int = 1, target_col: int = 0):
    """Okna wejsciowe bez zagladania w przyszlosc.

    data: tablica (T,) lub (T, d). Przyklad i: X = data[i : i+L] (kroki t-L+1..t), y = data[i+L+h-1, target]
    (krok t+h). Zwraca X (n, L, d), y (n,) i pozycje celu w szeregu (n,). Okna z brakiem w wejsciu lub celu
    sa pomijane."""
    a = np.asarray(data, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    L, h = int(window_size), int(horizon)
    n = len(a) - L - h + 1
    if n <= 0:
        return np.empty((0, L, a.shape[1])), np.empty(0), np.empty(0, dtype=int)
    idx = np.arange(n)
    X = np.lib.stride_tricks.sliding_window_view(a, (L, a.shape[1]))[:n, 0]
    tgt = idx + L + h - 1
    y = a[tgt, target_col]
    ok = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y)
    return X[ok], y[ok], tgt[ok]


def windows_by_split(h_scaled: pd.DataFrame, split: pd.Series, cols: list[str], window_size: int, horizon: int):
    """Okna przypisane do zbiorow wg czasu CELU (wejscie moze siegac do konca poprzedniego zbioru)."""
    X, y, tgt = make_windows(h_scaled[cols].to_numpy(), window_size, horizon, target_col=0)
    lab = split.to_numpy()[tgt]
    return {name: (X[lab == name], y[lab == name], h_scaled.index[tgt[lab == name]]) for name in SPLITS}


def window_shape_table(h_scaled, split, cols, windows, horizons) -> pd.DataFrame:
    rows = []
    for L in windows:
        for hz in horizons:
            w = windows_by_split(h_scaled, split, cols, L, hz)
            row = {"okno L": L, "horyzont h": hz}
            for name in SPLITS:
                row[f"X_{name}"] = str(tuple(w[name][0].shape))
            rows.append(row)
    return pd.DataFrame(rows)


def check_windows(split: pd.Series) -> pd.DataFrame:
    """Testy make_windows na szeregu arytmetycznym 0, 1, 2, ...: cel = ostatni krok wejscia + h, brak przecieku."""
    series = np.arange(len(split), dtype=float)
    rows = []
    for L, hz in [(30, 1), (30, 6), (24, 12)]:
        X, y, tgt = make_windows(series, L, hz)
        ok_target = bool(np.all(y == X[:, -1, 0] + hz))
        ok_order = bool(np.all(X[:, -1, 0] < y))
        lab = split.to_numpy()[tgt]
        first_val = np.where(lab == "val")[0]
        crosses = bool(len(first_val) and split.to_numpy()[int(X[first_val[0], 0, 0])] == "train")
        rows.append({"okno L": L, "horyzont h": hz, "cel = ostatni krok wejścia + h": ok_target,
                     "wejście przed celem": ok_order, "1. okno val sięga do train (dozwolone)": crosses})
    return pd.DataFrame(rows)


def plot_window(h_scaled: pd.DataFrame, split: pd.Series, cols: list[str], scaler, cont, spec: DatasetSpec,
                window_size: int = 30, horizon: int = 1, out_dir=None):
    """Ostatnie okno zbioru uczacego (skala oryginalna) i wartosc prognozowana."""
    import matplotlib.pyplot as plt
    w = windows_by_split(h_scaled, split, cols, window_size, horizon)
    X, y, times = w["train"]
    i = cont.index(spec.target)
    inv = lambda v: v * scaler.scale_[i] + scaler.mean_[i]
    step = pd.Timedelta(hours=1) if spec.freq == "h" else pd.DateOffset(months=1)
    t_end = times[-1] - horizon * step
    t_in = [t_end - (window_size - 1 - k) * step for k in range(window_size)]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t_in, inv(X[-1, :, 0]), color=SERIES[0], marker="o", ms=3, lw=1.2, label=f"okno wejściowe ({window_size} kroków)")
    ax.plot([times[-1]], [inv(y[-1])], "o", color=SERIES[1], ms=9, label=f"wartość prognozowana (t + {horizon})")
    ax.axvline(t_end, color=GRID, lw=1)
    ax.set_title(f"{spec.name}: przykładowe okno i cel (L = {window_size}, h = {horizon}, zbiór uczący)")
    ax.set_ylabel(spec.unit)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    _save(fig, out_dir, f"{spec.key}_okno_L{window_size}_h{horizon}")
    return fig


def window_counts(y: pd.Series, split: pd.Series, windows, horizons) -> pd.DataFrame:
    """Liczba uzytecznych przykladow (okno L, horyzont h) w kazdym zbiorze.

    Przyklad nalezy do zbioru wg czasu celu; wejscie moze siegac do konca poprzedniego zbioru
    (bylo znane w chwili prognozy). Przyklad jest odrzucany, gdy cel albo dowolny krok okna
    wejsciowego jest brakiem (dluga luka) - tak samo jak w eksperymentach."""
    nan = y.isna().to_numpy().astype(int)
    csum = np.concatenate([[0], np.cumsum(nan)])
    pos = np.arange(len(y))
    lab = split.to_numpy()
    rows = []
    for L in windows:
        for h in horizons:
            start, end = pos - h - L + 1, pos - h            # wejscie: kroki start..end
            full = start >= 0
            nan_in = np.zeros(len(y), dtype=int)
            nan_in[full] = csum[end[full] + 1] - csum[start[full]]
            ok = full & (nan_in == 0) & (nan == 0)
            row = {"okno L": L, "horyzont h": h}
            for name in SPLITS:
                row[name] = int((ok & (lab == name)).sum())
            row["odrzucone przez luki"] = int((full & ~ok).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def lagged_correlations(h_train: pd.DataFrame, spec: DatasetSpec, max_cols: int = 20) -> pd.DataFrame:
    """Korelacja Pearsona: zmienna w chwili t vs target w chwili t+h (train). Wiersz targetu = odniesienie."""
    y = h_train[spec.target]
    cols = [c for c in h_train.columns
            if c not in ("coverage",) and pd.api.types.is_numeric_dtype(h_train[c]) and h_train[c].nunique() > 1]
    cols = [spec.target] + [c for c in cols if c != spec.target][:max_cols]
    out = pd.DataFrame(index=pd.Index(cols, name="zmienna w chwili t"))
    for h in spec.horizons:
        fut = y.shift(-h)
        out[f"corr z y(t+{h})"] = [round(h_train[c].corr(fut), 3) for c in cols]
    return out

# ---------------------------------------------------------------------------
# Wykresy (matplotlib)
# ---------------------------------------------------------------------------

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SPLIT_SHADE = {"val": "#ecebe7", "test": "#dcdbd6"}


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
        "figure.dpi": 110,
    })


def _save(fig, out_dir, name):
    if out_dir is not None:
        p = Path(out_dir) / "figures"
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.png", dpi=150, bbox_inches="tight")


def _shade_splits(ax, split: pd.Series):
    v0 = split.index[split == "val"].min()
    t0 = split.index[split == "test"].min()
    end = split.index.max()
    for name, a, b in (("val", v0, t0), ("test", t0, end)):
        ax.axvspan(a, b, color=SPLIT_SHADE[name], zorder=0, lw=0)
        ax.text(a, 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=9, color=INK2)


def plot_series(h: pd.DataFrame, split: pd.Series, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    y = h[spec.target]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    _shade_splits(ax, split)
    ax.plot(y.index, y, color=SERIES[0], lw=0.4 if spec.freq == "h" else 1.4, label=spec.target)
    if spec.freq == "h":
        roll = y.rolling(24 * 30, min_periods=24 * 10).mean()       # srednia kroczaca wstecz (30 dni)
        ax.plot(roll.index, roll, color=SERIES[1], lw=1.6, label="średnia z ostatnich 30 dni")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    ax.set_title(f"{spec.name}: przebieg szeregu i podział chronologiczny")
    ax.set_ylabel(spec.unit)
    _save(fig, out_dir, f"{spec.key}_przebieg")
    return fig


def plot_zoom(y: pd.Series, start, periods: int, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    part = y.loc[pd.Timestamp(start):].iloc[:periods]
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(part.index, part, color=SERIES[0], lw=1.2, marker="o" if spec.freq != "h" else None, ms=3)
    ax.set_title(f"{spec.name}: fragment {part.index.min():%Y-%m-%d} – {part.index.max():%Y-%m-%d} (train)")
    ax.set_ylabel(spec.unit)
    _save(fig, out_dir, f"{spec.key}_fragment")
    return fig


def plot_hist(y_train: pd.Series, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(y_train.dropna(), bins=60 if spec.freq == "h" else 20, color=SERIES[0], edgecolor=SURFACE, lw=0.5)
    ax.set_title(f"{spec.name}: rozkład targetu (train)")
    ax.set_xlabel(spec.unit)
    ax.set_ylabel("liczba kroków")
    _save(fig, out_dir, f"{spec.key}_histogram")
    return fig


def plot_profiles(profiles: dict[str, pd.DataFrame], spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    labels = {"dzień tygodnia": ["pn", "wt", "śr", "cz", "pt", "so", "nd"]}
    fig, axes = plt.subplots(1, len(profiles), figsize=(4 * len(profiles), 3.2), squeeze=False)
    for ax, (name, p) in zip(axes[0], profiles.items()):
        ax.fill_between(p.index, p["q25"], p["q75"], color=SERIES[0], alpha=0.15, lw=0, label="kwartyle 25–75%")
        ax.plot(p.index, p["średnia"], color=SERIES[0], lw=2, marker="o", ms=3, label="średnia")
        ax.set_title(f"wg: {name}")
        ax.set_ylabel(spec.unit)
        if name in labels:
            ax.set_xticks(range(7), labels[name])
        elif name == "godzina":
            ax.set_xticks(range(0, 24, 3))
        else:
            ax.set_xticks(p.index)
    axes[0][0].legend(loc="best")
    fig.suptitle(f"{spec.name}: profile sezonowe (train)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_profile")
    return fig


def plot_acf(acf_vals: pd.Series, n: int, spec: DatasetSpec, title_suffix: str = "", out_dir=None, name="acf"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    lags = acf_vals.index[1:]
    ax.vlines(lags, 0, acf_vals.iloc[1:], color=SERIES[0], lw=0.8 if len(lags) > 60 else 2)
    conf = 1.96 / np.sqrt(n)
    ax.axhspan(-conf, conf, color=GRID, zorder=0, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    for s in spec.seasonal_candidates:
        if s < len(acf_vals):
            ax.plot([s], [acf_vals.iloc[s]], "o", color=SERIES[1], ms=6, zorder=3)
            ax.annotate(f"lag {s}: {acf_vals.iloc[s]:.2f}", (s, acf_vals.iloc[s]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9, color=INK)
    ax.set_ylim(top=max(1.15, acf_vals.iloc[1:].max() + 0.15))
    ax.set_title(f"{spec.name}: autokorelacja na zbiorze uczącym{title_suffix}")
    ax.set_xlabel(f"opóźnienie [{spec.step_word}]")
    ax.set_ylabel("ACF")
    _save(fig, out_dir, f"{spec.key}_{name}")
    return fig


def plot_missing(h_before_fill: pd.DataFrame, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    miss = h_before_fill[spec.target].isna().resample("MS").sum()
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.bar(miss.index, miss.to_numpy(), width=20, color=SERIES[0])
    ax.set_title(f"{spec.name}: brakujące godziny targetu w miesiącu (przed uzupełnieniem)")
    ax.set_ylabel("godziny")
    _save(fig, out_dir, f"{spec.key}_braki")
    return fig


def save_table(df: pd.DataFrame, out_dir, name: str, index: bool = True):
    p = Path(out_dir) / "tables"
    p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / f"{name}.csv", index=index)
    return df

# ---------------------------------------------------------------------------
# Charakterystyka rozszerzona (Raport 2): dekompozycja, stacjonarnosc, widmo,
# zmienne pomocnicze, rozklad w zbiorach. Wszystko liczone na zbiorze uczacym,
# z wyjatkiem porownania rozkladow train / val / test (opisowe).
# ---------------------------------------------------------------------------


def longest_segment(y: pd.Series) -> pd.Series:
    """Najdluzszy ciagly fragment bez brakow. Dekompozycja, testy stacjonarnosci i periodogram wymagaja
    szeregu bez luk; zamiast interpolowac dlugie przerwy (np. 307 dni w Metro) bierzemy najdluzszy fragment."""
    ok = y.notna().to_numpy().astype(int)
    edges = np.diff(np.concatenate([[0], ok, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    i = int(np.argmax(ends - starts))
    return y.iloc[starts[i]:ends[i]].astype(float)


def decompose(y_train: pd.Series, spec: DatasetSpec):
    """Dekompozycja STL/MSTL fragmentu zbioru uczacego: trend + skladowe sezonowe + reszta.
    Zbiory godzinowe: MSTL z okresami 24 i 168 h (cykl roczny trafia do trendu); Air Passengers: STL
    logarytmu z okresem 12 (sezonowosc multiplikatywna staje sie addytywna).
    Sila skladowej (Wang, Smith i Hyndman, 2006): F = max(0, 1 - Var(R) / Var(X + R)), X = trend lub sezon."""
    from statsmodels.tsa.seasonal import MSTL, STL
    seg = longest_segment(y_train)
    if spec.freq == "h":
        res = MSTL(seg, periods=(24, 168), stl_kwargs={"robust": True}).fit()
        seas = res.seasonal if isinstance(res.seasonal, pd.DataFrame) else res.seasonal.to_frame()
        seas.columns = [f"sezon {c.split('_')[-1]} h" for c in seas.columns]
        x = seg
    else:
        x = np.log(seg)
        res = STL(x, period=12, robust=True).fit()
        seas = res.seasonal.to_frame("sezon 12 mies.")
    comp = pd.concat([x.rename("szereg" if spec.freq == "h" else "log(szereg)"), res.trend.rename("trend"),
                      seas, res.resid.rename("reszta")], axis=1)
    R = comp["reszta"]
    strength = {"fragment": f"{seg.index[0]:%Y-%m-%d} – {seg.index[-1]:%Y-%m-%d}", "kroki": len(seg),
                "siła trendu F_T": round(float(max(0.0, 1 - R.var() / (comp["trend"] + R).var())), 3)}
    for c in seas.columns:
        strength[f"siła: {c}"] = round(float(max(0.0, 1 - R.var() / (comp[c] + R).var())), 3)
    strength["udział wariancji reszty"] = round(float(R.var() / comp.iloc[:, 0].var()), 3)
    return comp, strength


def plot_decomposition(comp: pd.DataFrame, spec: DatasetSpec, out_dir=None, last: int | None = None):
    """Skladowe dekompozycji; dla zbiorow godzinowych ostatnie `last` krokow fragmentu (np. 4 tygodnie)."""
    import matplotlib.pyplot as plt
    part = comp if last is None else comp.iloc[-last:]
    fig, axes = plt.subplots(len(part.columns), 1, figsize=(11, 1.6 * len(part.columns)), sharex=True)
    for ax, c in zip(axes, part.columns):
        ax.plot(part.index, part[c], color=SERIES[0] if c != "reszta" else INK2, lw=0.9)
        ax.set_ylabel(c, fontsize=9)
    rng = f"{part.index[0]:%Y-%m-%d} – {part.index[-1]:%Y-%m-%d}"
    axes[0].set_title(f"{spec.name}: dekompozycja {'MSTL' if spec.freq == 'h' else 'STL logarytmu'} ({rng}, train)")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_dekompozycja")
    return fig


def stationarity_table(y_train: pd.Series, spec: DatasetSpec, s: int) -> pd.DataFrame:
    """Testy ADF (H0: pierwiastek jednostkowy, czyli niestacjonarnosc) i KPSS (H0: stacjonarnosc wokol stalej)
    na najdluzszym ciaglym fragmencie zbioru uczacego: poziomy, pierwsze roznice i roznice sezonowe (lag s).
    Dla Air Passengers testujemy logarytm. Szereg uznajemy za stacjonarny, gdy ADF odrzuca H0 (p < 0,05),
    a KPSS jej nie odrzuca (p > 0,05)."""
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    seg = longest_segment(y_train)
    if spec.freq != "h":
        seg = np.log(seg)
    variants = {"poziomy": seg, "pierwsze różnice": seg.diff(), f"różnice sezonowe (lag {s})": seg.diff(s)}
    rows = []
    for name, x in variants.items():
        x = x.dropna().to_numpy()
        maxlag = None if spec.freq != "h" else 48
        adf = adfuller(x, maxlag=maxlag, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kp = kpss(x, regression="c", nlags="auto")
        ok = adf[1] < 0.05 and kp[1] > 0.05
        rows.append({"wariant": name, "n": len(x), "ADF statystyka": round(adf[0], 2), "ADF p": round(adf[1], 4),
                     "KPSS statystyka": round(kp[0], 3), "KPSS p": round(kp[1], 3),
                     "wniosek": "stacjonarny" if ok else ("niejednoznaczny" if adf[1] < 0.05 or kp[1] > 0.05
                                                          else "niestacjonarny")})
    return pd.DataFrame(rows)


def periodogram(y_train: pd.Series, spec: DatasetSpec, top: int = 6):
    """Periodogram najdluzszego ciaglego fragmentu zbioru uczacego (z usunietym trendem liniowym; Air:
    logarytm, aby sezonowosc byla addytywna). Rozpatrujemy okresy do polowy dlugosci fragmentu, wiec dla
    Jena Climate miesci sie tez cykl roczny. Zwraca widmo i najsilniejsze okresy (lokalne maksima)."""
    from scipy.signal import find_peaks, periodogram as pg
    seg = longest_segment(y_train)
    x = np.log(seg) if spec.freq != "h" else seg
    f, p = pg(x.to_numpy(), fs=1.0, detrend="linear", scaling="spectrum")
    f, p = f[1:], p[1:]
    period = 1 / f
    keep = period <= len(x) / 2
    period, p = period[keep], p[keep]
    pk, _ = find_peaks(p)
    best = pk[np.argsort(p[pk])[::-1][:top]]
    peaks = pd.DataFrame({"okres": np.round(period[best], 1), "udział mocy [%]": np.round(100 * p[best] / p.sum(), 2)})
    return pd.Series(p, index=pd.Index(period, name="okres")), peaks


def plot_periodogram(spec_series: pd.Series, peaks: pd.DataFrame, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3))
    s = spec_series.sort_index()
    ax.plot(s.index, s.values, color=SERIES[0], lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    marks = (12, 24, 168, 8766) if spec.freq == "h" else (3, 4, 6, 12)
    for m in marks:
        if s.index.min() <= m <= s.index.max():
            ax.axvline(m, color=SERIES[1], lw=0.8, ls="--")
            ax.text(m, 1.0, f" {m}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=SERIES[1])
    ax.set_xlabel(f"okres [{spec.step_word}]")
    ax.set_ylabel("moc widma")
    ax.set_title(f"{spec.name}: periodogram (train)")
    _save(fig, out_dir, f"{spec.key}_periodogram")
    return fig


def feature_correlations(h_train: pd.DataFrame, spec: DatasetSpec, max_cols: int = 16) -> pd.DataFrame:
    """Macierz korelacji Pearsona zmiennych liczbowych na zbiorze uczacym (redundancja zmiennych)."""
    cols = [c for c in h_train.columns
            if c != "coverage" and pd.api.types.is_numeric_dtype(h_train[c]) and h_train[c].nunique() > 2]
    cols = [spec.target] + [c for c in cols if c != spec.target][:max_cols - 1]
    return h_train[cols].corr().round(3)


def plot_corr_matrix(corr: pd.DataFrame, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.45 * n + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    ax.grid(False)
    if n <= 16:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{spec.name}: korelacje zmiennych (train)")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_korelacje_zmiennych")
    return fig


def plot_lagged_correlations(lc: pd.DataFrame, spec: DatasetSpec, out_dir=None, top: int = 10):
    """Korelacja zmiennej w chwili t z targetem w chwili t + h; najsilniejsze zmienne (bez samego targetu)."""
    import matplotlib.pyplot as plt
    other = lc.drop(index=spec.target)
    order = other.abs().max(axis=1).sort_values(ascending=False).index[:top]
    part = pd.concat([lc.loc[[spec.target]], other.loc[order]])
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(part) + 1.2))
    yy = np.arange(len(part))
    w = 0.8 / len(part.columns)
    for k, c in enumerate(part.columns):
        ax.barh(yy + k * w, part[c], height=w, color=SERIES[k % 3], label=c.replace("corr z ", ""))
    ax.set_yticks(yy + w, part.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("korelacja Pearsona")
    ax.legend(ncol=3, loc="lower right", fontsize=8)
    ax.set_title(f"{spec.name}: zmienna w chwili t a target w chwili t + h (train)")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_korelacje_opoznione")
    return fig


def split_distribution(h: pd.DataFrame, split: pd.Series, spec: DatasetSpec) -> pd.DataFrame:
    """Rozklad targetu w zbiorach (oryginalne jednostki) i odleglosc od zbioru uczacego: statystyka
    Kolmogorowa-Smirnowa (0 = identyczne rozklady, 1 = rozlaczne) i udzial wartosci poza zakresem train."""
    from scipy.stats import ks_2samp
    y = h[spec.target]
    tr = y[split == "train"].dropna()
    rows = []
    for name in SPLITS:
        x = y[split == name].dropna()
        q = x.quantile([0.05, 0.5, 0.95])
        rows.append({"zbiór": name, "n": len(x), "średnia": round(x.mean(), 3), "odch. std.": round(x.std(), 3),
                     "min": round(x.min(), 3), "kwantyl 5%": round(q[0.05], 3), "mediana": round(q[0.5], 3),
                     "kwantyl 95%": round(q[0.95], 3), "max": round(x.max(), 3),
                     "KS vs train": round(ks_2samp(tr, x).statistic, 3) if name != "train" else 0.0,
                     "poza zakresem train [%]": round(100 * ((x < tr.min()) | (x > tr.max())).mean(), 1)})
    return pd.DataFrame(rows)


def plot_split_distribution(h: pd.DataFrame, split: pd.Series, spec: DatasetSpec, out_dir=None):
    import matplotlib.pyplot as plt
    y = h[spec.target]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [y[split == s].dropna() for s in SPLITS]
    bins = np.histogram_bin_edges(y.dropna(), bins=50 if spec.freq == "h" else 15)
    for k, (s, x) in enumerate(zip(SPLITS, data)):
        ax.hist(x, bins=bins, density=True, histtype="step", lw=1.6, color=SERIES[k], label=s)
    ax.set_xlabel(spec.unit)
    ax.set_ylabel("gęstość")
    ax.legend()
    ax.set_title(f"{spec.name}: rozkład targetu w zbiorach train / val / test")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_rozklad_zbiorow")
    return fig


# ---------------------------------------------------------------------------
# Wartosci odstajace (Raport 2): IQR dla zmiennych i odchylenia od profilu sezonowego
# ---------------------------------------------------------------------------


def outlier_iqr_table(h_train: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Metoda IQR (Tukey): wartosc odstajaca lezy ponizej Q1 - 1,5 IQR albo powyzej Q3 + 1,5 IQR.
    Liczona na zbiorze uczacym dla zmiennych liczbowych o wiecej niz dwoch wartosciach (bez kolumn 0/1)."""
    cols = [c for c in h_train.columns
            if c != "coverage" and pd.api.types.is_numeric_dtype(h_train[c]) and h_train[c].nunique() > 2]
    cols = [spec.target] + [c for c in cols if c != spec.target]
    rows = []
    for c in cols:
        x = h_train[c].dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        nlo, nhi = int((x < lo).sum()), int((x > hi).sum())
        rows.append({"zmienna": c, "Q1": round(q1, 3), "Q3": round(q3, 3), "IQR": round(iqr, 3),
                     "dolna granica": round(lo, 3), "górna granica": round(hi, 3),
                     "poniżej": nlo, "powyżej": nhi, "odstające %": round(100 * (nlo + nhi) / len(x), 2),
                     "min": round(x.min(), 3), "max": round(x.max(), 3), "udział zer %": round(100 * (x == 0).mean(), 1)})
    return pd.DataFrame(rows)


def seasonal_anomalies(h_train: pd.DataFrame, spec: DatasetSpec, z: float = 4.0, top: int = 10, min_group: int = 10):
    """Anomalie wzgledem sezonowosci, wyznaczane WYLACZNIE na zbiorze uczacym (opisowo, nie jako cecha modelu).

    Zbiory godzinowe: profil = mediana targetu dla pary (miesiac, godzina) dla Jena i (dzien tygodnia, godzina)
    dla Household i Metro, na train; grupy z mniej niz
    min_group obserwacjami sa pomijane (brak wiarygodnego profilu). Air Passengers: reszta STL logarytmu na train;
    STL korzysta z sasiednich obserwacji (takze pozniejszych), wiec nadaje sie tylko do opisu EDA.
    Odporna skala: sigma_robust = 1,4826 * MAD odchylen; anomalia, gdy |odchylenie - mediana| > z * sigma_robust.
    Gdy MAD = 0 (np. wiele identycznych odchylen), stosujemy zastepczo 1,2533 * srednie odchylenie bezwzgledne;
    gdy i ono jest zerem, anomalii nie wyznaczamy."""
    y = h_train[spec.target].dropna().astype(float)
    if spec.freq == "h":
        # profil zgodny z sezonowoscia zbioru: Jena - miesiac x godzina (cykl roczny i dobowy, brak tygodniowego);
        # Household i Metro - dzien tygodnia x godzina (okres s = 168 h)
        first = y.index.month if spec.key == "jena" else y.index.dayofweek
        g = y.groupby([first, y.index.hour])
        expected = g.transform("median")
        size = g.transform("size")
        ok = size >= min_group
        y, expected = y[ok], expected[ok]
        dev = y - expected
    else:
        from statsmodels.tsa.seasonal import STL
        r = STL(np.log(y), period=12, robust=True).fit()
        expected = np.exp(r.trend + r.seasonal)
        dev = np.log(y) - (r.trend + r.seasonal)
    med = dev.median()
    mad = (dev - med).abs().median()
    scale, scale_name = 1.4826 * mad, "1,4826·MAD"
    if scale == 0:
        scale, scale_name = 1.2533 * (dev - med).abs().mean(), "1,2533·średnie odchylenie bezwzględne (MAD = 0)"
    rz = (dev - med) / scale if scale > 0 else dev * 0.0
    flag = rz.abs() > z
    summ = {"kroki": len(y), "skala": scale_name, "sigma_robust": round(float(scale), 4),
            f"anomalie (|z| > {z:g})": int(flag.sum()), "anomalie %": round(100 * flag.mean(), 2),
            "w górę": int((rz > z).sum()), "w dół": int((rz < -z).sum())}
    if spec.freq == "h":
        summ["dni z anomaliami"] = int(pd.Series(y.index[flag].normalize()).nunique())
    order = rz.abs().sort_values(ascending=False).index[:top]
    tops = pd.DataFrame({"czas": order, "dzień tygodnia": [["pn", "wt", "śr", "cz", "pt", "so", "nd"][t.dayofweek] for t in order],
                         "wartość": y[order].round(3).values,
                         "oczekiwana (profil)": np.asarray(expected[order]).round(3),
                         "odporny z": rz[order].round(1).values})
    if spec.key == "metro":               # zmienne ze zbioru, ktore pozwalaja sprawdzic mozliwy kontekst
        ctx = h_train.loc[order]
        tops["święto"] = ctx["holiday"].values
        for c in ("temp (degC)", "rain_1h", "snow_1h"):
            tops[c] = ctx[c].round(2).values
        wcols = [c for c in h_train.columns if c.startswith("w_")]
        tops["typ pogody"] = [", ".join(c[2:] for c in wcols if row[c] > 0) for _, row in ctx[wcols].iterrows()]
    return summ, tops, rz


def plot_outliers(h_train: pd.DataFrame, spec: DatasetSpec, rz: pd.Series, z: float = 4.0, out_dir=None):
    """Lewy panel: wykres pudelkowy standaryzowanych zmiennych liczbowych (train). Prawy panel: odporny z-score
    odchylenia targetu od profilu sezonowego w czasie, z progiem |z| = z."""
    import matplotlib.pyplot as plt
    cols = [c for c in h_train.columns
            if c != "coverage" and pd.api.types.is_numeric_dtype(h_train[c]) and h_train[c].nunique() > 2]
    cols = [spec.target] + [c for c in cols if c != spec.target]
    X = h_train[cols]
    Xs = (X - X.mean()) / X.std()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [max(2, len(cols)) * 0.5 + 1, 6]})
    ax = axes[0]
    ax.boxplot([Xs[c].dropna() for c in cols], vert=True, whis=1.5, widths=0.6, showfliers=True,
               flierprops=dict(marker=".", ms=2, alpha=0.3, color=SERIES[1]),
               medianprops=dict(color=SERIES[1]), boxprops=dict(color=SERIES[0]))
    ax.set_xticks(range(1, len(cols) + 1), [c.split(" (")[0] for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("wartość standaryzowana (train)")
    ax.set_title("Wykresy pudełkowe (IQR, wąsy 1,5·IQR)", fontsize=10)
    ax = axes[1]
    rzp = rz.reindex(h_train.index)            # luki pokazujemy jako przerwy w linii
    ax.plot(rzp.index, rzp.values, color=SERIES[0], lw=0.4 if spec.freq == "h" else 1.2)
    for s in (z, -z):
        ax.axhline(s, color=SERIES[1], lw=0.9, ls="--")
    big = rz[rz.abs() > z]
    ax.plot(big.index, big.values, "o", color=SERIES[1], ms=2.5)
    ax.set_ylabel("odporny z-score")
    ax.set_title("Odchylenie targetu od profilu sezonowego (train)" if spec.freq == "h"
                 else "Reszta STL logarytmu (train)", fontsize=10)
    fig.suptitle(f"{spec.name}: wartości odstające", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out_dir, f"{spec.key}_odstajace")
    return fig
