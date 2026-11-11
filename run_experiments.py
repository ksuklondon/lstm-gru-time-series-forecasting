"""Uruchamianie treningow Projektu 2A z terminala (z wznawianiem).

Przyklady:
    python run_experiments.py --blok A              # porownanie podstawowe (obowiazkowe)
    python run_experiments.py --blok A B C D        # wszystkie bloki po kolei
    python run_experiments.py --blok E              # rozszerzenie: TCN na Jena Climate (9 treningow)
    python run_experiments.py --blok A --zbior air  # tylko jeden zbior
    python run_experiments.py --plan                # tylko lista i liczba treningow, bez uczenia
    python run_experiments.py --pomiar              # pomiar czasu epoki na tym komputerze + szacunek czasu blokow
    python run_experiments.py --blok A --szybki     # przebieg testowy kodu (2 epoki, osobny folder)
    python run_experiments.py --zamroz              # zamrozenie konfiguracji na walidacji (final_configs.json)
    python run_experiments.py --test --potwierdzam-test   # jednorazowa ocena testowa (Raport 4)

Kazdy zakonczony trening jest od razu dopisywany do outputs/eksperymenty/wyniki.csv. Po przerwaniu
(Ctrl+C, uspienie, brak pradu) wystarczy uruchomic to samo polecenie ponownie: gotowe treningi sa pomijane.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging

import pandas as pd

import ts_models as tm

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec // 60} min {sec % 60:02d} s"


BASE_COLS = ["run_id", "block", "key", "model", "h", "L", "features", "variant", "params", "train_time_s",
             "n_train", "n_val", "val_MAE", "val_MSE", "val_RMSE", "val_MAPE", "val_sMAPE", "val_skill"]


def append_row(path: Path, row: dict, cols=None):
    """Dopisuje wiersz z kolumnami w stalej kolejnosci (naglowek pliku albo cols)."""
    if path.exists():
        cols = list(pd.read_csv(path, nrows=0).columns)
    df = pd.DataFrame([row])
    if cols:
        df = df.reindex(columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_baselines(block, out, only):
    path = out / "modele_bazowe_val.csv"
    done = set(pd.read_csv(path).run_id) if path.exists() else set()
    for key, h, L, feats, var in tm.baseline_cells(block):
        if only and key not in only:
            continue
        rid = f"{key}_linreg_h{h}_L{L}_{feats}_{var}"
        if rid in done:
            continue
        rows = tm.baselines_one(key, h, L, feats, var, out)
        for r in rows:
            if r["run_id"] not in done:
                append_row(path, r, BASE_COLS)
                done.add(r["run_id"])


def benchmark(out: Path, epochs_assumed: int = 40):
    """Pomiar czasu epoki na tym komputerze (3 epoki na pelnych danych, bez zapisu wynikow) i szacunek
    czasu calych blokow. Zalozenie: srednio epochs_assumed epok do wczesnego zatrzymania."""
    import keras
    import numpy as np

    class EpochTimer(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.times = []

        def on_epoch_begin(self, epoch, logs=None):
            self.t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.perf_counter() - self.t)

    rows = []
    for key in ("jena", "household", "metro"):
        for model, L in (("Dense", 72), ("LSTM", 72), ("GRU", 72), ("LSTM", 168), ("GRU", 168)):
            arr = tm.build_arrays(key, "uni", "std", 1)
            W = tm.windows(arr, L, 1)
            Xtr, ytr, _ = W["train"]
            Xva, yva, _ = W["val"]
            keras.utils.set_random_seed(42)
            m = tm.build_model(model, L, 1)
            cb = EpochTimer()
            m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=3, batch_size=256, verbose=0, callbacks=[cb])
            s = float(np.mean(cb.times[1:]))
            rows.append({"key": key, "model": model, "L": L, "n_train": len(ytr), "s_per_epoch": round(s, 2)})
            print(f"{key:10s} {model:5s} L={L:3d}: {s:6.2f} s/epokę ({len(ytr)} okien uczących)", flush=True)
            keras.backend.clear_session()
    bench = pd.DataFrame(rows)
    bench.to_csv(out / "pomiar_czasu.csv", index=False)

    def per_epoch(key, model, L):
        if key == "air":
            return 0.1
        b = bench[(bench.key == key) & (bench.model == model)]
        x = b.set_index("L").s_per_epoch
        if L in x.index:
            return x[L]
        if model == "Dense":
            return x.iloc[0]
        return float(np.interp(L, [72, 168], [x.get(72, x.iloc[0]), x.get(168, x.iloc[-1])]) * (1 if L >= 72 else L / 72))

    print(f"\nSzacunek (średnio {epochs_assumed} epok na trening):")
    total = 0
    for b in "ABCD":
        sec = sum(per_epoch(r.key, r.model, r.L) * epochs_assumed * (2 if r.layers == 2 or r.units == 128 else 1)
                  for r in tm.registry(b))
        total += sec
        print(f"  Blok {b}: ok. {fmt(sec)}")
    print(f"  Razem: ok. {fmt(total)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blok", nargs="*", default=[], choices=list("ABCDE"))
    ap.add_argument("--zbior", nargs="*", default=[], choices=list(tm.KEYS))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--szybki", action="store_true")
    ap.add_argument("--zamroz", "--zamroz-A", dest="freeze", action="store_true",
                    help="zamrozenie konfiguracji na podstawie walidacji (final_configs.json)")
    ap.add_argument("--test", "--test-A", dest="test", action="store_true",
                    help="ocena na zbiorze testowym (Raport 4); wymaga dodatkowo --potwierdzam-test")
    ap.add_argument("--potwierdzam-test", dest="confirm_test", action="store_true")
    ap.add_argument("--nadpisz-konfiguracje", dest="overwrite_cfg", action="store_true",
                    help="ponowne zamrozenie mimo istniejacego final_configs.json (tylko przed ocena testowa)")
    ap.add_argument("--pomiar", action="store_true", help="pomiar czasu epoki i szacunek czasu bloków")
    ap.add_argument("--dane", default="data")
    args = ap.parse_args()

    out = Path("outputs/eksperymenty_szybki" if args.szybki else tm.OUT)
    out.mkdir(parents=True, exist_ok=True)
    tm.prepare.__defaults__ = (args.dane,)

    if args.pomiar:
        print("Środowisko:", tm.environment())
        benchmark(out)
        return

    if args.plan or not (args.blok or args.freeze or args.test):
        for b in "ABCDE":
            runs = tm.registry(b)
            per = pd.Series([r.key for r in runs]).value_counts().to_dict()
            print(f"Blok {b}: {len(runs)} treningów sieci {per}")
        return

    results_path = out / "wyniki.csv"
    if args.blok:
        env = tm.environment()
        print("Środowisko:", env)
        pd.Series(env).to_json(out / "srodowisko.json", force_ascii=False, indent=1)
    for block in args.blok:
        runs = [r for r in tm.registry(block) if not args.zbior or r.key in args.zbior]
        done = tm.completed_runs(out)             # tylko przebiegi z wynikiem, plikami i ta sama konfiguracja
        todo = [r for r in runs if not tm.is_done(r, done)]
        print(f"\n=== Blok {block}: {len(runs)} treningów, gotowe {len(runs) - len(todo)}, do zrobienia {len(todo)} ===")
        print("Modele bazowe (persistence, seasonal naive, regresja liniowa) ...")
        run_baselines(block, out, args.zbior)
        t_block = time.time()
        for i, run in enumerate(todo, 1):
            t0 = time.time()
            tm.mark_status(out, run, "started")
            try:
                row = tm.train_one(run, out, quick=args.szybki)
            except KeyboardInterrupt:
                tm.mark_status(out, run, "interrupted")
                print("\nPrzerwano. Uruchom to samo polecenie ponownie, aby kontynuować.")
                sys.exit(1)
            except Exception as e:                # blad jednego przebiegu nie zatrzymuje calego bloku
                tm.mark_status(out, run, "failed", repr(e))
                print(f"[{block} {i}/{len(todo)}] {run.run_id}: BŁĄD {e!r} (zapisany w status.csv)", flush=True)
                continue
            append_row(results_path, row)          # wynik dopisujemy dopiero po zapisaniu wszystkich plikow
            tm.mark_status(out, run, "done")
            el = time.time() - t_block
            eta = el / i * (len(todo) - i)
            print(f"[{block} {i}/{len(todo)}] {run.run_id}: {fmt(time.time() - t0)}, epoki {row['epochs']}, "
                  f"val MAE {row['val_MAE']:.4f}, skill {row['val_skill']:+.3f} | pozostało ok. {fmt(eta)}", flush=True)
        print(f"Blok {block} zakończony.")
    if args.freeze:
        if (out / tm.CONFIG_FILE).exists() and not args.overwrite_cfg:
            print(f"{out / tm.CONFIG_FILE} już istnieje: konfiguracje są zamrożone i nie zostaną nadpisane.")
            return
        cfg = tm.freeze_configs(out, overwrite=args.overwrite_cfg)
        bad = [c for c in cfg["komorki"] if not c["kompletne"]]
        print(f"Zamrożono {len(cfg['komorki'])} konfiguracji ({out / tm.CONFIG_FILE})."
              + (f" UWAGA: niekompletne: {len(bad)}" if bad else ""))
    if args.test:
        if not args.confirm_test:
            print("Ocena na zbiorze testowym jest zaplanowana dopiero w Raporcie 4. "
                  "Aby ją wykonać, dodaj --potwierdzam-test.")
            return
        if not (out / tm.CONFIG_FILE).exists():
            print("Brak final_configs.json: najpierw zamroź konfiguracje (--zamroz).")
            return
        if (out / "wyniki_test.csv").exists():
            print(f"{out / 'wyniki_test.csv'} już istnieje: ocena testowa została wykonana i nie zostanie powtórzona.")
            return
        res = tm.evaluate_test(tm.CONFIG_FILE, out, confirm=True)
        print(f"Ocena testowa: {len(res)} wierszy -> {out / 'wyniki_test.csv'}")

if __name__ == "__main__":
    main()
