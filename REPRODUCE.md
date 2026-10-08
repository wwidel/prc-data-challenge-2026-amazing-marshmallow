# Reproducing the best submission (v62, leaderboard 251.2454)

We don't expect anyone to rerun this end to end. If you do, plan for **about two and a half days
of wall-clock time on one 8-vCPU, 32 GB machine**. Most of that is the ADS-B download and the two
CatBoost members. All timings below were measured on the machine every v62 member ran on: a Google
Compute Engine `c2-standard-8` (8 vCPUs, 31 GiB RAM, 4 physical cores).

For a result in minutes rather than days, see [the light tier](#the-light-tier) at the end.

## What you need

- **The competition data.** It is not in this repository and cannot be redistributed. The
  organisers provide 14 parquet files, which go unchanged into `data/raw/`:

  | file | rows |
  |---|---:|
  | `training_2025-MM-01_2025-MM+1-01.parquet` × 12 | 4,167,797 in total |
  | `ranking.parquet` | 689,534 |
  | `submitting.parquet` | 344,841 |

  `tests/test_pipeline.py::test_raw_files_are_complete` checks each file's row count.
- **Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).** `uv.lock` pins the exact versions
  the members were fitted with: polars, LightGBM, CatBoost, scikit-learn, scipy and the rest.
- **Disk.** About 6 GB for the ADS-B cut, plus room for each day's raw archive while it is cut.
  Roughly 1 GB more for the feature store and the predictions.
- **Memory.** 32 GB. The workstation we developed on had 3 GB usable, and it could not run the
  feature builds or the fits.

## The steps

`make help` lists the targets. In order:

| step | command | what it does | time on our VM |
|---|---|---|---|
| 1 | `make setup` | installs the locked environment | minutes |
| 2 | `make external` | METAR (IEM), stand and runway tables (X-Plane Scenery Gateway, OpenStreetMap), EUROCONTROL series, OPDI flight lists | minutes |
| 3 | `make adsb` | adsb.lol traces for 427 days (all of 2025, January and July 2026), cut to the ten airports' boxes | **~27 h**, resumable; 6.0 GB kept |
| 4 | `make spine` | the row spine every block is aligned to | ~1 min |
| 5 | `make features` | every cacheable feature block (the target-dependent ones are computed per fold during steps 6–7) | not timed separately |
| 6 | `make cv` | cross-validates the nine configs on the Jan and Jul 2025 folds; writes `data/predictions/` | **~22 h** as logged (table below) |
| 7 | `make refit` | refits the nine on all of 2025 and predicts the 2026 rows; writes `data/rank_preds/` | **~8 h** |
| 8 | `make overrides` | writes the four override files (a050, a060, a111, a125) | minutes |
| 9 | `make stack` | builds `data/output/v62.parquet` | minutes |
| 10 | `make verify` | independent checks: ids, dtypes, bounds, every override row landed | seconds |

Cross-validation and refit time per config, as logged on the VM:

| config | model | CV (two folds) | refit |
|---|---|---:|---:|
| `t05_huber_alpha[alpha=600]` | LightGBM, Huber loss | 0.96 h | not timed |
| `f25_queue2` | LightGBM | 0.17 h | not timed |
| `q04_c12_min_leaf30` | LightGBM | 0.47 h | ~17 min |
| `ab07_ab02_adsb_calib` | LightGBM | 0.73 h | ~23 min |
| `rb01_lirf_route_ab02` | LightGBM, routed by airport | 2.00 h | ~65 min |
| `cb03_k20_combo_adsb` | CatBoost | 8.07 h* | 2 h 54 min |
| `cb04_cb03_adsb_calib` | CatBoost | 9.01 h* | 2 h 57 min |
| `r14_rules_plus_unmatched_route` | rules + routed LightGBM (Rome layer) | 0.30 h | not timed |
| `lg01_lirf_gate_delay` | CatBoost on LIRF only (Rome override) | 0.07 h | not timed |

\* Both CatBoost runs were paused for part of that time while another job used the machine. Each
needs about 5–6 h of compute on its own. The runs were sequential; jobs running side by side on 8
vCPUs (4 physical cores) slowed each other down about 2.5×.

**Run the commands as given, without `--profile`.** The v62 members were fitted with LightGBM at its
default of 4 threads (`taxiout.models.lgbm.DEFAULTS`) and CatBoost at the 8 threads its configs pin.
LightGBM's result can change in the last digits with the thread count.

## What to check along the way

`make stack` prints the NNLS weights it fits on the out-of-fold rows. v62's were:

| member | NM-matched rows (312,922) | NM-unmatched rows (4,969) |
|---|---:|---:|
| `t05_huber_alpha[alpha=600]` | 0.0171 | 0 |
| `f25_queue2` | 0 | 0.0013 |
| `q04_c12_min_leaf30` | 0 | 0.0123 |
| `rb01_lirf_route_ab02` | 0.2857 | 0.2960 |
| `ab07_ab02_adsb_calib` | 0.2251 | 0 |
| `cb03_k20_combo_adsb` | 0.1807 | 0.6716 |
| `cb04_cb03_adsb_calib` | 0.2930 | 0.0684 |

The stack v62 was actually built from had twenty members. Thirteen of them got a weight of exactly
zero in both groups, so this repository leaves them out. A non-negative least-squares solution
stays the solution when the columns it gives zero weight are removed, so the seven weights don't
change. The Rome matched-row blend (`a111`) works the same way: it should print r14 0.8946 and ab07
0.1285, fitted on 26,131 rows.

The finished file has 344,841 rows and a maximum of 111,654 s. `make verify` fails if any override
row did not land. Our v62 file's sha256 starts with `bff918f7` and ends with `a4802b2c`.

## Why a rebuild will be close, not byte-identical

- **Live sources.** adsb.lol, OpenStreetMap, the Scenery Gateway's recommended packs and
  EUROCONTROL's 2026 files are maintained datasets, and any of them can change after our pull.
  The METAR archive and the 2025 OPDI flight lists are the most stable of them.
- **The cell shift is a fixed parameter here.** `config/cellshift_v54.csv` holds 11 airport-month
  shifts that `scripts/analysis/a120_cell_shift.py` computed on an earlier build, v54, whose members
  are not all in this repository. v62 applies them unchanged. You can point a120 at your own build
  (`make stack-noshift`, then `A120_PRED=data/output/v62_noshift.parquet`). It will print your
  build's estimated per-cell errors, not v54's.
- **Feature caches do not see external files.** A block's cache key hashes its code, not the files
  it reads, so after re-pulling a source rebuild its blocks with
  `uv run python scripts/build_features.py --overwrite <block>`.
- **Numerics.** The members were fitted on one CPU type at fixed seeds. Another CPU, BLAS or thread
  count can move individual predictions by fractions of a second.

## The light tier

`scripts/lite.py` fits about thirty numbers. Per airport, it runs a least squares of the label on
`takeoff − aobt_3` and `takeoff − eobt_1`. Rows missing either stamp fall back to stand and airport
medians. It then applies the full build's two exact rules, its bounds and the LOBT window. It needs
only `data/raw/` and the spine, and runs in minutes on a laptop with enough memory for the 2.4M-row
spine:

```bash
make setup && make spine && make lite      # writes data/output/lite.parquet
```

We measured this variant in our working repository (`a104`, variant "S0b") on the same Jan + Jul
2025 folds:

| | light tier | our stack at the time (v39: 13 members + the r14 Rome layer) |
|---|---:|---:|
| trimmed RMSE without Rome (the number we steered by) | 297.84 | 216.53 |
| untrimmed RMSE, Rome only | 1,035.1 | 625.1 |
| untrimmed RMSE, all rows | 440.49 | 320.82 |

v39 scored 276.79 on the leaderboard and v62 251.25, so the full stack's lead over the light tier
grew after this comparison was made. The light tier was never submitted and has no leaderboard
score. `scripts/lite.py` is a stand-alone extraction of that code. Its logic is tested on synthetic
data (`tests/test_lite.py`), but this extraction itself has not been run on the competition data.
