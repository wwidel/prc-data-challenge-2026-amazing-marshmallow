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
- **An internet connection, `curl` and `bzip2`.** Every open source is downloaded by a script in
  `scripts/pull_*.py` (`make external`, `make adsb`); three of them call `curl`, and the
  EUROCONTROL delay-by-cause files are decompressed with `bzcat`. Nothing is downloaded by hand.
- **Disk.** About 7 GB for the ADS-B cut (485 days), plus room for each day's raw archive while it
  is cut; 0.7 GB for the 17 OPDI flight lists; under 0.1 GB for METAR, the Gateway and OSM stands
  and the EUROCONTROL files. Roughly 1 GB more for the feature store and the predictions.
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
- **Numerics.** The members are fitted at fixed seeds, but a tree learner depends on the order of
  floating-point sums: another CPU, BLAS or thread count can change a split, and every prediction
  below that split moves with it. We did not measure this alone. Between our two phases, refits
  of the same configs differed by about 31 s RMS on the leaderboard rows, through changed inputs
  and numerics together (`a132`, `a133`).

## The final phase

The organisers added a final phase after the leaderboard closed: one submission predicting
`final_ranking.parquet`, which holds January, February, June and July 2026 (670,790 departures).
They rank Jan/Jul, Feb/Jun and all four months. The template is `final_submitting.parquet`. Put
both files in `data/raw/`, then run everything with `PHASE=final`, which sets
`TAXIOUT_PHASE=final`:

```bash
make PHASE=final spine features cv refit overrides stack-noshift cellshift stack verify
```

What changes in that phase:

- **Inputs.** The final file and its template replace the leaderboard's, and the scored months
  become `2026-01, 2026-02, 2026-06, 2026-07`. Every 2026 output gets its own directory
  (`data/*_final/`), so the two phases never mix in one checkout. `make adsb` and `make external`
  already pull February and June 2026.
- **Folds.** CV holds out January + July and February + June 2025, each trained on the other ten
  months (fold set `final_pairs`). That mirrors the organisers' rankings; the four-month score is
  the two folds pooled. `scripts/analysis/a129_stack_cv.py` scores the stack on them with the
  weights fitted on the other pair, and `a130_member_months.py` scores single members per month.
- **The organisers' randomness.** In the final file every takeoff time, and every arrival's
  in-block time, moved by an integer number of seconds in [−12, 12]; nothing else changed
  (`a126_final_randomness.py`). The LOBT window, the schedule-gap bound and the LEBL/LEMD ceilings
  widen by that much (`paths.TIME_NOISE`).
- **Overrides become rules.** The four override files are regenerated from the final file by the
  same scripts, and a120 computes the cell shift for all four months against the build without
  the shift (`make PHASE=final stack-noshift cellshift`).
- **The final submission's members and stack.** _To be filled in when the final submission is
  chosen: its composition file in `config/compositions/`, and `--unmatched-without-month 02`
  in the stack flags (the NM-unmatched vector for January, June and July fitted without February
  2025; [README](README.md#the-final-phase))._
- **A CatBoost CV can run one fold per job.** The fold sets `final_pairs_janjul` and
  `final_pairs_febjun` are the two folds of `final_pairs`, so a long CV can run as two jobs; a
  spot preemption then costs one fold, not two. `scripts/merge_oof.py <name>` joins the two
  out-of-fold files, and the stacker and the analyses then read the merged one:

  ```bash
  TAXIOUT_PHASE=final uv run python scripts/run_experiments.py --folds final_pairs_janjul config/experiments/cb04_cb03_adsb_calib.yaml
  TAXIOUT_PHASE=final uv run python scripts/run_experiments.py --folds final_pairs_febjun config/experiments/cb04_cb03_adsb_calib.yaml
  TAXIOUT_PHASE=final uv run python scripts/merge_oof.py cb04_cb03_adsb_calib
  ```

**How long it took us.** On a GCE n2-standard-32 (32 vCPUs, 128 GB, spot), running four CatBoost
jobs of 8 threads side by side, or LightGBM jobs of 4 threads: a CatBoost fold took 2.3–2.9 h at
3,000 iterations and about 3.6 h at 4,500; a CatBoost twelve-month refit 2.8–3.4 h at 3,000
iterations. A LightGBM CV (both folds) took 16–72 min, the routed rb01 about 1.6 h per fold; a
LightGBM refit 7–45 min, rb01's about 3 h. The four override scripts, the two stack builds, a120 and the
verification take a few minutes together.

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
score. `scripts/lite.py` is a stand-alone extraction of that code; run on the competition data
(2026-10-08) it reproduces a104's numbers above exactly.
