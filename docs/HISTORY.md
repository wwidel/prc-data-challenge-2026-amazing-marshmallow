# History: every submission, finding and experiment

Our project as it happened, in three indexes. Each line is a pointer, not a write-up: the full
texts are in [`SUBMISSIONS.md`](SUBMISSIONS.md) (every upload, verbatim) and
[`archive/LEARNINGS_LOG.md`](archive/LEARNINGS_LOG.md) (every finding, verbatim). The
[README](../README.md) is the digest: [where the score came from](../README.md#where-the-score-came-from),
[what did not work](../README.md#what-did-not-work) and [the final phase](../README.md#the-final-phase).

The working records cite analysis scripts (`aNNN`) and configs that are not all in this
repository: we shipped the ones behind the README's claims and the two builds. Cross-validation
scores of single experiments were kept in a run log on our build machine and are not reproduced
here; those that mattered are in the submission rows and the findings.

- [1. Submissions](#1-submissions)
- [2. Findings](#2-findings)
- [3. Experiments](#3-experiments)

## 1. Submissions

Every row of the log: v1 to v66 on the leaderboard (January and July 2026), then the final-phase
file. Leaderboard RMSE in seconds, as logged; v4, v20 and v22 were never uploaded. v63 onwards
are final-phase builds, scored on their January and July half. Full rows with every CV figure:
[`SUBMISSIONS.md`](SUBMISSIONS.md).

| v | date | build | leaderboard | what it was |
|---|---|---|---:|---|
| 1 | 2026-09-23 | `b05_match_aware_groups` | **604.41** | best CV baseline; match-aware grouping. |
| 2 | 2026-09-23 | `b02_group_stand_runway` | **629.22** | control for v1: identical but blind to the NM match flag |
| 3 | 2026-09-23 | `x02_lgbm_deep` | **433.80** | deep LightGBM: 511 leaves, 511 bins, lr 0.03 |
| 5 | 2026-09-24 | `x07_deep_per_airport` | **430.38** | per-airport deep. |
| 6 | 2026-09-24 | `x1_maxbin_2047` | **437.04** | **CV pooled missed by +46.45.** Best pooled CV ever recorded, worst board score here. |
| 7 | 2026-09-24 | `f03_both` | **428.02** | schedule + congestion features. |
| 8 | 2026-09-24 | `r10_f03_rules_D` | **427.71** | f03_both + rule D — **but the upload cap clipped every rule value (≥60,049) to 20,000, so rule D never reached the board; L023 is void (L033)** |
| 9 | 2026-09-24 | `c10_trust_aobt` | **425.96** | modes TD block: per-cell P(block=aobt actual), P(block=sched); best ex-LIRF 293.32; no schedule block |
| 10 | 2026-09-25 | `stack_et_members` | **422.76** | ExtraTrees stack over 16 refitted members (a031/a036), LIRF rows verbatim from v9 |
| 11 | 2026-09-25 | `stack_et_members` | **422.60** | ExtraTrees stack over 16 refitted members (a031/a036), LIRF rows verbatim from v9 |
| 12 | 2026-09-25 | `probe_LIRF_unmatched` | **430.38** | P-measure probe: v11 +2000s on LIRF unmatched (n=383). |
| 13 | 2026-09-25 | `probe_LIRF_unmatched_2026-01` | **424.42** | P-measure probe: v11 +2000s on LIRF unmatched Jan only (n=107). |
| 14 | 2026-09-25 | `probe_LIRF_unmatched_2026-01_ids` | _rejected_ | Shift probe on the 69 trace-covered Jan unmatched rows. |
| 15 | 2026-09-25 | `stack_et_members` | **421.66** | ExtraTrees stack over 18 members (f15/f16 weather joined; a038 cross-fold 288.60), LIRF rows verbatim from v9 |
| 16 | 2026-09-26 | `pop27_mixture` | **326.60** | v15 + a050 mixture values on the 27 uncapped pop27 rows (gap; 0.593·gap+0.407·87,500 on the 8 day-crossers). |
| 17 | 2026-09-26 | `pop27_hedge074` | **334.87** | v15 + the 0.74-hedge on pop27. |
| 18 | 2026-09-26 | `pop27_gap` | **319.86** | v15 + pure gap on all 27 pop27 rows — the L035-informed predictor. |
| 19 | 2026-09-26 | `subcap_hedge` | **317.72** | v18 + a051's calibrated copy-mode hedge on the 348 sub-cap unmatched Rome rows (LOMO AUC 0.863; monthly level = probe means). |
| 20 | 2026-09-26 | `stack_et_members` | _burned_ | 18-member stack v2 (+f17a geometry, −n15; cross-fold 288.30). |
| 21 | 2026-09-26 | `rome_overrides` | _superseded_ | v20 + Rome overrides; hit the daily limit, then superseded by v23 (better stack) before its rescheduled upload. |
| 22 | 2026-09-26 | `stack_et_members` | _burned_ | 20-member stack (+t02_cap7200, +f21_geometry3). |
| 23 | 2026-09-27 | `rome_overrides` | **316.59** | v22 (20-member stack: +t02_cap7200, +f21_geometry3, −n15) + pop27 gaps + sub-cap hedge. |
| 24 | 2026-09-27 | `aobt_identity_1row` | **290.91** | v23 with ONE row (`mvt_id 213147416`, LIRF, matched, `takeoff − aobt_3` = 86,340) set to that proxy instead of 13,056. |
| 25 | 2026-09-27 | `capped_stack_26` | **287.26** | 26-member capped stack + all overrides + bounds. |
| 26 | 2026-09-27 | `trimmed_stack_5` | **285.67** | **Five members only** (CatBoost-capped, LGBM-capped, linear-tree-capped, Huber-600, per-airport-capped) — one per function class, replacing 26. |
| 27 | 2026-09-27 | `trimmed_stack_5_nnls` | **283.98** | v26's five members with the **NNLS combiner** replacing ExtraTrees (400 trees, depth 6). |
| 28 | 2026-09-28 | `trimmed_stack_6_nnls` | **283.41** | v27's members **plus `f25_queue2` and `h06_mlp`**, both of which earned NNLS seats in `a081` (weights 0.211 and 0.092, together −0.725 to 219.181). |
| 29 | 2026-09-28 | `trimmed_stack_8_nnls` | **281.98** | v28's six members **plus `k05_catboost_depth10` and `x35_lo_card_l2_grid[lambda_l2=50.0]`**, both of which earned NNLS seats in `a081` cross-fold (−0.806 and −0.546 singly, **−1.169 together** → 218.012). |
| 30 | 2026-09-29 | `trimmed_stack_8_nnls_rome_r14` | **281.10** | v29 with **layer B swapped** from `f21_geometry3` to `r14_rules_plus_unmatched_route` — one flag, `--rome-from`. |
| 31 | 2026-09-29 | `trimmed_stack_10_nnls_rome_r14` | **280.17** | v30 with **`d05_catboost_modes5` and `x36_lo_card_modes5` added** to the member list. |
| 32 | 2026-09-30 | `trimmed_stack_10_nnls_rome_r14_noD` | **278.10** | v31 with **layer D (`subcap_hedge.parquet`) dropped** — PLAN §5 row 0c; one flag fewer, no refit. |
| 33 | 2026-10-01 | `trimmed_stack_11_nnls_rome_r14_noD` | **278.00** | v32 with **`d06_catboost_modes5_cap14400` added** (`d05`'s recipe at label cap 14,400) — the owner's queue item 4. |
| 34 | 2026-10-01 | `trimmed_stack_12_nnls_rome_r14_noD` | **278.11** | v33 with **`d07_catboost_depth12_modes5` added** — the owner's queue item 5. |
| 35 | 2026-10-01 | `trimmed_stack_12_nnls_rome_r14_noD` | **277.73** | **v33** (not v34) with **`h09_x36_modes6` added** — `x36` with the two-stage `modes6` classifier in place of `modes5`. |
| 36 | 2026-10-01 | `trimmed_stack_12_nnls_rome_r14_noD_cal2400` | **277.79** | v35 with **`--calibrate-above 2400`** (PLAN §5 item 1a): stacked body predictions above 2,400 s × **1.0248**, the scalar fitted on 2,124 out-of-fold rows above the same threshold. |
| 37 | 2026-10-01 | `trimmed_stack_12_nnls_grouped_unmatched_rome_r14_noD` | **276.83** | v35 with **`--weight-groups unmatched`** (PLAN §5 item 1b): one NNLS weight vector for NM-matched rows and one for unmatched, instead of one global. |
| 38 | 2026-10-02 | `trimmed_stack_12_nnls_grouped_unmatched_only_rome_r14_noD` | **276.88** | v35 with **`--weight-groups unmatched --global-for matched`**: v35's global vector on matched rows, v37's unmatched vector on the 5,283 unmatched ranking rows. |
| 39 | 2026-10-02 | `trimmed_stack_13_nnls_grouped_unmatched_rome_r14_noD` | **276.79** | v37 + **`d10_catboost_modes6`** (`docs/compositions/v39_members.txt`, `--weight-groups unmatched`). |
| 40 | 2026-10-02 | `trimmed_stack_13_nnls_grouped_unmatched_calunm_rome_r14_noD` | **277.00** | v39 with **`--calibrate-unmatched`**: unmatched rows' stacked predictions × **0.9769**, the least-squares scalar on the 4,967 unmatched OOF rows with labels ≤ 12 h (untrimmed it is exactly 1.0000: the group's own NNLS … |
| 41 | 2026-10-03 | `trimmed_stack_13_nnls_grouped_unmatched_rome_r14_noD_unm12` | **276.66** | v39 + **`--overrides data/external/unmatched12.parquet`**: the 4,900 non-LIRF unmatched ranking rows re-predicted with an NNLS vector fitted on **twelve months** of unmatched OOF rows (20,937 vs 4,969) over six members … |
| 42 | 2026-10-03 | `trimmed_stack_13_nnls_grouped_unmatched_rome_r14blend_noD_unm12` | **276.30** | v41 + **`--overrides data/external/rome_matched_blend.parquet`** (first in override order): the matched LIRF ranking rows take a Rome-only NNLS blend fitted on 26,131 matched LIRF OOF rows — **r14 0.9189, d06 0.0977, … |
| 43 | 2026-10-04 | `trimmed_stack_14_nnls_grouped_unmatched_rome_r14blend_noD_unm12` | **276.26** | v42's exact command with **`q04_c12_min_leaf30` added** (`c12`'s recipe — x36 + arrival_regime on the fixed calendar — at min_data_in_leaf 30; CV 219.002). |
| 44 | 2026-10-04 | `trimmed_stack_14_nnls_grouped_unmatched_rome_r14blend_noD_unm12v43_lobt` | **275.49** | v43's command with **`--lobt-window`** (project every prediction onto `mvt − lobt ± 3606 s`, last; L060, `docs/BORROWED_FROM_OUTGOING_BOAT.md` item 1) and `unmatched12_v43.parquet` in place of `unmatched12.parquet`. |
| 45 | 2026-10-05 | `trimmed_stack_15_nnls_grouped_unmatched_rome_r14blend_noD_unm12v43_lobt` | **275.40** | v44's command with **`k20_d10_border1024` added** (`d10`'s recipe at border_count 1024, fixed calendar; CV 217.111 vs `d10` 217.72). |
| 46 | 2026-10-05 | `trimmed_stack_16_nnls_grouped_unmatched_rome_r14blend_noD_unm12v43_lobt` | **272.97** | v45's command with **`cb01_q04_combo` added** (`q04` + context_window + gate_retention + eurocontrol_daily — three outgoing-boat ideas, `docs/BORROWED_FROM_OUTGOING_BOAT.md` items 4, 6, 7; CV 213.544, −5.46 vs `q04`). |
| 47 | 2026-10-05 | `trimmed_stack_17_nnls_grouped_unmatched_rome_r14blend_noD_unm12v43_lobt` | **260.38** | v46's command with **`ab02_cb01_adsb` added** — `cb01` + the **adsb.lol ADS-B block** (each departure's own ground segment and takeoff; outgoing-boat idea, `docs/BORROWED_FROM_OUTGOING_BOAT.md` item 3), full 427-day … |
| 48 | 2026-10-06 | `trimmed_stack_17_nnls_grouped_unmatched_rome_r14blend47_noD_unm12v43_lobt` | **260.02** | v47's command with the **Rome matched-row blend refitted over v47's members** (`a110`/`a111` with `A110_COMPOSITION=v47`: r14 + `ab02` 0.1273 + …; cross-fold grouped NNLS on LIRF OOF rows trimmed 551.7 vs r14 verbatim … |
| 49 | 2026-10-06 | `trimmed_stack_17_nnls_grouped_unmatched_rome_r14blend47_noD_unm12v43_lobt_ab03swap` | **261.86** | **Board test of the Discord drift claim** ("features encoding where an aircraft was heard didn't transfer 2025→2026; movement features did"): v48's command with **`ab02` swapped for `ab03_cb01_adsb_move`** — the same … |
| 50 | 2026-10-06 | `trimmed_stack_17_nnls_grouped_unmatched_rome_r14blend47_noD_unm12v47_lobt` | **255.64** | v48's command with **`unmatched12_v47.parquet`** in place of `unmatched12_v43.parquet`: the unmatched rows' NNLS vector refitted on **twelve months** of OOF over v47's members with a twelve-month OOF (`a109`, … |
| 51 | 2026-10-06 | `trimmed_stack_18_nnls_grouped_unmatched_rome_r14blend47_noD_unm12v47_lobt` | **255.00** | v50's command with **`ab07_ab02_adsb_calib` added** as an 18th member: ab02's recipe + `adsb_calib` (the ADS-B trace re-expressed against its own airport-month, label-free — coverage, the month's typical lag of the … |
| 52 | 2026-10-06 | `trimmed_stack_18_nnls_grouped_unmatched_rome_r14blend51_noD_unm12v47_lobt` | **254.99** | v51's command with the **Rome matched-row blend refitted over v51's members** (`a110`/`a111` with `A110_COMPOSITION=v51`): r14 0.8946 + **ab07 0.1285**, ab02 0 — the calibrated ADS-B member replaces ab02 inside the … |
| 53 | 2026-10-07 | `trimmed_stack_19_nnls_grouped_unmatched_rome_r14blend51_noD_unm12v47_lobt` | **253.61** | v52's command with **`cb03_k20_combo_adsb` added** as a 19th member: `k20`'s CatBoost recipe (d10 + border 1024) + context_window + gate_retention + eurocontrol_daily + **adsb** — the second model family on ab02's … |
| 54 | 2026-10-07 | `trimmed_stack_19_nnls_grouped_unmatched_rome_r14blend51_noD_lobt` | **252.05** | v53's command **without the `unmatched12_v47.parquet` override**: the 4,900 non-LIRF unmatched rows take the stack's own NM-unmatched NNLS vector (fitted on the 4,969 unmatched Jan + Jul OOF rows over v53's members: … |
| 55 | 2026-10-07 | `trimmed_stack_19_nnls_grouped_unmatched_rome_r14blend53_noD_lobt` | 252.17 | v54's command with the Rome matched-row blend refitted over **v53's members + `rb01_lirf_route_ab02`** (`rome_matched_blend_v53.parquet`, a110/a111): weights r14 0.8907, ab07 0.0083, ab02 0, rb01 0 — the first blend to … |
| 56 | 2026-10-07 | `trimmed_stack_19_nnls_grouped_unmatched_rome_r14blend51_noD_lobt_cellshift` | **251.68** | v54's command + `--overrides data/external/cellshift_v54.parquet`: **a120's per airport-month shift towards EUROCONTROL's published monthly taxi-out** (APDF indicator, organiser-approved). |
| 57 | 2026-10-07 | `trimmed_stack_19_nnls_grouped_unmatched_rome_r14blend51_noD_lobt_cellshift_rb01` | **251.63** | v56's layers (relative `--cell-shift data/external/cellshift_v54.csv`) with **`ab02_cb01_adsb` swapped for `rb01_lirf_route_ab02`** (the same LightGBM routed by airport: the body model sees LIRF rows at 0.1 weight) — … |
| 58 | 2026-10-08 | `..._rb01_lirfum_lg01` | **251.45** | v57 + **`--overrides data/external/lirf_um_lg01.parquet`**: the 356 LIRF NM-unmatched ranking rows outside `pop27_gap` take lg01 (LIRF gate-delay CatBoost, T = D − G; `a124`: beats r14 on both 2025 months, λ = 1 chosen … |
| 59 | 2026-10-08 | `..._rb01_an01` | 251.72 | v57 + **`an01_ab02_aobt_anchor`** (ab02 fitted on y − clip(MVT − AOBT_3, 0, 7200); solo CV 200.213, the best single member; `a081` beside v57 −0.509, transfer gate passed). |
| 60 | 2026-10-08 | `..._rb01_pr01` | 251.69 | v57 + **`pr01_ab02_plan_residual`** (ab02 + `plan_residual`; CV 200.23; `a081` −0.108, weight 0.278). |
| 62 | 2026-10-08 | `..._rb01_lirfum_lg01_cb04` | **251.25** | v58's layers (ROME_UNM = the lg01 override, cell shift) + **`cb04_cb03_adsb_calib`** (cb03 + adsb_calib; CV 200.349 vs cb03 200.47; `a081` beside v57 −0.146, transfer gate passed). |
| 63 | 2026-10-09 | `F0` (final phase, its Jan/Jul half) | 251.85 | **The final-phase pipeline on v62's seven members**, refitted on `final_ranking.parquet` (`TAXIOUT_PHASE=final`; the organisers' ±12 s on takeoff times, `a126`). |
| 64 | 2026-10-09 | `F1` (final phase, its Jan/Jul half) | 251.82 | **F0 with the `snow` block on the five LightGBM members** (q04s, ab07s, rb01s, t05s, f25s; cb03 and cb04 unchanged). |
| 65 | 2026-10-09 | `F2a` (final phase, its Jan/Jul half) | **251.11** | **F1 with the NM-unmatched rule and cb04 at 4,500 iterations.** Members cb03, hp_cb04_it4500, q04s, ab07s, rb01s, t05s, f25s; `build_stack.py --unmatched-without-month 02`: the unmatched vector for January, June and … |
| 66 | 2026-10-09 | `F2b` (final phase, its Jan/Jul half) | **250.86** | **F2a with cb03 at 4,500 iterations.** Members hp_cb03_it4500, hp_cb04_it4500, q04s, ab07s, rb01s, t05s, f25s; `build_stack.py --unmatched-without-month 02`. |
| final | 2026-10-10 | `F2b` — `amazing-marshmallow_final.parquet` | — (hidden) | **The final-phase submission**: F2b unchanged, the file whose January and July half scored v66 = 250.8618 (all 344,841 values identical). |

## 2. Findings

73 entries, L001 to L073, appended as the work went, never rewritten:
a finding that turned out wrong had its status changed and a note added. Each names the script
that measured it. Statuses: *confirmed*, *measured*, *open*, *corrected*, *withdrawn* (the
argument behind it did not hold).

| id | date | finding | status |
|---|---|---|---|
| [L001](archive/LEARNINGS_LOG.md#l001) | 2026-09-22 | The ranking set hides far less than it looks like it does | confirmed |
| [L002](archive/LEARNINGS_LOG.md#l002) | 2026-09-22 | `aobt_3` is the strongest single signal, and part of it is fabricated | argument withdrawn 2026-09-22 |
| [L003](archive/LEARNINGS_LOG.md#l003) | 2026-09-22 | 34 rows carry 28% of the squared error, and 31 are at one airport | confirmed |
| [L004](archive/LEARNINGS_LOG.md#l004) | 2026-09-22 | Taxi-in proxies taxi-out only where arrivals and departures share runways | confirmed |
| [L005](archive/LEARNINGS_LOG.md#l005) | 2026-09-22 | Out-of-sample baselines: the numbers to beat | confirmed |
| [L006](archive/LEARNINGS_LOG.md#l006) | 2026-09-22 | Does `MVT_ID_mvt` encode anything? | open |
| [L007](archive/LEARNINGS_LOG.md#l007) | 2026-09-22 | The extreme outliers are a data defect at one airport, not an event | confirmed, partly corrected 2026-09-22 |
| [L008](archive/LEARNINGS_LOG.md#l008) | 2026-09-22 | Does LIRF's contamination reach the 2026 evaluation set? | resolved: yes — see [L019](#l019) |
| [L009](archive/LEARNINGS_LOG.md#l009) | 2026-09-22 | Weather moves the target enormously; the drone closure did not | corrected 2026-09-22 |
| [L010](archive/LEARNINGS_LOG.md#l010) | 2026-09-22 | LFPG closed a runway for four months, and 2026 uses runways 2025 did not | confirmed |
| [L011](archive/LEARNINGS_LOG.md#l011) | 2026-09-22 | Group-median baselines remove a quarter of the variance and leave a biased residual | confirmed, partly corrected 2026-09-22 |
| [L012](archive/LEARNINGS_LOG.md#l012) | 2026-09-22 | A missing Network Manager flight match is worth about 22 seconds | confirmed |
| [L013](archive/LEARNINGS_LOG.md#l013) | 2026-09-22 | A stand-runway combo is nearly white noise; the structure is at the airport and the runway | confirmed |
| [L014](archive/LEARNINGS_LOG.md#l014) | 2026-09-22 | Delay is not taxi-out, and the problem reduces to estimating off-block time | confirmed |
| [L015](archive/LEARNINGS_LOG.md#l015) | 2026-09-22 | The anonymised operator is recoverable from the callsign | confirmed |
| [L016](archive/LEARNINGS_LOG.md#l016) | 2026-09-22 | Judge a proxy calibrated, not raw: `eobt_1` beats `aobt_3` | confirmed |
| [L017](archive/LEARNINGS_LOG.md#l017) | 2026-09-23 | Neither trimming training labels nor a two-stage classifier helps | confirmed |
| [L018](archive/LEARNINGS_LOG.md#l018) | 2026-09-23 | Runway configuration adds nothing over the static runway code | confirmed |
| [L019](archive/LEARNINGS_LOG.md#l019) | 2026-09-23 | Cross-validation predicts the leaderboard to within three seconds | corrected |
| [L020](archive/LEARNINGS_LOG.md#l020) | 2026-09-23 | `max_bin` was the entire gain, and the old default was a memory compromise | confirmed |
| [L021](archive/LEARNINGS_LOG.md#l021) | 2026-09-23 | 1.6% of rows carry 58% of the error, and they are flagged in advance | confirmed |
| [L022](archive/LEARNINGS_LOG.md#l022) | 2026-09-24 | ~~Steer by `rmse_ex_lirf`~~ **CORRECTED**: Rome transfers when the gain is a covariate, not capacity | corrected 2026-09-24, hours after it was written |
| [L023](archive/LEARNINGS_LOG.md#l023) | 2026-09-24 | A deterministic rule on a verified 2026 population still did not transfer | confirmed |
| [L024](archive/LEARNINGS_LOG.md#l024) | 2026-09-24 | OpenSky's REST `firstSeen` is the takeoff, not the off-block | confirmed |
| [L025](archive/LEARNINGS_LOG.md#l025) | 2026-09-24 | OpenSky does not receive the airport surface at LIRF or LFPG | confirmed (one-day snapshot; agrees with L024's independent measurement) |
| [L026](archive/LEARNINGS_LOG.md#l026) | 2026-09-25 | The N-batch: no single change moves ex-LIRF; an honest stacker does, a little | confirmed |
| [L027](archive/LEARNINGS_LOG.md#l027) | 2026-09-25 | adsb.lol hears the surface at Rome; the reconstruction route is open | confirmed |
| [L028](archive/LEARNINGS_LOG.md#l028) | 2026-09-25 | The stack resists improvement: no combiner variant beats plain ET-over-members | confirmed |
| [L029](archive/LEARNINGS_LOG.md#l029) | 2026-09-25 | The stack's remaining error sits in three places, and one of them is Paris | confirmed |
| [L030](archive/LEARNINGS_LOG.md#l030) | 2026-09-25 | Trace reconstruction works, with a per-airport accuracy set by receiver geography | confirmed |
| [L031](archive/LEARNINGS_LOG.md#l031) | 2026-09-25 | Probe algebra works: the unmatched Rome rows are over-predicted, not under | confirmed |
| [L032](archive/LEARNINGS_LOG.md#l032) | 2026-09-25 | The 2025 garbage process is the schedule gap, and matched rows are dirty too | confirmed |
| [L033](archive/LEARNINGS_LOG.md#l033) | 2026-09-25 | Every submission is capped at 20,000 s, and that rewrites three earlier learnings | confirmed |
| [L034](archive/LEARNINGS_LOG.md#l034) | 2026-09-25 | Two rows. The stack's non-LIRF error autopsy was dominated by two CV rows | confirmed |
| [L035](archive/LEARNINGS_LOG.md#l035) | 2026-09-26 | pop27 solved: the 2026 labels are the schedule gaps, and the cap was worth 95 s | confirmed |
| [L036](archive/LEARNINGS_LOG.md#l036) | 2026-09-26 | The sub-cap hedge: gate passed, board gain smaller than the law predicted | confirmed |
| [L037](archive/LEARNINGS_LOG.md#l037) | 2026-09-26 | Training-label winsorization is the body's missing lever; the other three were flat | confirmed |
| [L038](archive/LEARNINGS_LOG.md#l038) | 2026-09-27 | The frontier after the Rome fix: 348 rows are still the cheapest seconds on the board | confirmed |
| [L039](archive/LEARNINGS_LOG.md#l039) | 2026-09-27 | Matched-Rome is already optimal, and conditioning on the realized branch fakes a bias | confirmed |
| [L040](archive/LEARNINGS_LOG.md#l040) | 2026-09-27 | The label cap helps the body and hurts Rome, because the defect rows live in one place | confirmed |
| [L041](archive/LEARNINGS_LOG.md#l041) | 2026-09-27 | The refit path ignored every label treatment, so the cap's gain never reached a submission | confirmed |
| [L042](archive/LEARNINGS_LOG.md#l042) | 2026-09-27 | Rome's remaining headroom is at most −11.6 board seconds, and my earlier −13.0 had no script behind it | confirmed |
| [L043](archive/LEARNINGS_LOG.md#l043) | 2026-09-27 | "Layer D does not pay" was a baseline artefact; the same error as L040, by a different agent | confirmed |
| [L044](archive/LEARNINGS_LOG.md#l044) | 2026-09-28 | The runner labelled the untrimmed ex-LIRF "steer by this", and the gap is two rows | confirmed |
| [L045](archive/LEARNINGS_LOG.md#l045) | 2026-09-28 | An accidental replication: the same configuration twice, 0.10 apart | confirmed |
| [L046](archive/LEARNINGS_LOG.md#l046) | 2026-09-28 | v29 was built wrong twice before it was built right, and a sanity bound caught it | confirmed |
| [L047](archive/LEARNINGS_LOG.md#l047) | 2026-09-28 | a069's floor excluded the only region the evidence supports, and Rome has never been modelled | confirmed |
| [L048](archive/LEARNINGS_LOG.md#l048) | 2026-09-29 | v30 tested my correction to a069 and the correction lost | confirmed |
| [L049](archive/LEARNINGS_LOG.md#l049) | 2026-09-29 | A Manhattan stand→threshold distance is free to compute and does not beat the straight line | confirmed |
| [L050](archive/LEARNINGS_LOG.md#l050) | 2026-09-29 | Member-level deltas do not predict stack value, and I made L043's error arguing about it | confirmed |
| [L051](archive/LEARNINGS_LOG.md#l051) | 2026-09-29 | Layer D is now beaten by the base it overrides, because v30 replaced that base | confirmed |
| [L052](archive/LEARNINGS_LOG.md#l052) | 2026-09-30 | `modes5` is not underfitting, it is class-imbalance-blind: it finds `eobt_1` 8.5% of the time | confirmed |
| [L053](archive/LEARNINGS_LOG.md#l053) | 2026-10-01 | Fixing `modes5`'s class imbalance fixes the classifier and leaves the regression where it was | confirmed |
| [L054](archive/LEARNINGS_LOG.md#l054) | 2026-10-01 | Near-twin members earn `a081` seats whose board deltas are noise, sign included | confirmed (n = 3) |
| [L055](archive/LEARNINGS_LOG.md#l055) | 2026-10-01 | Two queued ideas closed by null results, and one LightGBM trap found on the way | confirmed |
| [L056](archive/LEARNINGS_LOG.md#l056) | 2026-10-01 | The stack's residuals hold nothing a second model can find, and an MLP needs two guards to show it | confirmed |
| [L057](archive/LEARNINGS_LOG.md#l057) | 2026-10-02 | The source classifier is used correctly: an audit on real data finds no defect | confirmed |
| [L058](archive/LEARNINGS_LOG.md#l058) | 2026-10-02 | v37's −0.90 was 94% the unmatched rows, which CV said the change made worse | confirmed |
| [L059](archive/LEARNINGS_LOG.md#l059) | 2026-10-03 | The arrivals reveal the day's `block_time` copy regime, and it is worth −0.74 on LightGBM | confirmed |
| [L060](archive/LEARNINGS_LOG.md#l060) | 2026-10-04 | `block_time` is always within ±3606 s of `lobt`, and a few of our predictions are not | confirmed |
| [L061](archive/LEARNINGS_LOG.md#l061) | 2026-10-04 | On our five single-member submissions, the body share of the CV gain ran opposite to the board | measured |
| [L062](archive/LEARNINGS_LOG.md#l062) | 2026-10-06 | adsb.lol's receivers changed between 2025 and 2026, most at EGLL and LEMD; the trace's lag behind the recorded off-block is airport-month specific | measured |
| [L063](archive/LEARNINGS_LOG.md#l063) | 2026-10-06 | A cross-fitted second stage on `ab02` learns nothing: the "corrector" edge needs features the base lacks | measured |
| [L064](archive/LEARNINGS_LOG.md#l064) | 2026-10-06 | On unmatched LIRF rows the sensor cannot predict the record; the arrivals say 2026 Rome records copies at least as often as 2025 | measured |
| [L065](archive/LEARNINGS_LOG.md#l065) | 2026-10-07 | EUROCONTROL's published monthly taxi-out is our labels minus a modelled offset; moving v54 halfway towards it paid −0.37 | measured |
| [L066](archive/LEARNINGS_LOG.md#l066) | 2026-10-08 | The dropout-repo levers on our board: one Rome override and one member landed, two seated members reversed | measured |
| [L067](archive/LEARNINGS_LOG.md#l067) | 2026-10-08 | The final file moved every takeoff time by up to ±12 s and nothing else; the bounds widen by that much | measured |
| [L068](archive/LEARNINGS_LOG.md#l068) | 2026-10-08 | February is the hardest month because of Istanbul's snow days, which no feature captured | measured |
| [L069](archive/LEARNINGS_LOG.md#l069) | 2026-10-09 | Summer thunderstorms leave little residual in the ADS-B member; no TS block was built | measured |
| [L070](archive/LEARNINGS_LOG.md#l070) | 2026-10-09 | F0 against v62 on the leaderboard's rows: the refits move ~31 s RMS, and the unmatched level is February's | measured |
| [L071](archive/LEARNINGS_LOG.md#l071) | 2026-10-09 | The NM-unmatched level does not transfer across months; February's is two Istanbul storm days | measured |
| [L072](archive/LEARNINGS_LOG.md#l072) | 2026-10-10 | Where F2b errs out of fold, and why the quick stack levers do not pay | measured |
| [L073](archive/LEARNINGS_LOG.md#l073) | 2026-10-10 | 2026 had more snow than 2025, and F2b already responds to it; short taxis are a coverage problem | measured |

## 3. Experiments

232 experiment configs, in name order. A config names its feature blocks, a model
and a fold set; the first sentence of its description says what it asked. 35 of them are in
[`config/experiments/`](../config/experiments) (marked ✓): the two builds' members and layers,
the final phase's variants and controls, and w08. The others ran on blocks or models we did not
ship. The prefix is the series: `b` baselines, `x` and `h` LightGBM feature work, `d`/`k`/`cb`
CatBoost, `r` Rome, `ab`/`cb` ADS-B, `hp` tuning; `family` is the tag each config carries.

| config | in repo | family | model | blocks | folds | what it asked |
|---|:-:|---|---|---:|---|---|
| `ab01_q04_adsb` |  | outgoing_boat | lgbm | 15 | seasonal | Does the ADS-B block help our best LightGBM recipe? adsb (docs/BORROWED_FROM_OUTGOING_BOAT.md item 3): each departure's own adsb.lol ground segment and takeoff -- time since first ground point / … |
| `ab02_cb01_adsb` |  | outgoing_boat | lgbm | 18 | seasonal | Does the full ADS-B block help the best LightGBM (cb01)? ab01 (q04 + adsb, on a block built from a partial pull: no 2025-10-14..12-31, no 2026) scored 205.53 against q04's 219.002 (-13.47; Jan … |
| `ab03_cb01_adsb_move` |  | outgoing_boat | lgbm | 18 | seasonal | ab02 with the ADS-B block reduced to its movement columns (adsb_move: times to takeoff from the first ground point and from the first movement, first ground speed, takeoff match error, match key), … |
| `ab04_ab02_adsb_context` |  | adsb_derived | lgbm | 19 | seasonal | Does adsb_context help the best LightGBM (ab02)? adsb_context: windowed means of the neighbouring departures' observed taxi (adsb_taxi_move) per airport and runway, ±15/60 min, the previous hour's … |
| `ab05_ab02_sensor_record` |  | adsb_derived | lgbm | 19 | seasonal | Does sensor_record help the best LightGBM (ab02)? sensor_record: record against sensor: for each stamp (aobt_3, eobt_1, lobt, iobt, sched_time) the implied taxi minus the observed one, the closest … |
| `ab07_ab02_adsb_calib` | ✓ | adsb_derived | lgbm | 19 | seasonal | The q04 recipe plus five blocks: context_window, gate_retention and eurocontrol_daily (ideas from team outgoing-boat), adsb (taxi evidence from adsb.lol traces) and adsb_calib (the trace … |
| `ab07p_ab07_placebo` | ✓ | control | lgbm | 20 | final_pairs | Control: ab07 plus the placebo block (three noise columns keyed on the movement ID). |
| `ab07s_ab07_snow` | ✓ | snow | lgbm | 20 | final_pairs | ab07 plus the snow block: the ADS-B recipe that would carry snow into the stack. |
| `an01_ab02_aobt_anchor` |  | dropouts | offset | 18 | seasonal | Anchored member: ab02's blocks and LightGBM parameters, fitted on y − clip(MVT − AOBT_3, 0, 7200) and served as anchor + fit (kind: offset, column ob_aobt_sec, clip [0, 7200]; a null anchor counts … |
| `b01_constant_mean` |  | baseline | constant | 0 | seasonal | Absolute floor: predict the training mean for every departure. |
| `b02_group_stand_runway` |  | baseline | group_stat | 1 | seasonal | The structural baseline: average taxi-out for this airport, stand and runway, backing off to airport x stand and then airport when a group is too thin. |
| `b03_aobt_proxy` |  | baseline | column | 2 | seasonal | Use Eurocontrol's reconstructed off-block time directly: predict takeoff minus aobt_3, with no fitting at all. |
| `b04_taxiin_proxy` |  | probe | column | 2 | seasonal | Test the taxi-in hypothesis end to end: predict a departure's taxi-out from the median arrival taxi time at the same stand, offset by the airport's own taxi-out/taxi-in gap via the group fallback. |
| `b05_match_aware_groups` |  | baseline | group_stat | 2 | seasonal | b02's structural grouping, but with the Network Manager match flag folded into the top-level key. |
| `c01_x02_capped` |  | capped_twins | lgbm | 6 | seasonal | A capped twin of an existing stack member. |
| `c02_counts_capped` |  | capped_twins | lgbm | 10 | seasonal | A capped twin of an existing stack member. |
| `c03_norms_capped` |  | capped_twins | lgbm | 10 | seasonal | A capped twin of an existing stack member. |
| `c04_lintree_capped` |  | capped_twins | lgbm | 9 | seasonal | A capped twin of an existing stack member. |
| `c05_perairport_capped` |  | capped_twins | per_group | 6 | seasonal | A capped twin of an existing stack member. |
| `c10_trust_aobt` |  | modes | lgbm | 9 | seasonal | E17. |
| `c11_x36_calendar_fixed` |  | calendar_fix | lgbm | 13 | seasonal | REVIEW-SUMMARY critical 1 (2026-10-03): x36_lo_card_modes5's recipe, unchanged, with the calendar block FIXED (minute-of-day was computed in Int8 and wrapped modulo 256; cal_local_minute_of_day, … |
| `c12_h12_calendar_fixed` |  | calendar_fix | lgbm | 14 | seasonal | REVIEW-SUMMARY critical 1 (2026-10-03): h12_x36_arrival_regime's recipe, unchanged, with the calendar block FIXED (minute-of-day was computed in Int8 and wrapped modulo 256; cal_local_minute_of_day, … |
| `c13_d05_calendar_fixed` |  | calendar | catboost | 13 | seasonal | Does the calendar fix carry to the CatBoost member that holds the weight? d05_catboost_modes5's exact recipe, built against the fixed calendar block (REVIEW-SUMMARY critical 1: … |
| `cb01_q04_combo` |  | outgoing_boat | lgbm | 17 | seasonal | Baseline: q04_c12_min_leaf30 = 219.002. |
| `cb02_k20_combo` |  | outgoing_boat | catboost | 16 | seasonal | The CatBoost version, on k20_d10_border1024's recipe (d10 + border 1024), so the refit round has a head start. |
| `cb03_k20_combo_adsb` | ✓ | outgoing_boat | catboost | 17 | seasonal | CatBoost (depth 10, 3,000 iterations, border_count 1,024) on seventeen blocks, among them modes6 and the context_window, gate_retention, eurocontrol_daily and adsb blocks; training labels capped at … |
| `cb03m_k20_combo_adsb_move` |  | outgoing_boat | catboost | 17 | seasonal | cb03 (cb02 + ADS-B, CatBoost) with the movement-only ADS-B block (adsb_move) instead of adsb. |
| `cb03s_cb03_snow` | ✓ | snow | catboost | 18 | final_pairs | cb03_k20_combo_adsb plus the snow block, nothing else changed. |
| `cb04_cb03_adsb_calib` | ✓ | outgoing_boat | catboost | 18 | seasonal | cb03 plus the adsb_calib block (see ab07). |
| `cb04s_cb04_snow` | ✓ | snow | catboost | 19 | final_pairs | cb04_cb03_adsb_calib plus the snow block, nothing else changed. |
| `cw01_q04_context_window` |  | outgoing_boat | lgbm | 15 | seasonal | Does the time-windowed context block help our best LightGBM recipe? context_window (docs/BORROWED_FROM_OUTGOING_BOAT.md item 4): neighbours' mvt - aobt_3 means in the past/future 15 and 60 min per … |
| `d01_xgboost_capped` |  | diversity | xgboost | 12 | seasonal | Representational diversity, at zero code cost: XGBoost is already a registered model kind and no XGBoost member has ever entered the stack. |
| `d02_catboost_capped` |  | diversity | catboost | 12 | seasonal | The stack's only non-LightGBM member (e02_catboost_solo) predates both the congestion and weather blocks and the label cap. |
| `d03_catboost_lo_card` |  | cardinality | catboost | 12 | seasonal | Does CatBoost want the same treatment LightGBM just gained 3.87 from? Widened from the original "CatBoost with x30's feature set": x32 showed the gain has TWO mostly independent parts, so testing … |
| `d04_catboost_rsm_only` |  | cardinality | catboost | 12 | seasonal | d03's control, and the arm that separates the two halves for CatBoost as x32 did for LightGBM. |
| `d05_catboost_modes5` |  | modes | catboost | 13 | seasonal | Does modes5 transfer to the member that actually carries the weight? h07 measured the classifier at -1.55 on LightGBM. |
| `d06_catboost_modes5_cap14400` |  | catboost | catboost | 13 | seasonal | The cap gradient on the champion. |
| `d07_catboost_depth12_modes5` |  | catboost | catboost | 13 | seasonal | Everything that paid, together: depth 12, cap 10,800, modes5. |
| `d08_catboost_mae_modes5` |  | diversity | catboost | 13 | seasonal | Buy decorrelation, not accuracy. |
| `d09_catboost_log_modes5` |  | diversity | catboost | 13 | seasonal | The second decorrelation axis: fit CatBoost on the LOG target rather than raw seconds. |
| `d10_catboost_modes6` |  | modes | catboost | 13 | seasonal | CatBoost confirmation for modes6. |
| `e01_family_blend` |  | ensemble | blend | 6 | seasonal | An equal blend of three model families rather than three LightGBMs. |
| `e02_catboost_solo` |  | ensemble | catboost | 6 | seasonal | CatBoost alone, for a like-for-like reading against x02. |
| `e03_linear_solo` |  | ensemble | linear | 6 | seasonal | Ridge alone. |
| `ec01_q04_eurocontrol_daily` |  | outgoing_boat | lgbm | 15 | seasonal | Does eurocontrol_daily help our best LightGBM recipe? eurocontrol_daily: EUROCONTROL's published daily series per airport x UTC day -- share of departures ATFM-regulated, off-slot late/early, all … |
| `f01_schedule` |  | feature | lgbm | 7 | seasonal | x02_lgbm_deep plus the schedule block, changing nothing else, so the difference is the feature. |
| `f02_congestion` |  | feature | lgbm | 7 | seasonal | x02_lgbm_deep plus the congestion block, changing nothing else. |
| `f03_both` |  | feature | lgbm | 8 | seasonal | Both new blocks at once, to see whether their gains are additive or the same gain twice. |
| `f05_consistency` |  | feature | lgbm | 9 | seasonal | The consistency block on top of f03's feature set. |
| `f07_no_schedule` |  | feature | lgbm | 8 | seasonal | Congestion and consistency, without the schedule block. |
| `f13_runway_history` |  | feature | lgbm | 10 | seasonal | Champion features plus the runway's own day: previous-departure wake, the wake pair, the gap's excess over that pair's global median, the day sequence number, whole-day runway counts and median gap, … |
| `f14_congestion2` |  | feature | lgbm | 11 | seasonal | f13's feature set plus the symmetric window family: arrivals around and after the takeoff, the next same-runway arrival and departure gaps, and the departure bank behind this one. |
| `f15_weather` |  | weather | lgbm | 10 | seasonal | Weather at the departure airport (open METAR, IEM archive), on the champion base. |
| `f16_weather_rh` |  | weather | lgbm | 11 | seasonal | Weather on the f13 base (runway_history included) -- the two January stories, de-icing and runway-day context, in one model. |
| `f17_geometry_identity` |  | geometry | lgbm | 13 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f17a_geometry` |  | geometry | lgbm | 12 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f17b_identity` |  | geometry | lgbm | 12 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f19_weather2` |  | weather | lgbm | 12 | seasonal | Weather v2 (wxcodes precipitation intensity and exposure, no dead columns) plus the geometry block, on the f13 base -- the successor to f17a (trimmed ex-LIRF 228.19). |
| `f20_geometry2` |  | geometry | lgbm | 12 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f21_geometry3` |  | geometry | lgbm | 12 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f22_modes3` |  | modes | lgbm | 13 | seasonal | Off-block ambiguity cells (P(block agrees with aobt_3), median gap) per airport x flight-number prefix x delay band, TARGET_DEPENDENT, on the f21 base. |
| `f23_queue` |  | geometry | lgbm | 13 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f24_runwaywind` |  | geometry | lgbm | 13 | seasonal | Stand-to-threshold distance (a047; sentinel bucket via sg_has_geo for LTFM/EDDM gaps) and movement-side identity (prefix frequency + garbling, REVIEW1 1.5-1.6) on the f16 base. |
| `f25_queue2` | ✓ | geometry | lgbm | 13 | seasonal | LightGBM on twelve blocks plus queue2: departures ahead of this one at its estimated off-block, still ahead at takeoff, and the same per runway. |
| `f25p_f25_placebo` | ✓ | control | lgbm | 14 | final_pairs | Control: f25 plus the placebo block (three noise columns keyed on the movement ID). |
| `f25s_f25_snow` | ✓ | snow | lgbm | 14 | final_pairs | Screen: f25 plus the snow block, the change q04s, ab07s, rb01s, cb04s and cb03s make to their members. |
| `g01_longtail` |  | longtail | lgbm | 13 | seasonal | The body's largest untargeted error mass. |
| `g01_x36_group_stop` |  | group_stop | lgbm_group_stop | 13 | seasonal | PLAN row 15: per-airport early-stopping rounds (ASHRAE rank 5). |
| `g02_longtail_nocap` |  | longtail | lgbm | 13 | seasonal | g01 without the training-label cap. |
| `g02_x36_group_stop_extended` |  | group_stop | lgbm_group_stop | 13 | seasonal | PLAN row 15, second arm. |
| `g10_s2c_border1024` |  | grid | catboost | 9 | seasonal | Phase A item 2 (PLAN §5 schedule): the CatBoost tiny grid on s2c. |
| `g11_s2c_ctr2` |  | grid | catboost | 9 | seasonal | Phase A item 2 (PLAN §5 schedule): the CatBoost tiny grid on s2c. |
| `g12_s2c_border1024_ctr2` |  | grid | catboost | 9 | seasonal | Phase A item 2 (PLAN §5 schedule): the CatBoost tiny grid on s2c. |
| `g13_s2c_stand_ap` |  | encoding | catboost | 9 | seasonal | Does keying the stand code by airport help CatBoost? REVIEW-SUMMARY major 37: stand_runway codes the bare stand name over all ten airports, so 526 shared codes cover 57.8% of rows, and s2c's … |
| `gr01_q04_gate_retention` |  | outgoing_boat | lgbm | 15 | seasonal | Does gate_retention help our best LightGBM recipe? gate_retention: gate retention at this flight's push clock: overdue departures (past EOBT, not pushed) per airport and runway, same-runway active, … |
| `h02_enroute` |  | geometry | lgbm | 13 | seasonal | The enroute block on the shipping twelve-block base: planned flight duration (arvt_1 - eobt_1), the arrival-time revision (arvt_3 - arvt_1, an ATFM-regulation fingerprint), a long-haul flag and a … |
| `h03_destination` |  | destination | lgbm | 14 | seasonal | The destination family on the shipping twelve-block base. |
| `h04_destination_labelfree` |  | destination | lgbm | 13 | seasonal | The label-free half of the destination family only, so the CONTEXT and TARGET_DEPENDENT contributions can be attributed separately. |
| `h05_mlp_probe` |  | diversity | mlp | 12 | quick | COST PROBE, not a candidate. |
| `h06_mlp` |  | diversity | mlp | 12 | seasonal | The only non-tree, non-linear function class in the set. |
| `h07_modes5` |  | modes | lgbm | 13 | seasonal | PLAN section 4.2's trained source classifier, as a feature block. |
| `h08_modes5_replaces_modes` |  | modes | lgbm | 12 | seasonal | Does the two-way cell table still add anything once the trained classifier is there? h07 added modes5 on top of modes and scored 224.44 against t03's 225.99, so the classifier is worth -1.55 as an … |
| `h09_x36_modes6` |  | modes | lgbm | 13 | seasonal | LightGBM screen for modes6, the two-stage source classifier (L052's remedy). |
| `h10_mlp_modes5` |  | diversity | mlp | 13 | seasonal | A low-correlation member, after L054: near-twin seats (corr 0.991-0.995) gave board deltas of -0.10, +0.11, -0.27. |
| `h11_mlp_modes5_wide` |  | diversity | mlp | 13 | seasonal | h10 with a wider, deeper network (512-256-128). |
| `h12_x36_arrival_regime` |  | arrival_regime | lgbm | 14 | seasonal | Phase A item 1 follow-up (PLAN §5 schedule): does the arrivals' day-level block_time copy regime add to x36? a108: per airport x day, the share of arrivals whose on-block time copies a stamp … |
| `hp_ab07_bag06` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: row subsampling: bagging_fraction 0.6 every iteration (default 0.8). |
| `hp_ab07_cat50` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: categorical smoothing: cat_smooth 50 and cat_l2 50 (LightGBM's defaults are 10 and 10), for the high-cardinality stand x runway codes. |
| `hp_ab07_ff04` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: feature fraction 0.4 (0.6 was the lowest tried). |
| `hp_ab07_huber` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: Huber loss, delta 600 s, on the ADS-B recipe (t05 is Huber on the oldest recipe). |
| `hp_ab07_lr015` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: learning rate 0.015 with up to 8,000 rounds (early stopping 400): the slower schedule was only ever tried on the first recipe. |
| `hp_ab07_xt` | ✓ | tuning | lgbm | 19 | final_pairs | ab07's recipe with one change: extremely randomised splits (extra_trees), a different bias for the same inputs. |
| `hp_ab07s_bag06` | ✓ | tuning | lgbm | 20 | final_pairs | ab07s's recipe (ab07 + the snow block) with one change: row subsampling: bagging_fraction 0.6 every iteration (default 0.8). |
| `hp_ab07s_cat50` | ✓ | tuning | lgbm | 20 | final_pairs | ab07s's recipe (ab07 + the snow block) with one change: categorical smoothing: cat_smooth 50 and cat_l2 50 (LightGBM's defaults are 10 and 10), for the high-cardinality stand x runway codes. |
| `hp_ab07s_ff04` | ✓ | tuning | lgbm | 20 | final_pairs | ab07s's recipe (ab07 + the snow block) with one change: feature fraction 0.4 (0.6 was the lowest tried). |
| `hp_ab07s_huber` | ✓ | tuning | lgbm | 20 | final_pairs | ab07s's recipe (ab07 + the snow block) with one change: Huber loss, delta 600 s, on the ADS-B recipe (t05 is Huber on the oldest recipe). |
| `hp_ab07s_xt` | ✓ | tuning | lgbm | 20 | final_pairs | ab07s's recipe (ab07 + the snow block) with one change: extremely randomised splits (extra_trees), a different bias for the same inputs. |
| `hp_cb03_it4500` | ✓ | tuning | catboost | 17 | final_pairs | cb03's recipe with one change: 4,500 iterations instead of 3,000 at the same learning rate -- the change that moved its sibling cb04 by -1.36 over four months (hp_cb04_it4500; every month better) … |
| `hp_cb04_it4500` | ✓ | tuning | catboost | 18 | final_pairs | cb04's recipe with one change: 4,500 iterations instead of 3,000 at the same learning rate: does the fixed 3,000 (no early stopping) under-fit? Baseline: cb04_cb03_adsb_calib on final_pairs, same … |
| `hp_cb04_l2_10` | ✓ | tuning | catboost | 18 | final_pairs | cb04's recipe with one change: l2_leaf_reg 10 (CatBoost's default, 3, was never varied on this recipe). |
| `hp_cb04_lr03` | ✓ | tuning | catboost | 18 | final_pairs | cb04's recipe with one change: learning rate 0.03 with 5,000 iterations: the slower schedule. |
| `hp_cb04_rs3` | ✓ | tuning | catboost | 18 | final_pairs | cb04's recipe with one change: random_strength 3 (default 1): more randomness in split scoring. |
| `k01_catboost_sweep` (grid) |  | catboost | catboost | 12 | seasonal | The night's finding, pursued. |
| `k02_catboost_cap10800` |  | catboost | catboost | 12 | seasonal | d02 at the cap the LightGBM sweep preferred (10,800 rather than 7,200). |
| `k03_catboost_nocap` |  | catboost | catboost | 12 | seasonal | The control d02 never had: CatBoost on the same blocks WITHOUT the cap. |
| `k04_catboost_depth6` |  | catboost | catboost | 12 | seasonal | CatBoost depth 6, as a SINGLE arm. |
| `k05_catboost_depth10` |  | catboost | catboost | 12 | seasonal | CatBoost depth 10, as a SINGLE arm. |
| `k06_catboost_depth12` |  | catboost | catboost | 12 | seasonal | Does CatBoost keep improving past depth 10? One knob against k05, and the longest single fit this project will have run. |
| `k07_catboost_depth10_cap10800` |  | catboost | catboost | 12 | seasonal | Does CatBoost at depth 10 want the cap at 10,800 rather than 7,200? One knob, one arm. |
| `k09_catboost_cap14400` |  | catboost | catboost | 12 | seasonal | The CatBoost cap gradient, one rung further. |
| `k10_catboost_depth12_cap10800` |  | catboost | catboost | 12 | seasonal | The last unexplored corner of the CatBoost grid: both knobs that paid, together. |
| `k20_d10_border1024` |  | grid | catboost | 13 | seasonal | Does g10's border_count 1024 carry to a weighted CatBoost member? g10_s2c_border1024 scored 217.552 against s2c's 218.28 (-0.73; both folds -0.73). |
| `k21_d06_border1024` |  | grid | catboost | 13 | seasonal | Does g10's border_count 1024 carry to a weighted CatBoost member? g10_s2c_border1024 scored 217.552 against s2c's 218.28 (-0.73; both folds -0.73). |
| `k22_d05_border1024` |  | grid | catboost | 13 | seasonal | Does g10's border_count 1024 carry to a weighted CatBoost member? g10_s2c_border1024 scored 217.552 against s2c's 218.28 (-0.73; both folds -0.73). |
| `lg01_lirf_gate_delay` | ✓ | openair | subset | 11 | seasonal | Gate-delay model for Rome, after team likable-eagle: taxi-out T = D - G, with D = takeoff - schedule (known on the ranking rows) and G the gate delay, which is what is learnt. |
| `lw01_q04_lobt_window` |  | outgoing_boat | lgbm | 15 | seasonal | Does lobt_window help our best LightGBM recipe? lobt_window: the LOBT window as inputs: its edges in taxi-seconds and, per candidate stamp (sched, eobt, iobt, aobt), whether a copy is feasible and … |
| `m00_lgbm_smoke` |  | smoke | lgbm | 5 | quick | A deliberately small LightGBM on a single fold. |
| `m01_lgbm_global` |  | model | lgbm | 5 | seasonal | First gradient-boosted model. |
| `m02_lgbm_per_airport` |  | model | per_group | 5 | seasonal | The same features, but ten models instead of one -- a LightGBM per airport, with a global model as fallback for any group below min_rows. |
| `m03_lgbm_blend` |  | model | blend | 5 | seasonal | Equal blend of the global model and the per-airport one. |
| `m04_lgbm_sweep` (grid) |  | sweep | lgbm | 5 | quick | Hyperparameter sweep over the global model. |
| `m05_lgbm_runway_config` |  | model | lgbm | 6 | seasonal | m01 plus the runway_config block. |
| `m06_lgbm_config_smoke` |  | smoke | lgbm | 6 | quick | The fast variant of m05: one fold, few trees. |
| `m10_min5_lgbm` |  | minimal | lgbm | 5 | seasonal | Minimal-solution screen S2 (owner's request, 2026-10-02): how much does pruning to five blocks cost? Five blocks chosen from evidence, not taste: offblock (the off-block stamps; dropping it costs … |
| `m11_min5_routed` |  | minimal | routed | 5 | seasonal | Minimal-solution screen S2, two populations: m10's five blocks, routed on flt_no_match so the unmatched rows get their own fit (as u02 and r14's base do). |
| `m12_min5_plus_stand` |  | minimal | lgbm | 6 | seasonal | Minimal-solution add-back arm. |
| `m13_min5_plus_context` |  | minimal | lgbm | 8 | seasonal | Minimal-solution add-back arm. |
| `m14_min5_plus_four` |  | minimal | lgbm | 9 | seasonal | Minimal-solution add-back arm. |
| `m7a_ab02_modes6` |  | modes | lgbm | 18 | seasonal | Does the copy classifier gain from ADS-B? Arm: the CONTROL: modes5 swapped for modes6 (the same two-stage classifier without ADS-B). |
| `m7b_ab02_modes7` |  | modes | lgbm | 18 | seasonal | Does the copy classifier gain from ADS-B? Arm: modes5 swapped for modes7 (modes6 + the departure's ADS-B as classifier inputs). |
| `n02_counts` |  | n_batch | lgbm | 10 | seasonal | N02. |
| `n03_norms` |  | n_batch | lgbm | 10 | seasonal | N03. |
| `n04_offset` |  | n_batch | offset | 9 | seasonal | N04. |
| `n05_modes2` |  | n_batch | lgbm | 9 | seasonal | N05. |
| `n06_lintree_127` |  | n_batch | lgbm | 9 | seasonal | N06. |
| `n06_lintree_63` |  | n_batch | lgbm | 9 | seasonal | N06. |
| `n08_seed_bag` |  | n_batch | blend | 9 | seasonal | N08. |
| `n12_lirf_downweight` |  | n_batch | routed | 9 | seasonal | N12. |
| `n13_unmatched_route_c10` |  | n_batch | routed | 9 | seasonal | N13. |
| `n14_schedule_rome_only` |  | n_batch | routed | 10 | seasonal | N14. |
| `n15_huber` |  | n_batch | lgbm | 9 | seasonal | N15. |
| `n18_runwayday` |  | n_batch | lgbm | 10 | seasonal | N18. |
| `o12_ab02` |  | oof12 | lgbm | 18 | rest10 | Twelve-month OOF piece for ab02_cb01_adsb: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_ab03` |  | oof12 | lgbm | 18 | rest10 | Twelve-month OOF piece for ab03_cb01_adsb_move (rest10), for the unmatched vector if the movement-only member wins on the board. |
| `o12_cb03` |  | oof12 | catboost | 17 | rest10 | Twelve-month OOF piece for cb03_k20_combo_adsb (rest10, ~20 h of CatBoost), so the unmatched vector can be refitted with cb03 among its members (a109 A109_EXTRA=cb03_k20_combo_adsb:o12_cb03). |
| `o12_d05` |  | oof12 | catboost | 13 | rest8 | Twelve-month OOF piece for d05_catboost_modes5: identical model, features, cap and treatments; only the fold set differs (rest8). |
| `o12_d06` |  | oof12 | catboost | 13 | rest10 | Twelve-month OOF piece for d06_catboost_modes5_cap14400: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_f25` |  | oof12 | lgbm | 13 | rest10 | Twelve-month OOF piece for f25_queue2: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_h06` |  | oof12 | mlp | 12 | rest10 | Twelve-month OOF piece for h06_mlp: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_h09` |  | oof12 | lgbm | 13 | rest10 | Twelve-month OOF piece for h09_x36_modes6: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_q04` |  | oof12 | lgbm | 14 | rest10 | Twelve-month OOF piece for q04_c12_min_leaf30: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_r14` |  | oof12 | rules | 8 | all_months | Twelve-month OOF for r14_rules_plus_unmatched_route: identical model, features, cap and treatments; only the fold set differs. |
| `o12_t05` |  | oof12 | lgbm | 12 | rest10 | Twelve-month OOF piece for t05_huber_alpha[alpha=600]: identical model, features, cap and treatments; only the fold set differs (rest10). |
| `o12_u03` |  | oof12 | routed | 18 | rest10 | Twelve-month OOF piece for u03_unmatched_route_ab02 (rest10), for the unmatched vector refit with u03 added. |
| `p00_catboost_search_control` |  | hyperparams | catboost | 12 | seasonal | The control p03 needs: CatBoost at the reduced search settings (iterations 1000, lr 0.15) with every other parameter at its default. |
| `p01_lgbm_capacity` (grid) |  | hyperparams | lgbm | 12 | seasonal | STAGE A of the first real hyperparameter search in this project. |
| `p02_lgbm_regularisation` (grid) |  | hyperparams | lgbm | 12 | seasonal | STAGE A, second half: a sweep AROUND the current regularisation, not the introduction of it. |
| `p02_lintree_smoke` |  | smoke | lgbm | 5 | quick | m00_lgbm_smoke with linear_tree: true and nothing else changed. |
| `p03_catboost_search` (grid) |  | hyperparams | catboost | 12 | seasonal | STAGE B: the first hyperparameter search ever run on CatBoost, which is our best function class (d02 = 220.94 trimmed ex-LIRF) and has had exactly ONE parameter setting tried across three runs (a082). |
| `p04_l2_extended` (grid) |  | hyperparams | lgbm | 12 | seasonal | The one hyperparameter edge left unresolved. |
| `pr01_ab02_plan_residual` |  | dropouts | lgbm | 19 | seasonal | ab02 + plan_residual: the plan's off-block-to-arrival time minus its (adep, ades, type) median (kind-mango plan_nm_taxi, v65 −1.22 live; docs/DROPOUTS_RECONCILED.md item 4). |
| `q01_x36_min_leaf` (grid) |  | hyperparams | lgbm | 13 | seasonal | Does a smaller min_data_in_leaf help the shipping LightGBM recipe? REVIEW-SUMMARY major 10: p01_lgbm_capacity's four unreported arms show min_data_in_leaf 50 beating 100 at every num_leaves (511: … |
| `q02_x36_stop_days` |  | hyperparams | lgbm | 13 | seasonal | Does the early-stopping slice matter? REVIEW-SUMMARY minor (lgbm.py:85-91): the stopping slice is the last 15% of training rows by month, i.e. |
| `q03_x36_lo_cat` |  | encoding | lgbm | 13 | seasonal | Are the lo-card codes better as numbers or as categoricals? REVIEW-SUMMARY major 38: stand_runway_lo emits lo_* names, so its seven codes reach LightGBM as numeric columns, and the x30-x36 ablations … |
| `q04_c12_min_leaf30` | ✓ | hyperparams | lgbm | 14 | seasonal | The main LightGBM recipe: fourteen blocks, including the low-cardinality stand x runway encoding (stand_runway_lo), the modes5 off-block-source features and the arrival regime; training labels … |
| `q04s_q04_snow` | ✓ | snow | lgbm | 15 | final_pairs | Screen: q04 plus the snow block (snow reports over 3-48 h, heavy snow, freezing rain, the day's count, hours since snow). |
| `q05_ab02_capacity` (grid) |  | hyperparams | lgbm | 18 | seasonal | Capacity of the best LightGBM recipe now that it carries ~170 columns (ab02). |
| `r00_rome_split_fw0.0` |  | rome | routed | 6 | seasonal | Rome on its own model, the other nine sharing one, with foreign_weight = 0.0. |
| `r015_rome_split_fw0.15` |  | rome | routed | 6 | seasonal | Rome on its own model, the other nine sharing one, with foreign_weight = 0.15. |
| `r04_rome_split_fw0.4` |  | rome | routed | 6 | seasonal | Rome on its own model, the other nine sharing one, with foreign_weight = 0.4. |
| `r10_f03_rules_D` |  | rules | rules | 8 | seasonal | f03_both with one deterministic override: unmatched departures at LIRF whose schedule gap exceeds twelve hours. |
| `r11_f03_rules_DA` |  | rules | rules | 8 | seasonal | Rule D, plus the A-CDM identity when it is extreme. |
| `r12_D_threshold_08h` |  | rules | rules | 8 | seasonal | THRESHOLD VARIANT. |
| `r12_D_threshold_18h` |  | rules | rules | 8 | seasonal | THRESHOLD VARIANT. |
| `r14_rules_plus_unmatched_route` | ✓ | rules | rules | 8 | seasonal | The Rome (LIRF) layer: two rules over a LightGBM routed on the NM flight-match flag. |
| `r15_rules_route_adsb` |  | rules | rules | 19 | seasonal | The Rome layer with today's information. |
| `rb01_lirf_route_ab02` | ✓ | rules | routed | 18 | seasonal | ab07's recipe without adsb_calib, routed by airport: one LightGBM for Rome (LIRF) rows and one for the other nine airports, each also trained on the other route's rows at weight 0.1. |
| `rb01s_rb01_snow` | ✓ | snow | routed | 19 | final_pairs | rb01_lirf_route_ab02 plus the snow block, nothing else changed. |
| `rm01_m5_mix_column` |  | rules | column | 1 | seasonal | Rome probe (2026-10-06, after r15's null): the copy-mixture expected label m5_label_mix from modes5 used DIRECTLY as the prediction, no fitting -- the expectation-mixture structure outgoing-boat's … |
| `rm02_m6_mix_column` |  | rules | column | 1 | seasonal | Rome probe (2026-10-06, after r15's null): the copy-mixture expected label m6_label_mix from modes6 used DIRECTLY as the prediction, no fitting -- the expectation-mixture structure outgoing-boat's … |
| `rm03_m7_mix_column` |  | rules | column | 1 | seasonal | Rome probe (2026-10-06, after r15's null): the copy-mixture expected label m7_label_mix from modes7 used DIRECTLY as the prediction, no fitting -- the expectation-mixture structure outgoing-boat's … |
| `s00_drop_calendar` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s01_drop_offblock` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s02_drop_stand_runway` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s03_drop_taxi_in` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s04_drop_flight_match` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s05_drop_runway_config` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s06_drop_congestion` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s07_drop_consistency` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s08_drop_modes` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s09_drop_runway_history` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s10_drop_weather` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s11_drop_stand_geometry` |  | ablation | lgbm | 11 | seasonal | Leave-one-block-out on the capped baseline, to answer a question never asked in this project: do we need all twelve feature families? No feature selection of any kind has ever been run here -- no … |
| `s2a_catboost_nine_arrival` |  | arrival_regime | catboost | 10 | seasonal | Phase B, branch 1 (PLAN §5 schedule): the arrivals' day-level copy regime on the CatBoost. |
| `s2c_catboost_nine` |  | minimal | catboost | 9 | seasonal | Minimal-solution S2, the expensive step (owner, 2026-10-02): one CatBoost on nine blocks. |
| `t01_leaf_sweep` (grid) |  | sweep | lgbm | 6 | seasonal | How small a leaf the deep model is allowed, and whether bagging should be on at all. |
| `t02_cap7200` |  | label_surgery | lgbm | 12 | seasonal | The f21 model with TRAINING labels winsorized at 2 h; evaluation untouched. |
| `t02_deep_blend` |  | ensemble | blend | 6 | seasonal | The equal-weight mean of the three deep models, refit so it can be submitted. |
| `t03_cap_sweep` (grid) |  | label_surgery | lgbm | 12 | seasonal | Where does training-label winsorisation bite best? t02 fixed 7,200 and took trimmed ex-LIRF 228.35 -> 226.23 (L037); the threshold has never been tuned because expand_grid routed the swept key into … |
| `t04_logtarget` |  | target | lgbm | 12 | seasonal | L2 on log(taxi + 1) with Duan's smearing back-transform, against L2 on raw seconds -- CLAUDE.md's next-step 6, never run in this repo (n15's "huber" was misconfigured and never tested a robust loss, … |
| `t05_huber_alpha` (grid) | ✓ | loss | lgbm | 12 | seasonal | LightGBM with a Huber loss, delta (alpha) = 600 s, on twelve blocks. |
| `t05s_t05_snow` | ✓ | snow | lgbm | 13 | final_pairs | Screen: t05 plus the snow block, the change q04s, ab07s, rb01s, cb04s and cb03s make to their members. |
| `t06_relabel` |  | label_surgery | lgbm | 12 | seasonal | Per-mechanism training-label repair instead of the blunt cap: LATE rows (label < 120 s) and SCHED copies (label == the schedule gap) are relabelled to takeoff - aobt_3 where that lands in [120, … |
| `t07_relabel_cap` |  | label_surgery | lgbm | 12 | seasonal | Both label treatments together: the per-mechanism repair of t06, then winsorisation at t02's threshold for whatever defect mass the repair cannot reach. |
| `u01_unmatched_route` |  | rome | routed | 6 | seasonal | A separate model for the rows with no Network Manager match. |
| `u02_x36_unmatched_route` |  | unmatched | routed | 13 | seasonal | An unmatched-rows specialist, priced where it can actually earn: in the unmatched NNLS vector. |
| `u03_unmatched_route_ab02` |  | unmatched | routed | 18 | seasonal | A member for the rows where the board has answered: the NM-unmatched departures. |
| `v01_quarters_lgbm` |  | validation | lgbm | 12 | quarters | the champion feature set on four folds instead of two. |
| `w01_gap6_with_td` |  | validation | lgbm | 12 | gap6 | WITH the target-dependent modes block. |
| `w02_gap6_no_td` |  | validation | lgbm | 11 | gap6 | WITHOUT modes. |
| `w03_adjacent_with_td` |  | validation | lgbm | 12 | adjacent_dec | WITH modes, adjacent months -- w01's control, same 6 training months. |
| `w04_adjacent_no_td` |  | validation | lgbm | 11 | adjacent_dec | WITHOUT modes, adjacent months -- w02's control, same 6 training months. |
| `w06_x36_quarters` |  | validation | lgbm | 13 | quarters | Does the modes5 gain survive months it was not selected on? Everything in v31 was chosen on the seasonal pair, Jan and Jul. |
| `w07_h09_quarters` |  | validation | lgbm | 13 | quarters | Does modes6's null result on the seasonal pair hold across the year? h09 (x36 with modes6 in place of modes5) scored 220.508 against x36's 220.435 on Jan+Jul -- inside L045's floor -- and then … |
| `w08_x36_all_months` | ✓ | validation | lgbm | 13 | all_months | x36 -- the best LightGBM of the leaderboard phase and the modes5 host in every shipped stack since v31 -- cross-validated on all twelve months of 2025, each fold trained on the other eleven. |
| `w09_d05_quarters` |  | validation | catboost | 13 | quarters | The CatBoost champion family beyond Jan + Jul. |
| `x01_lgbm_allmonths` |  | benchmark | lgbm | 5 | all_months | m01's model over all twelve leave-one-month-out folds instead of two. |
| `x02_lgbm_deep` |  | benchmark | lgbm | 6 | seasonal | A deliberately large model: 511 leaves, 511 bins, 4000 rounds. |
| `x03_lgbm_allmonths_full` |  | benchmark | lgbm | 6 | all_months | Every feature block over all twelve folds. |
| `x04_ablate_maxbin` |  | ablation | lgbm | 5 | seasonal | m01 with one change: max_bin raised from 127 to 511. |
| `x05_ablate_leaves` |  | ablation | lgbm | 5 | seasonal | m01 with one change: num_leaves raised from 127 to 511, max_bin left at the 127 default. |
| `x06_ablate_slow` |  | ablation | lgbm | 5 | seasonal | m01 with one change: a slower learning rate and more rounds, so early stopping has room to find a deeper optimum. |
| `x07_deep_per_airport` |  | model | per_group | 6 | seasonal | The two things that worked, combined: x02's deep settings applied per airport rather than globally. |
| `x1_maxbin_1023` |  | ablation | lgbm | 5 | seasonal | m01's model with max_bin = 1023, nothing else changed. |
| `x1_maxbin_2047` |  | ablation | lgbm | 5 | seasonal | m01's model with max_bin = 2047, nothing else changed. |
| `x1_maxbin_255` |  | ablation | lgbm | 5 | seasonal | m01's model with max_bin = 255, nothing else changed. |
| `x20_lintree` |  | modes | lgbm | 9 | seasonal | E14. |
| `x30_lo_cardinality` |  | cardinality | lgbm | 12 | seasonal | The ablation's biggest finding, followed up. |
| `x31_minimal_blocks` |  | cardinality | lgbm | 6 | seasonal | The ablation's cumulative question. |
| `x32_lo_card_ff06` |  | cardinality | lgbm | 12 | seasonal | THE arm that decides the next composition. |
| `x35_lo_card_l2_grid` (grid) |  | cardinality | lgbm | 12 | seasonal | The joint optimum. |
| `x36_lo_card_modes5` |  | modes | lgbm | 13 | seasonal | Do the two things that worked on LightGBM combine? x35 (lo-card feature set, ff 0.6, l2 50) is the best LightGBM in the project at 222.03. |
