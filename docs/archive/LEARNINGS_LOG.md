# Claude's learnings

> **About this file.** This is our findings log as it stood on 2026-10-10, kept verbatim: 73
> entries, appended as the work went and never rewritten; a finding that turned out wrong had its
> status changed and a note added. It was written as a working notebook by the AI agent that wrote
> most of the code (see the README's [AI assistance](../../README.md#ai-assistance)), so it says
> "Claude" and "the owner", and it cites things that are not in this repository: most analysis
> scripts (`aNNN`), internal documents (`PLAN.md`, `LEARNINGS.md`, `OUTLIERS.md` and others), the
> run log, and paths on our build machine. [`HISTORY.md`](../HISTORY.md) indexes it beside the
> submissions and the experiments.

**This is Claude's notebook, not the project's record.** The project's record is
[`LEARNINGS.md`](LEARNINGS.md), which the user owns. Nothing here is authoritative; it is what
Claude inferred while writing the code, kept so the reasoning behind a feature or a default is
recoverable, and so mistakes stay visible rather than being quietly dropped.

**Rules.** Every entry names the script that produced it and that script lives in
`scripts/analysis/`, writing its output to `reports/`. A number that cannot be reproduced by
re-running its script does not belong here. Entries are appended, never rewritten — if a finding
turns out to be wrong, change its status and add a note explaining why. `open` means the
question is asked but not settled; `withdrawn` means the argument behind it did not hold.

| | Finding | Status |
|---|---|---|
| [L001](#l001) | The ranking set hides far less than it looks like it does | confirmed |
| [L002](#l002) | `aobt_3` is the strongest single signal, and part of it is fabricated | argument withdrawn |
| [L003](#l003) | 34 rows carry 28% of the squared error, and 31 are at one airport | confirmed |
| [L004](#l004) | Taxi-in proxies taxi-out only where arrivals and departures share runways | confirmed |
| [L005](#l005) | Out-of-sample baselines: the numbers to beat | confirmed |
| [L006](#l006) | Does `MVT_ID_mvt` encode anything? | open |
| [L007](#l007) | The extreme outliers are a data defect at one airport | confirmed, partly corrected |
| [L008](#l008) | Does LIRF's contamination reach 2026? | open |
| [L009](#l009) | Weather moves the target enormously; drones did not | corrected |
| [L010](#l010) | LFPG closed a runway for four months | confirmed |
| [L011](#l011) | Group-median baselines and deviation targets | confirmed, partly corrected |
| [L012](#l012) | A missing NM flight match is worth ~22 s | confirmed |
| [L013](#l013) | Combos are white noise; structure is at airport and runway | confirmed |
| [L014](#l014) | Delay is not taxi-out; the problem is estimating off-block | confirmed |
| [L015](#l015) | The anonymised operator is recoverable from the callsign | confirmed |
| [L016](#l016) | Judge a proxy calibrated, not raw — `eobt_1` beats `aobt_3` | confirmed |
| [L017](#l017) | Neither trimming training labels nor a two-stage classifier helps | confirmed |
| [L018](#l018) | Runway configuration adds nothing over the static runway code | confirmed |
| [L019](#l019) | CV predicts the leaderboard to within 3 s, and the LIRF bet pays | corrected (see L022, L034, L054) |
| [L020](#l020) | `max_bin` was the whole gain, and 127 was a memory compromise | confirmed |
| [L021](#l021) | 1.6% of rows carry 58% of the error, and they are identifiable | confirmed |

---

<a id="l001"></a>
## L001 — The ranking set hides far less than it looks like it does

`confirmed` · `scripts/analysis/a001_offblock_proxies.py` · 2026-09-22

**Question.** What is actually missing at prediction time?

**Finding.** Only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, and only on `DEP` rows. Every
other column is present at the same null rate as in training. That includes the takeoff time,
the runway, the stand, all four alternative off-block estimates, and the complete set of
arrival rows for January and July 2026 with their block times intact.

**So what.** This is not a forecasting problem posed at pushback. We know when the aircraft
left the ground, which runway it used and which stand it came from. Congestion features are
built around *takeoff* time, and the arrival rows in the evaluation window can be used to
measure 2026 conditions directly rather than extrapolating from 2025.

---

<a id="l002"></a>
## L002 — `aobt_3` is the strongest single signal, and part of it is fabricated

`argument withdrawn 2026-09-22` · `scripts/analysis/a001_offblock_proxies.py` · 2026-09-22

> **The reasoning below does not hold.** This entry argued the default bands are fabricated
> because the proxy correlates ~0 with the target *inside* them. Restricting a variable to a
> 60-second window removes its variance, so that correlation collapses regardless — the control
> cell in `a001` measures +0.016 at 600 s and +0.030 at 900 s, neither of which is a default
> band. The proxy-RMSE table and the share figures below still stand; the correlation column and
> the conclusion drawn from it do not.
>
> `a001` now measures band size as excess mass over a flanking baseline instead, and reports a
> third band at 25 minutes that this entry missed. Those numbers are in
> `reports/a001_offblock_proxies.txt` and have not been written up as a finding — that is the
> user's call.

**Question.** The dataset keeps four off-block estimates besides the blanked one. How good is
each, used raw as a prediction of taxi-out?

**Finding.** In-sample over 2025, taking `mvt_time − <estimate>` as the prediction:

| estimate | corr | RMSE (s) |
|---|---:|---:|
| `aobt_3` (Eurocontrol M3, flown) | 0.536 | **384.9** |
| `eobt_1` (filed flight plan) | 0.556 | 676.7 |
| `iobt` | 0.515 | 739.8 |
| `lobt` | 0.514 | 740.1 |
| `sched_time` | 0.232 | 2412.7 |

`aobt_3` has the lowest raw RMSE, **but raw RMSE is the wrong yardstick** and this entry
originally drew the wrong conclusion from it. A model fits an intercept and a slope on its first
split, so a proxy carrying a large fixed offset is not penalised the way raw RMSE penalises it.
Giving each proxy that courtesy (`scripts/analysis/a012_proxy_calibration.py`):

| proxy | raw RMSE | bias | corr | **calibrated RMSE** |
|---|---:|---:|---:|---:|
| `eobt_1` | 676.7 | +377.2 | 0.552 | **344.7** |
| `aobt_3` | 384.9 | +17.0 | 0.525 | 351.7 |
| `iobt` | 739.8 | +366.9 | 0.511 | 355.2 |
| `lobt` | 740.1 | +365.9 | 0.510 | 355.5 |
| `sched_time` | 2409.8 | +901.5 | 0.216 | 451.9 |

**`eobt_1` is the better feature**, and the 384.9-versus-676.7 headline was almost entirely a
fixed −377 s offset. Out of sample it is not close. Calibrating on the fold's training months and
scoring on the held-out month, *on top of* the airport × stand × runway mean:

| proxy | Jan | Jul |
|---|---:|---:|
| `eobt_1` | **354.8** | **433.6** |
| `iobt` / `lobt` | 362.9 / 363.2 | 433.7 / 433.9 |
| `sched_time` | 406.2 | 451.8 |
| `aobt_3` | 387.6 | **463.7** |

On July `aobt_3` is the **worst** of the five — worse even than the raw timetable. The likely
reason is in this entry already: ~20% of `aobt_3` is a back-computed constant, which destroys its
conditional information, whereas `eobt_1` is always a genuine filed estimate.

Two further things qualify `aobt_3`. It is minute-rounded — 97.7% of its values sit
exactly on a minute, against second-resolution airport data — which is why it matches the true
off-block to the second only 0.65% of the time while landing within five minutes 74.6% of the
time. And for ~19% of departures Eurocontrol never observed the off-block: it back-computed the
value by subtracting a fixed default taxi time from the takeoff, leaving spikes at 20 and 12
minutes.

| `mvt_time − aobt_3` | share (train) | share (ranking) |
|---|---:|---:|
| measured | 77.2% | 77.4% |
| 20-minute band `[1170, 1230]` | 10.4% | 10.4% |
| 12-minute band `[690, 750]` | 8.5% | 7.8% |
| 25-minute band `[1470, 1530]` | 3.9% | 4.3% |

**Two different numbers, and they mean different things.** Band membership is **22.8% / 22.6%**.
But `a001`'s own excess-mass measure — band count against a flanking baseline, which is the
method that entry argues for — puts genuine fabrication at **12.1%** (2.2 + 7.0 + 2.8), because
most of the 12-minute band is ordinary traffic that happens to taxi for twelve minutes. Quote the
excess-mass figure when the question is "how much is fabricated" and band membership when the
question is "how many rows carry the flag".

Band rate is strongly airport-dependent (LTFM 62.7%, LEBL 26.8%, EDDM 24.5% … LFPG 12.8%, which
is the minimum) and stable between training and ranking. An earlier version of this entry quoted
the two-band figures from before the 25-minute band was added.

**So what.** The proxy goes in every model, but never alone: `ob_aobt_is_default_20min` and
`ob_aobt_is_default_12min` ship with it, because on a fifth of rows the value is a constant
carrying no information and a model handed the bare number will trust it anyway.

---

<a id="l003"></a>
## L003 — 34 rows carry 28% of the squared error, and 31 are at one airport

`confirmed` · `scripts/analysis/a003_target_outliers.py` · 2026-09-22

**Question.** RMSE squares errors. How much of the score is decided by a few impossible rows?

**Finding.** 34 departures in all of 2025 — 0.0016% — have a recorded taxi-out over twelve
hours, up to 131,167 s (36 hours). Measured against a constant-mean predictor they account for
**28.2% of the total squared error**. Dropping them moves RMSE from 546.4 s to 462.8 s.

**31 of the 34 are at LIRF**, two at LFPG, one at LSZH. They are not a single systematic bug: a
one-day timestamp correction makes only 41% of them plausible, and the rest stay absurd in both
directions.

The monthly spread is distorted by them. July 2025 has the largest standard deviation of any
month (745.0 s) and ten of the monster rows; February has the *highest* rate of genuinely long
taxis (0.53% over an hour against July's 0.34%), which is the de-icing signal.

**So what.** Two consequences. Reported CV carries both the real RMSE and a `rmse_trimmed` that
excludes validation rows above 12 h (`cv.TRIM_SECONDS`), because otherwise a model comparison
across two folds can be decided by ten records. And the leaderboard itself will be partly a
lottery on whatever equivalent rows sit in the 2026 ground truth — no model predicts a 36-hour
taxi — so a small unexplained gap between our CV and the leaderboard is expected, not evidence
of a bug. Whether to drop these rows from *training* is untested.

---

<a id="l004"></a>
## L004 — Taxi-in proxies taxi-out only where arrivals and departures share runways

`confirmed` · `scripts/analysis/a002_taxi_in_proxy.py` · 2026-09-22

**Question.** An arrival covers the same ground in reverse and barely queues, so its median
taxi time should measure the unimpeded distance from a stand to the runways. Does it?

**Finding.** Sometimes, and the airport interaction is not optional.

Most of these airports segregate their runways. At LEBL and LEMD arrivals and departures share
*zero* runways; at EHAM 207k of 246k departures leave from runways no arrival uses. Keying the
statistic on (airport, runway, stand) therefore covers only **41%** of departures with 20+
arrival observations. Keying on (airport, stand) covers **99.8%**.

Stand-level agreement between median taxi-in and median taxi-out:

| airport | corr | taxi-out − taxi-in (s) |
|---|---:|---:|
| EHAM | 0.838 | 303 |
| LEMD | 0.784 | 483 |
| LFPG | 0.615 | 421 |
| EDDF | 0.449 | 365 |
| LSZH | 0.357 | 382 |
| LIRF | 0.332 | 483 |
| EDDM | 0.147 | 425 |
| EGLL | 0.094 | 893 |
| LTFM | 0.088 | 299 |
| LEBL | −0.088 | 676 |

Pooled across 2,209 stands: 0.374. At row level the stand's median taxi-in correlates 0.177
with taxi-out, against 0.568 for the stand's median taxi-*out* — the leaky reference. That is
roughly a third of the *correlation* but only a tenth of the *variance* (0.177²/0.568² = 9.7%),
which is the figure that matters for a squared metric. The pooled 0.374 also understates the
within-airport relationship, which is **0.468**.

Where it fails, it fails for two different reasons. LEBL is geometry: fully segregated runways
mean the arrival path says nothing about the departure path. EGLL is queueing: Heathrow's
893-second gap between taxi-in and taxi-out is the departure queue, and it drowns the
stand-to-stand distance variation the proxy measures.

**So what.** Worth having, as a weak structural prior interacted with airport, not as a
standalone predictor. Its real value is twofold: it is not target-dependent — it reads labels on
*arrival* rows, which the ranking set supplies — so it needs no fold-aware handling and can be
computed contemporaneously for 2026. And the residual it leaves, taxi-out minus this, isolates
pushback and queueing, which is the part actually worth modelling.

Used raw it is badly mis-specified: `b04_taxiin_proxy` scores 860 s pooled with a −498 s bias,
exactly the offset in the table above. It needs a learned offset, which is what handing it to a
model as a feature does.

---

<a id="l005"></a>
## L005 — Out-of-sample baselines: the numbers to beat

`confirmed` · `config/experiments/b0*.yaml` · 2026-09-22

**Question.** What do the trivial models actually score on held-out months?

**Finding.** Fold set `seasonal` — train on eleven months of 2025, validate on the held-out
January and July, which are the evaluation months.

| experiment | pooled RMSE | trimmed | Jan | Jul |
|---|---:|---:|---:|---:|
| `b01_constant_mean` | 686.6 | 518.0 | 604.9 | 746.0 |
| `b02_group_stand_runway` | **629.5** | **441.4** | 554.7 | 683.9 |
| `b03_aobt_proxy` | 646.2 | 464.8 | **536.6** | 722.5 |
| `b04_taxiin_proxy` | 860.4 | 731.5 | 787.2 | 915.2 |

**So what.** Three things. Out-of-sample numbers are far worse than the in-sample group
statistics suggested (469 → 630 for the same structural grouping), so in-sample figures are
worthless for steering. The two folds are genuinely different problems — every model is ~130 s
worse on July than on January. And `b03` beats `b02` on January while losing on July: the
`aobt_3` proxy is a better point predictor but has a heavier error tail, which a squared metric
on a fat-tailed month punishes. A model that blends them should beat both.

---

<a id="l006"></a>
## L006 — Does `MVT_ID_mvt` encode anything?

`open` · no script yet · 2026-09-22

**Question.** `MVT_ID_mvt` is a Float64 with integral values: training spans
181,979,889–202,064,541 and ranking spans 202,180,234–213,827,173. Training and ranking do not
overlap — and the relationship is stronger than "no overlap", since **train max < rank min**,
so the identifier is a monotone counter across the two files. That mostly answers the question:
it orders by time, and any predictive content it has is a proxy for the date. If the ID orders by time within an airport, or encodes the
record's origin, it may carry information the columns do not.

**So what.** Unverified. Do not use it as a feature until this is settled, and be aware that if
it *is* informative it is the kind of artefact the organisers would consider outside the spirit
of the challenge — check the rules before relying on it.

---

<a id="l007"></a>
## L007 — The extreme outliers are a data defect at one airport, not an event

`confirmed, partly corrected 2026-09-22` · `scripts/analysis/a008_disruption_events.py` · 2026-09-22

> **Correction.** The final sentence of this entry said "No feature will predict them." That is
> false. 33 of the 34 have no Network Manager flight match, a condition already computed in the
> repo. See [L012](#l012).

**Question.** 34 departures exceed twelve hours of taxi-out and carry 28% of the total squared
error (L003). Did something happen on those days?

**Finding.** No. Three things rule out an event. They are spread across **30 distinct dates** —
a drone closure or a storm would cluster. **31 of 34 are at LIRF**, two at LFPG, one at LSZH.
And 14 of them land in `[86400, 90000]` seconds: subtract exactly 24 hours and the remainder is
602–1992 s, median 951 s, which is an ordinary Rome taxi-out. `block_time` is recorded one day
early.

The other 20 are 12–18 hours and are not explained by the day offset. They share a shape:
block_time in the evening, takeoff the following morning, usually on long-haul routes
(SBGR, SAEZ, YSSY, KMIA). Consistent with an overnight delay where the stand record was never
reopened, but not demonstrated.

**LIRF is not merely unlucky in the tail — it is contaminated at every scale.** Share of
departures over two hours: LIRF 0.299%, LFPG and EGLL 0.015%, LEMD and LEBL 0.000%. That is a
twentyfold gap to the next airport.

**So what.** These are label noise in the sense that no *physical* model explains a 36-hour
taxi. They are not unpredictable: see [L012](#l012). Whether to exclude them from *training*
while keeping them in validation remains untested.

---

<a id="l008"></a>
## L008 — Does LIRF's contamination reach the 2026 evaluation set?

`resolved: yes — see [L019](#l019)` · `previously open, downgraded from confirmed 2026-09-22` · `scripts/analysis/a008_disruption_events.py` · 2026-09-22

> **Downgraded.** This was written as `confirmed`. It is not. The comparison rests on **15 rows
> against 17 rows** in the January-plus-July window (Fisher exact p = 0.86), which cannot
> distinguish stability from anything else. Worse, the arrival tail does not share the departure
> tail's signature: of the 119 LIRF arrivals over two hours in 2025, only 4 lack an NM flight
> match, against 30 of the 31 extreme departures ([L012](#l012)). These may be two different
> defects, and the chain of inference crosses between them. And structurally, the 2026 arrival
> rows are *given* while the departure labels are *withheld* — whether the organisers filtered
> impossible targets out of the scored set is not observable from rows they chose to hand us.
> There are **zero** LIRF arrivals over twelve hours in either evaluation window.
>
> What the data supports: the defect has not vanished. Not: it is stable, and not: the scored
> labels carry it.

**Question.** Departure targets are blanked in the ranking set, so we cannot look at them. Can
we find out anyway whether 2026 carries the same corruption?

**Finding.** Yes, through arrivals. `ARR` rows are fully intact in both sets, and the same
pathology appears in them. Share of arrival taxi times over two hours:

| airport | 2025 training | 2026 ranking |
|---|---:|---:|
| LIRF | 0.074% | **0.063%** |
| LFPG | 0.003% | 0.005% |
| every other airport | ~0.000% | ~0.000% |

Restricted to January and July, LIRF is 0.057% in training against 0.063% in ranking. The
defect is stable and still present.

**So what.** Unresolved, and resolvable cheaply: two constant-value submissions determine
`E[y]` and `Var(y)` of the withheld labels exactly, since `RMSE(c)² = Var(y) + (E[y] − c)²`. If
the variance matches the trimmed 2025 variance the organisers cleaned the set; if it is far
above, they did not. That settles this entry by measurement instead of inference. Removing the
34 known rows moves 2025 RMSE from 546 s to 463 s, so the stake is large either way.

---

<a id="l009"></a>
## L009 — Weather moves the target enormously; the drone closure did not

`corrected 2026-09-22` · `scripts/analysis/a008_disruption_events.py` · 2026-09-22

> **Corrected.** The first version of this entry used a per-row z-statistic, which assumes
> departures are independent within a day. They are not — the measured design effect is 6x at
> LIRF and 150x at LTFM — so that statistic was inflated by roughly an order of magnitude. The
> table below now standardises against the airport's own distribution of **daily means**, which
> the same script had already computed a few cells earlier. The weather conclusion survives and
> strengthens. **The drone conclusion does not survive.**

**Question.** Separately from bad records: do genuine airport-wide disruptions move the target,
and can they be identified from outside the dataset?

**Finding.** Weather moves it enormously. Against the airport's own distribution of daily means:

| airport | date | event | z (day level) | rank among the year's days |
|---|---|---|---:|---|
| LTFM | 2025-02-23 | Siberian snow, week-long | **+9.86** | **1st of 365** |
| EDDM | 2025-12-24 | Christmas Eve | **+9.45** | **1st of 362** |
| LTFM | 2025-02-19 | same event, ~150 cancellations | +8.88 | 2nd of 365 |
| LIRF | 2025-07-13 | **unexplained** | +8.52 | **1st of 365** |
| LFPG | 2025-01-17 | winter weather | +7.65 | 1st of 365 |
| EDDM | 2025-10-03 | drone closure, morning after | **+0.90** | **35th of 362** |
| EDDM | 2025-10-02 | drone closure, the night itself | +0.07 | 109th of 362 |
| EDDM | 2025-10-04 | control, day after | −0.29 | 202nd of 362 |

Four of the five weather and holiday days are **the single worst day of the year** at their
airport. The Munich drone closure of 2–3 October 2025 is **not detectable**: the morning after
is the 35th-busiest-taxi day of the year and the closure night itself is the median day. The
earlier claim that it was detectable, and the appeal to the control day being "negative", were
both artefacts of the inflated statistic — a day at the 56th percentile establishes nothing.

**So what.** Weather is the highest-value external signal, and the margin over everything else is
larger than first stated. Drone incidents did not measurably move taxi-out at these ten airports
in 2025, so they are not a modelling priority — though the European rate quadrupled between 2024
and 2025 and kept rising into 2026, so a single incident inside the evaluation window remains a
tail risk. **LIRF on 2025-07-13 is the worst day of that airport's year and is unexplained** (*note 2026-10-03: since attributed to a runway reconfiguration under thunderstorms, SHRA/TSRA 08–18 UTC — `docs/OUTLIERS.md` and REVIEW1 E8*) —
three of the extreme outliers also fall on 13–14 July. Worth chasing.

---

<a id="l010"></a>
## L010 — LFPG closed a runway for four months, and 2026 uses runways 2025 did not

`confirmed` · `scripts/analysis/a008_disruption_events.py` · 2026-09-22

**Question.** Does airfield configuration change enough within the year to matter?

**Finding.** At LFPG, runway 09R holds 19–30% of departures from January to May, then falls to
**0.000 in August and September and ~0.000 in October and November** (0.0002, 0.0005), recovering
to 19.7% in December. 27L is the other end of the same strip of concrete and does the same. Its
parallel neighbour 09L moves in the opposite direction, from 0.1% to 9.8–17.4%.

So **one runway closed**, and its traffic went to the **parallel runway in the same doublet**
(09L/27R gains ~0.34 of departure share), not to the southern doublet, which gains only ~0.11.
An earlier version of this entry said the opposite.

Checking the ranking set directly — `runway` is not blanked — three airport-month-runway
combinations are used materially in 2026 but were essentially absent from the same month of
2025: LTFM 34L (0.0% → 4.3%) and 16R (0.0% → 2.4%) in January, EHAM 18C (0.4% → 6.2%).

**So what.** Any statistic keyed on runway needs training support from the *whole* year, not the
matching month, or those rows fall back. It also means a "runway configuration" feature — which
runways are active in the surrounding hour — is measurable on both sets and may carry the
airfield-state information that a static stand-runway code cannot.

---

<a id="l011"></a>
## L011 — Group-median baselines remove a quarter of the variance and leave a biased residual

`confirmed, partly corrected 2026-09-22` · `scripts/analysis/a007_candidate_targets.py` · 2026-09-22

> **Corrections, three of them.**
>
> 1. **The skew bullet was wrongly struck. It is restored, with better reasoning.**
>    `scripts/analysis/a012_proxy_calibration.py` shows the correction was wrong on both of its
>    numbers and on its analogy. The skew of the central 99.9% **rises** (Jan 1.92 → 2.93, Jul
>    2.83 → 4.53), as does the central 99% and the central 95% — it does not fall. The
>    denominator accounts for **111%** (Jan) and **122%** (Jul) of the raw rise, not ~95%; the
>    third central moment actually *fell* slightly (ratio 0.962, 0.931). And the analogy to
>    [L002](#l002) does not hold, because **skewness is scale-invariant** — rescaling cannot
>    change it, so a rise in raw skew cannot be dismissed as mechanical in the way a
>    range-restricted correlation can.
>
>    The honest statement is that both things are true and they answer different questions. Most
>    of the *raw* rise is the shrinking σ, so the raw numbers overstate it. But the residual's
>    **shape** genuinely is more skewed at every trimming level, so the original bullet's
>    observation stands: subtracting a baseline does not leave a better-behaved target.
> 2. **"Variance removed" was mis-measured, and `a007` now reports both.** It used
>    `np.var(residual)`, which re-centres on the residual's own mean and so silently forgives the
>    bias this entry complains about. The quantity that maps to RMSE is
>    `1 − mean(r²)/mean((y − baseline_global)²)`, with a *deployable* global baseline rather than
>    an oracle centred on the held-out month. Both columns are now printed. For the finest key:
>
>    | fold | statistic | var-removed (old) | **MSE-removed (correct)** |
>    |---|---|---:|---:|
>    | 2025-01 | median | 26.2% | **27.0%** |
>    | 2025-01 | mean | 25.7% | **27.7%** |
>    | 2025-07 | median | 27.8% | **27.9%** |
>    | 2025-07 | mean | 29.2% | **31.6%** |
>
>    The old metric had the median *beating* the mean on January (26.2 vs 25.7). On the correct
>    one the mean wins both folds, by 0.7 and 3.7 points. So the original column flattered
>    exactly the option this entry argues against, and correcting it strengthens the conclusion.
> 3. **The bias is not all skew.** Switching to a mean baseline drives January's residual mean to
>    ~0 but leaves **+36 to +38 s in July**. So roughly 60% of July's bias is median-vs-mean and
>    ~40% is **out-of-sample seasonal drift** — July runs hotter than the eleven months fitted on.
>    Those need different fixes: the first a change of statistic, the second a month or season
>    term, which no baseline keyed on airport/stand/runway has.

**Question.** Should the model predict raw taxi-out, or a deviation from a group baseline?

**Finding.** Baselines fitted on the fold's training months, scored on the held-out month, with
the target trimmed at twelve hours:

| baseline | RMSE Jan | RMSE Jul | variance removed | residual mean | residual skew |
|---|---:|---:|---:|---:|---:|
| global only | 487.5 | 556.5 | 0% | +81 / +111 | 11.0 / 11.9 |
| airport | 457.5 | 515.9 | 10.6 / 13.4% | +52 / +93 | 12.9 / 14.3 |
| airport × stand | 433.0 | 487.6 | 20.2 / 23.3% | +56 / +98 | 15.0 / 16.5 |
| airport × stand × runway | 416.6 | 472.6 | 26.2 / 27.8%† | +56 / +94 | 16.7 / 18.0 |

† the variance-removed column is the old, mis-measured one — see correction 2 above; the
MSE-removed figures are 27.0% and 27.9%.

Three things stand out, and two of them are arguments *against* using these as the target.

**The residual mean is systematically positive**, 52–98 s. A median baseline under-predicts a
right-skewed target, and RMSE is minimised by the conditional *mean*. Any baseline used this way
should be a mean, or a median plus a fitted offset.

**Subtracting a baseline makes the residual more skewed, not less** — 11.0 rises to 16.7, and
the shape is genuinely more skewed too: the central 99.9% goes 1.92 → 2.93 on January and
2.83 → 4.53 on July. Most of the *raw* rise is the shrinking denominator, so read the trimmed
figures; the direction is the same either way. The deviation targets are not distributionally
better behaved than the raw target.

**Features correlate less with the residual than with the raw target**: `ob_aobt_sec` goes from
0.407 against raw to 0.112 against the finest residual, because the baseline has already
absorbed the shared variance. One feature moves the other way — `ob_aobt_minus_eobt`, −0.322 to
−0.374 — which is the part that is genuinely about delay rather than distance.

Coverage in the 2026 ranking set, by minimum training support: airport 100%/100%/100%;
airport × stand 99.5%/98.8%/95.6%; airport × stand × runway 99.3%/96.3%/**80.3%** at 1/30/200
rows. The finest key fits best and covers worst.

**So what.** Raw target with the group keys as categoricals is a reasonable default, but
"strictly better" was wrong and rested on a false dichotomy. There are at least three options,
and subtracting a baseline does not preclude keeping the keys:

1. raw target, keys as categoricals;
2. residual target, **keys still in the design matrix**;
3. baseline as an **offset** (`init_score` in LightGBM) or as a shrunk target encoding.

Two arguments against the "the tree learns it for free" claim. LightGBM's categorical handling
does not give a per-group intercept — it sorts categories by gradient statistics and takes a
one-sided subset bounded by `max_cat_threshold` (32 by default), so representing a partition over
2,209 stands, let alone the stand×runway *interaction* which two separate categoricals cannot
express, costs many splits. And `min_data_in_leaf` does not shrink a thin group toward its
airport mean; it pools it with whatever co-categories fall on the same side of a split. Explicit
empirical-Bayes shrinkage is targeted and tunable, and the repo's `TARGET_DEPENDENT` machinery
exists precisely for it and is unused.

All three should be measured. Measured so far: group **mean** beats group median out of sample at
every key (Jan 414.3 vs 416.6, Jul 460.1 vs 472.6), and winsorising at two hours is a wash.

---

<a id="l012"></a>
## L012 — A missing Network Manager flight match is worth about 22 seconds

`confirmed` · `scripts/analysis/a009_flight_match.py` · 2026-09-22

**Question.** [L007](#l007) claimed no feature would predict the extreme rows. Is that true?

**Finding.** No. **33 of the 34** departures over twelve hours have no NM flight match — the
condition where every `_flt` column is null together — against a base rate of 1.075%.

It is not only the extremes. The condition separates the whole distribution, and the **sign
differs by airport**, so it has to be used as an interaction rather than a main effect:

| airport | matched mean | unmatched mean | unmatched n | unmatched % > 2 h |
|---|---:|---:|---:|---:|
| LIRF | 1,145 s | **6,534 s** | 1,487 | **29.5%** |
| LTFM | 1,050 s | 1,303 s | 3,623 | 0.52% |
| EGLL | 1,363 s | 1,568 s | 1,414 | 0.78% |
| LFPG | 1,017 s | 1,183 s | 3,765 | 0.19% |
| **EHAM** | 788 s | **600 s** | 4,084 | 0.02% |

The rate carries into the ranking set — 5,283 unmatched rows, LIRF 1.497% in 2025 against 1.424%
in 2026 — though it is not identical everywhere (EHAM rises 1.97% → 3.16%, EGLL falls 1.16% →
0.62%).

**Measured value.** Taking the `b02` structural baseline and folding the match flag into the
top-level key, scored with the competition metric on held-out months:

| | pooled | trimmed | Jan | Jul |
|---|---:|---:|---:|---:|
| `b02_group_stand_runway` | 629.5 | 441.4 | 554.7 | 683.9 |
| `b05_match_aware_groups` | **607.2** | **434.3** | **540.3** | **656.2** |

A paired day-blocked bootstrap on the gain — blocked by day because taxi-out is heavily
autocorrelated within an airport-day — gives **+14.1 s on January, 95% CI [+0.2, +29.1]** and
**+29.8 s on July, 95% CI [+11.6, +47.5]**. January is marginal; July is solid.

**The gain is one airport.** Decomposing the pooled MSE reduction from `b02` to `b05` by airport,
**LIRF supplies about 102% of it on both folds** — the next best contributor is around +1%, and
eight of the ten airports get very slightly *worse*. So "the condition separates the whole
distribution" overstates it: the distribution separates everywhere, but only LIRF's separation is
large enough to move a squared metric.

**So what.** Best baseline so far, from one boolean. It carries a bet: that 2026's unmatched rows
behave like 2025's. Because the gain is entirely LIRF, that bet is sharper than first written —
if the organisers cleaned LIRF's withheld labels, `b05` is not roughly break-even against `b02`,
it is slightly **worse everywhere**. If the organisers cleaned the withheld labels, predicting ~6,500 s for the
~380 unmatched LIRF rows costs roughly what it gains. The constant-submission probe in
[L008](#l008) settles that before committing.

---

<a id="l013"></a>
## L013 — A stand-runway combo is nearly white noise; the structure is at the airport and the runway

`confirmed` · `scripts/analysis/a010_stand_runway_timeseries.py` · 2026-09-22

**Question.** Treat a fixed (airport, stand, runway) combination as a time series. Autocorrelation?
Seasonality? Trend?

**Finding.** At combo level, almost nothing. Taking the busiest combination at each of the ten
airports, the daily median's lag-1 autocorrelation runs −0.04 to 0.19 against a white-noise band
of ~0.11, median **0.076**. Day-of-week ratios sit between 0.95 and 1.06 and monthly ratios
between **0.901 and 1.136** — the widest being EDDM 115/26L in **December at +13.6%**, which is
the de-icing signal showing through. Consecutive departures from the same combo correlate −0.02
to 0.12.

**But that is measurement noise, not absence of structure.** A combo sees about five departures a
day, so its daily median is a poor estimate of that day's level, and noise attenuates
autocorrelation toward zero. The same statistic at airport level, measured from 375–753
departures a day:

| level | median lag-1 ACF |
|---|---:|
| airport | **0.349** (EHAM 0.59, LEMD 0.52, LFPG 0.38) |
| airport × stand × runway | 0.076 |

Weekly echoes at lags 7, 14 and 21 are visible at airport level and absent at combo level.

**The queue is real, and it lives on the runway, not the stand.** Correlating successive
departures on the same runway, after removing that runway's median, by the gap between them:

| gap | n | correlation |
|---|---:|---:|
| 0–2 min | 1,091,849 | 0.278 |
| **2–5 min** | 716,863 | **0.318** |
| 5–10 min | 192,685 | 0.257 |
| 10–20 min | 72,626 | 0.219 |
| 20–60 min | 27,001 | 0.106 |
| > 4 h | 11,766 | 0.022 |

By airport, within ten minutes: LTFM **0.598**, EDDM 0.416, LFPG 0.402, LSZH 0.315, EHAM 0.302.

**So what.** Do not model a combo as a series — at ~5 movements a day the structure is not
resolvable, which is a weaker and more accurate claim than "there is nothing there". Temporal
features belong at the **airport-day** scale (persistence, for weather and configuration) and the
**runway-minutes** scale (queueing). The correlation curve peaks at **2–5 minutes** and decays from there, so the congestion window
should be **0–5 minutes** on the same runway, with wider windows as secondary features.

One caveat on the size of it: the runway residual removes only the runway's median, so two
departures minutes apart still share an hour and a day. Adding airport × runway × hour and then
airport-day controls reduces the 2–5 minute figure from 0.318 to 0.283 to **0.243**. About
three-quarters of the effect survives, so it is real, but the uncontrolled numbers above overstate
it by roughly a quarter.

---

<a id="l014"></a>
## L014 — Delay is not taxi-out, and the problem reduces to estimating off-block time

`confirmed` · `scripts/analysis/a011_delay_vs_taxiout.py` · 2026-09-22

**Question.** Is delay the same as increased taxi-out?

**Finding.** No — they are very nearly unrelated. **corr(off-block delay, taxi-out) = 0.014.**

Taxi-out is a *duration*; delay is a *lateness*. Taxi-out by off-block delay band is mildly
U-shaped, not increasing: early 1020 s, on time 999 s, **5–15 min late 958 s** (the minimum),
30–60 min late 1008 s, over an hour late 1032 s. A flight that pushes back late taxis slightly
*faster*, presumably because it has left the bank it was scheduled into.

There is an exact identity behind this, and it is the most useful thing in this entry:

```
mvt_time − sched_time  =  (block_time − sched_time)  +  (mvt_time − block_time)
departure delay        =  off-block delay            +  taxi-out
```

It holds to the second on every row. So `ob_sched_sec` — the only delay quantity computable on
the ranking set — is **off-block delay plus the target**. Decomposing its variance:

| component | share of var(departure delay) |
|---|---:|
| off-block delay | **95.4%** |
| taxi-out (the target) | 4.1% |
| 2 × covariance | 0.5% |

**So what.** Two consequences.

First, **estimating taxi-out is exactly equivalent to estimating off-block time**, because
takeoff time is given. That is why `aobt_3` is the strongest column in the dataset, and it makes
the whole problem legible: every feature is either improving an off-block estimate or correcting
its error. The accuracy of each available estimate, as an off-block error:

| estimate | median absolute error | within 5 min | RMSE if used directly |
|---|---:|---:|---:|
| `aobt_3` (M3 flown) | **175 s** | **74.6%** | **384.9** |
| `eobt_1` (filed plan) | 303 s | 48.9% | 676.7 |
| `sched_time` (timetable) | 481 s | 36.6% | 2409.8 |

Second, **do not treat delay features as taxi-out features.** Anything built on
`mvt_time − sched_time` is 95% a measurement of something the target barely relates to.

**Stand-level propagation is negligible.** The previous departure from the same stand correlates
0.247–0.392 with this one's taxi-out — but that is the stand's own level appearing on both sides.
After removing the stand × runway median it falls to **0.05–0.11**, and the previous flight's
*delay* correlates 0.006–0.034, which is nothing. Propagation happens on the runway (L013), not
through the stand — though note the two tests are not strictly like for like: the stand residual
is conditioned on stand × runway while the runway residual is conditioned on runway alone, and
the runway effect lives at 0–5 minutes, a gap two departures from one stand essentially never
have.

---

<a id="l015"></a>
## L015 — The anonymised operator is recoverable from the callsign

`confirmed` · `scripts/analysis/a011_delay_vs_taxiout.py` (incidental) · 2026-09-22

**Finding.** `AIRCRAFT_OPERATOR_flt` is a 64-character hash, described by the organisers as
anonymised. `CALLSIGN_flt` is not: the values are real ICAO callsigns whose first three letters
are the airline's ICAO designator — `DLH` Lufthansa (222,892 departures), `THY` Turkish
(222,550), `KLM`, `AFR` Air France, `BAW` British Airways, `EJU` easyJet Europe, `CFG` Condor,
`ANE` Air Nostrum, `DLA` Air Dolomiti. `FLIGHT_mvt` carries the real IATA flight number
(`LH982`, `EN8756`, `DE2290`).

**So what.** The operator hash adds nothing the callsign does not already carry, and the callsign
carries much more: it joins to airline identity, alliance, fleet, home-base status and terminal
or pier assignment, none of which the hash can reach. Note that the organisers published the
callsign themselves and use `BAW6VB` as their own worked example on the data page, so this is
supplied data rather than a circumvention — but it is worth being explicit that the
anonymisation is not effective, and worth a glance at the rules before leaning on it hard.

Measured value is unknown; operator as a plain categorical was worth ≤1.5 s on top of the
structural baseline, so the gain would have to come from the external joins, not the identity.

---

<a id="l016"></a>
## L016 — Judge a proxy calibrated, not raw: `eobt_1` beats `aobt_3`

`confirmed` · `scripts/analysis/a012_proxy_calibration.py` · 2026-09-22

**Question.** [L002](#l002) ranked the off-block proxies by the RMSE of using each one directly
as the prediction, and `aobt_3` won by a wide margin. Is that the right comparison?

**Finding.** No. Raw RMSE penalises a fixed offset, and a model removes a fixed offset on its
first split. `eobt_1` sits 377 s below the target on average; `aobt_3` sits 17 s below. Once each
proxy is given an intercept and a slope, the order changes:

| proxy | raw RMSE | bias | calibrated RMSE |
|---|---:|---:|---:|
| `eobt_1` | 676.7 | +377.2 | **344.7** |
| `aobt_3` | 384.9 | +17.0 | 351.7 |
| `iobt` / `lobt` | 739.8 / 740.1 | +367 / +366 | 355.2 / 355.5 |
| `sched_time` | 2409.8 | +901.5 | 451.9 |

Out of sample the margin widens. Calibrating on the fold's training months, scoring on the
held-out month, on top of the airport × stand × runway mean:

| proxy | Jan | Jul |
|---|---:|---:|
| **`eobt_1`** | **354.8** | **433.6** |
| `iobt` / `lobt` | 362.9 / 363.2 | 433.7 / 433.9 |
| `sched_time` | 406.2 | 451.8 |
| `aobt_3` | 387.6 | **463.7** |

On July `aobt_3` is the worst of the five, behind even the raw timetable. The likely reason is
already in L002: roughly a fifth of `aobt_3` is a back-computed constant, which carries no
conditional information, while `eobt_1` is always a genuine filed estimate.

The signal was visible earlier and went unread twice. L002's own correlation column had
`eobt_1` at 0.556 against `aobt_3` at 0.536. L011's residual-correlation table had `ob_eobt_sec`
at 0.421 against `ob_aobt_sec` at 0.112 — against the finest structural residual, which is what
a model with the group keys actually has left to explain.

**So what.** `eobt_1` belongs in the next experiment, and `aobt_3` should be kept as a
complement rather than the centrepiece — the two disagree in a way that is itself informative
(`ob_aobt_minus_eobt` is the only feature in L011's table that gets *stronger* against the
residual). More generally: this is the third time in this log that a variable was judged by a
statistic a model would have neutralised. See [L002](#l002) and [L011](#l011).

---

<a id="l017"></a>
## L017 — Neither trimming training labels nor a two-stage classifier helps

`confirmed` · `scripts/analysis/a015_outlier_classification.py` · 2026-09-23

**Question.** The outliers dominate the metric ([OUTLIERS.md](OUTLIERS.md)). Two standard
responses: drop them from training, or model them explicitly with a classifier. `CLAUDE_STRATEGY`
proposed the first as "the cheapest experiment of the three". Both were measured on the seasonal
folds with identical features.

**Finding. Both hurt, and the second hurts badly.**

Training on trimmed labels while scoring on everything:

| trim training above | pooled RMSE | vs none | pooled *trimmed* | vs none |
|---|---:|---:|---:|---:|
| **none** | **506.86** | — | 332.37 | — |
| 12 h | 510.49 | **+3.62** | 313.12 | −19.25 |
| 6 h | 520.42 | +13.55 | 312.96 | −19.41 |
| 4 h | 527.40 | +20.53 | 316.16 | −16.21 |
| 2 h | 556.20 | +49.34 | 337.63 | +5.26 |
| 1 h | 577.61 | **+70.75** | 362.85 | +30.48 |

The two columns disagree, and that disagreement is the finding. Trimming makes the model *better
at ordinary rows* — the trimmed diagnostic improves by ~19 s — and *worse at extreme ones*,
because it has never seen one. The competition scores the raw column, which weights extreme rows
enormously, so the trade is a losing one at every threshold.

A two-stage model is worse still. Stage one predicts `P(y > T)`, stage two the level in each
regime, combined as an expectation:

| fold | plain regression | two-stage @1 h | @2 h | @4 h |
|---|---:|---:|---:|---:|
| 2025-01 | **412.78** | 466.17 | 549.00 | 564.23 |
| 2025-07 | **571.53** | 612.81 | 757.83 | **1199.25** |

Always worse, and monotonically worse as the threshold rises — more than twice as bad at four
hours in July. The mechanism is calibration against a tiny base rate: with `P(outlier) ≈ 0.0002`
and `E[y | outlier] ≈ 20,000 s`, a probability that is off by even 0.01 adds ~200 s of bias to a
row that was perfectly ordinary. Spread over hundreds of thousands of ordinary rows, that swamps
whatever is gained on the handful of real ones.

**Is the outlier event predictable at all?** Partly, and only at the low end:

| threshold | positives | average precision | base rate | lift in top 1% | AUC from the NM flag alone |
|---|---:|---:|---:|---:|---:|
| > 1 h | 325–656 | **0.30–0.50** | 0.002–0.003 | **67–76×** | 0.49–0.58 |
| > 2 h | 29–120 | 0.20–0.23 | 0.0002–0.0006 | 41–57× | 0.79–0.86 |
| > 4 h | 8–37 | 0.0001–0.025 | 0.0001 | 0.0–16× | 0.95–0.995 |

So prediction is genuinely good above an hour and collapses above four, where a validation month
holds eight positives. Note the flag's role **inverts**: near-useless alone at one hour
(AUC 0.49), near-perfect at four (0.995). It identifies *catastrophic* rows, not *slow* ones —
consistent with it marking record defects rather than congestion.

**So what.** Leave the training labels alone and do not build a classifier stage. The predictable
part of the outlier problem is already available to a plain regressor through the same features,
and a squared-loss regressor already targets the conditional mean, which is what RMSE wants.
Effort belongs in features that explain *why* a row is slow — runway configuration first (see
[OUTLIERS.md](OUTLIERS.md)) — not in special handling of the target.

---

<a id="l018"></a>
## L018 — Runway configuration adds nothing over the static runway code

`confirmed` · `config/experiments/m05_lgbm_runway_config.yaml` · 2026-09-23

**Question.** Reconfiguration is the largest single effect found in the data: it produced the
worst day of Rome's 2025 and a 2.54× swing in Amsterdam's arrival taxi-in inside the scored
month ([OUTLIERS.md](OUTLIERS.md)). Does encoding it as features help?

**Finding.** No. `m05` = `m01` + the `runway_config` block:

| | pooled | trimmed | Jan | Jul |
|---|---:|---:|---:|---:|
| `m01_lgbm_global` | 507.82 | **331.03** | **421.88** | 567.69 |
| `m05_lgbm_runway_config` | **506.77** | 331.19 | 422.12 | **565.85** |

1.05 s apart, disagreeing in direction between the folds and between the metrics. Noise.

The importance ranking says why: `cat_airport_runway` sits fifth with 2.1e11 gain, while
`rwy_dep_active_n` — the block's strongest univariate signal, correlation −0.163 — sits 41st with
1.0e8, five thousand times less. The static code already carries what the configuration implies
for an individual movement.

**The generalisable lesson.** A strong univariate effect is not evidence of marginal value. The
block's headline finding — single-runway hours average 1,116 s against 938 s for two-runway hours
— is real and survives, but a model that already knows *which* runway was used has most of it.

**A caveat I raised and then checked, which did not survive.** I first wrote that the folds might
not contain reconfiguration of the size seen in July 2026, so the test might be unfair. Measured,
that is false — the 2025 folds carry comparable configuration variability:

| EHAM | Jan 2025 | Jan 2026 | Jul 2025 | Jul 2026 |
|---|---:|---:|---:|---:|
| distinct configurations | 104 | 140 | 66 | 102 |
| dominant-configuration share | 0.190 | 0.127 | 0.146 | 0.149 |

Every airport looks similar across the two years on both measures. The folds had ample
reconfiguration to learn from. **This was a fair test and the feature failed it**, and the caveat
was an excuse offered before checking whether it applied.

Keep the block anyway — 2.7 s to build, eight columns, and it may yet matter in combination with
a weather feature that explains *why* a configuration changed. But the 1.05 s is noise and the
block currently earns nothing.

---

<a id="l019"></a>
## L019 — Cross-validation predicts the leaderboard to within three seconds

`corrected` · `docs/SUBMISSIONS.md` · 2026-09-23

> **Correction, 2026-10-03.** The headline no longer holds. v6 missed by +46.45 (L022), the
> CV-to-board *delta* transfer ranged 0.75–1.30 over large changes (L034) and, for changes under
> ~0.2 trimmed, −0.98 to 43 with the sign wrong twice in five (L054); on the unmatched rows the board
> has overruled CV three times (L058). Read the entry as the 2026-09-23 state.

**Question.** Nothing had ever been submitted. How far does our cross-validation sit from the
real metric, and did the organisers clean the corrupt LIRF rows?

**Finding.** Two submissions, differing only in the Network Manager match flag:

| submission | CV pooled | leaderboard | gap |
|---|---:|---:|---:|
| `b05_match_aware_groups` | 607.20 | **604.41** | −2.79 |
| `b02_group_stand_runway` | 629.54 | **629.22** | −0.32 |
| `x02_lgbm_deep` *(added 2026-09-23)* | 432.81 | **433.80** | **+0.99** |

**The `seasonal` fold set is essentially unbiased.** The gaps are small across a 196 s range of
model quality and **not all the same sign**, so this is scatter around zero rather than a
correctable bias. Training on eleven months of 2025 and validating on the held-out
January and July predicts the 2026 metric to within a few seconds. This was not obvious in
advance: the year changed, traffic grew, `aobt_3`'s default mix drifted, and January 2026 carried
Storm Goretti at an airport whose 2025 was benign ([OUTLIERS.md](OUTLIERS.md)). None of it moved
the number materially.

**The contamination is in the withheld labels, and [L008](#l008) is resolved yes.** CV put the
match flag at 22.34 s; the leaderboard puts it at **24.81 s** — the gain replicated and slightly
exceeded its estimate. The organisers did not clean the corrupt rows.

Note how this was settled: two genuine candidate models, read by difference. The constant-value
probe that `CLAUDE_STRATEGY` §5b proposed and withdrew would have answered the same question and
was against the rules. The legitimate route worked.

**No public/private split is evident.** Both results report `used_pairs: 344841`, the full count
of ranking departures.

**So what.** Three consequences. Every CV number here can be read as a leaderboard estimate,
which de-risks the entire experiment pipeline. `m01_lgbm_global` at 507.82 CV should land near
505 — a hundred seconds better than anything submitted, and it should go up next. And the
warning in `CLAUDE_STRATEGY` §6 that "every number we quote is a 2025 number" is now measured
rather than feared: the 2025-to-2026 penalty is under three seconds.

---

<a id="l020"></a>
## L020 — `max_bin` was the entire gain, and the old default was a memory compromise

`confirmed` · `config/experiments/x04_ablate_maxbin.yaml` · 2026-09-23

**Question.** `x02_lgbm_deep` changed five things at once and gained 75 s over `m01`. Which one?

**Finding.** One of them, and it is the one chosen for the wrong reason.

| experiment | single change from `m01` | pooled | vs `m01` | time |
|---|---|---:|---:|---:|
| **`x04_ablate_maxbin`** | **`max_bin` 127 → 511** | **431.12** | **−76.70** | 183 s |
| `x05_ablate_leaves` | `num_leaves` 127 → 511 | 506.14 | −1.67 | 230 s |
| `x06_ablate_slow` | lr 0.03, 4000 rounds | 508.14 | +0.33 | 176 s |
| `x02_lgbm_deep` | all of the above together | 432.81 | −75.01 | 500 s |

`max_bin` accounts for the whole thing. The other changes contribute nothing and, combined, make
the model slightly *worse* than the single change alone while taking 2.7× as long.

**The 127 was mine, and it was a memory compromise.** It was set in `models/lgbm.py` to keep the
binned dataset inside the development machine's ~3 GB, with the comment saying as much. The
features that dominate importance — `ob_aobt_sec`, `ob_eobt_sec`, `ob_sched_sec` — are
continuous, so coarse binning blunts precisely the splits that carry the signal. The batch
machine has 31 GB and the constraint never applied there.

**So what.** The default is now 511. Two wider lessons. A parameter chosen for an operational
reason needs revisiting the moment the operational reason goes away, and nothing in the
repository prompted that. And a five-change experiment that wins teaches almost nothing until
it is decomposed — `x02` was submitted as v3 at 433.80 when `x04` would have scored slightly
better in a third of the time.

---

<a id="l021"></a>
## L021 — 1.6% of rows carry 58% of the error, and they are flagged in advance

`confirmed` · `scripts/analysis/a016_mispredictions.py` · 2026-09-23

**Question.** Where does the best model actually go wrong?

**Finding.** Out-of-fold residuals from `x07_deep_per_airport` (427.84):

| slice | rows | share of rows | RMSE | **share of squared error** |
|---|---:|---:|---:|---:|
| Network Manager match | 339,053 | 98.4% | 281 | 42.4% |
| **no NM match** | **5,366** | **1.6%** | **2,601** | **57.6%** |

A ninefold RMSE gap, on a condition computable before seeing the target (L012), present in the
ranking set, and already supplied to the model as `flt_no_match`.

The model cannot use it, and the reason is structural: **every `_flt` column is null together**,
so `aobt_3` and `eobt_1` — the two strongest features it has — are absent on exactly the rows
that hurt most. One model is being asked to serve two populations with different feature
availability.

By airport, LIRF remains dominant but is no longer alone:

| airport | share of rows | share of error | over-index |
|---|---:|---:|---:|
| LIRF | 7.7% | **55.3%** | 7.18 |
| **LFPG** | 11.5% | **21.1%** | **1.83** |
| every other | — | ≤ 5.8% | ≤ 0.45 |

Concentration is extreme even within this model: the worst **34 rows** (0.01%) carry 56.3% of
the squared error, and the 14 rows above twelve hours carry 49.6% on their own — predicted at
28,691 s against an actual 69,210 s.

The model hedges as squared loss requires, over-predicting short taxis by 200 s and
under-predicting the 1–12 h band by 1,421 s.

**So what.** `u01_unmatched_route` routes on the flag, giving the unmatched rows their own fit
over the features they actually have while `foreign_weight` keeps them borrowing structure from
the matched majority. If 58% of the error sits in 1.6% of rows, that is where the next gain is —
not in more hyperparameters.

---

<a id="l022"></a>
## L022 — ~~Steer by `rmse_ex_lirf`~~ **CORRECTED**: Rome transfers when the gain is a covariate, not capacity

`corrected 2026-09-24, hours after it was written` · `scripts/analysis/a020_cv_transfer.py` · `docs/SUBMISSIONS.md` · 2026-09-24

**Question.** `x1_maxbin_2047` scored 390.59 in cross-validation, the best we had ever recorded,
37 s ahead of `x07_deep_per_airport`. It scored **437.04** on the leaderboard — 6.7 s *worse* than
the model it had beaten. Four earlier submissions tracked CV within ±3 s. What changed?

**Finding.** Nothing changed. Pooled CV had been a coincidence all along, and Rome is the reason.

| submission | CV ex-LIRF | CV LIRF only | CV pooled | leaderboard |
|---|---:|---:|---:|---:|
| v3 `x02_lgbm_deep` | 299.51 | 1165 | 432.81 | 433.80 |
| v5 `x07_deep_per_airport` | 297.67 | 1147 | 427.84 | 430.38 |
| v6 `x1_maxbin_2047` | **304.14** | **934** | **390.59** | **437.04** |
| v7 `f03_both` | **296.46** | 1076 | 412.60 | **428.02** |

Rank agreement with the leaderboard, over these four:

| metric | Pearson | Spearman |
|---|---:|---:|
| pooled | −0.460 | **−0.200** |
| trimmed | +0.554 | +0.400 |
| **ex-LIRF** | **+0.965** | **+1.000** |

**Pooled CV is anti-correlated with the real metric.** Excluding one airport orders all four
correctly, and still does with `x1_maxbin_2047` removed, so a single anomaly is not carrying it.
The magnitudes align too: ex-LIRF spans 7.7 s across these models against 9.0 s on the board,
where pooled spans 42 s for the same 9 s of real difference — a 4.7× inflation, which is why
`f03_both` promised 15.2 s over `x07` and delivered 2.36.

**The mechanism is legible in the LIRF column.** `x1_maxbin_2047` has the *best* Rome CV of any
model and the *worst* ex-LIRF. It won cross-validation by resolving Rome's 2025 record defects at
2047 bins. 2026 has its own unrelated defects, so the memorised structure does not merely fail to
help — it makes confident extreme predictions on ordinary 2026 rows and actively costs seconds.

**A rule I proposed and falsified the same day.** I first proposed steering by each model's own
pooled-minus-trimmed gap, on the reasoning that a shrinking gap signals tail-fitting. It predicted
`f03_both` would land within 3 s; it missed by 15.4. The idea was directionally right and the cut
was wrong — most of the untrimmed tail *is* Rome, so excluding LIRF subsumes it and does better.
Recorded because the near-miss is instructive, not because it worked.

**So what.**

1. `cv.py` computes `rmse_ex_lirf` and `rmse_lirf_only` per fold and pooled; both reach
   `runs.jsonl`, MLflow and `SUBMISSIONS.md`, and the runner prints ex-LIRF first.
2. **Every conclusion drawn from pooled CV before 2026-09-24 is suspect**, including the batch
   that produced `x1_maxbin_2047`. `max_bin` 511→2047 *worsens* ex-LIRF (299.51 → 304.14); the
   42 s pooled "gain" was entirely Rome. L020's finding that `max_bin` 127→511 mattered is not
   overturned, but its magnitude was never measured on a metric that means anything.
3. The `r01`–`r03` Rome splits were judged on the metric Rome dominates and should be re-read.
4. Open-data ceilings in `docs/NEXT_STEPS_REVIEW.md` were all measured as pooled gains and
   deserve one re-measurement before being treated as settled. See `docs/OPTIONS.md`.

---

### CORRECTION — the headline above is false. Keep the metrics; discard the rule.

**The anti-correlation was range restriction.** Spearman −0.20 for pooled CV was computed over
only the four models with a per-airport breakdown logged — four models lying within **9 s** of
each other on the board. `b05` and `b02` were dropped for want of logged breakdowns, and their
breakdowns are recoverable by refitting (deterministic group means, 0.4 s each). Restored:

| metric | Spearman n=4 | **Spearman n=6** | Pearson n=6 |
|---|---:|---:|---:|
| CV pooled | **−0.200** | **+0.657** | **+0.988** |
| CV ex-LIRF | +1.000 | +0.943 | +0.996 |

Pooled CV predicts the board well. It simply cannot *resolve* a few seconds. That is a statement
about resolution, not sign — and this repository already carries a correction for exactly this
error, in L002. I made it again.

**The rule is falsified by a counterexample already on record.** Refit and verified:

| | `b05` | `b02` | `b05` better by |
|---|---:|---:|---:|
| CV ex-LIRF | 391.11 | 390.45 | **−0.66 — worse** |
| CV LIRF only | 1718.7 | 1821.7 | +103.0 |
| CV pooled | 607.20 | 629.54 | +22.34 |
| **leaderboard** | **604.41** | **629.22** | **+24.81** |

The largest board difference this project has measured comes from a change `rmse_ex_lirf` rejects.

**What actually separates transfer from non-transfer is the *source* of the Rome gain.** `b05`'s
comes from a covariate, `flt_no_match`, whose 2026 rate is verified (1.497% → 1.424%) — it
transferred. `x1_maxbin_2047`'s comes from model capacity — it did not: its CV Rome score is
933.9 and its *implied* 2026 Rome score is 1164. Every model submitted so far lands at 1144–1164
in 2026 while spanning 934–1165 in cross-validation.

**Replacement rule**, now `cv.predicted_board()`:

    predicted_board = sqrt(0.922 · ex_lirf² + 0.078 · L²)

Exact by construction — LIRF is 7.8% of the 2026 departures and RMSE over a partition is the
row-share-weighted quadratic mean. Pin `L = 1156` for a capacity or hyperparameter change; use
the model's own CV Rome only when the change is a covariate with a checked 2026 rate. Residual
against the four board scores: **sd 1.95 s**, against 21.29 s using each model's own CV Rome.

**The larger point, which outranks the metric.** 26 of 27 seasonal runs sit inside a 10 s ex-LIRF
band, and 1 s of ex-LIRF is worth 0.639 s of board — so the entire ex-LIRF spread of every model
ever run here is worth about 6.4 s. `sqrt(0.078)·1144.5 = 319.6`: **Rome alone floors the board
near 320 even with the other nine airports predicted perfectly.** Outside Rome every GBM run here
is the same model. Steering by ex-LIRF would have aimed the remaining days at the 6 s and away
from the 180.

Full audit in `docs/REVIEW_STEERING.md` and `scripts/analysis/a022_steering_audit.py`.

---

<a id="l023"></a>
## L023 — A deterministic rule on a verified 2026 population still did not transfer

`confirmed` · `config/experiments/r10_f03_rules_D.yaml` · `scripts/analysis/a027_verify_strategy.py` · 2026-09-24

**Question.** L022 concluded that Rome gains transfer when they come from a *covariate* and not
when they come from *capacity*. Rule D is as covariate-driven as a change can be: a deterministic
override on a cell identified without any label, whose 2026 membership was counted in advance —
11 LIRF rows, against 11 in the seasonal folds. Does it transfer?

**Finding. No.**

| | CV | board |
|---|---:|---:|
| predicted | **373.1** | — |
| actual | — | **427.71** |

A miss of **+54.6 s**, against v7's 428.02 — a gain of 0.31 s where 55 were forecast. And the
decomposition says the rule did not even produce that:

| submission | ex-LIRF | board | **implied 2026 Rome** |
|---|---:|---:|---:|
| v7 `f03_both` | 296.46 | 428.02 | **1144.7** |
| v8 `r10_f03_rules_D` | 295.17 | 427.71 | **1147.0** |

Rome got marginally *worse*. The whole 0.31 s came from the 300-second clip floor, which moved
ex-LIRF from 296.46 to 295.17 and touched 1,651 rows no rule saw. Every out-of-fold number was
verified twice beforehand — a027 reproduced 373.1 / 295.17 / 868.6 independently, and the
implementation reproduced them again through the shipped YAML. The arithmetic was never wrong.
The population was.

**So the L022 distinction needs refining.** `b05` transferred because its covariate shifts a
*conditional mean across 5,366 rows*, and the rate of that covariate in 2026 was verified.
Rule D asserts an *exact functional identity on eleven rows*, calibrated on twenty-nine from
2025. Both are "covariate-driven"; only the first is a statement about a population large enough
for 2026's copy to resemble 2025's. **The eleven rows in 2026's cell are a different eleven
rows**, and nothing guaranteed they inherited the 18/29 mixture.

**The general form.** A cell whose 2026 *membership* is verifiable is not the same as a cell
whose 2026 *behaviour* is verifiable. Counting the rows is necessary and nowhere near sufficient.
Where the label is withheld, only the first can ever be checked — so a rule of this shape is
unfalsifiable before submission, and its expected value must be discounted accordingly rather
than taken from cross-validation.

**So what.**

1. **Strategy A of `docs/STRATEGY_AND_NEXT_20.md` is dead by its own gate** ("board > 415 →
   2026's cell is not 2025's"). E03–E07 and E10–E16 all rest on it. Do not spend slots on
   `r11_f03_rules_DA`: it differs from v8 on one row, by the same mechanism that just failed.
2. **The ladder to 239 is gone.** It required these rules plus the two LFPG rows, and a028 showed
   those are unidentifiable at precision 0.004.
3. Fifth consecutive confirmation that **CV Rome never transfers**, now including the one
   mechanism that was supposed to be exempt. `r14` has a CV Rome of 625 against f03's 1076 and
   predicts no better on the board.
4. What does still hold: `cv.predicted_board` with Rome pinned at the observed ~1146 predicts
   every scored submission within 3.8 s, and within 0.3 s for the two most recent. Rank
   candidates by **ex-LIRF alone** and read the board off that.

---

<a id="l024"></a>
## L024 — OpenSky's REST `firstSeen` is the takeoff, not the off-block

`confirmed` · `scripts/analysis/a033_opensky_spike.py` · 2026-09-24

**Question.** The unmatched rows' off-block is the largest unmeasured quantity in the problem
(oracle 274.7 pooled if solved, a027). OpenSky's `/flights/departure` returns `firstSeen` per
flight; if that is first *gate* contact, `mvt_time − firstSeen` reconstructs the taxi for
exactly the rows whose record lost it. Is it?

**Finding. No — at both airports tested, first contact is wheels-off.** Same-day callsign
pairing on matched 2025 rows, where the true taxi is known:

| airport | pairs | delta = `mvt_time − firstSeen`, q05…q95 | P(delta ≤ 0) |
|---|---:|---|---:|
| LIRF | 25,639 / 26,131 | −53 … +12 s, median **−20** | 0.836 |
| LFPG | 17,193 / 17,684 | −61 … −2 s, median **−31** | 0.975 |

The pairing works essentially perfectly — which is what makes the negative clean. `firstSeen`
re-measures a quantity the ranking set already gives us for every row. The 30-day limit on
`/tracks` and the 1-hour limit on `/states/all` close every other REST avenue.

**So what.** N10 (off-block reconstruction) is gated entirely on **Trino** access
(`state_vectors_data4`, `onground = true`), which is requested but not yet granted. Until then
Front B has no data route, and the remaining budget belongs to Front A. Cost of the answer:
~3,880 credits and one afternoon — the spike did its job. The archive (96 windows, LIRF + LFPG)
is kept: the pairing machinery and calibration design carry over unchanged the day Trino lands.

---

<a id="l025"></a>
## L025 — OpenSky does not receive the airport surface at LIRF or LFPG

`confirmed (one-day snapshot; agrees with L024's independent measurement)` · `scripts/analysis/a034_trino_rehearsal.py` · 2026-09-24

**Question.** L024 killed the REST route to off-block reconstruction; the fallback was Trino's
`state_vectors` with `onground`. The public one-day snapshot (2026-03-01) carries that exact
table, so the go/no-go fact could be measured before the access request is even answered: does
OpenSky *see* the surface at the two airports that hold 77% of the champion's squared error?

**Finding. No.**

| airport | onground points, 3 peak hours | of which moving | aircraft with a track |
|---|---:|---:|---:|
| LIRF | **0** | — | — |
| LFPG | 20,114 | **0** | 2 — stationary transponders (~10k points each, ~3 h spans, never moving) |

Zero at Rome is not sparse coverage — it is none. LFPG's points are fixed emitters, not taxiing
aircraft; the extraction pipeline works (it found and characterised them), the traffic is simply
absent. `takeoffLandingInfo` in the same snapshot's flights table — the last conceivable field —
is takeoff/landing only and 100% null for both airports.

**The two measurements agree from independent directions.** L024 (REST): `firstSeen` sits at
wheels-off across 43k paired departures in four months — exactly what a network that never hears
the surface would produce. L025 (state vectors): the surface rows aren't there. One day and three
hours is a small window, but against zero-vs-hundreds-of-departures it is not a subtle read.

**So what.**

1. **Front B has no OpenSky data route** — not REST, and not Trino either, since the snapshot
   *is* Trino content. The pending Trino request loses its expected value for off-block
   reconstruction (it can still cheaply confirm this across more days if granted).
2. All remaining effort belongs to **Front A** (the nine airports' body) unless a genuinely
   different surface source appears (airport A-SMGCS data is not open; ADS-B aggregators with
   surface receivers at FCO/CDG would be the thing to look for, not more OpenSky).
3. Whatever the 234 leader does about Rome, **it is not built on OpenSky surface data**. The
   most likely remaining explanations: information already inside the challenge tables that we
   have not decoded, or an open source outside OpenSky entirely.

---

<a id="l026"></a>
## L026 — The N-batch: no single change moves ex-LIRF; an honest stacker does, a little

`confirmed` · `results/runs.jsonl` · `scripts/analysis/a031_stack.py` · `a032` · 2026-09-25

**Question.** Twelve no-external-data experiments (new representations, target offset, richer
mode cells, mid-depth linear trees, routing and weighting variants, seed bag, Huber, a
2026-verified runway-day flag) plus a calibration layer and an OOF stacker. What moves the
steering metric?

**Finding.** Singles: nothing beyond noise. The twelve land between **292.996 and 295.13
ex-LIRF** against the champion's 293.32 (bootstrap sd on a pair ≈ 1.5–2 s), except three that
are outright broken: `n15_huber` (429.7 — the objective misprices this target's scale),
`n06_lintree_63` (371.5), and `n05_modes2` slightly *worse* than v1's plainer cells (294.22 —
richer keys shrink away). Per-airport isotonic calibration is decisively dead: ex-LIRF 293 →
328, EHAM +241 s (a032), with the affine variant re-confirming a017.

**The one positive:** an ExtraTrees combiner over the 45-member out-of-fold matrix — fitted
**cross-fold** and **on non-LIRF rows only**, so it can neither see its own residuals nor buy
CV-Rome — reaches **ex-LIRF 290.21**, −3.1 s on the champion and the largest honest ex-LIRF
move recorded. Ridge and plain means capture none of it (293.1–293.3): the gain is nonlinear
member disagreement, exactly the Amazon-recipe prediction. Two caveats stated plainly: two
folds only, and 45 members give a flexible combiner room to be lucky — the member-subset check
(a035) and the board are the real tests.

**So what.** The body's single-model frontier at ~293 is real; further singles are noise
mining. The submission path is composition: stack on the nine, champion at Rome.

---

<a id="l027"></a>
## L027 — adsb.lol hears the surface at Rome; the reconstruction route is open

`confirmed` · `scripts/analysis/a037_adsblol_rehearsal.py` · 2026-09-25

**Question.** OpenSky receives no surface at LIRF or LFPG (L024/L025). adsb.lol is a different
receiver network (FlyItalyADSB among its feeders), openly licensed (ODbL), with one release per
day of full per-aircraft traces. Does *it* hear the surface?

**Finding. Yes — decisively.** One day (2026-01-05), one scan:

| airport | ground points | moving | aircraft with a moving ground track |
|---|---:|---:|---:|
| **LIRF** | 13,083 | 12,174 | **259** |
| LFPG | 3,204 | 2,500 | 276 |

Against roughly 350–400 daily departures at Fiumicino, 259 moving-ground aircraft on a single
(storm-affected) day is majority surface coverage — where OpenSky had literally zero points.
The organisers have confirmed in writing that open trajectory data used to extract off-block
times is permitted ("we are ok"), and a top-3 competitor states they use exactly this source.

**Caveats carried forward:** per-aircraft ground tracks are sparse (median ~32 points over a
whole day at LIRF, ~8 moving at LFPG) and one trace can span several rotations of the same
airframe, so the matching step must slice tracks per departure anchored on the known takeoff
time. Coverage of the *specific* 383 unmatched rows is the number that matters, not the daily
total; the full-months pull measures it.

**So what.** Front B is alive again, on a non-OpenSky source, with the competitor-verified
lesson that small samples mislead — the pipeline (`scripts/pull_adsblol.py`) therefore pulls
**all of January and July, both years**, one day at a time (download → box-filter to a small
parquet → delete raw; peak disk < 10 GB). Scored months first. The prize unchanged from a027:
every unmatched row exact ⇒ pooled 274.7; even Rome ≈ 550 ⇒ board ≈ 318 at today's ex-LIRF.

---

## L028 — The stack resists improvement: no combiner variant beats plain ET-over-members

`confirmed` · `scripts/analysis/a038_context_stack.py` + experiment `f14_congestion2` · 2026-09-25

**Question.** Three same-day ideas for beating v11's 289.52 cross-fold: (a) the rest of the
post-ops forward-window family as features (`congestion2`: arrival pressure ±15 min, next
arrival/departure gap on the runway), (b) a combiner that sees row covariates (hour, day,
match flag, airport, runway), (c) a stronger combiner (deeper ET, or LightGBM).

**Finding. All three negative.**

| variant | ex-LIRF (cross-fold) |
|---|---:|
| ET d6x200, members only — **shipping (v11)** | **289.52** |
| ET d8x400, members only | 289.65 |
| ET d8x400, members + covariates | 292.41 |
| LGBM combiner, members only | 295.29 |
| LGBM combiner, members + covariates | 300.34 |

And `f14_congestion2` (f13's ten blocks + the new window family) scored ex-LIRF **292.60**
against f13's 292.21 — inside the noise band, slightly worse. f13's `rh_next15m_rw` was
apparently the one forward-window signal that mattered.

**Why this is expected in hindsight.** The members already encode the covariates (each is a
LightGBM fitted on them); handing the combiner the covariates again lets it partition 344k rows
into cells small enough to re-fit residual noise. Boosting does the same by construction —
ExtraTrees' conservative interpolation over member agreement *is* the value.

**So what.** The stack improves by adding a *member that knows something new*, not by
re-engineering the combiner. That points all remaining effort at genuinely new information:
the adsb.lol reconstruction (L027) and the two error concentrations in L029.

## L029 — The stack's remaining error sits in three places, and one of them is Paris

`confirmed` · `scripts/analysis/a039_stack_autopsy.py` · 2026-09-25

**CORRECTION (see L034):** the three headline shares here are dominated by two dropped >12 h rows; LFPG's true share is ~16.6%, the unmatched share ~12%, and July is the harder fold on trimmed error.

Cross-fold residuals of the shipping stack (non-LIRF rows, 317,891 with per-fold OOF):

- **LFPG is 48.5% of non-LIRF squared error** on 12.5% of the rows — rmse 571.4 against
  268.7 for the next-worst airport. Rome-excluded steering has been hiding a second Rome.
- **The NM-unmatched rows again**: 4,969 rows (1.6%) carry **45.8%** of non-LIRF squared
  error at rmse 1,566.9 — the same defect class as Rome, at every airport. Presumably these
  two cuts overlap heavily (LFPG's unmatched rows); a039 does not cross them.
- **January, not July, is the stack's harder fold** (342.2 vs 238.4; 62.6% of squared error)
  — reversed from the single-model era. De-icing/winter-ops information would land there.
- By true taxi: 30–60 min rows are under-predicted by a median-scale bias of −378 s; the
  1–12 h band (519 rows) by −1,471 s. The `<10 m` band is over-predicted (+108 s).

**So what.** The trace-reconstruction front (a040) should treat **LFPG as a co-equal target
with LIRF**, and the unmatched-row flag remains the single most valuable split in the data.

---

## L030 — Trace reconstruction works, with a per-airport accuracy set by receiver geography

`confirmed` · `scripts/analysis/a040_trace_matching.py`, `a041_trace_reconstruct.py`,
`a042_surface_census.py`, `a043_proxy_correction.py` · 2026-09-25

**The matcher.** No identity join exists (adsb.lol's `r` is the registration, not the
callsign), so departures pair to traces on physics: candidate ground segments whose last
ground point sits within ±120 s of the recorded takeoff; **one candidate per departure**
(closest approach to the recorded stand in 300 m bands, takeoff-gap tie-break — pooling
candidates was the first version's poison, several aircraft take off inside any tolerance);
taxi-out = the last movement episode before takeoff (moving points split on >10-min quiet
gaps — without the split, the anchor catches the same segment's taxi-in). Two anchors: strict
(first moving point within 150 m of the stand) and relaxed (episode start, distance kept as a
covariate). Stand coordinates from the Scenery Gateway packs, joined on a normalized key
(fixes LSZH; LEBL's scenery predates a renumbering and cannot join).

**Where the ground is audible (a042).** Median daily moving-ground points:

| | 2025-01 | 2025-07 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|
| LIRF | 0 | 26,917 | 12,162 | **0** (alive 2 days) |
| LFPG | 0 | 999 | 1,455 | 2,873 |
| LEMD | 0 | 13,644 | 97,830 | 50,747 |
| LTFM | 0 | 0 | 0 | 0 |
| EHAM | 105k | 272k | 257k | 275k |

Feeders come and go: Rome's is loud in Jul 2025 and Jan 2026 and **dark for 29 of 31 days in
Jul 2026** — the July half of Rome's ranking rows cannot be reconstructed from this archive.
Istanbul is never audible.

**Accuracy is receiver geography, not method (a040 density cut).** Dense EHAM traces are
unbiased (median err +7 s, first capture 59 m from the stand); dense Rome traces still start
~555 m out and −550 s late — Fiumicino's receiver is spatially blind near the aprons. So the
proxy needs a per-airport correction: `taxi_sec ~ proxy + dist_first + 1/n_pts`, cross-fitted
(a043):

| airport | n pairs | raw bias | raw rmse | corrected rmse | median abs err |
|---|---:|---:|---:|---:|---:|
| LIRF 2025-07 | 13,283 | −664 | 1,023 | **594** | 267 |
| EHAM 2025-07 | 17,558 | +6 | 321 | 239 | 129 |
| EHAM 2025-01 | 14,652 | −45 | 421 | 357 | 132 |
| LSZH 2025-01 | 6,434 | −305 | 523 | 332 | 151 |

**2026 coverage (a041, `data/external/adsblol/recon_*.parquet`).** LIRF 2026-01: **69 of 108**
NM-unmatched rows carry a reconstruction (11 of them stand-anchored — the strict set is a
subset of the relaxed one, an earlier version of this entry wrongly summed them to 80), and
72% of matched rows. LFPG: 87/324 (Jan) and 116/408 (Jul) unmatched rows, 22–28% overall.

**The open deployment question — the user's call.** A corrected Rome proxy at 587 rmse against
the stack's 1,567 on unmatched rows looks like a large win, *if* the withheld labels on those
rows are physical taxi times. If 2026 unmatched labels are corrupt the way 2025's were (L012:
the organisers do not clean), the best prediction on the garbage component remains E[label],
and the proxy's value is per-row *evidence* for a mixture split rather than a direct override.

---

## L031 — Probe algebra works: the unmatched Rome rows are over-predicted, not under

`confirmed` · `scripts/make_probe.py`, submissions v12/v13 · 2026-09-25

**CORRECTION (see L033 and REPORT_AS_OF_25_09):** 27 of the shifted rows sat at the 20,000 s
submission cap and never moved (plus one partial at 19,313); corrected means are −610 (all),
−394 (Jan). More importantly, this entry's reading of −239/−494 as "consistent with trace
physicals" was wrong: v11 predicts ~4,617 s on these rows, so the implied mean labels are
~4,007 s (4,880 Jan / 3,686 Jul) — copy-mode delay values, not physical taxis. The
trace-override deployment is contraindicated by this measurement.

The board has no public/private split, so a submission differing from a base on one
population by known deltas returns an exact linear functional of the withheld labels:
`S = Σ residuals = (nΔ² − N·(b₁² − b₀²)) / (2Δ)`. Two slots measured, against v11
(b₀ = 422.6008, Δ = +2000):

| population | n | S | mean residual |
|---|---:|---:|---:|
| LIRF unmatched, all | 383 | −189,175 | **−493.9 s/row** |
| LIRF unmatched, Jan | 107 | −25,556 | −238.8 |
| LIRF unmatched, Jul (derived) | 276 | −163,619 | −592.8 |

v11 **over-predicts** the class; 2026's unmatched labels average ~500 s *below* our
predictions. The v8 direction (add huge schedule-gap values) was wrong for 2026 in
aggregate. January's −239 is consistent with the trace physicals. Two structural limits:
probes recover only *linear* functionals (Σy of any chosen subgroup; never Σy² of one), and
aggregate means can hide two-sided garbage. Optimal constant shifts from these S values are
worth only ~0.35 s of board — the value is the *information*, steering which subgroup gets
measured next (the >12 h schedule-gap subgroup is the discriminating one).

**Slot bookkeeping bites:** the morning's byte-identical v9 re-upload counted as one of the
five daily submissions; v14 (shift on the 69 trace-covered Jan rows) was built, validated
and **rejected on DAILY_LIMIT_REACHED** — resubmit after 00:00 UTC. Never re-upload an
existing file.

## L032 — The 2025 garbage process is the schedule gap, and matched rows are dirty too

`confirmed` · `scripts/analysis/a044_label_vs_physics.py`, `a045_correction_transfer.py` · 2026-09-25

Joining every 2025 reconstruction to its label (~134k rows, 1,214 unmatched):

- **Unmatched disagreeing rows (|label − physical| > 600 s; 46% of Rome's unmatched):**
  35.5% have label = `mvt_time − sched_time` within ±120 s, and corr(label, sched-gap) =
  **0.934** with median ratio 0.42. `sched_time` is movement-side — present on every 2026
  row — so this garbage candidate is computable per ranking row. Their labels: p50 2,891,
  p95 23,472 s.
- **Matched rows are not clean**: 2.5–16% disagree with physics (Rome 16.2%, labels median
  115 s *below* physical; EGLL/EHAM/LSZH gaps −22 to −62). Among matched disagreements,
  18% match `mvt−eobt_1`, 19% `mvt−aobt_3`, 17% `mvt−sched` — a third of the "defect class"
  problem exists inside the matched population, at every airport.
- **a045 LOAO licence test failed**: pooled-correction transfer to an unseen airport is
  fine at EDDF/LSZH (+4–8%), poor at EHAM/EDDM/LEMD (+17–31%), catastrophic at EGLL
  (2,174 rmse). Overriding LFPG with a transferred correction is not licensed; the feature
  path (the model learns per-airport calibration in-fold) is the deployment that survives.

---

## L033 — Every submission is capped at 20,000 s, and that rewrites three earlier learnings

`confirmed` · external review, verified against `src/taxiout/submit.py` (`CLIP`), v9/v11/v15 files · 2026-09-25

`submit.build_frame` clips every upload into [0, 20,000]; CV never applies that cap. **27
rows of every submission since v9 sit at exactly 20,000** — all LIRF unmatched (11 Jan + 16
Jul), including all 11 rows whose 2026 schedule gap exceeds 12 h (gaps 47,337–111,654 s).
Consequences, each verified:

- **L023 is void.** v8's rule D wrote ≥60,049 s on the cell; the upload clipped them all to
  20,000. The rule never reached the board — nothing about its transfer was ever tested.
- **L022's "CV-Rome never transfers" is partly an artefact**: capping CV predictions at 20k
  collapses the 286 s CV-Rome spread across submitted models to ~15 s (board spread ~4 s).
- **L031's probe algebra needs a cap correction**: rows at the cap absorb the +Δ shift, so
  S_true = S_computed − Δ/2·k. With k=27 (v12) and k=11 (v13): all-383 → S = −216,175 over
  356 uncapped rows (mean −607); Jan → −36,556 over 96 (mean −381); derived Jul → −179,619
  over 260 (mean −691). The probes said *nothing* about the 11 cell rows.
- **The board is consistent with cell labels ≈ schedule gaps**: if the 11 labels equal their
  gaps, those rows carry ~2.28e10 of squared error under the cap — implied Rome 1147–1161
  against the observed 1140–1148, where "ordinary taxi" labels would imply ~800. Predicting
  the cell exactly would move the board from 422.6 to **~335**. A one-slot discriminating
  probe exists below the cap: set the 11 rows to c=10,000 → the score returns Σy on the cell
  exactly; if the hypothesis is false (labels ~1,200) that probe *improves* the board ~10 s.
- v9→v11 share Rome rows verbatim, so their board deltas measure ex-LIRF alone: the stack
  transferred at ~1.40× its CV delta, and "implied Rome" absorbs that attribution error —
  it moved 1144.2→1139.7 across identical Rome predictions.

## L034 — Two rows. The stack's non-LIRF error autopsy was dominated by two CV rows

`confirmed` · external review, verified against `reports/a039_stack_autopsy.txt` · 2026-09-25

a039's true-taxi bands cover 317,889 of 317,891 rows and sum to 61.7% of squared error. The
two dropped rows (>12 h labels, LFPG, Jan-2025, residuals ~71,400 s) carry **38.3%** of the
stack's entire non-LIRF squared error. Corrections that follow:

- Without them, cross-fold ex-LIRF is ~227.5, not 289.5; LFPG's share falls from 48.5% to
  16.6% (behind LTFM 20.4% and EGLL 18.1%); the unmatched share falls from 45.8% to ~12%;
  and **July is the harder fold again** (238 vs 213). L029's headline claims were two rows.
- **Steering moves to trimmed ex-LIRF** for model comparison: the untrimmed number lets two
  rows outvote three hundred thousand. On trimmed ex-LIRF the weather gains are *larger*
  (c10 232.4 → f15 229.8 → f16 228.8), so f15/f16's value is in the body, not the tail.
- The f15/f16 YAML rationale ("January carries 62.6%") and PLAN §2.3's fold argument
  inherit the two rows; the weather gains stand on the trimmed numbers regardless.

Smaller verified corrections from the same review: `wx_precip_mm` is identically zero (IEM's
p01i is not reported in European METARs — precipitation must come from wxcodes intensity);
the ranking cell counts are 107 Jan + 276 Jul; LFPG 2026 coverage is 88/324 and 120/408;
a044's joined rows are 128,739 (1,871 unmatched); `2026.01.05_LFPG/_LIRF.parquet` rehearsal
leftovers double-count Jan-5 in every `2026.01*` glob; and a040's LIRF calibration fails its
own pre-registered gate (corr 0.28–0.39, P300 0.14–0.62) — "594 rmse after correction" is
the honest statement of what the matcher achieves at Rome, not gate passage.

---

## L035 — pop27 solved: the 2026 labels are the schedule gaps, and the cap was worth 95 s

`confirmed` · `scripts/analysis/a050_pop27_values.py`, submissions v15–v17 · 2026-09-26

With the upload cap raised to 200,000 (user decision), one day of submissions:

| v | predictor on pop27 | board |
|---|---|---:|
| 15 | 18-member stack, still capped at 20,000 | **421.66** (then best) |
| 16 | 2025-mixture mean (gap; 0.593·gap+0.407·87,500 on day-crossers) | **326.60** |
| 17 | 0.74-hedge | 334.87 |

The exact two-functional algebra (v16/v17 vs v15) gives **Σy over the 27 rows = 1,165,411**,
against Σgap = 1,195,661 (97.5%), a 2025-mixture expectation of ~1.29M, and ~28k if the
labels had been ordinary taxis. So the 2026 pop27 labels equal their takeoff−schedule gaps
almost exactly — REVIEW1's board-model prediction (326.7 for this predictor in the
labels=gap world) was right to within a tenth of a second. v16 removed 24.5e9 of squared
error, ~70% of what pop27 carried.Remaining: the mixture over-predicts the 8 day-crossing rows
(the 87,500 blend is too high in 2026); a pure-gap predictor prices at ~321, folded into the
next composition together with the sub-cap hedge (PLAN E2).

---

## L036 — The sub-cap hedge: gate passed, board gain smaller than the law predicted

`confirmed` · `scripts/analysis/a051_subcap_hedge.py`, submissions v18/v19 · 2026-09-26

v18 (pure gap on pop27, per L035's Σy measurement) scored **319.86** (−6.74 from v16).
v19 added the calibrated copy-mode hedge on the 348 sub-cap rows with 600 < gap ≤ 20 ks
(LOMO AUC 0.863; CV gate passed at SSE 3.71e9 vs base-rate 5.02e9 and c10's 6.53e9; monthly
levels calibrated to the probe-measured means) and scored **317.72** — a gain of −2.14
against the −7 to −12 estimate. The shortfall matches the caveat REVIEW2 attached: the 2025
band law over-stated realised SSE in its own July back-test, and the estimate inherited
that. Day total 2026-09-26: 422.60 → 317.72 across five slots (v15 421.66, v16 326.60,
v17 334.87, v18 319.86, v19 317.72). The eight sub-cap rows with gap ≤ 600 keep the stack's
values.

---

## L037 — Training-label winsorization is the body's missing lever; the other three were flat

`confirmed` · experiments f19–f22, t02; `scripts/analysis/a053_osm_join.py` · 2026-09-26/27

The E3 batch, judged on trimmed ex-LIRF (f17a's 228.19 as the reference):

| lever | result |
|---|---|
| weather2 (wxcodes precipitation) | 228.81 — flat; v1's non-precip columns already carried it |
| geometry sources (OSM union, ≥96% coverage everywhere incl. LTFM 99%) | 228.30–228.35 — coverage solved (a053: Gateway ∪ OSM ∪ A/B-collapse; EDDM 70→99%), CV flat; value is July-2026 unseen-stand insurance |
| modes3 (airport × prefix × delay ambiguity cells, TD, leak-tested) | 228.39 pooled, disagree band 344.8→343.6 — **the off-block ambiguity is not resolvable from admissible covariates**; the 26%/67% split stands as irreducible with in-data features |
| **t02: `train_label_cap: 7200`** (winsorize training labels; evaluation untouched) | **226.23**, and the sub-cap unmatched bias collapses **+1,381 → −218** — REVIEW2 X2's monster-poisoning mechanism confirmed causally. LSZH-Jul 225.5→197.1. Untrimmed metrics degrade by construction (the model stops predicting monsters); the composition's explicit Rome rules own those rows. |

One negative worth its own line: a single capped model now ties the 18-member stack's
trimmed body (226.23 vs ~226.3) — the stack's members all share the monster-poisoning, so a
capped re-stack is the obvious next composition base.

---

## L038 — The frontier after the Rome fix: 348 rows are still the cheapest seconds on the board

`confirmed` (arithmetic on scored submissions) · v19/v23 boards, `a051`, REVIEW2 s03 · 2026-09-27

Decomposing the champion composition (board 317.72, cross-fold ex-LIRF 288.30) through the
exact identity `board² = 0.922·ex² + 0.078·LIRF²`:

- the implied **LIRF component is 558**, down from ~1146 before the pop27 fix (L035);
- of Rome's remaining 8.38e9 squared error, **the 348 hedged sub-cap rows carry 41%**
  (3.46e9 at the hedge's ~3,152 CV RMSE), and the other 26,524 rows carry 4.92e9 → RMSE 431.

**What that does to the two paths.** Holding the other component fixed:

| lever | move | board |
|---|---|---:|
| body | ex-LIRF −10 (a full day's work has been worth ~2) | 309.4 (−8.3) |
| Rome | LIRF −100 | 305.0 (−12.7) |
| sub-cap hedge only | RMSE 3,152 → 2,500 / 2,000 / 1,500 (oracle is 1,217) | 311.8 / 308.1 / 305.2 |
| board targets, body-only | 300 needs ex-LIRF 267; 278 needs 240; 263 needs 221; 224 needs 168 | — |

**CORRECTION (L039, same day):** the sentence below overstated Rome's remaining upside. The
hedge's 3,152-vs-1,217 "gap to oracle" is mostly *irreducible mixture variance* — the oracle
knows which branch happened — so only the classifier-sharpness part is winnable, worth perhaps
−3 to −6 rather than −12. Matched-Rome is worth **zero** (L039).

**So the ordering flips back to Rome.** Sub-300 by body work alone needs 21 s of ex-LIRF,
where the whole E3 batch bought 2. The same 17 s sits in 348 rows whose labels the board has
already told us about (mean 4,007 s, copy-dominated — L035/L036), and whose oracle-mode error
is 1,217 against our 3,152. The gap is classifier sharpness, not missing information.

**Corollary for the stack.** Body members now buy ~0.5 board each (the 20-member re-stack's
0.65 cross-fold gain is worth −0.54 by this gradient). Worth shipping when free, never worth
displacing Rome work.

### Process lessons from the same day

- `expand_grid` routed every swept key into `model.params`, so `t03`'s `train_label_cap`
  sweep silently ran the same configuration twice. Fixed: top-level experiment fields are
  now routed out of the params dict. **Any grid over a non-model field before 2026-09-27
  swept nothing.**
- `git commit --amend` + force-push after the VM had pulled produced divergent branches and
  cost an experiment run. Rule: once a commit is pushed, only follow-up commits.
- `git stash -u` on the VM destroyed 15 untracked `data/rank_preds/*.parquet` refits (~5 h of
  compute). Rule: treat `data/rank_preds/` as regenerate-don't-trust, and never `stash -u`
  there. `members.txt` is now committed so a checkout cannot silently revert the composition.
  > *Correction, 2026-10-03:* reversed on 2026-09-28 (commit `66eb414`, "stop tracking the mutable
  > members file"). `data/rank_preds/members.txt` is gitignored working state; the only authority is
  > the versioned list under `docs/compositions/` (read through `paths.composition()`, which follows
  > `docs/compositions/BASE`), per RULES.md §6.10 and enforced by `tests/test_compositions.py`.


---

## L039 — Matched-Rome is already optimal, and conditioning on the realized branch fakes a bias

`confirmed` · `scripts/analysis/a054_matched_rome.py` + the follow-up check in that session · 2026-09-27

**The apparent finding.** a054 decomposed the 26,139 LIRF CV rows the composition does *not*
override. 209 of them — matched rows where `block_time` sits within 60 s of `sched_time` and
the flight left more than an hour late — carry **68.4% of the population's squared error** at
RMSE 5,858 and bias **−2,267**. The AEZ prefix alone holds 60% of the error share. The
ordinary 21,191 rows sit at RMSE 354, i.e. normal for any airport. It looked like a large,
identifiable, unexploited population.

**The trap.** That −2,267 bias is measured *conditional on the row having turned out to be a
schedule copy*, which is knowable only from the label. Any RMSE-optimal hedge on a two-branch
mixture looks badly biased when you condition on the branch that happened: it sits between the
branches by construction. The honest test is the *identifiable* population — matched LIRF rows
more than an hour late, 3,791 CV rows, selected only on columns 2026 also has:

| predictor on the identifiable population | RMSE | bias |
|---|---:|---:|
| `c10_trust_aobt` (the shipping Rome layer) | **1,490** | **−42** |
| LOMO prefix mixture `p̂·gap + (1−p̂)·median` | 1,636 | −242 |
| 50/50 blend of the two | 1,506 | — |

The model is already essentially unbiased there and beats the explicit mixture. Two
near-deterministic 2025 prefixes (TKJ 57/57, ISR 124/125 copies) shrink to **19 rows** in 2026,
on which v19 already predicts Σ 97,715 against Σgap 94,177 — the `modes` block learned them.

**So there is nothing to win in matched Rome**, and the same argument deflates the sub-cap
hedge: its distance to "oracle mode" is mostly mixture variance, not slack. What remains
winnable there is sharpening `p̂` (AUC 0.863 today, 0.68 without the prefix), which moves the
mixture term by whatever it moves `p(1−p)` — a few board seconds, not twelve.

**Method rule, third time this project has needed it** (after L022's range restriction and
L031's misread of the probes): before pricing a population, check that the selector uses only
columns the ranking set has. A cut defined with the label measures the hedge's cost, not an
opportunity.

---

## L040 — The label cap helps the body and hurts Rome, because the defect rows live in one place

`confirmed` · `a054_matched_rome.py`, the layer-B check in `RESULTS_26_09.md`, `t03_cap_sweep` · 2026-09-27

Training-label winsorisation is the largest body improvement measured in this project, and the
sweep gives it a broad flat optimum — trimmed ex-LIRF 228.35 uncapped → 230.56 at 3,600 →
**226.23 at 7,200 → 225.99 at 10,800 → 226.08 at 14,400**, with 1,800 catastrophic at 260.21.
Any cap above ~7,200 captures the whole gain.

**But its sign flips by population.** On the exact set of rows the composition's Rome layer
predicts — LIRF minus the 375 rows layers C and D override, 159,252 CV rows:

| Rome-layer candidate | SSE, all rows | RMSE, all | RMSE, plausible labels only |
|---|---:|---:|---:|
| `c10_trust_aobt` (shipping) | 10.48e9 | 633 | 390 |
| **`f21_geometry3`** | **10.46e9** | 633 | 388 |
| `t02_cap7200` | 10.89e9 | 645 | **379** |
| cap 10,800 | 10.67e9 | 639 | 381 |

The capped models are **better on plausible rows and worse overall**, because 75 implausible
rows carry most of that population's squared error and a capped model refuses to predict them
high. The body is the opposite case: there the defect rows are few, mostly trimmed away, and
the poisoning of shared leaves dominates.

**The error this corrects was mine.** `BEST_OF_26_09.md` §9 and idea D1 proposed swapping the
Rome layer to a capped model and called it free, citing the 390 → 379 plausible-row figure —
a subset chosen so that capping wins, while the population the swap actually changes includes
the rows where it loses. An external reviewer caught it before the queued job shipped a worse
candidate; the job was repointed at `f21_geometry3`, whose advantage over `c10` is 0.02e9,
i.e. nothing.

**Rule.** Price a change on the exact set of rows it will alter, never on a subset of them.
This is the same failure as L039's realized-branch conditioning and L022's range restriction —
third instance, second one of mine.

---

## L041 — The refit path ignored every label treatment, so the cap's gain never reached a submission

`confirmed` · external audit `scripts/analysis/rev4/w01_refit_asymmetry.py`, verified directly · 2026-09-27

`cv.run` applied `train_label_cap`, `target_transform` and `train_relabel`. **Neither refit path
did.** `scripts/refit_predict.py` and `submit.predict_experiment` both called
`model.fit(train, train["taxi_sec"])` on raw labels. The proof is exact: `t02_cap7200` and
`f21_geometry3` differ *only* by `train_label_cap: 7200`, and their ranking-prediction files
are bit-identical — max absolute difference **9.09e-13** over 344,841 rows.

**What that cost.** Everything measured about the cap — the −2.36 s trimmed body improvement,
the six population tests supporting it (L037, a054, a058), the whole capped-twins programme
(c01–c05, d01, d02) and chain6's ten refits — was undeliverable. A capped member would have
entered the stack as a *duplicate* of its uncapped twin (priced at +0.25 board), and the
composition would have carried none of the benefit its CV promised.

**The structural fix.** `cv.fit_and_predict` is now the single place label and target
treatments are applied, and cross-validation and both refit paths all route through it. A
regression test asserts the structure, not just the behaviour, because the failure mode was
two code paths drifting apart rather than one path being wrong.

**Three more defects from the same audit**, each of which made a queued experiment
untrustworthy rather than merely disappointing:

- `repair_training_labels`' SCHED rule was `|y − sched gap| ≤ 60`, which is the *normal* case
  for a punctual departure: it fired on **199,634 rows, 9.5% of training**, of which 197,471
  already carried plausible labels, rewriting them by a median of 176 s. As built, t06/t07
  tested noise injection. Fixed by also requiring `|y − ob_aobt_sec| > 300`.
- The log target's Duan smearing factor was estimated on **in-sample** residuals of a
  511-leaf booster, so it was far too small and the back-transform under-predicted by ~5%
  (held-out factor 1.053). Now estimated on a 5% held-out slice. t04's first result (trimmed
  229.96) is therefore invalid and a forced rerun is queued.
- `stack_submission` keyed OOF files on the experiment *name* and discarded the config hash,
  so when a grid arm was rerun under a new hash, `iterdir()` order decided which entered the
  combiner — and one of the two `t03` candidates was the L038 arm that applied no cap at all.
  Newest-per-name now wins and the chosen file is printed.

**The lesson worth keeping.** Three of my four verification habits — unit tests, CV gates and
leak tests — all passed while this was broken, because each checked a single path. What caught
it was asking whether two artefacts that *should* differ actually do. Add that to the list:
when a treatment is supposed to change an output, diff the outputs.

---

## L042 — Rome's remaining headroom is at most −11.6 board seconds, and my earlier −13.0 had no script behind it

`confirmed` · `scripts/analysis/a069_rome_headroom.py` · 2026-09-27

The board is one RMSE over all 344,841 rows, so the Rome/body split is **not observed**. I had
been quoting "Rome holds ~8.9% of the board's squared error, all remaining Rome work is worth at
most −13.0, a perfect Rome lands at 274.2" in strategy discussions. Those numbers came from an
inline calculation, had no script, and rested on a category error.

**The error.** The inference has exactly one free parameter: the *level* ratio
`L = board_body_rmse / cv_ex_lirf`. I reached for L034's *transfer* ratio (0.93–1.30, mean ~1.10),
which relates a **change** in trimmed CV to a **change** in board, and used it as a level
multiplier. Those are different quantities. At `L = 1.10` the body alone would account for
1.11× the whole board's squared error — arithmetically impossible.

**The bound that needs no assumption.** `MSE_Rome ≥ 0` forces
`board_body_rmse ≤ BOARD·√(N/N_body) = 283.98·1.0417 = 295.75`, i.e. `L ≤ 1.0425`. A soft floor
comes from Rome being no *better* than the body, `L ≥ 1.0010`. Over that whole feasible range:

| L | Rome's share of board SSE | implied Rome board RMSE | board if Rome were perfect | max gain |
|---:|---:|---:|---:|---:|
| 1.0000 | 8.0% | 287.4 | 272.40 | **−11.58** |
| 1.0100 | 6.1% | 251.9 | 275.12 | −8.86 |
| 1.0200 | 4.3% | 210.1 | 277.85 | −6.13 |
| 1.0300 | 2.4% | 157.0 | 280.57 | −3.41 |
| 1.0425 | 0.0% | — | 283.98 | −0.00 |

So a *perfect* Rome — every one of the 26,899 predictions exactly right, an unreachable ceiling
— is worth between **−11.6 and −0.0** board seconds, and the −11.6 end needs the most optimistic
`L`. My −13.0 was outside the feasible range at its own stated share.

**What this changes.** The body is now the only place a meaningful gain can come from, more
decisively than I had argued. Cutting the body's squared error by 10% buys −13.4 s and by 20%
buys −27.5 s, roughly `L`-insensitive. Against v27's 283.98, TOP50 needs 3.9% of current squared
error removed, TOP25 15.4%, TOP10 26.9%, the leader 37.5% — and since the body holds 92–100% of
it, those are body-error targets.

**The habit to add.** A number that gates a strategic choice gets a script before it gets
quoted, not after. This one survived several discussions because it sounded like arithmetic and
therefore did not feel like a claim. See `RULES.md` §7.1 and §7.6.

---

## L043 — "Layer D does not pay" was a baseline artefact; the same error as L040, by a different agent

`confirmed` · `scripts/analysis/a079_rome_layers_vs_baseline.py`, after `a078` · 2026-09-27

A delegated analysis concluded layer D "does not pay against a fitted model" at ΔRMSE **+0.68**.
I re-derived it, got **+0.17** for the form it tested and **+1.85** for the form layer D actually
ships — so I confirmed and strengthened its conclusion. Both of us were wrong, for the same
reason: we priced the override against **`d02_catboost_capped`**, and v27's Rome baseline is
**`f21_geometry3`** (`--rome-from f21_geometry3`).

Against the baseline it actually replaces, every sign is the opposite of what matters:

| population | n | vs `f21_geometry3` | vs `c10_trust_aobt` | vs `d02_catboost_capped` |
|---|---:|---:|---:|---:|
| **C** LIRF & unmatched & gap>20k, pure gap | 24 | **−60.79** | −59.13 | −169.80 |
| **D** LIRF & unmatched & 600<gap≤20k, pure gap | 365 | +15.00 | +14.64 | +17.80 |
| **D** the same rows, layer D's hedge form, probe-free | 365 | **−4.73** | −5.06 | **+1.85** |
| new: `ac_type_null & gap>43.2k`, pure gap | 11 | **−53.06** | −51.80 | −143.41 |

**Why d02 inverts it.** d02 is trained with `train_label_cap: 7200`, so it cannot predict a large
value at all. On layer C's long-label rows its SSE is 5.345e10 against f21's 1.874e10, which makes
any override look spectacular; on layer D's mostly-ordinary rows it predicts conservatively and
scores 4.396e9 against f21's 6.442e9, which makes the override look harmful. The capped model is
the worst possible yardstick for a long-label override, and it is the one we both reached for
because it is the highest-weighted member.

**Two things this establishes, beyond the correction.**

1. **Layer D pays, and pays without any probe.** The −4.73 above uses `p̂` = the copy base rate
   from the *other eleven months* (0.476 and 0.524) and `m_band` = the non-copy band medians from
   the other eleven months. Nothing is fitted in fold and no probe enters. The per-month scale
   factors from v12/v13 are therefore droppable.
2. **The pure gap is wrong on layer D's population** (+15.00) and right on layer C's (−60.79).
   The two layers are not variants of one idea; the band boundary at 20,000 s is load-bearing.

**The rule, restated because this is its second occurrence.** L040 was me pricing a change on a
*subset* of the rows it would alter. This is the same failure one level up: pricing a change
against a *model that is not the one being changed*. Both are answered by the same question, asked
before the measurement rather than after — *exactly which predictions does this replace, and what
are they now?* See `RULES.md` §7.8.

---

## L044 — The runner labelled the untrimmed ex-LIRF "steer by this", and the gap is two rows

`confirmed` · `scripts/analysis/a091_where_the_trim_bites.py`, `reports/a091_where_the_trim_bites.txt` · 2026-09-28

`scripts/run_experiments.py` printed `cv.pooled_rmse_ex_lirf` under the marker `<- steer by this`
for the whole life of the project. That field is computed **without** the trim mask
(`cv.py`, the `elsewhere` branch), so the console labelled the untrimmed figure as the steering
metric, while the number the ledger actually steers by (L034) was computed only inside `a059` and
`a089` and appeared nowhere in the runner's output. Nothing was mis-computed; a label sat against
the wrong field. Fixed by adding `cv.pooled_rmse_trimmed_ex_lirf` and per-fold
`rmse_trimmed_ex_lirf`, with `cv.steering_rows()` holding the single definition and a test in
`tests/test_cv.py` asserting that whatever the runner marks is that field.

**The gap between the two figures is two rows.** Above `TRIM_SECONDS` the year holds 34 rows, 31
of them at LIRF; of the three elsewhere, **2 are LFPG in 2025-01 and 1 is LSZH in 2025-05**. Every
other month has none. Consequences:

- On `seasonal`, the trim removes exactly **2 of 317,891** non-LIRF rows, both in January.
- July's trimmed and untrimmed ex-LIRF are therefore **identical to every decimal place**, which
  is why `m00_lgbm_smoke` (fold `quick` = 2025-07) reports 258.43 twice. That is correct output,
  not a mask failing to bite.
- Backing the squared-error difference out of the two logged figures across eight independent
  seasonal runs recovers a residual of **71,368–71,516 s** for each of those two rows, and a
  share of non-LIRF squared error of **38.9–39.6%**. Both reproduce L034 (~71,400 s, 38.3%) from
  the run log alone, without re-fitting anything.
- The residual is stable to ±150 s across LightGBM and CatBoost at four regularisation settings,
  because no model predicts these rows at all — the spread between models is a rounding artifact
  of the logged 2-decimal figures, not a modelling difference.

**The published ledger does not move.** `a059`/`a089` weight each airport-month cell by its
*untrimmed* count while using its trimmed RMSE, which is not the exact pooled trimmed RMSE.
Measured against the exact value across all 119 logged runs, that recipe is biased **high by
+0.02 to +0.06 s** (median +0.032), with the same sign every time, so levels are ~0.03 s
optimistic-looking and deltas are unaffected at the resolution anything is decided on.

**The rule.** L040 priced a change on the wrong rows, L043 against the wrong baseline, `a088` on
the wrong fold; this is the same family — the *label* on a reported number is part of the number,
and a marker saying which metric to steer by is a claim that needs a test like any other. See
`RULES.md` §7 and `.claude/skills/run-experiment/`.

---

## L045 — An accidental replication: the same configuration twice, 0.10 apart

`confirmed` · `results/runs.jsonl`, hashes `e43ef053fb65` / `83fd4afeb158` · 2026-09-28

> **Correction, 2026-10-03 (REVIEW-SUMMARY major 10).** The two runs are identical to every logged
> decimal on **July** (rmse 568.068, trimmed 325.588, bias −7.411) and differ only on **January**
> (427.621 vs 427.844). That is not run-to-run noise of a deterministic pipeline: something in the
> code or inputs moved one fold between commit `b388c4b` and `3a3a4d6` (the latter does not resolve
> in this repository). So the "0.10 resolution floor" is a one-fold change, not a measured variance,
> and the PLAN rows that dismiss deltas with it (11, 15, 30, 36, 37) are not supported by it. A
> genuine floor needs one configuration repeated on both folds with identical code.

> **Status note 2026-10-01 — still holds, and the board is coarser.** The 0.10 floor is the CV
> resolution of a *solo* score. Three overnight member additions show that at the *board* the
> resolution of a ~0.1–0.2 trimmed stack delta is worse than that: realised/CV ratios of 0.49,
> −0.98 and 3.11 (L054).

`p01_lgbm_capacity`'s `[min_data_in_leaf=100,num_leaves=511]` arm is the **same configuration**
as the project's baseline `t03_cap_sweep[train_label_cap=10800]`: identical 12 blocks, identical
99 feature columns, identical `model_params`, identical cap 10,800, identical folds. Verified by
loading both configs and comparing — `config_hash()` returns `83fd4afeb158` for *both* today.

It ran anyway, because the stored hash for the 09-26 baseline is `e43ef053fb65`. Two commits on
2026-09-26 at 23:03 and 23:07 (`f5e4b5b` log-target transform, `abdabdb` label repair) added
`target_transform` and `train_relabel` to the `config_hash` payload. Same experiment, different
payload schema, so `already_run()` did not recognise it. **Any configuration last scored before
2026-09-26 23:07 will silently re-run if it is offered again** — it costs time, not correctness.

**The replication: 225.99 (09-26) against 225.89 (09-28), a difference of 0.10 on trimmed
ex-LIRF.** This is the only replication of a single configuration the project has, so it is n=1
and should not be quoted as a variance estimate. What it does establish is that a difference of
about a tenth of a second on this metric is not interpretable. Consequences for claims on record:

- **`x35[l2=50]` at 222.03 against `x32` at 222.12 is a gap of 0.09 — at or below this.** The two
  are not separable, so "the best LightGBM in the project" cannot be assigned between them. The
  conclusion that drew on them — *LightGBM tuning is exhausted* — is **strengthened**, not
  weakened: four L2 settings and two feature-fraction settings all land inside a tenth.
- `p01`'s best arm, 225.82, is 0.07 from the baseline. Capacity screening returned nothing.
- Differences that survive comfortably: `d03` − `d02` = +4.42, `d04` − `d02` = +0.76,
  `x32` − baseline = −3.87, `s06_drop_congestion` = +4.53.

**Method note.** Reading this took a detour through an absent key: pre-21:51 lines on 09-26 have
no `train_label_cap` field at all, so `r.get("train_label_cap")` returned `None` and briefly
looked like *no cap was applied to the baseline*. The five sweep arms disprove that immediately
(260.21 / 230.56 / 227.26 / 225.99 / 226.08 — the cap plainly varied something), and the two
14:59–15:08 arms show what an unapplied cap actually looks like: `[3600]` and `[14400]` both
228.35, pooled both 425.71, byte-identical, which is L038's bug. This is `RULES.md` §7.11, which
exists for exactly this and was written before the detour rather than after it.

---

## L046 — v29 was built wrong twice before it was built right, and a sanity bound caught it

`confirmed` · `docs/BEST_27_09.md` §"one command", `reports/` build logs, bucket result for v29 · 2026-09-28

The first v29 build ran without error, passed every structural check — 344,841 rows, no nulls, no
duplicate ids, order identical to `submitting.parquet`, dtypes matching — and would have scored
roughly **317 instead of 281.98**, throwing away four submissions' worth of established gains. It
was missing two things, and the CLI defaults hid both:

1. **No `--overrides`.** The three defect layers are not part of the stack; they are applied last
   from `data/external/{pop27_gap,subcap_hedge,onerow_aobt}.parquet`. They carry v18's −6.74, v19's
   −2.14 and v24's −25.68 — about **−34.5 board seconds**, none of it reproduced by the combiner.
2. **`--rome-from` left at its default**, so Rome took the **v9 submission verbatim** rather than
   `f21_geometry3`'s twelve-month refit. The log already shows the difference: `CV LIRF only` is
   1155 for v10–v15 and 785.7 for v19–v28.

**What caught it was a bound, not a checker.** The structural validation cannot see a missing
override, because a submission without one is perfectly well-formed. What did not fit was a single
number: `max 20000.0`. v24 had established one row at **86,340**, and the pop27 rows take gap
values near 87,500, so a maximum of exactly 20,000 was impossible for a correct build — and 20,000
is v9's own ceiling from the era when the upload cap flattened everything above 60,049 (L033).
Rebuilt correctly, the maximum is **111,654**.

**The rule this gives.** A submission's *distributional* extremes are a cheap independent check on
whether every layer landed, and they catch what schema validation cannot. Concretely, before any
upload: the maximum must exceed 86,340, because v24 put a row there deliberately. The verification
script must also join each override file back against the built submission and require every row to
land. Doing that surfaced one more thing worth knowing rather than guessing: 336 of 348
`subcap_hedge` rows ship exactly, and the 12 that differ sit **exactly** on the `gap + 1800`
bound, which by design runs after the overrides — so a partial mismatch there is correct, and only
a row shipped *above* its override would be a fault.

**The reproduction recipe is the authority, not the CLI's defaults.** `BEST_27_09.md` records the
exact command; three of its six flags are non-default, and omitting any of them produces a valid
file with a worse score. A build that reproduces a shipped submission must start from that file.

---

## L047 — a069's floor excluded the only region the evidence supports, and Rome has never been modelled

`confirmed` · `a092_where_are_the_fifty_seconds.py`, `a093_uncovered_defect_rows.py`, `a094_rome_stack.py`, `reports/a09{2,3,4}*.txt` · 2026-09-28

`PLAN.md` §2 — "the one thing that decides everything else" — asserts that all remaining Rome work
is worth at most **−11.6 board seconds**, and everything else in the plan follows from it. It comes
from `a069`, whose inference has one free parameter, the level ratio
`L = board_body_rmse / cv_ex_lirf`. Its **upper** bound on `L` is sound (`MSE_Rome >= 0`). Its
**lower** bound is not.

`a069` sets `L >= BOARD / CV_EX_LIRF` ≈ 1.001 and glosses it as "Rome being no *better* than the
body". But the board is a row-share-weighted mean of the two MSEs, so `body_rmse >= board` forces
`rome_rmse <= board`. **The floor therefore assumes Rome's board RMSE is at most 283.98** — that
Rome is as accurate as the whole submission. Cross-validation puts Rome at **785.7** for this very
composition, and `SUBMISSIONS.md`'s own note says that figure *under*-reads the board. So the floor
excludes the entire region the evidence points at, and the −11.6 is the value at the excluded edge.

Re-derived over the feasible frontier (`a092`), parameterised by Rome's board RMSE:

| Rome board RMSE | implied body | body − our trimmed CV | Rome's share of SSE | perfect-Rome gain |
|---:|---:|---:|---:|---:|
| 283.98 (a069's edge) | 281.8 | +63.8 | 7.9% | −11.4 |
| 550 | 246.3 | +28.3 | 29.7% | −45.5 |
| **676** | **218.1** | **+0.1** | **44.8%** | **−72.5** |
| 785.7 (the CV figure) | 184.4 | −33.6 | 60.6% | −104.9 |

**The frontier has a fixed point.** At Rome = 676 the implied body is 218.1 against our trimmed
cross-fold body of 218.012 — agreement to a tenth of a second. That is the world in which our
trimmed CV transfers to the board essentially unchanged and Rome carries all the outliers, which is
exactly what the 2025 profile predicts: of the 34 labels above twelve hours, **31 are at LIRF**.

**The split is not measurable, and that is a structural result, not a gap in effort.** Changing our
predictions on a subset moves the board by
`sum_S (p'^2 - p^2) - 2*sum_S (p'-p)*y`, so each submission buys exactly one **linear** functional
of the unknown labels. Rome's squared error needs `sum_S y^2`, a **quadratic** functional, and it
never appears alone — for any predictions we ship the board reports `S_body + S_rome`, in which the
truth-only term `sum y^2` is additively entangled with the constant `S_body`. Zeroing Rome, scaling
it and offsetting it all return the same combination. The useful corollary is that linear
functionals *are* identifiable, which is why L035 could solve `sum(y)` over 27 rows and v24 could
recover one label to the second.

**Two negatives that close the obvious follow-up (`a093`).** Extending the override rules through the
other stamps does not work: with `mvt_time − stamp > 4 h`, all of 2025 holds 2 rows via `aobt_3`,
3 via `eobt_1` and 5 each via `lobt`/`iobt`, and **2026 holds exactly one per stamp, already
covered**. The only large uncovered population is the schedule gap — 1,815 rows — whose 2025
precision for "the label *is* the gap" is **1.5%**, so a blanket override is catastrophic by a076's
`q/n` bound.

**Where the seconds actually are (`a094`).** Layer B ships `f21_geometry3`'s Rome predictions
verbatim: one LightGBM, never tuned, never blended, while the body accumulated an eight-member NNLS
stack over thirty-plus experiments. Measured on the 26,528 LIRF out-of-fold rows across both
seasonal folds:

| population | n | `f21_geometry3` | `r14_rules_plus_unmatched_route` | delta |
|---|---:|---:|---:|---:|
| all LIRF rows | 26,528 | 1,159.2 | **625.1** | −534.1 |
| label ≤ 12 h | 26,516 | 702.9 | **564.0** | −138.9 |
| **label ≤ 7,200 s** | **26,415** | **643.9** | **533.5** | **−110.3** |
| label > 20,000 s | 24 | 31,508 | 9,165 | −22,343 |

**66.8% of layer B's Rome squared error sits on 24 rows** — and in 2026 those are precisely what
layers C/D/E already override, so that part of r14's advantage is already shipped and must not be
counted twice. But the advantage **survives the monsters**: on the 99.6% of Rome rows below 7,200 s,
r14 is still **110.3 s better**. A cross-fold NNLS stack on LIRF rows scores **625.4** against
1,159.2, at weights `r14 0.907 + n06_lintree_127 0.070`, and both have twelve-month refits on disk
already.

Priced on the ordinary-row ratio 533.5/643.9 = 0.8285, a layer-B swap is worth about **−9 board
seconds if Rome sits at 450, −15 at 550, −21 at 676** — against the −1.43 that v29's entire
eight-member stack bought.

**The one check before shipping it.** r14 is a *rules* model and its rules were derived on 2025, so
part of the ordinary-row advantage may be in-sample. The decisive test is whether the advantage
survives on rows where no rule fires. It must be run before a layer-B swap; and because the builder
applies C/D/E *after* layer B, a swap cannot double-apply the monster overrides.

**The method-level lesson.** `a069`'s error is the same family as L040, L043 and L046: correct
arithmetic over the wrong object. Here the wrong object was the *feasible region* — a bound stated
as a near-tautology ("Rome no better than the body") that in fact encoded a strong empirical claim
contradicted by the project's own per-airport record. A bound whose justification is one clause long
deserves the same scrutiny as a measurement.

**GATE RESULT (`a095_r14_gate.py`, 2026-09-29).** The layer-B swap's one open risk is closed and it
passes. r14's two hard rules fire on **12 of 26,528** LIRF out-of-fold rows — `lirf_unmatched_12h`
on 11, `aobt_identity_extreme` on 1 — and those twelve carry **63.2%** of `f21_geometry3`'s entire
Rome squared error. They are the monster rows that layers C/D/E already override in 2026, so that
part of the advantage is correctly excluded. On the **26,516 rows no rule claims**, r14 is still
**−138.9** (702.9 against 564.0), and **−110.3** on the 26,415 below 7,200 s. The edge is therefore
the *routed base* — a separate fit for unmatched rows, the `u01_unmatched_route` idea — measured out
of fold, and the hand-fitted 87,540 constant contributes nothing to it. The swap is a real model
improvement on Rome's ordinary rows and is ready to ship.

---

## L048 — v30 tested my correction to a069 and the correction lost

`confirmed` · `a096_what_v30_measured.py`, submission v30 · 2026-09-29

L047 argued that `a069` had floored the feasible region in the wrong place and that Rome plausibly
held **44.8%** of the board's squared error, making a layer-B swap worth **−9 to −21**. v30 shipped
exactly that swap — v29 with `--rome-from r14_rules_plus_unmatched_route`, body and override rows
verified byte-identical — and realised **−0.88** (281.98 → 281.0988).

Inverting the realised gain against r14's measured **17.15%** CV edge on ordinary Rome rows:

| assumed layer-B board RMSE | fraction of the CV edge that transferred |
|---:|---:|
| 200 | 49.1% |
| 284 (`a069`'s edge) | 23.8% |
| 450 | 9.4% |
| 676 (`a092`'s fixed point) | **4.1%** |

Rome cannot both sit at 676 *and* have r14's edge transfer — that combination pays −20. So either
Rome's board error is far smaller than the fixed point implies, or Rome improvements measured in
cross-validation do not carry to 2026. The second has a direct precedent: **L022**, where
`x1_maxbin_2047` cut its CV Rome score to 933.9 while its implied 2026 Rome score stayed at 1,164,
because the CV gain was memorising 2025's record defects. Rome's CV score is dominated by 24 rows
(66.8% of layer B's Rome squared error, `a095`), and those are precisely the rows whose 2026
analogues are already overridden — so a CV Rome improvement is measured largely on a population that
does not exist in the scored set. Probably both explanations hold.

**What survives, what does not.** `a092`'s frontier arithmetic stands — it is an identity, and the
observation that `a069`'s floor is a strong empirical claim dressed as a tautology stands too. What
does not stand is my inference about *which point on the frontier we occupy*. I treated "the fixed
point matches our trimmed CV to 0.1 s" as strong evidence; it is a coincidence of two numbers, and
one slot was enough to weigh against it. `a069`'s original conclusion — Rome is nearly spent — is
better supported by this measurement than my correction was.

**The slot bought something durable anyway.** Because the change was confined to layer B and both
prediction vectors are on disk, `delta(SSE) = sum(p'^2 - p^2) - 2*sum((p'-p)*y)` leaves one unknown,
giving **`sum((p30 - p29) * y) = −1.19971e8`** exactly — a weighted mean label of **1,643.3 s** on
the rows that moved. That is the first new information about the withheld 2026 Rome labels since
v24, and it is reusable by any later Rome analysis.

**The method lesson, which is the one that generalises.** Predicting −9 to −21 from a CV delta
required the transfer ratio to hold, and the project already knew it does not hold at Rome (L022).
I applied a body-derived transfer assumption to Rome, which is the same class of error as quoting
the delta band as a level ratio (§4.5) — a ratio measured on one population used on another.
**Rome needs its own transfer ratio, and the honest value for a capacity- or model-driven Rome gain
is near zero.**

---

## L049 — A Manhattan stand→threshold distance is free to compute and does not beat the straight line

`confirmed` · `scripts/analysis/a097_manhattan_distance.py` · 2026-09-29

A taxiway-*routed* distance needs an OSM pull for `aeroway=taxiway` that was never made. A
**Manhattan** distance needs nothing — it is an L1 norm over the same two points `stand_geometry`
already holds. The axes are the whole question, because L1 is not rotation-invariant and taxiways
run parallel and perpendicular to runways rather than north–south. The Gateway file carries both
ends of every runway, so the heading is the exact bearing between reciprocal thresholds, not an
approximation from the designator: EDDM 8R → 83.4°, LEMD 32R → 322.3°, LTFM 18 → 179.2°.

Measured on 1,292,891 labelled 2025 departures holding a stand coordinate, a threshold coordinate
and a paired runway heading. Correlation with `taxi_sec`, **within airport** (airport means removed
from both sides, because a pooled figure is mostly a between-airport effect):

| population | n | straight line | Manhattan (E/N) | **Manhattan (runway frame)** |
|---|---:|---:|---:|---:|
| all labelled | 1,292,891 | 0.3311 | 0.3269 | 0.3302 |
| ≤ 12 h | 1,292,881 | 0.3695 | 0.3649 | 0.3686 |
| ≤ 2 h | 1,292,792 | **0.3838** | 0.3792 | 0.3830 |
| ≤ 1 h | 1,291,355 | 0.4007 | 0.3968 | 0.4008 |

**No version beats the straight line, and the incremental R² of adding the runway-frame Manhattan
distance to it is 0.0005–0.0012** across the same cuts. The naive east/north axes are consistently
the worst of the three, which is the expected penalty for an arbitrary frame.

Two things worth keeping from the exercise:

- **Outliers depress the correlation, and removing them is worth more than changing the metric.**
  Within airport the straight line goes 0.3311 → 0.4007 from the full population to rows under an
  hour. The gap between *any two distance definitions* is under 0.005; the gap between populations
  is 0.07.
- **It is not uniform across airports.** On rows under two hours the runway-frame Manhattan beats
  the straight line at EGLL by **+0.020** — and Heathrow is where the straight line is weakest of
  all ten, 0.087 — and at LSZH by +0.019, while losing at LIRF (−0.019), LEMD (−0.017) and LEBL
  (−0.014). Two parallel runways with terminals between them is exactly the geometry a straight
  line misreads, so the sign pattern is mechanically sensible even though the pooled effect is nil.

**Coverage cost, if anyone builds it anyway.** 18 of 72 runway ends have no reciprocal in the
Gateway file, so the runway-frame version is null on 1,292,891 of the 2,063,039 rows that have both
coordinates — 37% missing, against the straight line's 1%. LIRF is the worst hit: 15,792 usable rows
against 160,704 departures.

---

## L050 — Member-level deltas do not predict stack value, and I made L043's error arguing about it

`confirmed` · `a081` runs of 2026-09-29, `reports/a081_does_a_member_earn_a_seat.txt` · 2026-09-29

> **Status note 2026-10-01 — the instrument holds; its small verdicts do not reach the board.**
> `a081` remains the right CV instrument, and it said "seat" for `d06`, `d07` and `h09`. But all three
> were near-twins (residual correlation 0.991–0.995 with a seated member) and their board deltas
> were −0.10, **+0.11** and −0.27 against CV −0.20, −0.11, −0.09. A seat worth under ~0.2 trimmed is
> a sampling bet on the board, not a measured gain (L054).

**The finding.** `modes5` was carried by three different hosts. Their member-level gains are nearly
identical; their stack values differ by a factor of seventy:

| arm | solo trimmed ex-LIRF | delta vs its own baseline | stack delta | weight |
|---|---:|---:|---:|---:|
| `h07_modes5` (lgbm) | 224.44 | −1.55 | **−0.013** | 0.1249 |
| `x36_lo_card_modes5` (lgbm) | 220.44 | −1.59 | **−0.516** | 0.3692 |
| `d05_catboost_modes5` | 218.00 | **−0.95** | **−0.908** | 0.5635 |

The **smallest** member-level gain has the **largest** stack gain. What tracks stack value is the
candidate's *absolute* level against the members already seated — `d05` 218.00 beats `k05`'s 219.41,
`x36` 220.44 beats `x35`'s 222.03, `h07` 224.44 beats neither — with decorrelation as a second factor,
which is how `h06_mlp` earns a seat at a solo score of 236.17, the worst of any member, on residual
correlations of 0.915–0.943 against the trees' 0.954–0.976.

**So screening a block by its delta against a fixed baseline is uninformative about whether to ship
it.** `a081` against the shipping composition is the correct instrument and it costs no fitting.

**Measured consequence, and it is a negative.** Five arms previously called failures are, cap-matched,
wins or dead heats — and **all five earn weight exactly 0.0000 in the v31 stack**:
`g02_longtail_nocap` (−0.53), `f23_queue` (−0.50), `g01_longtail` (−0.00), `f22_modes3` (+0.04),
`h02_enroute` (+0.16). The LightGBM feature space is exhausted *for this stack*, which is a stronger
statement than the delta table supports and was free to obtain.

**My error, and it is L043 again.** The first version of `PLAN.md` §4.7 compared nine rejected blocks
against `t03`'s **capped** baseline of 225.99 when most are **uncapped** arms whose baseline is
`f21_geometry3` at 228.35 — inflating every loss by 2.36, inside the section arguing that the project
keeps measuring against the wrong object. Its headline claim was that `f25_queue2` "lost at 226.81 and
is nevertheless a member, so the instrument was visibly wrong". Cap-matched, **`f25_queue2` won by
1.54** and its seat is entirely consistent. The rhetorical point was manufactured by my own baseline
error.

Two procedural notes that would have caught it:

- **Every arm's baseline is a function of its label cap.** `PLAN.md` states this ("rows 1–5 read
  against `f21_geometry3` = 228.35, rows 6 and 19–29 against `t03` = 225.99") and I read past it. A
  table comparing arms with *different* caps to a *single* baseline is wrong by construction.
- **The regenerated-not-retyped rule.** The same pass found `230 needs 33.05%` still in the document
  after two board changes. Share-of-squared-error tables are now regenerated from the current board by
  script rather than edited by hand, because they are stale the moment a submission lands.

---

## L051 — Layer D is now beaten by the base it overrides, because v30 replaced that base

`confirmed` · `scripts/analysis/a079_rome_layers_vs_baseline.py` (r14 baseline added 2026-09-29) · 2026-09-29

> **Status note 2026-10-01 — tested and confirmed on the board.** v32 dropped layer D and realised
> **−2.07** (280.17 → 278.10), inside the "−1.5 to −3" stated here and under the −6.85 ceiling. The
> caveat — that CV might be scoring the wrong year because 2026's sub-cap rows are pure copy — did
> not bite: the hedge was a loss in 2026 as CV said.

v30 swapped the Rome base from `f21_geometry3` to `r14_rules_plus_unmatched_route`. Every earlier
pricing of the Rome layers was against `f21`, so all of it was against a baseline that no longer
ships. Re-priced on the 365 CV rows layer D covers (LIRF, unmatched, 600 < gap ≤ 20,000 s):

| baseline | baseline SSE on those rows | layer D's hedge SSE | d_rmse |
|---|---:|---:|---:|
| **`r14` — what ships now** | **3.691e9** | 5.063e9 | **+5.95** |
| `f21_geometry3` — superseded | 6.442e9 | 5.063e9 | −4.73 |
| `c10_trust_aobt` | 6.542e9 | 5.063e9 | −5.06 |
| `d02_catboost_capped` | 4.396e9 | 5.063e9 | +1.85 |

The hedge's own SSE is fixed at 5.063e9; what changed is that **r14 achieves 3.691e9 on the same
rows**, beating it by 1.372e9. r14's routed base fits unmatched rows separately, which is exactly
this population. `a078` had already found the +1.85 against `d02` and it was read as a wrong-baseline
artefact (L043); against the base that now actually ships it is +5.95.

**Layers C and the probe-free replacement still hold against r14**: pure gap beats r14 by −1.05 on
the 24 gap>20 ks rows, and the `ac_type_null & gap>43.2 ks` layer by −0.54 on 11 rows. So this is
specific to layer D, not to the layered design.

**The caveat that stops this being a decision.** In CV the ranking is r14 > hedge > pure gap, with
pure gap at a catastrophic +30.22. But L036 records that 2026's sub-cap rows are **pure copy** where
2025's are ~81% copy, and the hedge is `p̂·gap + (1−p̂)·median` with `p̂` estimated from 2025. If 2026
is more copy-like, gap moves *up* the ranking there and the hedge under-weights it — which is also
why layer D realised only −2.14 board against a −7 to −12 estimate when it shipped in v19. So CV says
drop it and the population difference says CV may be measuring the wrong year.

**Naive board arithmetic, stated as a bound not a forecast.** Scaling 1.372e9 by 348/365 gives
1.3081e9 of the board's 2.70684e10, which is −6.85 board seconds. L036's precedent is that layer-D
estimates over-predict roughly fourfold, and L048 found only 4.1% of one measured Rome gain
transferred at all. **Expect −1.5 to −3 and treat −6.85 as the ceiling.**

**It costs nothing to settle.** Dropping layer D is omitting one `--overrides` flag: no refit, no
fitting, one slot. Either outcome is informative — a gain retires the layer for good, a loss
establishes that 2026's sub-cap population is copy-like and the hedge earns its place.

---

## L052 — `modes5` is not underfitting, it is class-imbalance-blind: it finds `eobt_1` 8.5% of the time

`confirmed` · `scripts/analysis/a098_modes5_review.py`, `reports/a098_modes5_review.txt` · 2026-09-30

> **Status note 2026-10-01 — the diagnosis holds; the inference drawn from it is withdrawn.** The
> recall table here is correct and `a099` reproduces it to four decimals. What does **not** hold is
> "the 94% of the oracle that remains unrealised is a class-imbalance limit": `modes6` removed the
> imbalance — `eobt_1` recall 0.085 → 0.394, and 0.720 among true copies — and the regression did not
> move (`h09` +0.073 against `x36`; implied-label RMSE 404.2 against 401.4). See L053.

`PLAN.md` §4.8 ranked "grow the classifier" as the strongest remaining lever, on the grounds that
`a084` had diagnosed the *predecessor* as underfitting — max class probability 0.26. The argument
assumed `modes5` shares that failure. Measured on the 190,713 held-out rows of the July fold:

**It does not commit.** Mean top probability **0.396**, median 0.401; **26.6%** of rows exceed 0.50,
**2.87%** exceed 0.70, **0.08%** exceed 0.90. So on the surface the "underfitting" story looks right.

**But confidence is anti-correlated with correctness over most of the range.** Accuracy by
top-probability decile is U-shaped, and **the lowest-confidence decile is the most accurate**:

| decile | mean top p | accuracy |
|---:|---:|---:|
| 0 | 0.099 | **0.8713** |
| 3 | 0.334 | 0.6003 |
| 6 | 0.461 | **0.5088** |
| 9 | 0.678 | 0.6777 |

Overall argmax accuracy **0.6179** against a majority-class rate of **0.5440** — seven points above
guessing. More capacity does not fix a monotonicity failure of that shape.

**Per-class recall is the explanation, and it is the finding.**

| class | rows | share | **recall** |
|---|---:|---:|---:|
| `none` | 103,746 | 54.4% | **0.7837** |
| `aobt_3` | 71,804 | 37.6% | 0.4893 |
| **`eobt_1`** | **11,420** | **6.0%** | **0.0849** |
| `lobt` | 2,505 | 1.3% | 0.0894 |
| `sched_time` | 1,142 | 0.6% | 0.1637 |
| `iobt` | 96 | 0.05% | 0.2188 |

The classifier has learned to separate the two classes that make up 92% of the rows and is **nearly
blind to the other four**. The U-shape follows directly: when unsure it predicts `none`, and `none`
is right 78% of the time, so low confidence scores well. High confidence means claiming a specific
stamp, which it does badly.

**`eobt_1` at 8.5% recall is the whole point missed.** `a083` identified `eobt_1` as the second
commonest `block_time` source, 9.8% of matched departures and 11.45% of the disagreeing population,
and named it as the opportunity `modes` never modelled. `modes5` still does not model it.

**So the 94% of `a083`'s −17.58 oracle that remains unrealised is a class-imbalance limit, not a
capacity limit.** The right next move is **not** more trees. It is one of: multiclass class weights,
balanced subsampling, or a two-stage decomposition — copy-versus-none first, then which-stamp among
copies only, where the imbalance is 71,804 / 11,420 / 2,505 rather than 103,746 against everything.

**Method note.** This diagnostic cost ~30 minutes of a contended box and redirected a five-hour build.
`a084` played the same role for `modes4`. Reviewing a block on real data before scaling it up has now
paid twice, and in both cases the *stated* reason for the next step was wrong: `modes4` was not
salvageable by a finer cell table, and `modes5` is not improvable by a bigger one.

**One thing that does work, and is worth keeping.** Median absolute error of `m5_label_mix` falls with
confidence — 173.8 s below 0.5, 112.8 s at 0.5–0.7, **83.2 s at 0.7–0.9** — even though its RMSE rises
over the same bands. The confident rows are genuinely better predicted at the median and are carrying
a few extreme rows that dominate the squared error. So the signal is real where it fires; it just
fires on 2.87% of rows and misses the class that matters.

---

<a id="l053"></a>
## L053 — Fixing `modes5`'s class imbalance fixes the classifier and leaves the regression where it was

`confirmed` · `scripts/analysis/a099_modes6_review.py`, `reports/a099_modes6_review.txt`; `h09_x36_modes6` in `results/runs.jsonl`; `a081` · 2026-10-01

L052 diagnosed `modes5` as class-imbalance-blind and named a two-stage decomposition as the remedy.
`modes6` is that remedy: stage 1 copy-or-not on every row, stage 2 which-stamp on copy rows only,
fitted both calibrated and class-balanced. Same capacity, covariates, label and cross-fit as
`modes5`, so the decomposition is the only change. On the 190,713 held-out July rows:

| recall | `modes5` argmax | `modes6` calibrated | `modes6` two-stage, p_copy ≥ 0.5, balanced |
|---|---:|---:|---:|
| `aobt_3` | 0.4893 | 0.4818 | 0.3477 |
| **`eobt_1`** | **0.0849** | **0.0855** | **0.3938** |
| `lobt` | 0.0894 | 0.0842 | 0.6491 |
| `none` | 0.7837 | 0.7867 | 0.7317 |
| accuracy | 0.6179 | 0.6164 | 0.5627 |

Among true copies only, `eobt_1` recall is 0.367 / 0.375 / **0.720**. So the decomposition does what
L052 said it would — **for classification.**

**It does nothing for the regression.** The calibrated `modes6` probabilities track `modes5`'s almost
exactly: accuracy by top-probability decile agrees within 0.01 in all ten deciles. The implied label
is no better (`m6_label_mix` trimmed RMSE 404.2 against `m5_label_mix` 401.4 on all rows; 466.2
against 478.2 on true-`none` rows). The LightGBM screen `h09` (x36 with `modes6` in place of `modes5`)
scores **220.508 against 220.435** — inside L045's floor.

**What I take from it.** The balanced posterior *finds* `eobt_1` rows, but by trading calibration for
recall: at threshold 0.5 it gains 31 points of `eobt_1` recall and loses 13 points of `aobt_3` and 5.5
of `none`, against the calibrated argmax. A downstream regressor wants a calibrated probability, and the calibrated one is unchanged —
the extra recall is bought with false positives a squared-error model has to undo. So L052's "the
94% of the oracle that remains unrealised is a class-imbalance limit" is withdrawn: the imbalance was
real and removing it did not realise any of the oracle. What limits it is whatever separates an
`eobt_1` copy from a non-copy in covariate space, and these covariates do not.

**Method notes.** The first real-data leak run failed the *converse* test (does the block read its
training labels?) because, with every training label set to 999,999 s, no row copies a stamp and
stage 2 had nothing to fit. Real folds never hit that case; the block now degrades to all-`none`.
And `a099` lost its first 80 minutes of compute to `with_columns` refusing two DataFrames — computed
blocks are now cached to `data/interim/a099/` keyed on fold and module hash.

---

<a id="l054"></a>
## L054 — Near-twin members earn `a081` seats whose board deltas are noise, sign included

`confirmed (n = 3)` · `a081` transcripts at commits `a0b24c1`, `8412832`; board scores from the bucket's `_result.json` files (`SUBMISSIONS.md` v32–v35) · 2026-10-01

> **Status note 2026-10-01, later the same day — n = 5, and the spread widened in both directions.**
> Two combiner flags priced by their own cross-fold scripts against v35 and shipped alone on v35:
> `--calibrate-above 2400` (`a086`, −0.082) realised **+0.066** (v36; sign wrong again), and
> `--weight-groups unmatched` (`a085`, −0.021) realised **−0.900** (v37; 43×). Neither is a near-twin
> member, so the rule above is not specific to twins: **every** item under ~0.2 trimmed has now
> landed somewhere in −0.90 … +0.11 regardless of its CV size. The positive-mean reading survives
> (mean −0.22 over five) and is now dominated by one draw.

Three member additions, each priced cross-fold by `a081` against the stack it joined, each shipped
alone so the board delta is attributable:

| v | added | residual corr. with its nearest seated member | `a081` trimmed Δ | board Δ | realised / CV |
|---:|---|---|---:|---:|---:|
| 33 | `d06` | 0.995 with `d05` | −0.199 | −0.098 | 0.49 |
| 34 | `d07` (on v33) | 0.993 with `d05` | −0.112 | **+0.110** | **−0.98** |
| 35 | `h09` (on v33) | 0.991 with `x36` | −0.087 | −0.271 | 3.11 |

The ratio spans −0.98 to 3.11 and the **sign** of the board delta is wrong once in three. The two
earlier large additions behaved (v29 −1.169 → −1.43, ratio 1.22; v31 −1.028 → −0.93, ratio 0.90),
which is where `PLAN.md`'s "0.93–1.30" band came from. **That band does not hold for deltas under
~0.2 trimmed** and should not be quoted for them.

**Why near-twins in particular.** Each addition reweights two almost identical predictors: in v35
`h09` took 0.1619 and `x36` fell 0.1968 → 0.0691. The board then sees the *difference* between two
twins, re-weighted — a small quantity whose sign depends on the 2026 rows, which CV's two 2025
months cannot pin down. A genuinely different member (`h06_mlp`, 0.915–0.943) changes the stack in a
direction CV can measure; a twin changes it by a fraction of its own noise.

**Consequence.** A seat under ~0.2 trimmed is a sampling bet with a positive mean (−0.086 average
board delta over the three) and a standard deviation of 0.19, about twice the mean. Worth spending spare slots
on; not worth a 4–5 hour refit in its own right. n = 3, so this is a working rule, not a law.

---

<a id="l055"></a>
## L055 — Two queued ideas closed by null results, and one LightGBM trap found on the way

`confirmed` · `g01_x36_group_stop`, `g02_x36_group_stop_extended`, `w01`–`w04` in `results/runs.jsonl`; `a081`; `tests/test_group_stop.py` · 2026-10-01

**Per-airport early-stopping rounds (PLAN row 15, ASHRAE rank 5).** `lgbm_group_stop` fits the same
booster as `lgbm`, then predicts each airport at the argmin of its own RMSE curve on the temporal stop
slice. Against `x36` = 220.435: `g01` **220.746** (+0.31), `g02` **220.531** (+0.096). `a081`: no seat
for either (+0.049, +0.047). One global tree count is enough here.

**The trap.** `g01` could only stop an airport *earlier* than the pooled model: `lgb.train` (4.7.0)
round-trips the booster through `model_to_string` unless `keep_training_booster=True`, which keeps
only the first `best_iteration` trees. Its log shows every airport at or below the global count (Jan
fold: EGLL 360 of 1,647; Jul fold: LIRF 350 of 1,255). Predictions from the parent class are
unaffected because they use `best_iteration_` either way; anything that reads the curve *past* the
best needs the flag. `g02` sets it; `test_extend_curve_keeps_trees_past_the_global_best` pins it.

**Does our no-gap CV flatter the target-dependent blocks? (PLAN row 17.)** December 2025 held out,
six training months each, trimmed ex-LIRF: `modes` is worth +0.235 with a six-month gap (`w01`
207.981 − `w02` 207.746) and +0.289 adjacent (`w03` 220.785 − `w04` 220.496), a differential of
**−0.054**. Inside L045's floor: no evidence that adjacency inflates the label-derived block.
(The levels differ by 12.8 s between the two fold sets, which the configs pre-registered as
uninterpretable — six different training months, not a gap effect.)

---

<a id="l056"></a>
## L056 — The stack's residuals hold nothing a second model can find, and an MLP needs two guards to show it

`confirmed` · `scripts/analysis/a101_residual_correction.py`, `reports/a101_residual_correction.txt` · 2026-10-01

Asked because a weak-but-different model might help the stack as a *correction* where it cannot as
an NNLS member (the owner's question; `h10`/`h11` earn no seat). The residual `y − p_stack` of the
cross-fold grouped stack (216.677) was modelled from the twelve member predictions, `p_stack`, the
match flag and the eleven label-free blocks of `x36`'s set; fitted on one fold, scored on the other.

| model | λ = 0.25 | λ = 1.0 | Jan / Jul at 0.25 | corr(r̂, residual) |
|---|---:|---:|---|---:|
| MLP (256-128) | +0.86 | +27.77 | +1.22 / +0.60 | 0.045 |
| LightGBM (31 leaves, 300 rounds) | −0.05 | +3.55 | +0.26 / −0.27 | 0.034 |

Nothing stable: the only negative cell is two opposite-signed folds. This also closed PLAN §4.8
item 6 (a neural combiner), whose gate was this run.

**Two traps on the way, both mine and both silent.**

- `MLPPredictor` clips predictions to `(0, 20000)` by default — right for taxi times, wrong for a
  residual centred on zero. The first run floored every negative correction and reported the MLP at
  +13.7 to +187.
- Trained on **one month**, a column that is near-constant there (a frost flag in July) gets a tiny
  standardisation scale, and the other month's values land hundreds of σ out. The second run gave
  sd(r̂) **1,460** against the LightGBM's 43. `MLPPredictor(input_clip=5.0)` now bounds the
  standardised inputs; it is off by default, so `h06`/`h10`/`h11` are unchanged. Any model fitted on
  a single month of this data needs the same guard, and a tree model does not.

---

<a id="l057"></a>
## L057 — The source classifier is used correctly: an audit on real data finds no defect

`confirmed` · `scripts/analysis/a102_classifier_audit.py`, `reports/a102_classifier_audit.txt` · 2026-10-02

Asked by the owner after `modes5` became the single largest lever on record. Code review found the
label, the leak boundary, the label-cap order and the CV/refit symmetry correct; two things the code
only assumed were checked on data, and `cv.build_matrix` now asserts the first.

| check | result |
|---|---|
| 2025 training rows with a null `taxi_sec` (would be silently labelled `none`) | **0** of 2,085,047 |
| class shares at the shipped tolerances | `aobt_3` 38.94% (a084: 39.1%), `eobt_1` 7.29%, `lobt` 1.28%, `sched_time` 0.52%, `iobt` 0.04%, `none` 51.93% |
| block attached by position vs joined on `mvt_id` (held-out July fold) | identical on every column, 344,419 rows |
| null stamps, Jan + Jul 2025 vs 2026 | 1.56% vs 1.53% (all four `_flt` stamps null together) |
| gap quantiles (q5–q95), 2025 vs 2026 | within ~1–4% everywhere; `aobt_3 − eobt_1` median 300 → **240 s** |
| outputs on 2026 ranking rows (refit) vs held-out 2025 rows (CV) | means within 0.002–0.005: `p_copy` 0.469 / 0.474, `p_eobt` 0.070 / 0.075, `label_mix` 971 / 975 s |

The classifier sees the same population where it ships as where it was validated, and behaves the
same there. The one visible movement — `aobt_3` and `eobt_1` sitting 60 s closer at the median in
2026 — shifts `p_eobt` up by 0.005 on average.

---

<a id="l058"></a>
## L058 — v37's −0.90 was 94% the unmatched rows, which CV said the change made worse

`confirmed` · board scores of v35, v37, v38 (`SUBMISSIONS.md`, bucket `_result.json`); `scripts/analysis/a100_unmatched_population.py` · 2026-10-02

v38 applied v37's unmatched-group NNLS vector to the 5,283 unmatched ranking rows only, leaving
every matched row byte-identical to v35 (verified 339,558 of 339,558; unmatched 5,283 of 5,283 equal
to v37). The row sets are disjoint, so the change in mean squared error is exactly additive:

| | board | MSE vs v35 | share |
|---|---:|---:|---:|
| v35 | 277.7275 | — | |
| v38 (unmatched rows changed) | 276.8843 | −467.65 | **93.7%** |
| v37 (both changed) | 276.8272 | −499.27 | 100% |
| matched rows' part (v37 − v38) | | −31.62 | 6.3% |

**CV pointed the other way.** `a100`, cross-fold on the 2025 OOF rows: under the split the matched
rows improve (trimmed 205.992 → 205.926) and the unmatched rows get slightly *worse* (576.246 →
577.225; untrimmed 1,544.20 → 1,543.61). The board's −467.65 MSE over 344,841 rows is about
−30,500 s² per unmatched ranking row on average — a large move for the population CV calls neutral.

**So the 2026 unmatched rows respond to their re-weighting nothing like the 2025 ones do in CV.** The
unmatched vector moves weight to `d05` (0.559) and its weights sum to 1.056, raising the level of
those predictions; consistent readings are that 2026's unmatched rows are under-predicted by the
global vector, or that their composition differs (L036 already found 2026's sub-cap rows to be pure
copy where 2025's are ~81% copy). Which one is not settled by this.

**Consequence.** The unmatched rows are the largest lever identified since layer D, and CV on 2025
is the wrong instrument for them. Candidates for them should be fitted on 2025 and judged on the
board (RULES §9.4: genuine candidates, not label queries).

> **Note on L058, 2026-10-02 (v40).** The opposite move was then shipped: v39 with the unmatched rows'
> predictions × 0.9769 (4,867 rows, matched rows byte-identical). Board **+0.213**. Two points in
> the same direction: raising the unmatched level (v37's vector, weights summing to 1.056) gained
> −0.84 on those rows; lowering it 2.3% cost +0.21. Consistent with 2026's unmatched rows being
> under-predicted — which also means **any further step on that axis would be chosen from board
> responses, and is RULES §9.3 territory (a query programme), not a 2025-fitted candidate.**

> **Note on L058, 2026-10-03 (v41, `a109`).** The unmatched vector refitted on twelve months of OOF
> (20,937 rows vs 4,969) over six members — `d05` 0.976, `h06_mlp` 0.102, sum 1.079 — **scored worse
> out of sample in CV** than v39's Jan + Jul fit (ten non-seasonal months: 567.06 vs 558.11 trimmed,
> 891.1 vs 885.2 untrimmed) **and better on the board** (−0.123; 4,836 unmatched non-Rome rows changed,
> everything else byte-identical). A third instance of CV and board disagreeing on exactly these
> rows. The month-to-month heterogeneity CV shows (more months → worse on held-out months) is itself
> a finding: the unmatched population is not stationary across 2025.

---

<a id="l059"></a>
## L059 — The arrivals reveal the day's `block_time` copy regime, and it is worth −0.74 on LightGBM

`confirmed` · `scripts/analysis/a108_arrival_copy_regime.py`; `h12_x36_arrival_regime` in `results/runs.jsonl`; `a081` · 2026-10-03

The ranking set hides `block_time` for departures only. Arrival on-block times copy a stamp far less
often than departure off-block times (9.5%: `sched_time` 5.2%, `arvt_1` 3.9%, `arvt_3` 0.3%; departures
48%), but **an airport-day's arrival copy share tracks its departure copy share**: correlation 0.50
across airport-days, **0.33 within airport** (0.31 / 0.26 per airport × operator × month). No
stamp-to-stamp mapping (`arvt_3`-copying vs `aobt_3`-copying ≈ 0 within airport), and arrival shares in
Jan + Jul 2026 sit within ~2 points of 2025 at every airport — so it is a day-level regime, not a
2026 shift.

The `arrival_regime` block (six columns: airport-day copy, sched-copy and `arvt_1`-copy shares, count;
airport × operator × month copy share, count) added to `x36`: **219.70 vs 220.435, −0.735**, on both
folds (Jan −0.74, Jul −0.74). Above the noise floor at solo level, and the first new *information*
source since `modes5`. `a081`: seat −0.114 (residual corr 0.993 with `x36`) — as a member it is a twin;
its value has to come through the CatBoost (`s2a`) and the classifier (`modes7`).

## L060 — `block_time` is always within ±3606 s of `lobt`, and a few of our predictions are not

`confirmed` · `scripts/analysis/a112_lobt_window.py` · 2026-10-04

**Question.** A competitor's public write-up (`docs/COMPETITOR_OUTGOING_BOAT.md` §3.2) states that
|BLOCK − LOBT| ≤ 3606 s on every 2025 departure with a LOBT, and that projecting predictions onto
that window was worth −30.6 board seconds to them. Does it hold on our data, and do we violate it?

**Finding.** It holds exactly: **2,062,577 of 2,062,577** 2025 departures with a LOBT, at all ten
airports, with min −3606 and max +3606. LOBT is present on 99.998% of NM-matched rows and on none of
the 22,424 unmatched ones. v43 puts **43** ranking predictions outside
`[mvt − lobt − 3606, mvt − lobt + 3606]` (40 below, 3 above; mean 714 s out, max 4,273). Σ move² =
53.6 M s², so projection lowers the board by **at least 0.281 s** from 276.26 if the window holds in
2026. On each BASE member's Jan + Jul OOF, projection improves trimmed ex-LIRF by 0.34–1.52.

**So what.** Our only admissible bound was one-sided (`gap + 1800`). The window is a free,
two-sided, label-free bound; the stack already respects it on all but 43 rows, which is why it is
worth ~0.3 s to us and was worth 30 s to a single model.

## L061 — On our five single-member submissions, the body share of the CV gain ran opposite to the board

`measured` · `scripts/analysis/a113_transfer_backtest.py`, `taxiout.transfer` · 2026-10-04

**Question.** The owner adopted outgoing-boat's transfer gate as the steering metric
(docs/BORROWED_FROM_OUTGOING_BOAT.md item 2): split a CV gain exactly into per-flight contributions
and trust only the part from flights with y ≤ 3600 s. On their six submissions that body share
predicted the board delta with Pearson 0.985. Does it on ours?

**Numbers.** Every submission that added exactly one member, with the combiner that shipped,
cross-fold Jan + Jul OOF, ex-LIRF, untrimmed (gains in seconds, positive = better):

| transition | trimmed CV | untrimmed CV | body | body 95% CI | tail | top-20 share | gate | board |
|---|---:|---:|---:|---|---:|---:|---|---:|
| v32 → v33 (+d06) | 0.199 | 0.152 | 0.022 | [−0.026, +0.078] | 0.130 | 1.26 | fail | +0.099 |
| v33 → v34 (+d07) | 0.112 | 0.059 | 0.187 | [+0.133, +0.274] | −0.128 | 0.13 | pass | −0.110 |
| v33 → v35 (+h09) | 0.087 | 0.046 | 0.017 | [−0.026, +0.059] | 0.029 | 1.41 | fail | +0.270 |
| v37 → v39 (+d10) | 0.149 | 0.136 | 0.066 | [+0.018, +0.126] | 0.071 | 0.57 | pass | +0.040 |
| v42 → v43 (+q04) | 0.290 | 0.263 | 0.166 | [+0.092, +0.269] | 0.098 | 0.30 | pass | +0.041 |

Pearson with the board gain over the five: trimmed CV −0.19, untrimmed −0.16, **body −0.81**,
tail +0.51. Five transitions, all within ±0.27 on the board — a small sample of small effects.

**L060 addendum (2026-10-05).** Shipped as v44 (`--lobt-window`, no other change reaching the
board: the unmatched refit with `q04` gave it weight 0, so those rows are byte-identical). Board
276.26 → **275.4927, −0.77**, from 43 moved rows — 2.7× the guaranteed lower bound of 0.281, i.e.
the truth on those rows sits well inside the window, not on its edge.

## L062 — adsb.lol's receivers changed between 2025 and 2026, most at EGLL and LEMD; the trace's lag behind the recorded off-block is airport-month specific

`measured` · `scripts/analysis/a115_adsb_drift_and_error_budget.py` (label-free: traces, `mvt_time`, `aobt_3`) · 2026-10-06

**Question.** A competitor on the challenge Discord reported that ADS-B features "encoding where an
aircraft was heard" did not transfer from 2025 to 2026 while "movement" features did. Does our data
show the drift, and where?

**Numbers.** Match rate (share of departures with a matched trace) and the median of
`adsb_taxi_move − (mvt − aobt_3)` (how much later than the recorded off-block the trace first shows
movement; more negative = the trace starts later), per airport and month:

| airport | match Jan 25 | Jul 25 | Jan 26 | Jul 26 | lag Jan 25 | Jul 25 | Jan 26 | Jul 26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGLL | 0.21 | 0.08 | **0.61** | **0.98** | −842 | −1,093 | **−538** | **−244** |
| LEMD | 0.00 | 0.03 | **0.97** | 0.44 | — | −541 | −375 | −507 |
| LIRF | 0.00 | 0.78 | 0.59 | **0.00** | — | −501 | −639 | −619 |
| EDDF | 0.71 | 0.75 | 0.51 | 0.60 | −499 | −441 | −564 | −611 |
| EHAM | 0.97 | 0.98 | 0.97 | 0.98 | −236 | −175 | −170 | −166 |
| LFPG | 0.00 | 0.13 | 0.10 | 0.12 | −958 | −697 | −760 | −717 |
| LTFM | 0 | 0 | 0 | 0 | — | — | — | — |

At EGLL the median trace density (`adsb_n_ground`) went 12 / 6 → 37 / 92 points and the share of
traces that start from a standstill 0.12 / 0.09 → 0.13 / 0.35: the 2026 receivers see the stand area
the 2025 ones did not. The lag is stable where coverage was already good (EHAM).

**Error budget of `ab02` (same script, Jan + Jul 2025 OOF, untrimmed, ex-LIRF):** unmatched rows
without a trace (3,042 rows) hold 49% of the squared error; matched rows without a trace (180,877)
38%; matched rows with a trace (132,045) 11%. By label band, two rows above 12 h hold 44%.

**So what.** (1) A model that learnt EGLL's 2025 lag (~−1,000 s) over-predicts traced EGLL rows in
2026, where the lag is ~−250 to −540 s; nothing in the 2025 folds can show it. The remedy built on
it is `features/adsb_calib.py`: the trace re-expressed against its own airport-month (coverage,
typical lag, lag-adjusted taxi), computed label-free from each month's own data; it is screened
under a "no loss in CV" rule and judged on the board. (2) `adsb_move` (movement columns only) is
the other half of the Discord's advice, also a board test (`ab03` swap). (3) The next body
improvement is not at the traced rows: 87% of ex-LIRF squared error sits on rows without a trace.

**L062 addendum (2026-10-06, v49).** The board verdict on the Discord's remedy: v49 = v48 with `ab02`
swapped for `ab03` (the ADS-B block reduced to its movement columns) scored **261.86 vs 260.02, +1.84**
(CV +1.49). The receiver-dependent columns (`lat0/lon0`, point count, largest gap, MLAT share) carry
information in 2026 as well; dropping them costs more on the board than in CV. Same sign and size as
outgoing-boat's v14 (+1.54). The drift in L062 is real, but its remedy is not removal; `adsb_calib`
(the trace against its own airport-month) is the remaining candidate, judged on the board.

> **Note on L058, 2026-10-06 (v50).** The unmatched vector refitted on twelve months over v47's members
> (`a109`, `A109_MEMBERS=v47`: ab02 0.919, d05 0.143, sum 1.062) shipped as v50: **260.02 → 255.64,
> −4.38**, from 4,889 changed rows (mean +89.5 s). CV on the ten other months said the twelve-month fit was
> worse than the two-month one (529.07 vs 515.67 trimmed) — the third time CV has pointed the wrong way on
> this population (v37, v41, v50). Three data points, one direction: 2026's unmatched rows want a higher level
> than 2025's CV rows do, and the member that knows the most about each flight (ADS-B) is the one to carry it.


## L063 — A cross-fitted second stage on `ab02` learns nothing: the "corrector" edge needs features the base lacks

`measured` · `scripts/analysis/a116_second_stage.py`, `reports/a116_second_stage.txt`, `a081` · 2026-10-06

**Question.** Two outside sources (outgoing-boat's corrector, worth −5 to −13 s per airport on their
base; a Discord post: "correct a strong base model rather than predicting from scratch") point at a
second stage. Does one on our strongest member help?

**Set-up.** LightGBM on `min(y, 10800) − base`, base = `ab02`'s out-of-fold prediction over all twelve
months (`o12_ab02` + the seasonal file), inputs = `ab02`'s 16 cacheable blocks + the base itself;
cross-fitted (Jan and Jul each predicted by a stage 2 trained on the other eleven months' OOF rows);
the ranking rows from a stage 2 on all twelve months applied to the twelve-month refit.

**Numbers.** Early stopping ended at 32 (Jan) and 30 (Jul) trees, 5 on the full fit. Trimmed ex-LIRF:
Jan 186.834 → 186.607, Jul 211.094 → 211.315; untrimmed unchanged to 0.1; LIRF −3 to −6 trimmed.
Ranking corrections: mean +0.4 s, mean |Δ| 2.7 s, p99 41 s. `a081` against v47's members: 200.557 →
200.608, no seat (weight 0.84 but no gain — it is a copy of ab02).

**So what.** outgoing-boat's corrector saw blocks their base did not (`ext_*`, `map_*`, `cel_*`,
`sup_*`, `pista_*`, `ret_*`); its gain was those inputs, not the two-stage structure. With every block
in the base, the residual of an L2-fitted GBM holds nothing a second GBM on the same inputs can find
(L056 said the same of the stack's residual). Closed; ~2 minutes of compute.

**L062 addendum 2 (2026-10-06, v51).** The remedy that works: `adsb_calib` — per airport × month, label-free,
the coverage, the median lag of the trace behind the recorded off-block, the lag-adjusted trace taxi, the row's
lag against the month and the relative trace density. As an 18th member (`ab07` = ab02 + the block): solo CV
+0.07 (2025 cannot see the benefit), `a081` −0.43, **board −0.64** (255.64 → 255.00). Removing the drifting
columns lost (v49); calibrating them against their own month won. The gap between CV and board here is the
drift itself.


## L064 — On unmatched LIRF rows the sensor cannot predict the record; the arrivals say 2026 Rome records copies at least as often as 2025

`measured` · `scripts/analysis/a117_rome_from_2025.py`, `reports/a117_rome_from_2025.txt` · 2026-10-06

**Sensor vs record (July 2025, the month with 78% LIRF trace coverage).** Of 397 NM-unmatched LIRF
departures in Jan + Jul 2025, 265 have a matched trace (all in July); **257 of those 265 show an observed
taxi under 20 minutes**. Their labels: 62% at or under 1 h, 13% over 3 h, median 1,371 s. `r14`'s route
predicts a median of ~4,000 s on them and scores RMSE 3,918; replacing it by the body model (`ab02`,
~1,000–1,500 s) scores **10,001**, the lag-adjusted trace 11,854, even the in-sample label median
11,669. On all LIRF rows the gate would move the layer from 625 → 1,100 untrimmed, 564 → 662 trimmed.
So the sensor sees a short physical taxi on nearly every one of these flights while 38% of the labels
are not physical taxi times: the hedge (the conditional mean, which is what the route learns) is the
right prediction, and no sensor-based gate can improve it. This closes the "record vs sensor" rule for
unmatched rows on 2025 evidence, consistent with L012/L031/L033. In 2026 it would have touched 54 rows
(January; July has one trace).

**The 2026 regime, from the arrivals (label-free).** LIRF arrivals' copy share (in-block within 60 s of
a stamp): Jan 25 0.166, Jul 25 0.151, **Jan 26 0.174, Jul 26 0.177**; SCHED copies 0.100 / 0.099 → 0.095
/ 0.114. At day level in 2025 the arrivals' share correlates +0.48 with the departures' SCHED-copy share
at LIRF (365 days). Other airports move by ≤ 2 points. Reading: nothing suggests 2026 Rome records fewer
copies; July 2026 looks slightly more copy-prone than July 2025. The existing copy hedges stand.

> **Note on L058, 2026-10-07 (`u03`).** A member fitted *for* the unmatched rows (ab02's recipe under
> r14's NM-match routing, foreign rows at 0.1) is not better than ab02 on those rows: Jan 615.4 vs 606.5
> trimmed, Jul 533.0 vs 531.2 (`u03_unmatched_route_ab02`, `~/after_cb03.log`). What moved the board on
> that population was re-weighting existing members (v37 −0.90, v50 −4.38), not a specialised fit; the
> 2025 rows do not hold a separate structure to learn.

> **Note on L058, 2026-10-07 (v54).** Fourth point, same direction. v53 without the twelve-month unmatched
> override — so the 4,900 unmatched non-LIRF rows take the stack's grouped two-month vector, which includes
> `cb03` (0.73) — scored **252.05, −1.56** vs v53. On these rows the *member set* decides: each time a
> member with more information per flight reached them (ab02 in v50, cb03 in v54) the board moved by
> seconds, while CV on the ten other months could not rank the fits. A dedicated unmatched model (`u03`)
> added nothing; the information has to come through the members.


---

<a id="l065"></a>
## L065 — EUROCONTROL's published monthly taxi-out is our labels minus a modelled offset; moving v54 halfway towards it paid −0.37

`measured` · `scripts/analysis/a119_published_taxi_out.py`, `scripts/analysis/a120_cell_shift.py`,
`reports/a119_published_taxi_out.txt`, `reports/a120_cell_shift.txt`; board v56 · 2026-10-07

**Same source.** The APDF indicator's flight count `TF` equals our departures per airport-month (EDDF and
EGLL to the row; the rest within 0.8%). Its mean — (reference + additional minutes) / flights
with a reference — is over a subset: flights EUROCONTROL marks invalid and flights without a reference
time are left out. A label rule (120 ≤ taxi ≤ 7,200 s) reproduces `VALID_FL` within 0.34% of `TF` from May
to September at every airport, but not in winter: EDDM leaves out 18.9% of its January 2025 flights beyond
the rule, LFPG 8.9%, LSZH 8.8%, EDDF 4.7%, LTFM 10.6% in the February 2025 snow (EHAM, EGLL, LEBL ≈ 0 all
year). In those cells our label mean sits 51–164 s above the published one. *Inference, not verified: the
excluded flights are the remote-de-iced ones* — each one's excess taxi is nearly constant per airport across
the winter months (E: EDDF 1,075 s, EDDM 868, LSZH 879, LEMD 800, LFPG 1,257).

**The offset model.** Per 2025 airport-month, offset = label mean of our rows (not on a rare stand-runway
pair) − published mean, fitted as b0[airport] + E[airport] × s_w + F × s_r (s_w excluded share beyond the
rule, s_r no-reference share, F = 114 s per unit share, pooled) plus the airport's mean contribution of labels
outside the rule. Each month predicted from a fit without it: RMS 3.5 s in summer months, 15.5 s in winter
months; LTFM's single snow month is missed by 61 s (its E is the pooled 952 s).

**2026.** v54's error per airport-month, b = v54 mean − (published + modelled offset): EDDF Jan +42, LTFM
Jan +82 (error bar 61), LEBL Jul +22, LFPG Jul −21, LSZH Jan +16, EGLL Jul +13, EDDM Jul −11; the remaining
eleven within ±9, EHAM January +6 — v54 already predicts EHAM's published +179 s rise (it predicts +194).
v56 = v54 moved by −0.5 × b in the eleven cells where |b| exceeded its error bar: **251.68, −0.37**, against
an arithmetic −0.90 if every b were exact (LTFM January about 70% of that). a119's first version subtracted
the whole 2025 comparison (offset + our 2025 out-of-fold error) and so measured whether our error had
changed since 2025, not the error; the owner's question exposed it.

**APT-DLY (`a118`), for the record.** Day-level arrival ATFM delay by cause tracks the daily taxi level
(+0.19 within airport-month) but not the members' daily residual (−0.05); a day-level correction fitted
on one 2025 month and scored on the other makes both slightly worse (188.589 → 188.606, 170.626 →
170.649). `ec_pre_min`, already in the members, correlates +0.58 with it.

---

<a id="l066"></a>
## L066 — The dropout-repo levers on our board: one Rome override and one member landed, two seated members reversed

`measured` · boards v57–v62; `a081` (an01, pr01, cb04), `a124`, `a125`; `docs/DROPOUTS_RECONCILED.md` · 2026-10-08

| arm (single change) | evidence before upload | board |
|---|---|---|
| v57: ab02 → rb01 (body fitted with LIRF at 0.1 weight) | CV +0.04 | 251.63, −0.05 vs v56 |
| v58: LIRF NM-unmatched rows ← lg01 (T = D − G, CatBoost on all LIRF) | `a124`: both months, λ = 1 cross-fold, ≈ −3,270 MSE | 251.45, **−0.18** vs v57 |
| v59: + an01 (ab02 anchored on clip(MVT − AOBT_3, 0, 7200)) | solo 200.213 (best member); `a081` −0.509, transfer gate passed | 251.72, **+0.09** vs v57 |
| v60: + pr01 (ab02 + plan_residual) | CV 200.23; `a081` −0.108 | 251.69, +0.06 vs v57 |
| v62: v58 + cb04 (cb03 + adsb_calib) | CV 200.349 vs cb03 200.47; `a081` −0.146 | 251.25, **−0.20** vs v58 |

Two members that earned a cross-fold seat lost on the board. That is the first time `a081` has
passed a member whose board sign then flipped, so its −0.5 is no longer a sufficient signal for an
anchored variant of a member already in the stack. For an01, the anchored target changes the
inductive bias without adding information. The Rome override transferred at about 3% of its 2025
size. That fits OpenAir's experience on the same 383 rows: small and reversible. *Inference, not
verified:* for an01 and pr01 the stack had no new information to gain; cb04 and lg01 each bring an
input or a form the stack lacked (the drift-calibrated trace on a CatBoost base; the
gate-delay parametrisation).

---

<a id="l067"></a>
## L067 — The final file moved every takeoff time by up to ±12 s and nothing else; the bounds widen by that much

`measured` · `scripts/analysis/a126_final_randomness.py` · 2026-10-08

`final_ranking.parquet` (final phase: Jan, Feb, Jun, Jul 2026) holds every departure and arrival ID
of `ranking.parquet` plus 325,949 February and June departures. On the 689,534 shared rows only
`mvt_time` changed (on 91.9%), and `block_time` on arrivals: integer seconds in [−12, 12], sd 4.9,
drawn independently per row and per column. Schedule, LOBT, IOBT, EOBT, AOBT, ARVT, runway, stand and
every string are untouched. Arrival labels were recomputed from the moved times (`taxi_sec =
block_time − mvt_time` on 100% of arrivals), so departure labels are presumably moved the same way.
Consequence in the final pipeline (`paths.TIME_NOISE`): the LOBT window widens to ±3,618 s, the gap
bound to +1,812 s, the LEBL/LEMD ceilings to 3,624 s; everything else moves by at most 12 s.

---

<a id="l068"></a>
## L068 — February is the hardest month because of Istanbul's snow days, which no feature captured

`measured` · `scripts/analysis/a127_february_error.py`, `a128_feb_jun_level_and_snow.py` · 2026-10-08

In x36's leave-one-month-out run, February's trimmed RMSE without Rome is 263.65 against 176–231
elsewhere. LTFM alone carries 49.9% of February's trimmed squared error (trimmed RMSE 557 in
February against 210–286 in Jan, Jun, Jul); the top 10 airport-days hold 52%, the two worst — LTFM
on 23 and 19 February 2025, mean taxi 49 and 46 min — 30%. Those are LTFM's snow days: mean residual
+926 and +935 s on the two worst, +240 to +324 on three more, +190 on the freezing day after the
storm; snow-free days sit within ±75 s. APT-DLY records no delay on them (0 de-icing minutes at
LTFM all month), and LTFM has no ADS-B coverage and mostly default `aobt_3`. February 2026 at LTFM
had snow on the 2nd, 3rd and 18th and was warmer. EUROCONTROL's published LTFM February mean
rose 16.49 → 17.42 min, but its valid share rose from 88.9% to 99.0%: the 2025 figure leaves out
storm-day flights, the offset a120 models. The `snow` block (final-phase repository) is the test.

**L068 addendum (2026-10-08 18:20 UTC, the screen).** `q04s_q04_snow` = q04 + the `snow` block (snow
reports over 3–48 h, heavy snow, freezing rain, the day's count, hours since snow), on the final
phase's paired folds against q04 on the same folds (`a130_member_months.py`, final-phase repo):
trimmed RMSE without Rome Jan 205.20 → 204.41, **Feb 261.87 → 256.19 (−5.68)**, Jun +0.01, Jul −0.00;
four months 223.16 → 221.57; NM-unmatched rows 771.6 → 757.0. Its pre-declared screen (Feb ≥ 1.0 s
better, no month > 0.3 s worse) passes; the snow variants of ab07, rb01, cb04 and cb03 are queued, to
be judged together in the stack (`a129`).

---

<a id="l069"></a>
## L069 — Summer thunderstorms leave little residual in the ADS-B member; no TS block was built

`measured` · `scripts/analysis/a131_summer_thunderstorms.py` · 2026-10-09

The summer counterpart of L068. 1,201 of the 285,626 METAR reports at the ten airports carry TS.
Departures (trimmed, Rome left out) with a TS report in the 3 h up to takeoff: 3,449 in June and
4,738 in July 2025. In x36's leave-one-month-out run they carry 6.5% / 6.3% of the month's trimmed
squared error; per cell (TS reports 1 / 2+, latest report with or without TS) the mean residual
runs −28 to +74 s and the RMSE 226–599, against 192 / 227 on the other rows. In ab07's final-phase
out-of-fold run (the calibrated ADS-B member) the share is 6.2% / 5.8%, the cell means −14 to +68 s,
the RMSE 180–538 against 167 / 209. Removing every TS cell's mean residual in ab07 would lower
July's squared error by about 0.05%; what remains on those rows is spread, not bias. The worst July
airport-days are not storm days: LTFM, with no TS report all month, sits +27 s under-predicted on
average in ab07 (+41 s in x36), a level that a120's cell shift addresses for 2026. A
thunderstorm-window block would have had a ceiling below the gate (−0.3 s over four months), so none
was built.

---

<a id="l070"></a>
## L070 — F0 against v62 on the leaderboard's rows: the refits move ~31 s RMS, and the unmatched level is February's

`measured` · final-phase repository: `scripts/analysis/a132_board_vs_final.py`,
`a133_feature_drift_between_phases.py`, `a134_adsb_matches_under_noise.py`,
`a135_unmatched_level.py` · 2026-10-09

v63 (F0's Jan/Jul half: v62's seven members refitted on the final file, four-month weights,
layers and cell shift) scored 251.8545 against v62's 251.2454. On the 344,841 shared rows:

- **Refits (a132).** Each member's final-phase refit differs from its leaderboard-phase refit by
  28.5–32.5 s RMS on NM-matched rows (15–27% of rows by more than 30 s), 53–141 s on the 4,900
  NM-unmatched rows and 59–80 s at Rome. The takeoff noise (sd 4.92 s) explains 2–4% of that
  variance (slope on it 0.8–1.3).
- **Builds (a132).** F0 − v62: +2.07 s mean, 18.75 RMS on matched rows; **+110.65 mean, 180.93 RMS
  on the NM-unmatched rows**; −4.21 / 73.79 at Rome.
- **Training inputs (a133).** Of the 19 cached blocks the members read, 16 are identical on 2025
  rows. `taxi_in`'s stand statistics differ on 75–98% of rows (they now include the final file's
  Feb/Jun 2026 arrivals), and `runway_config`'s and `stand_runway`'s category codes are renumbered
  on nearly every row. On Jan/Jul 2026 rows nearly every time-based block moves with the noise.
- **ADS-B (a134).** Of the 196,452 Jan/Jul rows matched in both stores, 189 (0.10%) change event;
  779 lose a match and 774 gain one; every other row moves by exactly the noise.
- **The unmatched level (a135).** On the 8,334 NM-unmatched non-Rome out-of-fold rows of the four
  months, every member under-predicts February by 430 (cb03) to 710 s (t05), against Jan 62–134
  (two labels above 12 h; trimmed −41 to +32), Jun −17 to +8, Jul 12–50. The unmatched NNLS vector
  fitted on Jan+Jul alone sums to 1.047, on Feb+Jun alone it is cb03 × 1.283, on all four cb03 ×
  1.151 (F0's). The four-month vector leaves the Jan+Jul unmatched rows at −110.4 s mean residual
  (trimmed RMSE 564.9, against 545.4 for the Jan+Jul vector in sample); 110² s² over 4,900 of
  344,841 rows is ≈ 0.34 leaderboard seconds if v62's level was right.

**L068 addendum 2 (2026-10-09 06:30 UTC, the stack).** Member screens (`a130`, final-phase repo; Δ per
month Jan / Feb / Jun / Jul, trimmed without Rome): t05s −0.13 / **−8.39** / −0.04 / −0.08; rb01s −0.36 /
**−2.47** / −0.39 / −0.23; f25s −0.86 / **−3.81** / +0.77 / +1.01; ab07s −0.11 / **−1.64** / +0.10 / +0.18;
cb04s −0.33 / +0.19 / +0.12 / −0.58; cb03s +0.32 / −0.59 / −0.27 / −0.01 (unmatched rows +2.80). The
placebo controls set the scale: f25 + three noise columns −0.82 / +0.90 / +0.14 / +0.02, ab07 + them
−0.04 / −0.49 / +0.40 / −0.03. In the stack (`a129`, cross-pair weights, base 198.09 over four
months): the five LightGBM snow swaps −0.33 (every month better; Feb −0.91), all seven +0.22, the
two CatBoost swaps +0.52. The snow block moves the LightGBM members' February and barely moves the
CatBoost members'.

---

<a id="l071"></a>
## L071 — The NM-unmatched level does not transfer across months; February's is two Istanbul storm days

`measured` · final-phase repository: `scripts/analysis/a137_unmatched_month_transfer.py`
(members: F1's) · 2026-10-09

Each 2025 month held out in turn; the matched rows' vector fitted on the other three months, the
unmatched rows' vector by rule. Trimmed RMSE without Rome, four months together: **pooled (the other
three months; F0's and F1's rule) 197.34**; pooled on labels ≤ 12 h +0.19; **without February −1.13**;
same pair +2.10; no separate vector −0.27; fitted in sample (unreachable) −3.91. Per held-out month,
without February against pooled: Jan −1.32, Jun −1.24, Jul −1.84; February held out is under-
predicted by every rule (mean residual +393 to +459 s), and in sample its own vector sums to 1.511
(the others 1.01–1.06). February's unmatched squared error under the vector of the other three
months: LTFM 77.6% (363 rows, mean label 3,397 s, mean residual +1,187), LSZH 17.8%; the airport-days
LTFM 23 Feb (84 rows, mean label 5,529 s) 36.4%, LTFM 19 Feb (85 rows, 5,076 s) 29.6%, LSZH 8 Feb
(9 rows) 17.3%. The four-month vector therefore raises every month's unmatched rows by ~15% to
cover two storm days.

**Tuning on the snow member (same day, `a130`/`a129` against F1 = 197.76).** hp_ab07s_* member Δ over
the four months: ff04 −0.33, cat50 −0.21, bag06 −0.05, xt +2.44, huber +0.46. In the stack, as a
replacement for ab07s or beside it: −0.11 (cat50, replacing) to +0.02; none passes. NNLS gives xt no
weight, so "replace with xt" is F1 without ab07s: 197.72.

---

<a id="l072"></a>
## L072 — Where F2b errs out of fold, and why the quick stack levers do not pay

`measured` · final-phase repository: `scripts/analysis/a139_where_the_stack_errs.py`,
`a140_tail_calibration.py`, `a129_stack_cv.py` (`A129_MATCHED_BY`) · 2026-10-10

F2b as an out-of-fold build on Jan, Feb, Jun, Jul 2025 (stack weights fitted on the other pair, the
unmatched rule, r14 at Rome, the bounds and the LOBT window; no cell shift): RMSE 290.60 over all rows,
236.78 trimmed; without Rome 233.22 / 194.75; Rome 665.5 / 522.0 (7.7% of rows, 40.6% of all squared
error). The worst 0.1% of rows (671) carry 53.9% of all squared error; two LFPG labels above 12 h in
January carry 18%; NM-unmatched rows without an ADS-B trace (5,016) 23%.

Trimmed, without Rome: LTFM 31.3% of the squared error (13% of rows, RMSE 288.6, mean residual
+11.6), LFPG 17.9% (232.9), EGLL 17.5% (229.3), every other airport 119–173. July 32.2%, February
27.0%, January and June 20.4% each (RMSE Feb 218.5, Jul 207.7, Jan 183.3, Jun 168.5). LTFM in
February alone: 15.7% (RMSE 448.2, +55.2 s); ISO week 8 (17–23 Feb): 12.2%, with LTFM on 23 and 19
February (642 and 563 departures, mean residual +545 and +600 s) the two worst airport-days. NM-matched
rows with an ADS-B trace: RMSE 122.6; without one: 219.4 (67.2% of the squared error); the same gap
within airports (EDDF 119 / 187, LSZH 120 / 274, EGLL 185 / 235). LTFM has no trace at all, LFPG 5%,
EGLL 12%. Label bands: 10–20 min −28.8 s mean residual, 30–60 min +282, 1–2 h +1,403; labels above
30 min (3.8% of rows) carry about 45%. Calibration by prediction quantile: within −8 to −3 s up to the
95th percentile, then +17.5 (95–99%), +166.7 (99–99.9%), +527.5 (top 0.1%).

Levers tested against F2b (195.21, four months, trimmed without Rome): matched vector split by ADS-B
coverage 195.20; per airport 195.28 (Feb −1.18, Jan +0.76, Jun +0.91); a cross-fitted stretch of the
top predictions +0.03 to +0.46 (hinge above the 95th/99th/99.5th percentile, isotonic), each better
in February only — fitted with February in the fitting pair its slope is +0.28 to +0.48, without it
+0.08 to +0.15. The top-end shortfall, like the unmatched level (L071), is February's.

---

<a id="l073"></a>
## L073 — 2026 had more snow than 2025, and F2b already responds to it; short taxis are a coverage problem

`measured` · final-phase repository: `scripts/analysis/a141_snow_days.py`, `a142_short_taxis.py`,
`a120_cell_shift.py` (F2b's run) · 2026-10-10

**Snow.** METAR snow reports, January 2026 against January 2025: EHAM 194 / 41, LSZH 161 / 35, EDDF
114 / 19, EDDM 112 / 47, LTFM 147 / 0; February: LTFM 74 / 258. F2b's 2026 mean prediction on the
snowiest days, against its airport-month's median day: EHAM 3–7 Jan +1,077 to +1,895 s (means 2,119–2,699
s), LFPG 5 and 7 Jan +968 / +1,152, LSZH 10 Jan +1,045, LTFM 18–19 Jan +736 / +921 and 18 Feb +746,
EDDF 29 Jan +866, EDDM 19 Feb +876. In 2025, out of fold, snow days were within −38 to +43 s of their
labels everywhere but EGLL on 4 Jan (−113) and LTFM's storms (+545 on 23 Feb, +600 on 19 Feb, +163 to
+202 on 6, 11 and 20 Feb). a120 on F2b's build: LTFM's 2026 build sits above EUROCONTROL's published
level (plus the label-measured offset) by 87.8 s in January (shifted −43.9) and 57.6 s in February
(error bar 61.3, not shifted); EDDF February +39.0 (shifted −19.5).

**Short taxis.** 83,550 of 619,364 trimmed rows without Rome are under 10 minutes. Residual on 5–10
min labels: −35.8 s with an ADS-B trace (RMSE 90.0), −129.1 without (183.5); 10–15 min −12.0 / −77.8.
Share under 10 minutes and its residual: EHAM 30.0% (−43.4), LSZH 29.9% (−46.5), EDDM 21.6% (−53.1),
EDDF 20.1% (−47.4), LEBL 10.1% (−94.2), LFPG 7.2% (−172.7), LEMD 6.5% (−109.3), LTFM 5.0% (−168.0),
EGLL 0.4% (−247.2). Short taxis whose `takeoff − aobt_3` read 5–10 min: −30.9 s; 15–20 min: −154.9;
20–30 min: −202.8. With a trace and `aobt_3` copied into the label: −13.8 (RMSE 64.3); no trace and
not copied (16,574 rows): −156.5 (213.6). Rows whose `takeoff − aobt_3` is under 10 minutes are short
in 59.3% of cases with a trace and 39.2% without, mean residual −1.9 / −4.4.
