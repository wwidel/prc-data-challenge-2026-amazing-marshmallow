# Taxi-out prediction — PRC Data Challenge 2026, team amazing-marshmallow
#
# `make help` lists every target. REPRODUCE.md walks through them in order, with timings.
# Recipes use tabs. uv may live in ~/.local/bin, which is not on a fresh shell's PATH, so every
# recipe goes through $(UV) rather than assuming it.

UV  := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
RUN := $(UV) run

#: `board` (the leaderboard set, Jan + Jul 2026) or `final` (the final phase: Jan, Feb, Jun and Jul
#: 2026). The phase picks the input files and gives every 2026 output its own directory.
PHASE ?= board
export TAXIOUT_PHASE := $(PHASE)
ifeq ($(PHASE),final)
FOLDS     ?= final_pairs
LAYER_DIR := data/overrides_final
CELLSHIFT := $(LAYER_DIR)/cellshift.csv
OUT       := data/output/final
else
FOLDS     ?=
LAYER_DIR := data/external
CELLSHIFT := config/cellshift_v54.csv
OUT       := data/output/v62
endif

#: The seven members of v62's stack, then the Rome layer (r14) and the Rome-unmatched model (lg01).
MEMBERS := t05_huber_alpha f25_queue2 q04_c12_min_leaf30 ab07_ab02_adsb_calib \
           rb01_lirf_route_ab02 cb03_k20_combo_adsb cb04_cb03_adsb_calib
LAYERS  := r14_rules_plus_unmatched_route lg01_lirf_gate_delay
REFITS  := "t05_huber_alpha[alpha=600]" f25_queue2 q04_c12_min_leaf30 ab07_ab02_adsb_calib \
           rb01_lirf_route_ab02 cb03_k20_combo_adsb cb04_cb03_adsb_calib $(LAYERS)

#: v62's override files, in the order they are applied (later files win).
OVERRIDES := $(LAYER_DIR)/rome_matched_blend.parquet $(LAYER_DIR)/pop27_gap.parquet \
             $(LAYER_DIR)/onerow_aobt.parquet $(LAYER_DIR)/lirf_um_lg01.parquet
STACK_FLAGS := --rome-from r14_rules_plus_unmatched_route --weight-groups unmatched \
               $(foreach f,$(OVERRIDES),--overrides $(f)) --lobt-window

.DEFAULT_GOAL := help
.PHONY: help setup external adsb spine features cv refit overrides stack stack-noshift cellshift \
        verify lite test lint fmt check

help:  ## show this list
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

## ---- environment ------------------------------------------------------------------------

setup:  ## install the exact dependency versions from uv.lock
	$(UV) sync

## ---- data (the competition files go in data/raw/ by hand; see REPRODUCE.md) -------------

external:  ## pull the small open data: METAR, stands and runways (Gateway, OSM), EUROCONTROL, OPDI
	$(RUN) python scripts/pull_metar.py
	$(RUN) python scripts/pull_gateway.py
	$(RUN) python scripts/pull_gateway.py --runways
	$(RUN) python scripts/pull_osm_stands.py
	$(RUN) python scripts/pull_eurocontrol.py
	$(RUN) python scripts/pull_opdi.py

adsb:  ## pull and cut the adsb.lol traces for every day used (~31 h, ~7 GB kept; resumable)
	$(RUN) python scripts/pull_adsb.py $(foreach m,01 02 03 04 05 06 07 08 09 10 11 12,--month 2025-$(m)) \
		$(foreach m,01 02 06 07,--month 2026-$(m))

## ---- pipeline ---------------------------------------------------------------------------

spine:  ## build the row spine every feature block is aligned to (~1 min)
	$(RUN) python scripts/build_spine.py

features:  ## build every cacheable feature block (target-dependent ones are built per fold)
	$(RUN) python scripts/build_features.py

cv:  ## cross-validate the nine configs (board: Jan, Jul 2025; final: the two month pairs)
	$(RUN) python scripts/run_experiments.py --build $(if $(FOLDS),--folds $(FOLDS)) \
		$(foreach e,$(MEMBERS) $(LAYERS),config/experiments/$(e).yaml)

refit:  ## refit the nine on all of 2025 and predict the phase's 2026 rows
	$(RUN) python scripts/refit_predict.py $(REFITS)

overrides:  ## write the four override files into $(LAYER_DIR)/ (order matters: a125 reads a050's)
	$(RUN) python scripts/analysis/a050_pop27_values.py
	$(RUN) python scripts/analysis/a060_aobt_identity_extreme.py
	$(RUN) python scripts/analysis/a111_rome_matched_blend.py
	$(RUN) python scripts/analysis/a125_lirf_um_override.py

stack:  ## build the prediction file -> $(OUT).parquet (board: v62; final: the final file)
	$(RUN) python scripts/build_stack.py $(STACK_FLAGS) --cell-shift $(CELLSHIFT) --out $(OUT).parquet

stack-noshift:  ## the same without the cell shift -> $(OUT)_noshift.parquet (input to a120)
	$(RUN) python scripts/build_stack.py $(STACK_FLAGS) --out $(OUT)_noshift.parquet

cellshift:  ## final phase: a120's per airport-month shifts on $(OUT)_noshift.parquet -> $(CELLSHIFT)
	A120_PRED=$(OUT)_noshift.parquet $(RUN) python scripts/analysis/a120_cell_shift.py

verify:  ## independent checks of $(OUT).parquet (ids, bounds, every override landed)
	$(RUN) python scripts/verify_stack.py $(OUT).parquet $(foreach f,$(OVERRIDES),--overrides $(f))

lite:  ## the light tier: ~30 least-squares numbers + the rules (needs data/raw and the spine)
	$(RUN) python scripts/lite.py --out data/output/lite.parquet

## ---- quality ----------------------------------------------------------------------------

test:  ## run the test suite (data-gated tests skip without data/raw and a spine)
	$(RUN) pytest -q

lint:  ## check formatting and lint rules
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt:  ## apply formatting and safe lint fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

check: lint test  ## lint and test
