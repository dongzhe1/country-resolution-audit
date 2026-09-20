# Country Resolution Audit

This project measures **which countries a global impact assessment is able to
report on individually**, and what is lost where it cannot. Assessments built on
general-equilibrium databases report results for regions, some of which are
single countries and some of which are groups; the choice of which is which is
made by the study, is rarely stated, and determines what its published results
can be read to say.

The toolkit does three things, and they are independent of each other:

1. **Counts** how many economies a study merged that its own database carries
   separately — a statistic that needs no model.
2. **Measures** an external, physically observed proxy for each country from AIS
   ship movements, so that group members can be compared without using the
   study's own output.
3. **Regroups the study's own published results** into the categories its group
   rows are named after, which tests whether those groups are informative using
   nothing but the study's numbers.

It was built around one assessment of a global shipping carbon price, but
nothing in `analysis/` is specific to that case: the inputs are a table of
reported regions, a set of country indicators, and optionally a measured
exposure column.

---

## System Requirements

### Software Dependencies

All Python package dependencies are listed in `requirements.txt`. Key libraries:

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |

> **Note:** Python 3.10 or above is required. The statistical routines are
> written out in full and use only `numpy` and the standard library — no scipy,
> no statsmodels, no R — so the dependency surface stays small and every
> estimator can be read.

### Tested Operating Systems

- macOS 26.2
- Ubuntu 22.04
- Windows 11

### Hardware

No non-standard hardware is required. Everything runs on CPU, and run time
depends on which of the two data sources in this repository you are running
it against:

- **Synthetic demo data** (`data/`, from `generate_fake_data.py`): the four
  pipeline steps take **two to three minutes** end to end, and the analysis
  scripts run in under a minute. Any laptop is enough.
- **Real data** (`results/`, already included) vs. **regenerating it
  yourself** from a full AIS pull: the analysis scripts still run in under a
  minute on the included `results/`, but rebuilding that from raw AIS —
  `pull_port_visits.py` through `state_exposure.py` on a full fleet, millions
  of port-call events — is a **multi-hour job**, dominated by GFW API
  throughput rather than CPU, and can stretch to a day or more depending on
  network speed and how many cores `--jobs` gets. It parallelises across
  ships, so it is a natural fit for a cluster or any multi-core machine you
  can leave running; there is nothing in the code that requires one.

---

## Installation

```bash
pip install -r requirements.txt
```

**Typical installation time:** under a minute on a standard desktop computer.

`searoute` is listed as an optional dependency. It is used only by
`pipeline/route_distances.py` to replace great-circle distance with a sea route;
without it the rest of the pipeline runs and falls back to great-circle
distance, saying so as it goes. If installing it fails on Windows with a
`UnicodeDecodeError`, that is `pip` reading one of its files under a non-UTF-8
system code page, not a problem with this repository; set the interpreter to
UTF-8 mode first and retry:

```powershell
$env:PYTHONUTF8="1"        # PowerShell
```
```cmd
set PYTHONUTF8=1           # cmd.exe
```
```bash
export PYTHONUTF8=1        # bash / WSL / macOS / Linux (rarely needed there)
```

If that is not worth doing, skip `searoute` — nothing downstream requires it.

---

## Demo

> **Note on input data:** the AIS extraction behind the published results is
> 7.9 million port-call events for 28,501 ships, retrieved under an API key, and
> the vessel register it joins to is licensed commercially. Neither can ship
> with this repository. `generate_fake_data.py` writes files with the same
> names, columns and dtypes so that the whole pipeline can be executed and
> inspected. **The synthetic values are random**: they exercise the code and
> reproduce no substantive result.

Run the four steps below in order. The whole demo takes about **two to three
minutes** on a normal desktop computer.

### Step 1 — Generate demo data

```bash
python generate_fake_data.py --out data --vessels 400 --shards 2
```

Creates, under `data/`:

```
port_visits_0000.csv.gz    AIS port-call events, sharded as pulled
port_visits_0001.csv.gz
ship_info.csv              vessel particulars
mrv/2022_renamed.csv       reported annual CO2 per ship, one file per year
mrv/2023_renamed.csv
mrv/2024_renamed.csv
```

Expected output:

```
wrote 10,848 port-call events for 400 vessels in 2 shard(s)
  data/port_visits_*.csv.gz
  data/ship_info.csv          400 vessels
  data/mrv/*.csv              368 ships x 3 years

These values are random. They run the code; they reproduce nothing.
```

### Step 2 — Build voyages

```bash
cd pipeline
python build_voyages.py ../data --jobs 4
```

A voyage is the sea passage from the end of one port visit to the start of the
next by the same ship. Four filters remove artefacts rather than passages: sea
time under an hour or over 120 days, straight-line distance under one nautical
mile, implied speed above 30 knots.

Expected output:

```
reading 2 shard(s) on 2 process(es)
  10,848 port visits, 400 vessels
building voyages on 4 process(es) (400 vessels)

voyages built: 9,933 (from 10,448 candidate legs)
  dropped         0  sea_hours_too_short
  dropped         0  sea_hours_too_long
  dropped       515  distance_too_short
  dropped         0  implied_speed_too_high

  vessels with >=1 voyage : 400
  distinct departure ports: 20
  distinct countries      : 20
  median distance (nm)    : 5,097
  median implied speed    : 12.0 kn

wrote ../data/voyages.csv.gz
```

### Step 3 — Estimate emissions and attribute them to states

```bash
python state_exposure.py ../data --voyages voyages.csv.gz
```

`state_exposure.py` looks for `ship_info.csv` next to the data it was given, so
no setup is needed here. If your register lives somewhere else, point at it
directly instead of relying on shell-specific environment-variable syntax:
`--seaweb /path/to/ship_info.csv`.

Main-engine load is the cube of voyage-average speed over service speed, clipped
to 5–90% of installed power; fuel is converted at 190 g/kWh and burned fuel at
3.114 t CO₂ per tonne. Each voyage is charged half to the state it departed and
half to the state it arrived in.

Expected output includes:

```
voyages: 9,933  vessels: 400
matched to particulars: 9,933/9,933 voyages (100.0%)
voyages with CO2 estimate: 9,933  total 21.4 Mt CO2

wrote ../data/state_exposure.csv   (20 states)
```

### Step 4 — Check the estimate against reported emissions

```bash
python validate_mrv.py ../data --voyages voyages.csv.gz
```

Compares the bottom-up estimate with reported annual emissions per ship-year.
On real data this is what the per-ship-type calibration is fitted from.

Expected output includes:

```
MRV total: 1,104 ship-years (0 duplicate rows dropped)

matched ship-years: 254   ships: 182

estimate / reported
  p50    0.23
  mean   0.31   geometric mean   0.21
```

**Expected total demo run time:** two to three minutes.

---

## Generating the key statistics

The demo above shows the pipeline running. This section produces the numbers
themselves.

`data/` and `results/` hold **the same kinds of file in the same formats**. The
difference is their provenance: `data/` is whatever `generate_fake_data.py`
invented, and `results/` is the derived tables from the real extraction, which
are included in this repository. Only the second yields substantive statistics.

```bash
cd analysis
python resolution_gap.py          ../results
python merged_economies.py        ../results
python resolution_model.py        ../results
python validate_firth.py          ../results
python aggregate_rows.py          ../results
python placebo_resolved.py        ../results
python disbursement_resolution.py ../results
python exposure_denominators.py   ../results
python exposure_vs_model.py       ../results
python ais_visibility.py          ../results
python temporal_coverage_tests.py ../results
python decision_consequence.py    ../results
```

**Expected run time:** under a minute for all eleven.

Each writes a `facts_*.json` beside its output. Those files are the interface:
every quantity reported in the write-up is generated from them, never typed by
hand, so a number that changes here changes there and a number that cannot be
produced cannot be printed.

### What each script establishes

| Script | Statistic |
|---|---|
| `merged_economies.py` | How many economies the study merged that its database carries separately, and how many of those are least developed or small island states. Needs no model. |
| `resolution_model.py` | Firth penalised logistic regression for which states receive a region of their own, with penalised profile intervals and likelihood-ratio tests. |
| `validate_firth.py` | Checks that estimator against an independent optimiser — Nelder–Mead on the penalised log-likelihood written from its definition. |
| `aggregate_rows.py` | Reconstructs group membership under the constraint that aggregation can only combine the database's regions, never split them, and enumerates every structurally possible mapping. |
| `placebo_resolved.py` | Sorts the economies the study *did* resolve into the categories its group rows are named after, and compares within-group spread with between-group spread. Uses no external measurement. |
| `disbursement_resolution.py` | The same comparison weighted by output and taken over every scenario and every admissible assignment, together with the modelled redistribution outcome by reported row. Uses no external measurement. |
| `exposure_denominators.py` | The resolved/unresolved intensity contrast under four denominators, with sensitivity to the vintage of the output figures. |
| `exposure_vs_model.py` | How much of the study's own country-level result is recoverable from the external measure, by paired bootstrap over countries. |
| `ais_visibility.py` | Equivalence tests against an independent yardstick, to test whether the record sees some states less well than others. |
| `decision_consequence.py` | Holds the measure fixed and moves only the resolution, showing that a group reading selects a different set of states. |
| `temporal_coverage_tests.py` | Whether the contrast is stable across periods and how complete the record is per state. |

### Reading a result

```bash
python -c "import json; print(json.load(open('../results/facts_merged.json')))"
```

`model_constants.py` imports the emission model's constants from `pipeline/`
rather than restating them, and runs anywhere.

---

## Getting real data

Unlike the demo above, this is a multi-hour undertaking dominated by API
throughput and network speed, not CPU — see [Hardware](#hardware).

```bash
python download_data.py --out data
```

Fetches the public inputs and prints where the rest comes from.

| Input | Availability |
|---|---|
| AIS port-call events | Global Fishing Watch API, free, registration required. Use `pipeline/pull_port_visits.py` with `GFW_TOKEN` set. A full fleet pull is millions of events and costs two to three days of free-tier quota. |
| Reported emissions | Public, from the EU MRV register (THETIS-MRV). Save the annual CSVs under `data/mrv/`. |
| Country indicators | Public, World Bank WDI. Already in `reference/` at the vintage used. |
| Country and status lists | Public, UN M49. Already in `reference/country_status.csv`. |
| Database region lists | Public GTAP documentation. Already in `reference/gtap11_regions.csv` and `gtap11_composite_members.csv`. |
| Vessel particulars | **Commercial licence; cannot be redistributed.** Schema below. |
| The impact assessment itself | Public IMO documents, not redistributed here. See below. |

### The assessment documents

Two IMO documents are the subject of the audit. They are public but large, and
are not carried here. Everything derived from them is already in `reference/`
and `results/`, so nothing in `analysis/` needs them; they are listed so the
derived tables can be checked against the source.

| Document | Used for |
|---|---|
| `MEPC 82/INF.8/Add.2` — Task 3, impacts on States | the 111 reported rows, their simulated GDP effects, the revenue-disbursement scenarios, and the reviewer exchange quoted in the paper |
| `MEPC 82/7/4` — report of the Steering Committee | the modelling limits the Committee recorded |

Three counts the paper quotes are taken from Add.2 directly and are not
reproduced by anything here, because they are properties of the document rather
than of the data: it runs to 305 pages; 14 of the states inside its group rows
were searched for by name; none of the 14 appears. Any text search of the
converted document repeats them.

Both are on the IMO's public document site under those numbers. Convert the PDF
to text before use:

```bash
mkdir -p docs/sources
pdftotext -layout MEPC82-INF8-Add2.pdf docs/sources/MEPC82-INF8-Add2.txt
```

`extract_gdp_impact.py` rebuilds the GDP table from that text:

```bash
python extract_gdp_impact.py docs/sources/MEPC82-INF8-Add2.txt reference/
```

It writes `reference/gtap_gdp_impact.csv` with one row per economy per scenario
and columns `scenario, gtap_region, resolution, disbursement, gdp_2030,
gdp_2040, gdp_2050`, and exits non-zero if any scenario does not come back with
the expected 111 rows. The copy already in `reference/` was produced this way,
so a rebuild should reproduce it byte for byte.

### Vessel register schema

`pipeline/` expects a CSV with these columns. Any register carrying the same
fields will work.

| Column | Meaning |
|---|---|
| `lrnoimo_ship_no` | IMO number, the join key |
| `gross_tonnage` | GT |
| `deadweight` | DWT |
| `shiptype_group` | type label used for the per-type calibration |
| `total_kilowattsof_main_engines` | installed main-engine power, kW |
| `speedservice` | service speed, knots |
| `year_of_build` | build year |
| `flag_name` | flag state |

---

## Reference tables

`reference/` holds the published lookups, frozen at the vintage used, so the
statistics do not move when an upstream source is revised.

| File | Source |
|---|---|
| `country_status.csv` | UN M49: sovereign states, least developed / small island / landlocked flags |
| `country_regions.csv` | UN M49 regions, sub-regions, intermediate regions |
| `country_indicators.csv`, `country_indicators_wdi.csv` | World Bank WDI: GDP, population |
| `gdp_ppp.csv` | World Bank WDI: GDP at purchasing power parity |
| `gtap_regions_mepc82.csv` | the assessment's reported regions |
| `gtap_gdp_impact.csv` | the assessment's published GDP effects for 2030, 2040 and 2050: ten scenarios without revenue disbursement, and the four of those that also model disbursement under three eligibility schemes. Rebuilt by `extract_gdp_impact.py` |
| `gtap11_regions.csv` | GTAP 11: 141 individual economies, 19 composite regions |
| `gtap11_composite_members.csv` | which countries each composite region comprises |
| `port_throughput_teu.csv` | World Bank container port throughput |
| `transit_decisions.csv` | recorded judgements on ambiguous waiting anchorages |

`country_indicators.csv`, `country_indicators_wdi.csv`, `gdp_ppp.csv` and
`port_throughput_teu.csv` carry World Bank World Development Indicators
content, © World Bank, licensed CC BY 4.0 — redistribution in modified form
is permitted with attribution, which this line provides.

`country_status.csv` and `country_regions.csv` reproduce the UN M49
classification, used here as a factual statistical standard rather than as
licensed content; no restrictive terms were found attached to it.

`transit_decisions.csv` is the only file here containing a human judgement
rather than a published figure. A handful of anchorages could not be classified
on evidence, so each was decided by hand and the reason written down. The
pipeline fails loudly if it meets an unrecorded case, because a silent default
is how an undocumented judgement ends up in a published ranking.

---

## Repository structure

```
generate_fake_data.py       synthetic inputs with the real schema
download_data.py            public inputs; pointers for the rest
requirements.txt

pipeline/                   needs AIS events and a vessel register
  pull_port_visits.py       API extraction, vessel by vessel
  build_voyages.py          port visits -> voyages
  route_distances.py        sea-route distance per port pair
  transit_candidates.py     find anchorages that may be waiting areas
  transit_evidence.py       classify them on visit-sequence evidence
  finalise_transit_list.py  combine with the recorded human decisions
  state_exposure.py         emissions per voyage, attributed to both ends
  state_temporal_coverage.py  exposure by year; per-state completeness
  validate_mrv.py           compare with reported emissions
  calibrate_types.py        per-ship-type correction, fitted then held out
  analyze.py                sensitivity grid over model assumptions
  compare_transit.py        what changes between two voyage tables
  diagnose_*.py             targeted checks on distance, anchorages, endpoints
  profile_sparse.py         where coverage is thin
  export_derived_tables.py      bundle the derived tables with a manifest

analysis/                   needs only results/
  sample.py                 who is in the sample; one rule, used everywhere
  facts.py                  write a statistic as JSON; refuses NaN and infinity
  resolution_gap.py         which states received a region of their own
  resolution_model.py       Firth logistic regression on resolution
  validate_firth.py         cross-check that estimator independently
  merged_economies.py       which economies the study merged
  aggregate_rows.py         reconstruct group membership under the constraint
  placebo_resolved.py       the same question, in the study's own output
  disbursement_resolution.py  within against between, over all scenarios
  exposure_denominators.py  the contrast under four denominators
  exposure_vs_model.py      external measure against the study's own effects
  ais_visibility.py         equivalence tests on record completeness
  temporal_coverage_tests.py  stability over time
  decision_consequence.py   what resolution costs a selection rule
  development_gradient.py   exposure across income groups
  disbursement.py           how a group figure spreads over its members
  model_constants.py        the emission model's constants, imported not retyped

reference/                  published lookup tables
results/                    derived tables from the real extraction
```

`results/` holds three kinds of file: the derived tables the analysis reads, the
outputs it writes back including every `facts_*.json`, and a few tables only the
pipeline stage can produce but which the write-up cites. Plotting intermediates
are not included.

---

## Notes on the code

**No reported number is typed.** Every quantity in the write-up is emitted by
these scripts into a macro file and read from there.

**`facts.py` refuses to record NaN or infinity.** A silent NaN reaching a
document is indistinguishable from a real value at the point a reader sees it.

**`sample.py` is the only definition of who is in the sample.** It follows the
UN M49 list: an entity the list carries within another State is outside the
sample, and so is an entity the list does not carry separately. The rule is
applied uniformly and its output names the reason for each exclusion. Deferring
to a published statistical list settles membership without the authors
classifying anyone.

**The statistical routines are written out rather than imported.** Firth's
penalised likelihood, the profile intervals, Welch tests with Satterthwaite
degrees of freedom, and the incomplete beta function behind the t distribution
are all in the source. `validate_firth.py` maximises the same objective by an
independent method and compares, which catches an estimator that converges to
the wrong place. It does not establish that the objective is Firth's; for that,
run R's `logistf` on the same design.

---

## Licence

Code: MIT (see `LICENSE`).
Derived data in `results/` and `reference/`: CC BY-NC 4.0.

Vessel particulars are licensed from a commercial provider and are not included.
Every quantity derived from them is.

This work uses data from Global Fishing Watch, Copyright 2026, Global Fishing
Watch, Inc.
