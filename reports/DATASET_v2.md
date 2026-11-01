# GBEM-UA v2: dataset

Built by `python scripts/build_v2.py` (seed 42) from `data/raw/full_v1_{acc,prob}.csv`. All checks below are
asserted by the script; the machine-readable record is `data/v2/build_report.json`.

## Construction rules

1. **One block per profession.** A block is 8 rows: 2 experiences (relevant, irrelevant) × 2 genders × 2 job forms
   (base, feminitive). The raw file is split into 8-row chunks of consecutive rows of the same profession, and the
   **first complete block in file order** is kept. That is the first generation pass. The second pass is not
   used, because its positives duplicate the first pass for 341/349 professions.
2. **Dropped:** 350 second-pass blocks, 1 extra «урядовець» block in pass 1 (rows 2588–2595), and 2 incomplete
   4-row «амбасадор» chunks (rows 2088–2091 and 4892–4895).
3. **`strip()`** on `sentence`, `profession`, `experience` (v1 test had 1324 rows with a leading space in `experience`).
4. **`fem_identical`** marks professions whose feminitive equals the base form. Only «рибалка» qualifies. It stays
   in the data but is excluded from every comparison involving a feminitive (`metrics/scores.py`).
5. **Group split by profession:** `GroupShuffleSplit(train_size=175/349, random_state=42)`, which gives 175 train and
   174 test professions. The lists are in `data/v2/split_professions.json`. `full` contains all 349 professions
   and is used for zero-shot methods. `test` is a subset of `full`.
6. **New columns:** `row_id` (`profession|rel/irr|group`), `item_id` (`profession|rel/irr`, shared by the 4 gender
   variants), `job_form` (exact surface form of the profession in the sentence, used as the debiasing target),
   `group` (`man_nofem`, `woman_nofem`, `woman_fem`, `man_fem`), `fem_identical`, `split`, and `raw_row`
   (the row in the raw v1 file, for traceability).

Stratifying the split by `fem_identical` would be a no-op with a single such profession, so it is not done.
Stratifying by industry needs sector labels the data does not have.

## T1. v1 vs v2

| | v1 (UNLP 2025) | v2 |
|---|---|---|
| Professions | 349 (paper: 351) | 349 |
| Rows per profession | 16 (two passes; «урядовець» 32, «амбасадор» 24) | 8 |
| Rows train / test / full | 2804 / 2800 / – | 1400 / 1392 / 2792 |
| Professions train / test | 349 / 349 | 175 / 174 |
| Profession overlap train ∩ test | 349 | **0** |
| Test sentences found in train | **1368** (all positives) | **0** |
| Duplicate sentences within a split | 4 / 4 («рибалка») | 4 in train («рибалка», flagged `fem_identical`), 0 in test |
| Positives (relevant experience) | 1402 / 1400 | 700 / 696 / 1396 |
| Leading spaces in `experience` (test) | 1324 | 0 |
| acc and prob configs aligned | yes | yes (same `row_id` order; asserted) |
| Native-speaker validation (Cohen's κ) | – (paper: "manually reviewed") | **pending**, see below |

Identical numbers hold for the prob config. Its sentences use «Посаду X він BLANK отримав.»; `BLANK` becomes «» or «не ».

## Native-speaker validation (plan 1.3): ready, not done

`data/v2/annotation/annotation_sheet.csv` has one row per profession (349 rows). Each row holds both job forms, both
experiences, the 4 relevant-experience sentences, and empty columns for the criteria:
(a) relevant experience is relevant, (b) irrelevant experience is irrelevant, (c) the feminitive is correct
(gendergid.org.ua), (d) the four sentence types are grammatical. Two annotators fill copies independently.
`scripts/annotation_kappa.py <a.csv> <b.csv>` computes Cohen's κ per criterion and lists disagreements. Professions
with unfixable problems are then excluded and listed in the dataset card. Until this is done, papers and the
thesis must say that the data has not been validated by native speakers.

## Fine-tuning data

`python debias/finetune.py --build-data-only` writes `data/finetune/{train,val}.jsonl`. Train-split professions
only, with 10% of them (group split, seed 42) held out for checkpoint selection, which gives 942 / 108 examples
(157 / 18 professions). Groups man_nofem, woman_nofem and woman_fem are used, and both classes:
relevant → «Так», irrelevant → «Ні». No test profession appears (asserted in `tests/test_finetune.py`).
