# data/

Training data lives here as dated snapshots, with `data/current/` as a
symlink to whichever one the SFT / predict / eval scripts should use.

```
data/
├── 2026-06-24/         <— this snapshot is committed (~100 MB JSONL)
│   ├── train.jsonl     chat-format SFT input (66,892 rows)
│   ├── eval.jsonl      chat-format held-out, stratified by chapter (7,433)
│   ├── flat.jsonl      legacy (prompt, completion, code, chapter, source) (74,325)
│   ├── stats.json      per-chapter and per-source counts
│   └── README.md       dataset card from the builder
└── current  ──►  2026-06-24/
```

## Switching snapshots

Drop a new snapshot folder in and repoint the symlink:

```bash
ln -snf 2026-09-15 data/current
```

The Makefile + sft_phase0 + predict scripts all reference
`data/current/...` so flipping the symlink rotates them all at once.

## Provenance

This snapshot was built by
[`shelley-data-ingest/us-ingest`](https://github.com/raahulrahl)
(sibling repo, not public) out of the live Postgres `bindu_db`:

- **HTSUS schedule** — 19,670 declarable 10-digit codes from
  `htsus.nomenclature` (sourced from `hts.usitc.gov`).
- **CROSS rulings** — 187,567 valid binding rulings from
  `precedents.rulings WHERE source='cross'`, scraped from
  `rulings.cbp.gov` (chapter-prefix search sweep across all 99 HS
  chapters).
- Capped at 1,000 CROSS rows per chapter to prevent apparel +
  electronics from dominating; HTSUS schedule rows guarantee every
  chapter is represented.
- Split 90/10 stratified by chapter — every chapter appears in both
  train and eval.

## To regenerate

You only need this if (a) CBP publishes new CROSS rulings, (b) USITC
publishes a new HTSUS revision, or (c) you want richer per-row
reasoning (run the optional `text` stage in us-ingest to fetch full
ruling bodies):

```bash
cd ../shelley-data-ingest/us-ingest
DATABASE_URL=... .venv/bin/python -m src.cross.run sweep
DATABASE_URL=... .venv/bin/python -m src.cross.run ingest
DATABASE_URL=... .venv/bin/python -m src.dataset.build

# Copy new snapshot into this repo + repoint symlink
cp -r ../../regulations/dataset/us/$(date +%F) ../../hs-code-llm/data/
ln -snf $(date +%F) ../../hs-code-llm/data/current
```
