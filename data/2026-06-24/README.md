# US HS-Code Classification Dataset

Generated: 2026-06-24  
Sources: CROSS binding rulings (`precedents.rulings WHERE source='cross'`) + HTSUS schedule (`htsus.nomenclature WHERE is_declarable`)

## Files

- `train.jsonl` — chat-format (system/user/assistant), instruction-tuning ready
- `eval.jsonl`  — chat-format, held-out split
- `flat.jsonl`  — legacy (prompt, completion, code, chapter, source) for SFT pipelines that don't take chat
- `stats.json`  — per-chapter and per-source counts

## Build parameters

- Per-chapter cap (CROSS only): 1000
- Eval fraction: 0.1
- RNG seed: 20260624

## Counts

| Source | Candidates loaded |
|---|---|
| cross | 187,567 |
| htsus_schedule | 19,670 |

- CROSS rows after per-chapter cap: **54,655**
- Total examples (train + eval): **74,325**
- Train: **66,892**
- Eval:  **7,433**

## Example

```json
{"messages":[
  {"role":"system","content":"You are an expert US customs classifier..."},
  {"role":"user","content":"Country: USA\nProduct: Men's cotton t-shirt, knitted, 95% cotton 5% spandex"},
  {"role":"assistant","content":"HTSUS Code: 6109.10.00.27\n\nReasoning: ..."}
]}
```

## Disclaimer

Decision support only. CROSS rulings are public CBP records; the HTSUS schedule is published by the USITC. Final classification authority rests with US Customs and Border Protection.