# Trade Compliance HS Code Classifier — Project Plan

> A reinforcement-learning fine-tuned LLM for predicting Harmonized System (HS) codes from product descriptions, scoped initially to USA (HTSUS) with a path to multi-country support.

---

## 1. Project Goal

Build a domain-expert LLM that:

1. Takes a **product description** in natural language.
2. Predicts the correct **HS / HTS code** (e.g. `6109.10.0027`).
3. Provides **legal reasoning** grounded in HTS chapter notes / CROSS rulings.
4. Works across multiple jurisdictions (USA first, EU/UK/China next) via swappable country adapters.

**Use case:** decision support for customs brokers, importers, and compliance teams — *not* a replacement for human review.

---

## 2. The Big Picture: Architecture

We use a **shared base model + country-specific LoRA adapters** pattern:

```
              ┌────────────────────────────────────┐
              │   BASE MODEL (international)       │
              │   Knows WCO 6-digit logic          │
              │   "What chapter & heading is this?"│
              │   Trained ONCE, used by all        │
              └────────────────────────────────────┘
                              │
        ┌───────────────┬─────┴─────┬───────────────┐
        ▼               ▼           ▼               ▼
   ┌────────┐      ┌────────┐  ┌────────┐      ┌────────┐
   │ 🇺🇸 USA │      │ 🇪🇺 EU  │  │ 🇬🇧 UK  │      │ 🇨🇳 China│
   │ 10 dig │      │ 8/10dig│  │ 10 dig │      │ 10 dig │
   │ adapter│      │ adapter│  │ adapter│      │ adapter│
   └────────┘      └────────┘  └────────┘      └────────┘
```

**Why this design:**

| Benefit | Impact |
|---|---|
| Train international logic **once** | 90% of difficulty is shared — don't repeat it per country |
| Country adapters are **cheap** (~$20 each) | Adding a new country is almost free |
| **Modular deployment** | Ship base + only the adapters customers need |
| **Easy maintenance** | WCO updates → retrain base. Country tariff change → retrain just that adapter |

---

## 3. Background: How HS Codes Work Globally

### Structure
- **WCO Harmonized System** = international standard, **6 digits**, used by 200+ countries.
- Each country extends with **national digits** (statistical suffixes).

### Code anatomy
```
8471.30.0100
└┬┘ └┬┘ └┬┘ └┘
 │   │   │   └─ National (US-specific) — 10 digits
 │   │   └───── Subheading — 8 digits
 │   └───────── Heading — 6 digits (international)
 └───────────── Chapter — 2 digits
```

### Country digit counts

| Country | System | Digits |
|---|---|---|
| 🇺🇸 USA | HTSUS | 10 |
| 🇪🇺 EU | CN / TARIC | 8 / 10 |
| 🇬🇧 UK | UK Global Tariff | 10 |
| 🇨🇳 China | China HS | 10–13 |
| 🇯🇵 Japan | Japan HS | 9 |
| 🇨🇦 Canada | Canadian Customs Tariff | 10 |
| 🇮🇳 India | ITC(HS) | 8 |

### Important reality
Even when two countries have the same 6-digit code, **duty rates and trade rules differ**. Same code ≠ same regulation.

---

## 4. Data Sources (All Free, All Public)

### 🇺🇸 USA
| Source | Description | URL |
|---|---|---|
| HTSUS | Full US tariff schedule (CSV/JSON downloads) | hts.usitc.gov |
| CROSS | ~200K+ binding rulings with reasoning | rulings.cbp.gov |

### 🇪🇺 EU
| Source | Description |
|---|---|
| TARIC | EU tariff database with duties + measures (public API) |
| EBTI | Binding Tariff Information rulings |
| CN nomenclature | 8-digit Combined Nomenclature |

### 🌐 International / Others
| Source | Description |
|---|---|
| WCO HS Nomenclature | 6-digit international standard |
| UK Trade Tariff API | gov.uk/trade-tariff |
| WITS (World Bank) | Tariff data for 200+ countries |

**Recommended starting point:** scrape CROSS — it has the most data with full reasoning text.

---

## 5. The Phased Roadmap

### Phase 0 — Smoke Test (free)
**Goal:** prove the workflow works end-to-end. Catch bugs cheaply.

| Item | Detail |
|---|---|
| Model | Qwen 2.5 **0.5B** |
| Data | ~500–1,000 CROSS examples |
| Hardware | Kaggle / Google Colab (free T4 GPU) |
| Time | 1–2 hours |
| Cost | **$0** |
| Expected accuracy | 20–40% (sanity check only) |

**Success criterion:** the training loop runs, model produces correctly-formatted output, you can debug your data pipeline.

---

### Phase A — Build the International Base Model

**Goal:** a model that's accurate at the universal 6-digit WCO level.

#### A1. Data preparation
- Collect ~50K (product description → 6-digit code) pairs from CROSS, EBTI, and WCO docs.
- Normalize codes to 6 digits.
- Format as instruction-tuning pairs with reasoning.

#### A2. SFT (Supervised Fine-Tuning)
| Item | Detail |
|---|---|
| Base model | Qwen 2.5 **7B** (or Llama 3.1 8B) |
| Hardware | 1× A100 80GB on RunPod |
| Time | 6–12 hours |
| Cost | **~$15** |
| Expected accuracy | 60–75% chapter-level |

#### A3. RL with PRIME-RL
| Item | Detail |
|---|---|
| Framework | PRIME-RL + verifiers library |
| Algorithm | GRPO |
| Hardware | 1–2× A100 80GB |
| Time | 12–24 hours |
| Cost | **~$40** |
| Expected accuracy | 80–90% at 6-digit level |

#### A4. Evaluation
- Held-out set of ~500 hand-checked international classifications.
- Measure accuracy at each level (chapter / heading / subheading).

**Phase A total: ~$55**

---

### Phase B — Country-Specific Adapters

**Goal:** per-country LoRA adapters that extend the base to national digit codes.

**Repeat this for each country (start with USA):**

#### B1. Country data
- Scrape country-specific rulings.
- Format with country flag in prompt:
  ```
  Country: USA
  Product: Cotton T-shirts, men's, knitted, 95% cotton 5% spandex
  → 6109.10.0027
  ```

#### B2. SFT the adapter
| Item | Detail |
|---|---|
| Approach | LoRA on top of frozen base |
| Adapter size | ~50–200MB |
| Hardware | 1× A100 |
| Time | 2–4 hours |
| Cost | **~$5** |

#### B3. RL the adapter
| Item | Detail |
|---|---|
| Reward | Full 10-digit (or country-specific) match scoring |
| Hardware | 1× A100 |
| Time | 4–8 hours |
| Cost | **~$15** |

**Per-country total: ~$20**

---

## 6. The Reward Function (Heart of RL)

For HS codes, the reward function uses **hierarchical partial credit**:

| Match level | Example | Reward |
|---|---|---|
| Full 10-digit match | `8471.30.0100` = `8471.30.0100` | **+1.0** |
| 8-digit match | predicted `8471.30.01XX` | **+0.7** |
| 6-digit match | predicted `8471.30.XXXX` | **+0.5** |
| 4-digit match | predicted `8471.XX.XXXX` | **+0.3** |
| 2-digit match (chapter only) | predicted `84XX.XX.XXXX` | **+0.1** |
| Wrong format / garbage | "I think it's a computer" | **−0.1** |

### Bonus signals
| Reward | Purpose |
|---|---|
| **+0.2** for citing a CROSS ruling number | Grounding in real sources |
| **+0.1** for using HTS chapter notes in reasoning | Legal-style reasoning |
| **−0.3** for "I'm not sure" with no attempt | Discourages refusal-as-safety |
| **−0.5** for hallucinated chapter (e.g. Chapter 99) | Catches made-up codes |

### Why this matters
- Naive 0/1 reward gives no gradient — model can't learn to improve incrementally.
- Hierarchical reward lets model climb: wrong chapter → right chapter → right heading → right code.

---

## 7. How One RL Training Step Works

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Pick a prompt                                       │
│  "Cotton T-shirt, men's, knitted, 95% cotton 5% spandex"     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Model generates 8 different answers (rollouts)      │
│  A: "6109.10.0027" ✓                                         │
│  B: "6109.10.0040" (close — same heading)                    │
│  C: "6110.20.2000" (wrong heading, right chapter)            │
│  D: "6109.10.0027" ✓                                         │
│  E: "0207.14.0020" (very wrong)                              │
│  F: "6109.10.0011" (close)                                   │
│  G: "I think this is apparel..." (incomplete)                │
│  H: "6109.10.0027" ✓                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Reward function scores each                         │
│  A: +1.0    B: +0.7    C: +0.1    D: +1.0                   │
│  E: 0.0     F: +0.7    G: -0.1    H: +1.0                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Algorithm (GRPO) nudges the model                   │
│  "Do more of what A, D, H did. Do less of what E, G did."    │
│  Tiny weight updates flow into the model.                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    Repeat 10,000+ times
```

---

## 8. Total Cost Estimate

### For a 4-country production model

| Component | Cost |
|---|---|
| Phase 0 (smoke test) | $0 |
| Phase A (international base) | $55 |
| Phase B × 4 countries (USA, EU, UK, China) | $80 |
| **Total** | **~$135** |
| Iteration buffer (3–5 reruns) | +$200 |
| **Realistic project budget** | **~$300–$400** |

### GPU rental options (2026 prices)

| Provider | GPU | $/hr |
|---|---|---|
| Kaggle / Colab | T4 | **Free** (30 hrs/week) |
| Vast.ai | RTX 4090 | $0.09–$0.59 |
| RunPod | RTX 4090 | $0.34–$0.69 |
| RunPod | A100 40GB | ~$0.60 |
| RunPod | A100 80GB | ~$0.79 |
| Lambda Labs | A100 | $1.29 |
| Lambda Labs | H100 | $2.49 |

**Recommendation:** Kaggle for Phase 0, RunPod (A100 80GB) for everything else.

---

## 9. Common Pitfalls to Avoid

| Pitfall | What happens | How to avoid |
|---|---|---|
| **Reward hacking** | Model gets high reward via nonsense | Unit-test reward function on edge cases before RL |
| **Mode collapse** | Model predicts same code for everything | Add KL penalty; ensure diverse training data |
| **Forgetting reasoning** | Accuracy up, explanations become garbage | Add small reward for well-structured reasoning |
| **Reward bugs** | Weird loss curves; model gets worse | Test reward function on 50 hand-picked examples first |
| **Skipping SFT** | RL doesn't improve much | Ensure SFT baseline ≥60% before RL |
| **Multi-country too early** | Can't debug which layer is wrong | Build one country end-to-end before adding more |

---

## 10. Pre-Flight Checklist Before Phase A RL

- [ ] Phase 0 ran successfully on Kaggle (smoke test passed)
- [ ] Phase A SFT model achieves ≥60% chapter-level accuracy
- [ ] You have ≥10,000 product descriptions for RL prompts
- [ ] Reward function written and **unit-tested** on 50 examples
- [ ] Eval set of ~500 hand-checked examples (kept separate)
- [ ] Budget confirmed (~$50 minimum for a real RL run)
- [ ] PRIME-RL `verifiers` quickstart read at least once

---

## 11. Production / Deployment Considerations

### Legal / compliance
- **Always include a disclaimer:** "Decision support only. Not official classification advice."
- **Require human review** for any binding classification decision.
- **Cite sources** in every answer (CROSS ruling # or HTS chapter note).
- **Log everything** — audit trail is critical for regulated use.

### Versioning
- Track base model version (e.g. `base-v1.0-wco2022`).
- Track adapter versions per country (e.g. `usa-adapter-v1.2`).
- When WCO releases new revision, all adapters need re-validation.

### Updating
- HS revisions every 5 years (WCO).
- US HTSUS updates ~annually.
- EU CN updates annually (Jan 1).
- Plan for annual retraining of adapters.

---

## 12. Tech Stack Summary

| Layer | Tool |
|---|---|
| Base model | Qwen 2.5 7B / Llama 3.1 8B (Hugging Face) |
| SFT framework | Hugging Face TRL or Unsloth |
| RL framework | PRIME-RL ([github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)) |
| Environment library | verifiers ([github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)) |
| Adapter format | LoRA (via PEFT) |
| Inference | vLLM |
| Compute | RunPod / Vast.ai |
| Free smoke testing | Kaggle Notebooks / Google Colab |

---

## 13. Glossary

| Term | Plain meaning |
|---|---|
| **HS / Harmonized System** | International product classification system from WCO (6 digits). |
| **HTSUS** | US version of HS, 10 digits. |
| **CN / TARIC** | EU versions (8 / 10 digits). |
| **CROSS** | Public US Customs database of binding rulings. |
| **EBTI** | EU equivalent of CROSS. |
| **SFT** | Supervised Fine-Tuning — teach by example. |
| **RL** | Reinforcement Learning — teach by reward signal. |
| **GRPO** | Group Relative Policy Optimization — modern RL algorithm. |
| **LoRA** | Low-Rank Adaptation — small trainable layers on top of a frozen base. |
| **PEFT** | Parameter-Efficient Fine-Tuning (the library that does LoRA). |
| **vLLM** | High-performance LLM inference engine. |
| **FSDP2** | PyTorch's distributed training framework. |
| **Rollout** | One sampled answer from the model during RL. |
| **Reward function** | Code that scores a model's output. |
| **Verifiable reward** | A reward that can be checked programmatically (yes/no, match/no-match). |

---

## 14. Immediate Next Steps

In order:

1. **Set up Kaggle account** and confirm GPU access (free).
2. **Write a CROSS scraper** — collect 1,000 rulings as starter data.
3. **Format data** as `(product description → 6-digit code)` JSON.
4. **Phase 0 notebook on Kaggle** — fine-tune Qwen 0.5B end-to-end.
5. **Hand-test** with 20 product descriptions you write yourself.
6. **Decide what's missing** before scaling to Phase A.

Don't skip ahead. Phase 0 catches the bugs that would cost real money in Phase A.

---

*Document version: 1.0 — initial project plan*# Trade Compliance HS Code Classifier — Project Plan

> A reinforcement-learning fine-tuned LLM for predicting Harmonized System (HS) codes from product descriptions, scoped initially to USA (HTSUS) with a path to multi-country support.

---

## 1. Project Goal

Build a domain-expert LLM that:

1. Takes a **product description** in natural language.
2. Predicts the correct **HS / HTS code** (e.g. `6109.10.0027`).
3. Provides **legal reasoning** grounded in HTS chapter notes / CROSS rulings.
4. Works across multiple jurisdictions (USA first, EU/UK/China next) via swappable country adapters.

**Use case:** decision support for customs brokers, importers, and compliance teams — *not* a replacement for human review.

---

## 2. The Big Picture: Architecture

We use a **shared base model + country-specific LoRA adapters** pattern:

```
              ┌────────────────────────────────────┐
              │   BASE MODEL (international)       │
              │   Knows WCO 6-digit logic          │
              │   "What chapter & heading is this?"│
              │   Trained ONCE, used by all        │
              └────────────────────────────────────┘
                              │
        ┌───────────────┬─────┴─────┬───────────────┐
        ▼               ▼           ▼               ▼
   ┌────────┐      ┌────────┐  ┌────────┐      ┌────────┐
   │ 🇺🇸 USA │      │ 🇪🇺 EU  │  │ 🇬🇧 UK  │      │ 🇨🇳 China│
   │ 10 dig │      │ 8/10dig│  │ 10 dig │      │ 10 dig │
   │ adapter│      │ adapter│  │ adapter│      │ adapter│
   └────────┘      └────────┘  └────────┘      └────────┘
```

**Why this design:**

| Benefit | Impact |
|---|---|
| Train international logic **once** | 90% of difficulty is shared — don't repeat it per country |
| Country adapters are **cheap** (~$20 each) | Adding a new country is almost free |
| **Modular deployment** | Ship base + only the adapters customers need |
| **Easy maintenance** | WCO updates → retrain base. Country tariff change → retrain just that adapter |

---

## 3. Background: How HS Codes Work Globally

### Structure
- **WCO Harmonized System** = international standard, **6 digits**, used by 200+ countries.
- Each country extends with **national digits** (statistical suffixes).

### Code anatomy
```
8471.30.0100
└┬┘ └┬┘ └┬┘ └┘
 │   │   │   └─ National (US-specific) — 10 digits
 │   │   └───── Subheading — 8 digits
 │   └───────── Heading — 6 digits (international)
 └───────────── Chapter — 2 digits
```

### Country digit counts

| Country | System | Digits |
|---|---|---|
| 🇺🇸 USA | HTSUS | 10 |
| 🇪🇺 EU | CN / TARIC | 8 / 10 |
| 🇬🇧 UK | UK Global Tariff | 10 |
| 🇨🇳 China | China HS | 10–13 |
| 🇯🇵 Japan | Japan HS | 9 |
| 🇨🇦 Canada | Canadian Customs Tariff | 10 |
| 🇮🇳 India | ITC(HS) | 8 |

### Important reality
Even when two countries have the same 6-digit code, **duty rates and trade rules differ**. Same code ≠ same regulation.

---

## 4. Data Sources (All Free, All Public)

### 🇺🇸 USA
| Source | Description | URL |
|---|---|---|
| HTSUS | Full US tariff schedule (CSV/JSON downloads) | hts.usitc.gov |
| CROSS | ~200K+ binding rulings with reasoning | rulings.cbp.gov |

### 🇪🇺 EU
| Source | Description |
|---|---|
| TARIC | EU tariff database with duties + measures (public API) |
| EBTI | Binding Tariff Information rulings |
| CN nomenclature | 8-digit Combined Nomenclature |

### 🌐 International / Others
| Source | Description |
|---|---|
| WCO HS Nomenclature | 6-digit international standard |
| UK Trade Tariff API | gov.uk/trade-tariff |
| WITS (World Bank) | Tariff data for 200+ countries |

**Recommended starting point:** scrape CROSS — it has the most data with full reasoning text.

---

## 5. The Phased Roadmap

### Phase 0 — Smoke Test (free)
**Goal:** prove the workflow works end-to-end. Catch bugs cheaply.

| Item | Detail |
|---|---|
| Model | Qwen 2.5 **0.5B** |
| Data | ~500–1,000 CROSS examples |
| Hardware | Kaggle / Google Colab (free T4 GPU) |
| Time | 1–2 hours |
| Cost | **$0** |
| Expected accuracy | 20–40% (sanity check only) |

**Success criterion:** the training loop runs, model produces correctly-formatted output, you can debug your data pipeline.

---

### Phase A — Build the International Base Model

**Goal:** a model that's accurate at the universal 6-digit WCO level.

#### A1. Data preparation
- Collect ~50K (product description → 6-digit code) pairs from CROSS, EBTI, and WCO docs.
- Normalize codes to 6 digits.
- Format as instruction-tuning pairs with reasoning.

#### A2. SFT (Supervised Fine-Tuning)
| Item | Detail |
|---|---|
| Base model | Qwen 2.5 **7B** (or Llama 3.1 8B) |
| Hardware | 1× A100 80GB on RunPod |
| Time | 6–12 hours |
| Cost | **~$15** |
| Expected accuracy | 60–75% chapter-level |

#### A3. RL with PRIME-RL
| Item | Detail |
|---|---|
| Framework | PRIME-RL + verifiers library |
| Algorithm | GRPO |
| Hardware | 1–2× A100 80GB |
| Time | 12–24 hours |
| Cost | **~$40** |
| Expected accuracy | 80–90% at 6-digit level |

#### A4. Evaluation
- Held-out set of ~500 hand-checked international classifications.
- Measure accuracy at each level (chapter / heading / subheading).

**Phase A total: ~$55**

---

### Phase B — Country-Specific Adapters

**Goal:** per-country LoRA adapters that extend the base to national digit codes.

**Repeat this for each country (start with USA):**

#### B1. Country data
- Scrape country-specific rulings.
- Format with country flag in prompt:
  ```
  Country: USA
  Product: Cotton T-shirts, men's, knitted, 95% cotton 5% spandex
  → 6109.10.0027
  ```

#### B2. SFT the adapter
| Item | Detail |
|---|---|
| Approach | LoRA on top of frozen base |
| Adapter size | ~50–200MB |
| Hardware | 1× A100 |
| Time | 2–4 hours |
| Cost | **~$5** |

#### B3. RL the adapter
| Item | Detail |
|---|---|
| Reward | Full 10-digit (or country-specific) match scoring |
| Hardware | 1× A100 |
| Time | 4–8 hours |
| Cost | **~$15** |

**Per-country total: ~$20**

---

## 6. The Reward Function (Heart of RL)

For HS codes, the reward function uses **hierarchical partial credit**:

| Match level | Example | Reward |
|---|---|---|
| Full 10-digit match | `8471.30.0100` = `8471.30.0100` | **+1.0** |
| 8-digit match | predicted `8471.30.01XX` | **+0.7** |
| 6-digit match | predicted `8471.30.XXXX` | **+0.5** |
| 4-digit match | predicted `8471.XX.XXXX` | **+0.3** |
| 2-digit match (chapter only) | predicted `84XX.XX.XXXX` | **+0.1** |
| Wrong format / garbage | "I think it's a computer" | **−0.1** |

### Bonus signals
| Reward | Purpose |
|---|---|
| **+0.2** for citing a CROSS ruling number | Grounding in real sources |
| **+0.1** for using HTS chapter notes in reasoning | Legal-style reasoning |
| **−0.3** for "I'm not sure" with no attempt | Discourages refusal-as-safety |
| **−0.5** for hallucinated chapter (e.g. Chapter 99) | Catches made-up codes |

### Why this matters
- Naive 0/1 reward gives no gradient — model can't learn to improve incrementally.
- Hierarchical reward lets model climb: wrong chapter → right chapter → right heading → right code.

---

## 7. How One RL Training Step Works

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Pick a prompt                                       │
│  "Cotton T-shirt, men's, knitted, 95% cotton 5% spandex"     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Model generates 8 different answers (rollouts)      │
│  A: "6109.10.0027" ✓                                         │
│  B: "6109.10.0040" (close — same heading)                    │
│  C: "6110.20.2000" (wrong heading, right chapter)            │
│  D: "6109.10.0027" ✓                                         │
│  E: "0207.14.0020" (very wrong)                              │
│  F: "6109.10.0011" (close)                                   │
│  G: "I think this is apparel..." (incomplete)                │
│  H: "6109.10.0027" ✓                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Reward function scores each                         │
│  A: +1.0    B: +0.7    C: +0.1    D: +1.0                   │
│  E: 0.0     F: +0.7    G: -0.1    H: +1.0                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Algorithm (GRPO) nudges the model                   │
│  "Do more of what A, D, H did. Do less of what E, G did."    │
│  Tiny weight updates flow into the model.                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    Repeat 10,000+ times
```

---

## 8. Total Cost Estimate

### For a 4-country production model

| Component | Cost |
|---|---|
| Phase 0 (smoke test) | $0 |
| Phase A (international base) | $55 |
| Phase B × 4 countries (USA, EU, UK, China) | $80 |
| **Total** | **~$135** |
| Iteration buffer (3–5 reruns) | +$200 |
| **Realistic project budget** | **~$300–$400** |

### GPU rental options (2026 prices)

| Provider | GPU | $/hr |
|---|---|---|
| Kaggle / Colab | T4 | **Free** (30 hrs/week) |
| Vast.ai | RTX 4090 | $0.09–$0.59 |
| RunPod | RTX 4090 | $0.34–$0.69 |
| RunPod | A100 40GB | ~$0.60 |
| RunPod | A100 80GB | ~$0.79 |
| Lambda Labs | A100 | $1.29 |
| Lambda Labs | H100 | $2.49 |

**Recommendation:** Kaggle for Phase 0, RunPod (A100 80GB) for everything else.

---

## 9. Common Pitfalls to Avoid

| Pitfall | What happens | How to avoid |
|---|---|---|
| **Reward hacking** | Model gets high reward via nonsense | Unit-test reward function on edge cases before RL |
| **Mode collapse** | Model predicts same code for everything | Add KL penalty; ensure diverse training data |
| **Forgetting reasoning** | Accuracy up, explanations become garbage | Add small reward for well-structured reasoning |
| **Reward bugs** | Weird loss curves; model gets worse | Test reward function on 50 hand-picked examples first |
| **Skipping SFT** | RL doesn't improve much | Ensure SFT baseline ≥60% before RL |
| **Multi-country too early** | Can't debug which layer is wrong | Build one country end-to-end before adding more |

---

## 10. Pre-Flight Checklist Before Phase A RL

- [ ] Phase 0 ran successfully on Kaggle (smoke test passed)
- [ ] Phase A SFT model achieves ≥60% chapter-level accuracy
- [ ] You have ≥10,000 product descriptions for RL prompts
- [ ] Reward function written and **unit-tested** on 50 examples
- [ ] Eval set of ~500 hand-checked examples (kept separate)
- [ ] Budget confirmed (~$50 minimum for a real RL run)
- [ ] PRIME-RL `verifiers` quickstart read at least once

---

## 11. Production / Deployment Considerations

### Legal / compliance
- **Always include a disclaimer:** "Decision support only. Not official classification advice."
- **Require human review** for any binding classification decision.
- **Cite sources** in every answer (CROSS ruling # or HTS chapter note).
- **Log everything** — audit trail is critical for regulated use.

### Versioning
- Track base model version (e.g. `base-v1.0-wco2022`).
- Track adapter versions per country (e.g. `usa-adapter-v1.2`).
- When WCO releases new revision, all adapters need re-validation.

### Updating
- HS revisions every 5 years (WCO).
- US HTSUS updates ~annually.
- EU CN updates annually (Jan 1).
- Plan for annual retraining of adapters.

---

## 12. Tech Stack Summary

| Layer | Tool |
|---|---|
| Base model | Qwen 2.5 7B / Llama 3.1 8B (Hugging Face) |
| SFT framework | Hugging Face TRL or Unsloth |
| RL framework | PRIME-RL ([github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)) |
| Environment library | verifiers ([github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)) |
| Adapter format | LoRA (via PEFT) |
| Inference | vLLM |
| Compute | RunPod / Vast.ai |
| Free smoke testing | Kaggle Notebooks / Google Colab |

---

## 13. Glossary

| Term | Plain meaning |
|---|---|
| **HS / Harmonized System** | International product classification system from WCO (6 digits). |
| **HTSUS** | US version of HS, 10 digits. |
| **CN / TARIC** | EU versions (8 / 10 digits). |
| **CROSS** | Public US Customs database of binding rulings. |
| **EBTI** | EU equivalent of CROSS. |
| **SFT** | Supervised Fine-Tuning — teach by example. |
| **RL** | Reinforcement Learning — teach by reward signal. |
| **GRPO** | Group Relative Policy Optimization — modern RL algorithm. |
| **LoRA** | Low-Rank Adaptation — small trainable layers on top of a frozen base. |
| **PEFT** | Parameter-Efficient Fine-Tuning (the library that does LoRA). |
| **vLLM** | High-performance LLM inference engine. |
| **FSDP2** | PyTorch's distributed training framework. |
| **Rollout** | One sampled answer from the model during RL. |
| **Reward function** | Code that scores a model's output. |
| **Verifiable reward** | A reward that can be checked programmatically (yes/no, match/no-match). |

---

## 14. Immediate Next Steps

In order:

1. **Set up Kaggle account** and confirm GPU access (free).
2. **Write a CROSS scraper** — collect 1,000 rulings as starter data.
3. **Format data** as `(product description → 6-digit code)` JSON.
4. **Phase 0 notebook on Kaggle** — fine-tune Qwen 0.5B end-to-end.
5. **Hand-test** with 20 product descriptions you write yourself.
6. **Decide what's missing** before scaling to Phase A.

Don't skip ahead. Phase 0 catches the bugs that would cost real money in Phase A.

---

*Document version: 1.0 — initial project plan*