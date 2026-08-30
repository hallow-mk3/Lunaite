# Empirical Study: Retrieval-Based Tool Selection vs. Naive In-Context Schema Packing at Scale

**Project:** Lunaite Harness Research  
**Author:** Swasthik Shetty ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com))  
**Date:** August 2026  
**Status:** Complete  

---

## 1. Executive Summary & Research Question

Modern tool-augmented LLM architectures typically follow a **naive selection strategy**: when an agent is provided with $N$ callable tools, all $N$ JSON schema definitions are injected into the system prompt context on every invocation.

While this approach works for small tool collections ($N \le 5$), real-world agent environments frequently demand access to dozens or hundreds of specialized tools across diverse operational domains.

### Research Question
> When an agent has access to a large, varied tool library ($N \in \{10, 25, 50\}$), does pre-filtering tools via **retrieval-based semantic selection** (dense embeddings with cosine similarity) before calling the model produce:
> 1. **Higher correct-tool-selection accuracy**,
> 2. **Higher correct-argument-construction accuracy**, and
> 3. **Lower prompt token cost and latency**,  
> compared to the naive approach of dumping all tool schemas in context?

---

## 2. System Architecture & Methodology

### 2.1 The Lunaite Harness
The experiment is powered by the `lunaite.harness` orchestration pipeline:

```
[ User Query: q ]
       │
       ▼
 ┌────────────────────────────────────────┐
 │ Selector: S(q, Registry, k)            │
 ├────────────────────────────────────────┤
 │ • Naive: Returns all N tools           │
 │ • Retrieval: Returns Top-k (MiniLM-L6) │
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ LLM Function Calling Completion        │
 │ (JSON Schema + OpenAI-compatible API)  │
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ Structured Telemetry & Validation      │
 │ • Tool Selection Correctness           │
 │ • Argument AST Normalization & Matching│
 │ • Pure Execution & Error Handling      │
 └────────────────────────────────────────┘
```

### 2.2 Tool Library ($N=50$)
The study uses 50 deterministic, mock tool implementations across 10 functional domains to prevent external network flakiness:
1. **Math & Arithmetic** (`add_numbers`, `divide_numbers`, `raise_to_power`, etc.)
2. **Unit Conversion — Length** (`convert_km_to_miles`, `convert_inches_to_cm`, etc.)
3. **Unit Conversion — Weight & Mass** (`convert_kg_to_lbs`, `convert_grams_to_ounces`, etc.)
4. **Unit Conversion — Temperature** (`celsius_to_fahrenheit`, `fahrenheit_to_celsius`, etc.)
5. **Date & Time** (`get_current_time`, `calculate_date_difference`, `add_days_to_date`, etc.)
6. **Currency & Forex** (`convert_usd_to_eur`, `get_exchange_rate`, etc.)
7. **String Operations** (`reverse_string`, `count_vowels`, `word_count`, etc.)
8. **Statistics** (`calculate_mean`, `calculate_median`, `calculate_variance`, etc.)
9. **Geography & Geocoding** (`get_city_coordinates`, `get_country_capital`, etc.)
10. **General Knowledge & Science** (`get_element_atomic_number`, `get_planet_moons`, etc.)

#### Deliberately Confusable Pairs
To stress-test semantic discrimination rather than shallow keyword matching, the library incorporates confusable pairs:
- `convert_km_to_miles` vs `convert_miles_to_km`
- `convert_kg_to_lbs` vs `convert_lbs_to_kg`
- `word_count` vs `character_count`
- `get_city_population` vs `get_country_population`

### 2.3 Evaluation Benchmark Dataset (`eval_tasks.jsonl`)
150 ground-truth evaluation queries classified into three behavioral categories:
- **Unambiguous (70%)**: Single obvious tool match with strict argument ground truth.
- **Ambiguous / Confusable (20%)**: Query could plausibly match multiple similar tools; only one matches exact intent.
- **Negative / Trick (10%)**: Queries where no available tool is appropriate (model must correctly decline to invoke a tool).

---

## 3. Experimental Setup & Conditions

- **Registry Sizes ($N$)**: 10, 25, 50 tools.
- **Selectors**:
  - `NaiveSelector`: $k = N$ (all tools exposed).
  - `RetrievalSelector`: $k \in \{3, 5, 10\}$ using `sentence-transformers/all-MiniLM-L6-v2`.
- **Target LLMs**: `qwen2.5:7b` / OpenAI-compatible function-calling engines.
- **Metrics**:
  - $\text{Acc}_{\text{tool}}$: Exact match on selected tool name (or successfully returning no tool for trick cases).
  - $\text{Acc}_{\text{args}}$: Exact normalized match on parameter keys and values.
  - $\text{Acc}_{\text{full}}$: Joint success ($\text{Acc}_{\text{tool}} \land \text{Acc}_{\text{args}} \land \text{Exec}_{\text{success}}$).
  - $\bar{T}_{\text{prompt}}$: Mean prompt tokens per query.
  - $\bar{L}$: Mean end-to-end latency in seconds.

---

## 4. Key Findings

### 1. Token Cost Scales Linearly in Naive, Flat in Retrieval
As the registry grows from 10 to 50 tools:
- **Naive Selection** prompt token consumption scales from **~900 tokens** to **~4,300 tokens** per request.
- **Retrieval Selection ($k=3$)** remains constant at **~380–420 tokens** regardless of total registry size.
- **Net Cost Reduction**: **85% to 91% token reduction** at $N=50$.

### 2. Accuracy Retention & Confusable Pairs
- In small registries ($N=10$), Naive and Retrieval ($k \ge 3$) perform comparably in tool accuracy.
- In larger registries ($N=50$), Naive selection suffers from "distractor overload," where the presence of multiple similar function signatures increases false-positive routing on ambiguous tasks.
- Setting $k=5$ provides the optimal Pareto balance: it captures the ground-truth tool in the top-$k$ set across 98%+ of queries while avoiding schema dilution.

### 3. Latency & Computational Overhead
- The local embedding and cosine similarity calculation for 50 tools via `all-MiniLM-L6-v2` executes in **< 4 ms on CPU**.
- The reduction in prompt token length substantially reduces LLM time-to-first-token (TTFT) and inference latency, resulting in a net **~1.8× faster end-to-end response time**.

---

## 5. Summary Table

| Selection Condition | Registry Size ($N$) | Mean Prompt Tokens | Estimated Token Savings | Full Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (All Tools)** | 10 | ~920 tok | Baseline (0%) | 88.0% |
| **Retrieval ($k=3$)** | 10 | ~390 tok | **57.6% savings** | 87.5% |
| **Naive (All Tools)** | 25 | ~2,150 tok | Baseline (0%) | 82.0% |
| **Retrieval ($k=5$)** | 25 | ~580 tok | **73.0% savings** | 86.0% |
| **Naive (All Tools)** | 50 | ~4,280 tok | Baseline (0%) | 76.5% |
| **Retrieval ($k=5$)** | 50 | ~610 tok | **85.7% savings** | 85.0% |

---

## 6. How to Reproduce

```bash
# 1. Run the benchmark evaluation matrix
python research/tool_selection/run_eval.py \
  --model qwen2.5:7b \
  --registry-sizes 10 25 50 \
  --k-values 3 5 10

# 2. Analyze raw JSONL logs and generate visualization plots
python research/tool_selection/analyze.py
```

Generated plots will be saved to `research/tool_selection/results/`:
- `plot_accuracy_vs_registry_size.png`
- `plot_accuracy_vs_token_cost.png`
- `plot_token_consumption.png`

---

## 7. Deliberate Stance & Open Questions

1. **When Retrieval Is Not Enough**: Pure cosine similarity on tool descriptions can struggle with queries that require multi-step reasoning before tool selection is obvious. Future work should investigate **hybrid lexical + semantic retrieval** (e.g. BM25 + Dense) and small pre-selector LLM routers.
2. **Read-Only Safety**: The standard harness deliberately restricts tools to safe, non-destructive callables. Extending to state-modifying tools requires reversible transactions or human-in-the-loop gates.
