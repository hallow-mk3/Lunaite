# PROJECT: Lunaite Harness — Tool Selection at Scale for LLM Agents

## Objective
Build a clean LLM tool-access harness, then answer one measured research
question on top of it: as the number of available tools grows, does
retrieval-based tool selection improve tool-call correctness and reduce
cost compared to naive "dump all tool schemas in context" selection?

## Background
The original project attempted to be a universal cognitive architecture
(MoE adapters, LoRA, deliberation, memory, voice, GUI) attached to any
model. That scope is being discarded. The one part worth keeping is the
underlying idea: a harness that gives an LLM access to external tools.
Ignore all prior architecture, training code, and claims — this is a
fresh rebuild scoped to tool access and tool selection only.

## Research Question
When an agent has access to a large, varied tool library (20-50+ tools),
does selecting a relevant subset via retrieval (embedding similarity
between the query and tool descriptions) before calling the model produce:
  (a) higher correct-tool-selection accuracy,
  (b) higher correct-argument-construction accuracy, and
  (c) lower token cost,
compared to giving the model all tool schemas at once (the naive/default
approach most function-calling APIs use today)?

## Method

### 1. Build the harness (engineering component)
- `ToolRegistry`: holds tool definitions (name, description, JSON schema
  for arguments, and the callable itself). Read-only/non-destructive tools
  only for the builtin set (explicitly exclude shell/filesystem/clipboard
  access — document this as a deliberate security stance, not an omission).
- `Harness`: given a user query, a `ToolRegistry`, and a `Selector`, calls
  the LLM with (query + selected tool schemas), parses the tool call,
  executes it, and returns the result. Must log every step (which tools
  were shown, which was picked, what arguments were generated, whether
  execution succeeded) for later analysis.
- Two selector strategies implementing the same interface:
  - `NaiveSelector`: returns all tools in the registry, unfiltered.
  - `RetrievalSelector`: embeds the query and all tool descriptions,
    returns only the top-k most similar tools (k configurable).
- Support at least one real LLM backend (OpenAI-compatible API or Ollama)
  with real function-calling/tool-call parsing — not a mocked stub.

### 2. Build the tool library (needed for a meaningful test)
- 20-50 tools across varied domains (e.g. math, unit conversion, date/time,
  weather, search, currency conversion, string manipulation, unit lookup,
  trivia/wiki, geocoding) — enough that naive selection starts to strain
  context and plausibility of picking correctly.
- Some tools should be deliberately similar/confusable (e.g. two different
  "convert X to Y" tools) to test whether selection actually discriminates
  well, not just filters by obvious keyword.

### 3. Build the eval set
- ~100-150 synthetic tasks, each mapped to a known correct tool + expected
  arguments (ground truth), spanning:
  - unambiguous cases (only one tool obviously fits)
  - ambiguous cases (2+ plausible tools, only one is actually correct)
  - trick cases (no tool fits — correct behavior is to say so, not force a
    call)
- Document how these were generated (hand-written, or LLM-assisted with
  manual review) and include the full set in the repo for reproducibility.

### 4. Conditions
  A. Naive selection: all tools shown every time.
  B. Retrieval selection: top-k tools shown, k swept over a few values
     (e.g. 3, 5, 10) to show the accuracy/cost tradeoff curve, not just one
     point.

### 5. Metrics (per condition, per k where applicable)
- Correct-tool-selection rate (did it pick the right tool, or correctly
  decline when no tool fits?)
- Correct-argument rate (given right tool was picked, were arguments valid
  and correct?)
- Full-task success rate (both of the above correct)
- Prompt tokens consumed per task (this is where naive selection is
  expected to lose as the registry grows — show this directly)
- Latency per task

### 6. Deliverables
- `research/tool_selection/run_eval.py`: runs both conditions across the
  full eval set and logs raw per-task results (JSONL: task id, condition,
  k, tools shown, tool selected, arguments generated, success/failure per
  metric, tokens, latency).
- `research/tool_selection/analyze.py`: computes aggregate metrics, and
  produces:
  - accuracy vs. tool-registry-size chart (this is the core plot — run the
    eval at a few registry sizes, e.g. 10/25/50 tools, to show how naive
    selection degrades while retrieval holds up, if that's what happens)
  - accuracy vs. token-cost tradeoff for different k values
- `research/tool_selection/REPORT.md`: states the exact question, method,
  results (with plots), and an honest interpretation — including stating
  plainly if retrieval does NOT help, or only helps past some registry
  size threshold, if that's what the data shows.

## Constraints / Non-negotiables
- No claims beyond what's measured; state exact model, embedding model,
  and k values used for every reported number.
- Naive selection is the real baseline to beat — most existing
  function-calling systems today just dump all tools in context, so this
  is the practically relevant comparison, not a strawman.
- Ignore/do not carry over anything from the previous version of this
  project (MoE, LoRA, training scripts, voice, GUI, deliberation module,
  desktop telemetry) — this is a clean rebuild scoped only to the harness
  and this research question.
- Keep `research/` clearly separated from the core `lunaite` package so the
  library stays a usable, lightweight harness independent of this specific
  study.

## Suggested repo structure
    lunaite/
    ├── harness/          # core Harness class, orchestration + logging
    ├── tools/             # ToolRegistry, Tool definition, builtin tool set
    ├── selection/          # NaiveSelector, RetrievalSelector (shared interface)
    research/
    └── tool_selection/
        ├── tools_library.py     # the 20-50 tool definitions for the study
        ├── eval_tasks.jsonl     # the ~100-150 ground-truth tasks
        ├── run_eval.py
        ├── analyze.py
        ├── results/
        └── REPORT.md

## First implementation step
Confirm which LLM backend and embedding model are available/affordable for
running (task count) x (conditions) x (registry sizes) total completions,
before writing any code.