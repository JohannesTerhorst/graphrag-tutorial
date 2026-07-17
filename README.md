# GraphRAG Tutorial — From Local to Global

A from-scratch, four-notebook implementation of **GraphRAG** for the TUM seminar
*Large Language Models: Bausteine, Training und Steuerung* (IN2107 / IN2396 / IN4816).

It reproduces the pipeline of **Edge et al. (2025)** — *From Local to Global: A GraphRAG
Approach to Query-Focused Summarization* (arXiv:2404.16130v2) — end to end on a single
Wikipedia article (the 2022 FIFA World Cup), using a **local LLM via Ollama**. No API keys,
no cloud, no cost.

The tutorial contrasts GraphRAG against naive vector-style RAG on a *global sensemaking*
question that flat retrieval cannot answer well:

> *"What are the main storylines and themes of the 2022 FIFA World Cup?"*

---

## What you need

| Requirement | Notes |
|---|---|
| **Python 3.12** | A virtual environment is strongly recommended. |
| **[Ollama](https://ollama.com/download)** | Runs the local model. Install for your OS, then pull the model (below). |
| **~6 GB VRAM or RAM** | `qwen2.5:7b` fits in 8 GB VRAM; it also runs on CPU, just slower. |

### 1. Install Ollama and pull the model

Install Ollama from <https://ollama.com/download>, then:

```bash
ollama pull qwen2.5:7b
```

Leave Ollama running (it serves on `http://localhost:11434` by default). If 7B is too heavy
for your machine, `ollama pull qwen2.5:3b` and change `MODEL` in the notebook setup cells.

### 2. Create the environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Jupyter and run the notebooks in order

```bash
jupyter lab
```

Run **`notebooks/00` → `01` → `02` → `03`** in sequence. Each notebook reads the previous
stage's output from `data/output/` and writes its own, so the order matters. Use
*Kernel → Restart & Run All* per notebook.

---

## Connecting to Ollama (portability)

The setup cell in each notebook locates the Ollama server automatically, in this order:

1. **`OLLAMA_BASE_URL`** environment variable, if set — use this for a remote host, e.g.
   `export OLLAMA_BASE_URL=http://192.168.1.50:11434`
2. **WSL2** — resolves the Windows-host IP (Ollama running on Windows, notebook in WSL2).
3. **Fallback** — `http://localhost:11434` (native Linux / macOS / Windows).

If a notebook cannot reach the model, set `OLLAMA_BASE_URL` explicitly and re-run the setup
cell.

---

## The pipeline

```
data/input/wc2022_trimmed.pdf
        │
        ▼  Notebook 00 — Graph Construction   (§3.1.1–3.1.3, App. A, E.1)
   chunk → LLM entity/relationship extraction + glean pass → merge → weighted graph
        │   → data/output/wc2022_trimmed_graph.json
        │   → data/output/wc2022_trimmed_graph.html   (interactive pyvis graph)
        ▼
   Notebook 01 — Community Detection          (§3.1.4, App. B)
   Leiden partitioning at two resolution levels (C0 coarse, C1 fine)
        │   → data/output/wc2022_trimmed_communities.json
        ▼
   Notebook 02 — Community Summaries          (§3.1.5, App. E.2)
   LLM writes a structured JSON report per C1 community
        │   → data/output/wc2022_trimmed_summaries.json
        ▼
   Notebook 03 — GraphRAG Query               (§3.1.6, App. E.3–E.4)
   Map: score each community 0–100 for the question · Reduce: synthesise a global answer
   Side-by-side comparison against a naive TF-IDF RAG baseline
```

The output filenames are derived from the input PDF's stem, so changing `PDF_PATH` in
Notebook 00 propagates through the whole pipeline.

---

## Project layout

```
graphrag-tutorial/
├── README.md                 ← this file
├── CLAUDE.md                 ← detailed implementation guide (paper section mapping)
├── requirements.txt          ← pinned dependencies
├── data/
│   ├── input/
│   │   └── wc2022_trimmed.pdf         (tutorial input — 2022 FIFA World Cup)
│   └── output/                        (created/overwritten by the notebooks)
└── notebooks/
    ├── 00_graph_construction.ipynb
    ├── 01_community_detection.ipynb
    ├── 02_community_summaries.ipynb
    └── 03_graphrag_query.ipynb
```

The `data/output/` directory ships with a pre-computed run so you can read the results
without waiting for the LLM. Re-running the notebooks overwrites these files.

---

## Notes

- **Determinism.** All LLM calls use `temperature=0.0`. Leiden uses a fixed seed. Runs are
  reproducible up to Ollama/model-version differences.
- **Runtime.** On an RTX 3070 (`qwen2.5:7b`, ~12 tok/s) a full pipeline run is a few minutes,
  dominated by Notebook 00's per-chunk extraction. CPU-only is slower but works.
- **Paper faithfulness.** Each design choice is traceable to a section of Edge et al. (2025);
  see `CLAUDE.md` for the section-by-section mapping.
