# GraphRAG Tutorial — Implementation Guide

## Goal

A from-scratch GraphRAG tutorial for the LLM Seminar at TUM. Students run four sequential
notebooks that implement the full pipeline from raw text to global query answering. The
implementation follows the original paper faithfully:

> **Edge et al. (2025)** — *From Local to Global: A GraphRAG Approach to Query-Focused
> Summarization*. arXiv:2404.16130v2

Each design decision should be traceable back to a section of that paper.

---

## Setup

### Environment

- Python venv at `.venv/` — activate with `source .venv/bin/activate`
- Jupyter: `jupyter lab` from the repo root
- All dependencies in `requirements.txt`

### LLM (Ollama via WSL2)

The local LLM runs in Ollama on the Windows host. The WSL2 host IP is resolved dynamically:

```python
import subprocess
windows_ip = subprocess.check_output(
    "ip route list default | awk '{print $3}'", shell=True
).decode().strip()

from langchain_community.chat_models import ChatOllama
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url=f"http://{windows_ip}:11434",
    temperature=0.0,
    num_predict=512,
)
```

Model: **qwen2.5:3b** (confirmed working in `notebooks/hello_world.ipynb`, ~5 tok/s).
Model: **qwen2.5:7b** (fits entirely in 8 GB VRAM, ~12 tok/s on RTX 3070).
Use `temperature=0.0` throughout for deterministic extractions.

---

## Dataset

**File**: `data/input/wc2022_trimmed.pdf`

Trimmed from the full 2022 FIFA World Cup Wikipedia article
(`data/input/2022_FIFA_World_Cup.pdf`). The 2022 tournament (hosted by Qatar, 32 teams)
was chosen over 2026 because it is already complete — the Wikipedia article contains
richer context: match results, top scorers, the champion (Argentina), and player
performances. This makes entity extraction more meaningful and community summaries
more interesting to read.

**Expected entity types**: `TEAM`, `NATION`, `STADIUM`, `CITY`, `CONFEDERATION`,
`COMPETITION`, `PERSON`, `EVENT`

---

## Pipeline Overview

```
wc2022_trimmed.pdf
       │
       ▼  Notebook 00
  Text Chunks (600 tok, 100 overlap)
       │  LLM extraction + glean pass (§3.1.1–3.1.2, Appendix A)
       ▼
  Entities + Relationships
       │  Merge by exact name, edge weight = duplicate count (§3.1.3)
       ▼
  data/output/wc2022_trimmed_graph.json
       │
       ▼  Notebook 01
  Leiden community detection — hierarchical (§3.1.4)
       ▼
  data/output/wc2022_trimmed_communities.json
       │
       ▼  Notebook 02
  LLM community summaries — structured JSON (§3.1.5, Appendix E.2)
       ▼
  data/output/wc2022_trimmed_summaries.json
       │
       ▼  Notebook 03
  Map-Reduce query with helpfulness scoring (§3.1.6, Appendix E.3–E.4)
       ▼
  Global Answer
```

Output filenames use the input PDF stem as a prefix so that changing `PDF_PATH`
in notebook 00 automatically propagates through the entire pipeline.

---

## Notebook 00 — Graph Construction

**Paper sections**: §3.1.1 (chunking), §3.1.2 (extraction), §3.1.3 (graph), Appendix A & E.1

### Step 1: Load and chunk the PDF

Use `pymupdf` (`fitz`) to extract text. Clean citation markers (`[2]`, `[A]`, etc.) with
a simple regex before chunking.

Chunk at **600 tokens with 100-token overlap** (paper's default, §3.1.1). Use a character-
based splitter (1 token ≈ 4 chars) or `langchain_text_splitters.RecursiveCharacterTextSplitter`.

### Step 2: Entity and relationship extraction

Use the paper's **exact tuple format** (Appendix E.1). Entity types for this dataset:
`TEAM, NATION, STADIUM, CITY, CONFEDERATION, COMPETITION, PERSON, EVENT`

Prompt structure (adapt from Appendix E.1):

```
---Goal---
Given a text document, identify all entities of the types listed and all
relationships among the identified entities.

---Steps---
1. For each entity extract:
   - entity_name (capitalized)
   - entity_type (one of: {entity_types})
   - entity_description (comprehensive)
   Format: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<description>)

2. For each pair of clearly related entities extract:
   - source_entity, target_entity (from step 1)
   - relationship_description
   - relationship_strength (1–10 integer)
   Format: ("relationship"{tuple_delimiter}<src>{tuple_delimiter}<tgt>{tuple_delimiter}<desc>{tuple_delimiter}<strength>)

3. Output all entities then relationships as a single list using {record_delimiter}.
4. End with {completion_delimiter}

---Examples---
[2–3 soccer-domain examples here, e.g. Mbappe/France/FIFA]

---Real Data---
Entity types: {entity_types}
Input: {input_text}
Output:
```

Use `##` as `record_delimiter`, `<|COMPLETE|>` as `completion_delimiter`,
`<|>` as `tuple_delimiter`.

### Step 3: Self-reflection / glean pass (Appendix A.2)

After the initial extraction, run a second prompt asking the LLM whether it missed any
entities. Force a yes/no answer, and if yes, extract the missed entities. One iteration
is sufficient for this tutorial dataset.

Glean prompt:
```
Many entities may have been missed in the last extraction. Answer YES | NO: were
important named entities of types {entity_types} missed from the text above?
```
If YES → prompt: `Add the missed entities using the same format as above.`

### Step 4: Parse and build the graph

Parse tuples, merge duplicate entity names (exact string match, case-insensitive), and
aggregate descriptions. Build a `networkx.Graph` where:
- Nodes: entities with attributes `type`, `description`
- Edges: relationships with attributes `description`, `weight` (= duplicate count)

Save with `networkx.node_link_data` to `data/output/wc2022_trimmed_graph.json`.

### Visualisation

Use `pyvis.network.Network` for an interactive HTML graph. Color nodes by entity type.
Save to `data/output/wc2022_trimmed_graph.html`.

---

## Notebook 01 — Community Detection

**Paper sections**: §3.1.4, Appendix B

### Algorithm

Use **Leiden** (paper's choice, citing Traag et al. 2019), not Louvain.
Library: `leidenalg` + `python-igraph`.

```python
import igraph as ig
import leidenalg

# Convert networkx graph to igraph
g_ig = ig.Graph.from_networkx(G)
# Run at two resolution levels to get hierarchy
partition_c0 = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
partition_c1 = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=1.5)
```

Produce two levels (C0 = root communities, C1 = sub-communities), mirroring Figure 4 of
the paper.

### Output

`data/output/wc2022_trimmed_communities.json`:
```json
{
  "C0": { "0": ["ARGENTINA", "MESSI", ...], "1": [...] },
  "C1": { "0": ["ARGENTINA", "MESSI"], "1": ["FRANCE", "MBAPPE"], ... }
}
```

### Visualisation

Reproduce a version of Figure 4: two side-by-side `matplotlib` plots with nodes colored
by community assignment at C0 and C1. Node size proportional to degree.

---

## Notebook 02 — Community Summaries

**Paper sections**: §3.1.5, Appendix E.2

### Input

For each community in C1 (leaf level), collect:
- All node descriptions
- All edge descriptions within the community
- Prioritise by edge weight (combined source + target degree, descending)

### Prompt

Use the paper's **structured JSON output** format (Appendix E.2):

```
---Goal---
Write a comprehensive report of a community, given a list of entities that
belong to the community and their relationships.

---Report Structure---
Return a JSON object with:
{
  "title": "<short, specific title with named entities>",
  "summary": "<executive summary of the community>",
  "rating": <float 0–10, importance of this community>,
  "rating_explanation": "<one sentence>",
  "findings": [
    {"summary": "<insight title>", "explanation": "<2–3 sentences>"},
    ...  (3–5 findings)
  ]
}

---Community Data---
Entities: {node_descriptions}
Relationships: {edge_descriptions}
Output:
```

### Output

`data/output/wc2022_trimmed_summaries.json`: list of community report objects, each with
the JSON structure above plus a `community_id` field.

Display each summary in a rendered markdown cell so students can read the discovered themes.

---

## Notebook 03 — GraphRAG Query

**Paper sections**: §3.1.6, Appendix E.3–E.4

### Query

Use this example query throughout the notebook:
> *"What are the main storylines and themes of the 2022 FIFA World Cup?"*

This is a global sensemaking question — the kind vector RAG cannot answer well (§1).

### Map step (Appendix E.4 / Community Answer Generation)

For each community summary, prompt the LLM:

```
---Goal---
Generate a response to the user's question based on the community report below.
At the start of your response, output an integer score 0–100 indicating how
helpful this community report is for answering the question.
Format: <ANSWER_HELPFULNESS>score</ANSWER_HELPFULNESS>

---Question---
{question}

---Community Report---
{community_summary}

Output:
```

Filter out responses with score = 0.

### Reduce step (Appendix E.4 / Global Answer Generation)

Sort partial answers by helpfulness score (descending). Concatenate the top answers into
a context window and prompt the LLM for a final synthesis:

```
---Goal---
Generate a comprehensive response to the question below by synthesising the
analyst reports provided. Style the response in markdown.

---Question---
{question}

---Analyst Reports (ranked by helpfulness)---
{top_partial_answers}

Output:
```

### Comparison

Show a side-by-side comparison:
- **GraphRAG answer** (from community summaries)
- **Naive RAG answer** (top-3 most similar text chunks, retrieved by keyword overlap
  or simple TF-IDF — no embeddings needed to illustrate the contrast)

The GraphRAG answer should be notably more comprehensive on themes; the naive answer
will name specific facts but miss the big picture.

---

## Dependencies to Add

These are not yet in `requirements.txt`:

```
networkx
pymupdf          # fitz — PDF text extraction
leidenalg        # Leiden community detection
python-igraph    # required by leidenalg
pyvis            # interactive graph visualisation
scikit-learn     # TF-IDF for naive RAG baseline in notebook 03
```

Install in the venv:
```bash
pip install networkx pymupdf leidenalg python-igraph pyvis scikit-learn
```

---

## File Structure

```
graphrag-tutorial/
├── CLAUDE.md                        ← this file
├── requirements.txt
├── data/
│   ├── input/
│   │   ├── 2022_FIFA_World_Cup.pdf            (full Wikipedia article)
│   │   └── wc2022_trimmed.pdf                 (trimmed tutorial input)
│   └── output/                                (created by notebooks)
│       ├── wc2022_trimmed_graph.json
│       ├── wc2022_trimmed_graph.html
│       ├── wc2022_trimmed_communities.json
│       └── wc2022_trimmed_summaries.json
└── notebooks/
    ├── hello_world.ipynb                 (Ollama connectivity test)
    ├── 00_graph_construction.ipynb
    ├── 01_community_detection.ipynb
    ├── 02_community_summaries.ipynb
    └── 03_graphrag_query.ipynb
```

---

## Session Notes

- **Each notebook is a pipeline stage** — it reads from the previous stage's output file
  and writes its own output file to `data/output/`.
- **Implement one notebook per session.** Start each session by reading this file and the
  target notebook's section above.
- **Prompts should be shown to students** — display the prompt template in a markdown cell
  before the code cell that calls the LLM, so they can read it alongside the paper.
- **Explain design choices** — each notebook should have a markdown cell explaining *why*
  this step is done this way (citing the paper section), not just *what* it does.
- The `hello_world.ipynb` confirms the Ollama/WSL2 setup works. If the LLM is unreachable,
  check the Windows host IP with `ip route list default | awk '{print $3}'`.
