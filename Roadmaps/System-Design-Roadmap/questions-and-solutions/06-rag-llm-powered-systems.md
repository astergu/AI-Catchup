# Deep Dive: LLM-Powered Systems (RAG Architecture)

*Deep dive into Question #6. A strong candidate understands the full RAG stack — from ingestion to retrieval to generation — and has a concrete plan for evaluating and mitigating hallucinations.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Data sources:** How many documents? What formats (PDF, HTML, Confluence, Notion)? How frequently do they update?
- **Scale:** How many users querying simultaneously? Expected QPS?
- **Latency:** Acceptable response time? (RAG pipelines typically take 2–5 seconds)
- **Success Metric:** Factual accuracy? User satisfaction score? Reduction in support tickets?

### 2. Why RAG Instead of Fine-Tuning?

| Approach | When to Use | Downside |
|---|---|---|
| **RAG** | Knowledge changes frequently, need citations, want to control what the model "knows" | Adds retrieval latency and complexity |
| **Fine-tuning** | Teaching the model a style, domain language, or task format | Expensive to retrain when knowledge updates; model can still hallucinate |
| **RAG + Fine-tuning** | Best of both worlds for high-stakes production systems | Most complex |

For a company's internal documentation Q&A, **RAG is the right choice** because docs change frequently and you need verifiable citations.

### 3. Architecture Overview

```
[OFFLINE: Ingestion Pipeline]

  Raw Documents (PDF, HTML, Markdown)
          |
  [Document Parsing & Cleaning]
          |
  [Chunking]
          |
  [Embedding Model]  → chunk embeddings (e.g., 1536-dim)
          |
  [Vector Database]  (Pinecone / Milvus / Weaviate / pgvector)
  + Keyword Index    (Elasticsearch / BM25)


[ONLINE: Query Pipeline]

  User Question
       |
  [Query Rewriting]   (optional: expand, clarify, decompose)
       |
  [Hybrid Retrieval]  → Top 50 candidates
    ├─ Vector Search (ANN)
    └─ BM25 Keyword Search
       |
  [Re-ranking]        → Top 5 most relevant chunks (Cross-Encoder)
       |
  [Prompt Assembly]   → System prompt + context chunks + user question
       |
  [LLM Generation]    (GPT-4o / Claude / Llama)
       |
  [Response + Citations]
```

### 4. Ingestion Pipeline (The "Offline" Part)

#### Document Parsing
- Handle multiple formats: PDFs (use `pdfplumber` or `pymupdf`), HTML (BeautifulSoup), DOCX, Notion/Confluence exports.
- Strip boilerplate: headers, footers, nav menus.
- Preserve structure: headings and section titles are valuable context.

#### Chunking Strategy
Don't just split by a fixed number of characters. A strong answer mentions:

| Strategy | How | Best For |
|---|---|---|
| **Fixed-size with overlap** | 512 tokens, 50-token overlap between chunks | Simple baseline |
| **Recursive Character Splitting** | Split on `\n\n`, then `\n`, then `.`, then ` ` — preserves natural boundaries | General text |
| **Semantic Chunking** | Split where the semantic meaning shifts (detected by embedding similarity drop) | Long documents with varied topics |
| **Document-aware** | Respect section headers; keep a section together if small enough | Structured docs (wikis, manuals) |

**Chunk size trade-off:**
- **Too small:** Loses context needed to answer the question.
- **Too large:** Dilutes the relevance signal; more irrelevant text is fed to the LLM.

#### Metadata Enrichment
Each chunk should be stored with metadata:
- `source_url`, `page_number`, `section_title`
- `last_updated` — enables filtering to recent docs only
- `doc_type` — enables filtering by document category
- `author` — useful for attribution

This metadata enables **filtered retrieval** (e.g., "only search docs updated in the last 6 months").

#### Embedding Model Selection
| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Cost-efficient, strong performance |
| `text-embedding-3-large` (OpenAI) | 3072 | Best performance, higher cost |
| `BGE-M3` (BAAI) | 1024 | Open-source, multilingual, competitive |
| `E5-large` | 1024 | Open-source, strong on retrieval benchmarks |

Choose based on latency budget, cost, and whether multilingual support is needed.

### 5. Retrieval & Re-ranking (The "Search" Part)

#### Hybrid Search
Combining both approaches dramatically improves recall:
- **Vector Search:** Catches semantic matches ("remote work policy" matches "work from home guidelines").
- **BM25 / Keyword Search:** Catches exact matches for proper nouns, acronyms, version numbers (e.g., "SOC 2 Type II").
- **Fusion:** Use **Reciprocal Rank Fusion (RRF)** to merge ranked lists from both sources without needing to tune score scales.

#### Re-ranking with a Cross-Encoder
A **Bi-Encoder** (Two-Tower) is used for fast retrieval but scores query and document independently. A **Cross-Encoder** sees both query and document together and produces a much higher-quality relevance score — but is too slow to run over all documents.

**The pattern:** Retrieve Top 50 with fast ANN → Re-rank with Cross-Encoder → Keep Top 5 for the prompt.

Popular Cross-Encoders: Cohere Rerank, `ms-marco-MiniLM` (open-source).

#### Query Rewriting (Advanced)
Before retrieval, improve the query:
- **HyDE (Hypothetical Document Embeddings):** Ask the LLM to generate a hypothetical answer to the question, then embed *that* and use it for retrieval. Works surprisingly well.
- **Query Decomposition:** Break "What is the vacation policy and how does it compare to the sick leave policy?" into two sub-queries.
- **Step-back prompting:** Generalize a specific question to a broader one for better retrieval.

### 6. Generation & Grounding (The "AI" Part)

#### Prompt Engineering
```
System: You are a helpful assistant for [Company] internal documentation.
Answer ONLY using the provided context below.
If the answer is not contained in the context, respond with:
"I don't have enough information in the documentation to answer this question."
Always cite the source document using [Source N] notation.

Context:
[Source 1]: {chunk_1_text}
[Source 2]: {chunk_2_text}
...

User: {question}
```

Key instructions:
- **Grounding constraint:** "Answer ONLY using the provided context" reduces hallucination.
- **Explicit fallback:** Tell the model what to say when it doesn't know.
- **Citation requirement:** Forces the model to link every claim to a source.

#### LLM Choice
- `temperature = 0` for factual Q&A (deterministic, reduces hallucination).
- Larger context window allows more retrieved chunks, but more tokens = higher cost and latency.

### 7. Evaluation (The "Crucial" Part)

#### RAGAS Metrics — The RAG Triad

| Metric | Question it Answers | How Measured |
|---|---|---|
| **Context Relevance** | Did retrieval return the right documents? | LLM-as-judge rates each retrieved chunk's relevance to the query |
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? (No hallucination) | LLM decomposes answer into claims, checks each against context |
| **Answer Relevance** | Does the answer actually address what the user asked? | Embed the answer and compare to the original question |

All three can be automated using an **LLM-as-a-judge** pipeline (a second LLM evaluates the output of the first).

#### Evaluation Dataset
Build a golden test set:
- 50–100 representative questions with known correct answers.
- Run the full pipeline against this set and measure RAGAS metrics.
- Use this as a regression test — any pipeline change must not degrade these scores.

### 8. Handling Document Updates

Documents change. A naive system re-indexes everything from scratch nightly — expensive and slow.

**Better approach — Incremental Indexing:**
1. Detect changed documents via `last_modified` timestamp or webhook from the source system.
2. Delete old chunks for that document from the vector DB.
3. Re-parse, re-chunk, re-embed only the changed document.
4. Insert new chunks.

**Handle deletions:** If a policy document is removed, its chunks must also be deleted from the index to avoid returning stale answers.

**Handle permissions:** If some docs are restricted to certain teams, attach permission metadata to chunks and filter at retrieval time. Never return chunks to users who don't have access.

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Architecture** | Mentions a basic "DB + LLM" setup. | Includes Ingestion Pipeline and Re-ranking step. | Discusses Query Rewriting (HyDE, decomposition) and caching layers. |
| **Chunking** | "Split text into pieces." | Mentions recursive splitting with overlap. | Discusses semantic chunking, chunk size trade-offs, and metadata enrichment. |
| **Retrieval** | Only vector search. | Hybrid search with BM25 + vector. | Discusses Reciprocal Rank Fusion and Cross-Encoder re-ranking. |
| **Hallucination** | "LLMs are usually accurate." | Suggests strict prompting and `temperature = 0`. | Proposes an automated RAGAS evaluation pipeline (LLM-as-a-judge) as a regression test. |
| **Data Updates** | Ignores how documents are updated. | Mentions a cron job to re-index everything. | Proposes incremental indexing, deletion handling, and permission-based filtering. |
| **Trade-offs** | Doesn't consider cost or latency. | Notes that more context = higher cost and latency. | Proposes small models for re-ranking and large models for final generation to balance cost and quality. |

---

## How to "Fail" a Candidate (Red Flags)

- **"The Magic Box" Bias:** Treating the LLM as a magic box that "just knows" company information — they don't understand why retrieval is needed.
- **Ignoring Latency:** Proposing a 10-step pipeline without considering that each LLM call adds 1–3 seconds. A strong candidate thinks about where to add caching (e.g., cache embeddings of common queries).
- **No Evaluation Plan:** No idea how to tell if the system is getting better or worse after a change. Without a golden test set and RAGAS-style metrics, every change is a gamble.
- **Re-indexing everything on every update:** Shows lack of systems thinking. At scale, full re-indexing is prohibitively expensive.
