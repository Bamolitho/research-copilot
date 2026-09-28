# Research Copilot

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![UV](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv\&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/Ruff-linter%20%26%20formatter-D7FF64?logo=ruff\&logoColor=black)](https://docs.astral.sh/ruff/)
[![MyPy](https://img.shields.io/badge/MyPy-type%20checking-2A6DB2?logo=python\&logoColor=white)](https://mypy.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Pytest-testing-0A9EDC?logo=pytest\&logoColor=white)](https://docs.pytest.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent%20orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-persistence-336791?logo=postgresql\&logoColor=white)](https://www.postgresql.org/)

**Research Copilot** is an AI-powered research and technology-watch assistant designed to help users retrieve, analyze, verify, and synthesize scientific and technical information.

The project started as a fully hardcoded proof of concept built manually with Python. The current version aims to transform that prototype into a modular, extensible agentic system built around graph-based orchestration, persistent storage, multiple research tools, and an evaluation and refinement loop.

> **Work in progress:** I am currently migrating the project to a new, more extensible architecture designed to make it easier to add new features, tools, agents, and evaluation components without having to redesign the existing workflow.

The new version will also introduce an **LLM Judge** responsible for evaluating the quality of generated answers across multiple dimensions such as relevance, factuality, faithfulness to sources, completeness, and citation quality. Failed evaluations will be able to trigger a feedback loop to improve the research and generation process.

## Overview

Research Copilot is an AI research assistant built with **LangGraph** and **LangChain**.

The current workflow allows an agent to:

1. Understand the user's question.
2. Plan a research query.
3. Search scientific literature on **arXiv**.
4. Collect structured research sources.
5. Generate a final answer grounded in the retrieved sources.

The project is designed around a graph-based architecture so that additional tools, agents, evaluation steps, and research sources can be added progressively.

## Architecture

```text
The complete architecture will be added soon
```


## Current Components

### Planner

The planner uses an LLM with access to the arXiv search tool.

Its responsibility is to determine whether research is required and generate an appropriate search query.

The planner does not generate the final response.

### arXiv Search

The project currently uses the arXiv API to retrieve scientific papers.

Each result is converted into a structured `Source` object containing:

* title
* authors
* abstract
* arXiv ID
* URL

The number of retrieved papers and abstract length are limited to keep the context manageable.

### Tool Node

A custom LangGraph tool node executes the planner's tool calls.

It:

* executes `arxiv_search`
* stores the returned `Source` objects in `state["sources"]`
* returns a lightweight `ToolMessage`

Source serialization is deliberately handled later by the answer generator instead of duplicating the same information inside the tool message.

### Answer Generator

The generator receives:

* the original question
* the retrieved research sources

It is explicitly instructed to:

* use only the provided sources
* avoid unsupported claims
* use only provided arXiv IDs for citations
* remain concise
* return a Markdown answer

The generator therefore operates on structured research data rather than relying on the planner's tool-call messages as its source of context.

## Source-Grounded Generation

The final answer is generated from an explicit `sources_context` built from the structured sources:

```text
Question
+
Research sources
    ├── Title
    ├── Authors
    ├── Abstract
    ├── arXiv ID
    └── URL
```

This prevents the model from inventing citations or relying on information that was not retrieved during the research step.

## LLM Judge

The next major component is a structured **LLM Judge**.

Instead of returning a single subjective score, the judge evaluates several independent criteria:

| Criterion        | Weight |
| ---------------- | -----: |
| Relevance        |    20% |
| Factuality       |    30% |
| Faithfulness     |    30% |
| Completeness     |    15% |
| Citation quality |     5% |

Each criterion receives a score between `0.0` and `1.0`.

The global score is calculated deterministically in Python:

```text
global_score =
    relevance × 0.20
  + factuality × 0.30
  + faithfulness × 0.30
  + completeness × 0.15
  + citation_quality × 0.05
```

The current target threshold is:

```text
0.80
```

The LLM provides the qualitative evaluation, while Python is responsible for calculating the final score and determining whether the answer passes.

### Judge Feedback

The judge also produces:

* identified issues
* missing information
* unsupported claims
* actionable feedback for the planner
* a suggested next action

Possible actions include:

```text
accept
new_search
refine_search
regenerate
```

This makes the evaluation useful not only as a quality gate, but also as feedback for future iterations.


## Development

The project uses **uv** for Python environment and dependency management.

### Run the agent

```bash
make run
```

Equivalent to:

```bash
uv run python -m scripts.one_agent
```

### Run tests

```bash
make test
```

### Type checking

```bash
make mypy
```

### Linting

```bash
make lint
```

Automatically fix linting issues with:

```bash
make lint-fix
```

### Formatting

Format the project with:

```bash
make format
```

Check formatting without modifying files:

```bash
make format-check
```

### Run all checks

```bash
make check
```

The `check` target performs:

```text
lint
mypy
format-check
tests
```

## Development Workflow

For local development:

```bash
make lint-fix
make format
make check
make run
```

The goal is to keep the codebase automatically formatted, linted, type-safe, and tested before running or committing changes.

## Roadmap

### Current

* [x] LangGraph-based workflow
* [x] LLM planner
* [x] arXiv research tool
* [x] Structured research sources
* [x] Custom tool execution node
* [x] Source-grounded answer generation
* [x] Automated linting
* [x] Automated formatting
* [x] MyPy type checking
* [x] Makefile development commands

### Next

* [ ] LLM Judge
* [ ] Multi-criteria answer evaluation
* [ ] Deterministic weighted scoring
* [ ] Judge feedback loop
* [ ] Iterative research and regeneration
* [ ] Test workflow
* [ ] Knowledge base integration
* [ ] Additional web research tools
* [ ] Web interface
* [ ] Additional agents and research tools

## Design Goals

The new architecture is being built around four main principles:

**Extensibility**
New tools, agents, sources, and evaluation components should be easy to integrate.

**Modularity**
Each component should have a clearly defined responsibility.

**Source grounding**
Generated answers should be based on retrieved research rather than unsupported model knowledge.

**Evaluability**
Answer quality should be measurable through explicit criteria rather than relying only on subjective inspection.

The long-term goal is to turn the current research workflow into a modular research system where new capabilities can be added without fundamentally changing the existing architecture.
