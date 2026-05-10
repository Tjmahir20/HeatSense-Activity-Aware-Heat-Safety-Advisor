# HeatSense: Activity Aware Heat Safety Advisor

## Overview

HeatSense is a web application that helps outdoor workers plan safer work shifts in hot conditions. It combines a live weather forecast, a heat-stress index (Wet-Bulb Globe Temperature, WBGT), and short passages from a NIOSH/OSHA knowledge base. The app turns those inputs into an hour-by-hour shift plan with work and rest guidance, hydration reminders, and plain-language notes.

Many heat tools show only a single number or a generic tip. HeatSense ties the forecast to the worker’s job intensity, sun exposure, and shift window. It also grounds written advice in retrieved reference text instead of relying on the model’s memory alone.

This project is submitted as the **final project for CSC 7644: Applied LLM Development** (Louisiana State University).

## Key features and capabilities

- Collects shift details through a browser questionnaire (location, job intensity, sun or shade, shift start and end).
- Resolves locations with **Open-Meteo Geocoding** or accepts **latitude and longitude** typed directly.
- Fetches hourly forecast fields with **Open-Meteo Forecast** and computes **WBGT** with standard-style formulas (wet-bulb, globe, and outdoor WBGT weighting).
- Answers guideline questions over **fixed project data** using **retrieval-augmented generation (RAG)**.
- Uses a **hybrid retriever**: **dense vectors** (OpenAI embeddings + **ChromaDB**) and **lexical search** (**BM25** via `rank-bm25`), then fuses scores before returning context to the model.
- Runs an **LLM agent** with **multiple tools**: one tool returns the WBGT table; another returns retrieved guideline passages. Tool arguments are checked with **JSON Schema** (`jsonschema`) before execution.
- Produces a **structured shift plan** (JSON) validated with **Pydantic** so the UI can render consistent hour cards.
- **Does not** store full conversation transcripts in a database. Request timing is logged with Flask’s logger when a plan is generated.

## Tech stack and architecture (high level)

**LLMs and APIs**

- **OpenAI Chat Completions** (`gpt-4o-mini`) for the two-step agent (tool selection, then final JSON plan).
- **OpenAI Embeddings** (`text-embedding-3-small`) to embed knowledge-base chunks for ChromaDB.
- **Open-Meteo** Geocoding API and Forecast API (no API key required for these public endpoints).

**Libraries and frameworks**

- **Flask** for the web server and routes.
- **Jinja2** templates and static CSS (no separate React or Node front end).
- **ChromaDB** (persistent client) as the **vector store**.
- **rank-bm25** for lexical retrieval.
- **Pydantic** for the output schema (`ShiftPlan`, `HourPlan`).
- **jsonschema** for validating tool arguments.
- **requests** for HTTP calls to Open-Meteo.
- **python-dotenv** for loading environment variables from a `.env` file.
- **numpy** is listed for numeric support used with the weather stack.

**Main components**

- **Back end:** `app.py` handles HTTP, geocoding, and calls the agent. `pipeline/weather.py` computes WBGT from forecast data. `pipeline/rag.py` loads chunks, builds or opens the Chroma index, and runs hybrid retrieval. `pipeline/agent.py` implements the two-turn tool workflow and parses the final plan. `pipeline/schema.py` defines the validated output shape.
- **Front end:** `templates/index.html` and `static/style.css` serve the questionnaire and results in the browser.
- **Vector store and data:** `data/niosh_osha.txt` is the knowledge source, structured as self-contained semantic blocks (one per work intensity category plus hydration, risk classification, heat illness, and supervisor guidance). The chunker in `pipeline/rag.py` splits on `--- BLOCK: ---` markers so each block is retrieved intact. On first use, embeddings are written under `data/chroma_db/` (created automatically).

## Setup instructions

**Prerequisites**

- **Python:** 3.10 or newer recommended (3.11+ tested in development).
- **Operating system:** Windows, macOS, or Linux. Paths in examples use PowerShell on Windows; use your shell’s equivalent where needed.
- **Tools:** `python`, `pip`, and a web browser. A virtual environment (`venv` or `conda`) is recommended but not required.

**Install dependencies**

1. Open a terminal and change into this project folder (the directory that contains `app.py` and `requirements.txt`).

2. (Optional) Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install packages:

   ```text
   pip install -r requirements.txt
   ```

**Configure environment variables**

1. Copy the example env file and rename the copy to `.env`:

   ```powershell
   copy .env.example .env
   ```

   On macOS or Linux:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your OpenAI key:

   ```text
   OPENAI_API_KEY=sk-...your_real_key_here...
   ```

   The first run of RAG will call the Embeddings API once to index `data/niosh_osha.txt`.

## Running the application

**Start the server**

From the project root:

```text
python app.py
```

By default the development server listens at **http://127.0.0.1:5001** (port 5001 avoids clashing with other local Flask apps on 5000).

**How to use it**

1. Open the URL above in a web browser.
2. Complete the questionnaire steps (location, job intensity, sun or shade, shift times).
3. Submit the form. The server geocodes the location, runs the agent (tools plus final JSON), then shows hourly cards with WBGT, work and rest, water, and notes.

Stop the server with `Ctrl+C` in the terminal.

## Repository organization

| Path | Purpose |
|------|---------|
| `app.py` | Flask app: routes, form handling, geocoding, calls `run_agent`, renders templates. |
| `pipeline/` | Core LLM and data logic (import as the `pipeline` package). |
| `pipeline/agent.py` | Two-turn OpenAI agent, tool schemas, JSON Schema validation, shift-plan JSON parsing. |
| `pipeline/rag.py` | Block-aware chunking, embeddings, ChromaDB persistence, BM25 search, hybrid fusion, `retrieve()`. |
| `pipeline/weather.py` | Open-Meteo fetch and WBGT computation for shift hours. |
| `pipeline/schema.py` | Pydantic models for `ShiftPlan` and `HourPlan`. |
| `pipeline/__init__.py` | Package marker. |
| `templates/index.html` | Multi-step questionnaire and results layout. |
| `static/style.css` | Visual styling for the web UI. |
| `data/niosh_osha.txt` | Knowledge-base text for RAG. Structured in `--- BLOCK: ---` sections covering NIOSH REL thresholds by work intensity (Light / Moderate / Heavy / Very Heavy), hydration guidance, risk level classifications, heat illness response, and supervisor planning. High-risk stop-work thresholds follow ACGIH/DOD/NIOSH 2016 guidance. |
| `data/chroma_db/` | Created at runtime: persistent Chroma database files (safe to delete to force a rebuild). |
| `requirements.txt` | Python dependency pins for `pip`. |
| `.env.example` | Template for required environment variables. |
| `evaluation/` | Evaluation artifacts: latency timing results (30 automated queries), human rubric scores (20 shift plans rated on accuracy, clarity, and completeness), and any supporting scripts. |
| `README.md` | This document. |


## Attributions and citations

**Course and student-authored reuse**

Design patterns in `pipeline/agent.py`, `pipeline/rag.py`, are adapted ideas from the author’s **own prior CSC 7644 coursework** (applied LLM modules on OpenAI clients, RAG with hybrid retrieval, and agentic tool calling). Reusing one’s own submitted work is documented here for transparency.

**External services and documentation**

- **OpenAI** API and **OpenAI Python SDK** for chat completions and embeddings. Usage follows current official SDK patterns. See [OpenAI Platform documentation](https://platform.openai.com/docs).
- **Open-Meteo** Geocoding and Forecast APIs. See [Open-Meteo API documentation](https://open-meteo.com/en/docs).

**Open-source libraries**

- **Flask:** [Flask documentation](https://flask.palletsprojects.com/).
- **ChromaDB:** [Chroma documentation](https://docs.trychroma.com/).
- **rank-bm25:** PyPI package `rank-bm25` for Okapi BM25.
- **Pydantic**, **jsonschema**, **requests**, **python-dotenv**, **numpy:** standard library usage per each project’s documentation.

**Scientific and regulatory basis**

Wet-bulb approximations and WBGT composition follow common references used in occupational heat literature, including **Stull (2011)** for wet-bulb estimation, simplified **globe** temperature treatment in the spirit of **Hunter and Minyard (1999)**, and **Yaglou and Minard (1957)** style outdoor WBGT weighting. The text file `data/niosh_osha.txt` is structured as semantic blocks and grounds advice in NIOSH thresholds for acclimatized workers (**NIOSH 2016**) and work-rest guidance from **ACGIH** and **DOD (2007)**. 

This file is **not** a substitute for official agency publications or site-specific occupational health programs.

**References**
Bernard, T. E., & Iheanacho, I. (2015). Heat index and adjusted temperature as surrogates for wet bulb globe temperature to screen for occupational heat stress. Journal of Occupational and Environmental Hygiene, 12(5), 323–333. https://doi.org/10.1080/15459624.2014.989365
Hunter, C. H., & Minyard, C. O. (1999). Estimating wet bulb globe temperature using standard meteorological measurements. Proceedings of the 2nd Conference on Environmental Applications, American Meteorological Society.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459–9474. https://arxiv.org/abs/2005.11401
National Institute for Occupational Safety and Health. (2016). Criteria for a recommended standard: Occupational exposure to heat and hot environments (Publication No. 2016-106). U.S. Department of Health and Human Services, CDC. https://www.cdc.gov/niosh/docs/2016-106/pdfs/2016-106.pdf
Occupational Safety and Health Administration. (2023). Heat illness prevention. U.S. Department of Labor. https://www.osha.gov/heat-exposure
Open-Meteo. (2024). Open-Meteo weather forecast API documentation. https://open-meteo.com/en/docs
OpenAI. (2024). GPT-4o mini model card and API documentation. https://platform.openai.com/docs/models/gpt-4o-mini
Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. Journal of Applied Meteorology and Climatology, 50(11), 2267–2269. https://doi.org/10.1175/JAMC-D-11-0143.1
Yaglou, C. P., & Minard, D. (1957). Control of heat casualties at military training centers. A.M.A. Archives of Industrial Health, 16(4), 302–316.

