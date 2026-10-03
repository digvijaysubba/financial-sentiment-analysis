# Financial Sentiment Analysis Pipeline

This project builds an end-to-end NLP pipeline to analyze earnings call transcripts.  
It ingests raw transcripts, cleans and structures them into Prepared Remarks and Q&A sections, computes sentiment using multiple models, and generates structured outputs and visualizations.

---

## Project Overview

The pipeline performs the following steps:

1. Ingest raw `.txt` earnings transcripts
2. Store transcripts in a SQLite database
3. Clean and split text into:
   - Prepared Remarks
   - Q&A
4. Compute sentiment using:
   - VADER
   - TextBlob
5. Generate:
   - A final structured dataset (`outputs/results.csv`)
   - Sentiment comparison plots (`outputs/plots/`)

---

## Project Structure
Financial sentiment/
├── data/
│ └── raw/ # Raw earnings transcripts (.txt)
├── db/
│ └── pipeline.db # SQLite database (auto-generated)
├── outputs/
│ ├── results.csv # Final structured dataset
│ └── plots/ # Generated visualizations
├── scripts/
│ ├── ingest.py
│ ├── clean_text.py
│ ├── sentiment_models.py
│ ├── build_results.py
│ ├── make_plots.py
│ ├── summary_report.py
│ ├── fetch_prices.py
│ ├── compute_returns.py
│ ├── quant_analysis.py
│ └── export_powerbi.py
├── api/
│ └── main.py # FastAPI REST layer
├── tests/
│ └── test_api.py
├── docs/
│ └── powerbi_dashboard.md # Power BI build guide
├── requirements.txt
└── README.md

## Data

Raw transcripts are **not included** in this repo because they are third-party copyrighted text.
To run the pipeline, place earnings-call transcripts in `data/raw/` named `TICKER_YYYY-MM-DD.txt`
(e.g. `AAPL_2025-07-31.txt`, where the date is the earnings call date). The analysis here used
20 calls from AAPL, MSFT, GOOGL, NVDA and TSLA (2025–2026). Price data in `data/prices/` is
included and can be refreshed with `scripts/fetch_prices.py`.

## Setup Instructions

## 1️ Navigate to the project root

```powershell
cd "D:\projects\Financial sentiment"
```

## 2️. Create and activate virtual environment (if not already created)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

## 3️. Install dependencies
pip install -r requirements.txt
Pipeline Run Order (Start-to-Finish)

## Run the following commands from the project root:

python scripts/ingest.py
python scripts/clean_text.py
python scripts/sentiment_models.py
python scripts/build_results.py
python scripts/make_plots.py
python scripts/summary_report.py
python scripts/fetch_prices.py
python scripts/compute_returns.py
python scripts/quant_analysis.py
python scripts/export_powerbi.py
Outputs

## After running the full pipeline:

📄 outputs/results.csv

Contains one row per:

Company

Report Date

Columns include:

vader_prepared

vader_qa

vader_shift

textblob_prepared

textblob_qa

textblob_shift

## Where:

shift = Q&A sentiment − Prepared sentiment

Interpretation:

Negative shift → Q&A tone is more negative than prepared remarks

Positive shift → Q&A tone is more positive than prepared remarks

📊 outputs/plots/

## Generated visualizations include:

Prepared vs Q&A comparisons

Tone shift plots

Model agreement comparisons

## REST API

Serves pipeline outputs over HTTP (read-only against `db/pipeline.db`).

```powershell
uvicorn api.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service and data availability |
| GET | `/companies` | Tickers with transcript counts and date range |
| GET | `/transcripts?company=` | Transcript metadata and section lengths |
| GET | `/transcripts/{id}?include_text=` | One transcript with its sentiment scores |
| GET | `/sentiment?company=&model=&section=` | Per-section scores (vader/textblob × prepared/qa) |
| GET | `/results?company=` | Per-call tone shift and post-earnings returns |
| GET | `/summary` | Per-company averages |
| GET | `/correlations?model=` | Tone shift vs return correlations |
| POST | `/analyze` | Score arbitrary text: `{"text": "..."}` |

Run tests: `python -m pytest tests`

## Power BI Dashboard

See [docs/powerbi_dashboard.md](docs/powerbi_dashboard.md) for the 3-page dashboard
(KPI cards, sentiment-vs-return scatter plots, slicers, drill-down) built on
`outputs/financial_sentiment_powerbi.csv`.

## Tech Stack

Python

SQLite

pandas

numpy

matplotlib

vaderSentiment

TextBlob

scipy, yfinance

FastAPI, uvicorn, pytest

Power BI

## Purpose

This project demonstrates:

1. Data ingestion and cleaning

2. SQL-backed pipelines

3. Sentiment modeling and experimentation

4. Reproducible analytics workflows

5. Structured output generation

6. Visualization of model differences

7. Designed as a real-world data science workflow for messy, unstructured text data.