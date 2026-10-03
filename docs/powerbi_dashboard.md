# Power BI Dashboard — Build Guide

Builds a 3-page interactive dashboard from `outputs/financial_sentiment_powerbi.csv`
(one row per earnings call). Save the result as `dashboard/financial_sentiment.pbix`.

## 0. Refresh the data

```powershell
python scripts/compute_returns.py
python scripts/export_powerbi.py
```

## 1. Load and model

1. Power BI Desktop → **Get data → Text/CSV** → `outputs/financial_sentiment_powerbi.csv` → **Transform Data**.
2. In Power Query set types:
   - `date` → Date
   - `sentiment_score`, `*_shift`, `*_prepared`, `*_qa` → Decimal number
   - `gap_ret`, `return_1d`, `return_3d`, `return_5d` → **Percentage**
   - `transcript_id` → Whole number
3. Rename the query to `Calls` → **Close & Apply**.
4. In Model view, select `date` → **Mark as date table** is *not* needed; just ensure the
   date hierarchy is on (Year → Quarter → Month).
5. Create a hierarchy for drill-down: right-click `sector` → **Create hierarchy** →
   add `company`, then `quarter`. Name it `Sector Hierarchy`.

## 2. DAX measures

**Home → New measure**, one per line:

```DAX
Calls Analysed = COUNTROWS ( Calls )

Avg Tone Shift = AVERAGE ( Calls[sentiment_score] )

Avg VADER Shift = AVERAGE ( Calls[vader_shift] )

Avg TextBlob Shift = AVERAGE ( Calls[textblob_shift] )

Avg Gap Return = AVERAGE ( Calls[gap_ret] )

Avg 1D Return = AVERAGE ( Calls[return_1d] )

Avg 5D Return = AVERAGE ( Calls[return_5d] )

% Negative Shift =
DIVIDE (
    COUNTROWS ( FILTER ( Calls, Calls[sentiment_score] < 0 ) ),
    [Calls Analysed]
)

Model Agreement =
DIVIDE (
    COUNTROWS ( FILTER ( Calls, SIGN ( Calls[vader_shift] ) = SIGN ( Calls[textblob_shift] ) ) ),
    [Calls Analysed]
)

Shift–Return Correlation (1D) =
VAR x = AVERAGE ( Calls[sentiment_score] )
VAR y = AVERAGE ( Calls[return_1d] )
VAR cov = SUMX ( Calls, ( Calls[sentiment_score] - x ) * ( Calls[return_1d] - y ) )
VAR sx  = SQRT ( SUMX ( Calls, ( Calls[sentiment_score] - x ) ^ 2 ) )
VAR sy  = SQRT ( SUMX ( Calls, ( Calls[return_1d] - y ) ^ 2 ) )
RETURN DIVIDE ( cov, sx * sy )
```

Format the return and `%` measures as Percentage (1 decimal); shifts as Decimal (3 places).

## 3. Pages

Put the same three **slicers** on every page — `company` (dropdown, multi-select),
`sector` (tiles), `quarter` (dropdown) — then **View → Sync slicers** to sync them across pages.

### Page 1 — Overview

| Visual | Fields |
|---|---|
| 5 × Card (top row) | `Calls Analysed`, `Avg Tone Shift`, `Avg 1D Return`, `% Negative Shift`, `Model Agreement` |
| Clustered bar | Y: `company` · X: `Avg VADER Shift`, `Avg TextBlob Shift` |
| Clustered column | X: `company` · Y: `Avg Gap Return`, `Avg 1D Return`, `Avg 5D Return` |

Conditional formatting on the bar chart: **Data colors → fx → Rules**, red if < 0, green if ≥ 0.

### Page 2 — Sentiment vs Return

| Visual | Fields |
|---|---|
| Scatter chart (large) | X: `sentiment_score` · Y: `return_1d` · Values/Details: `transcript_id` · Legend: `company` · Tooltips: `date`, `quarter` |
| Analytics pane on scatter | Add **Trend line** |
| Scatter chart | X: `vader_shift` · Y: `gap_ret` · Details: `transcript_id` · Legend: `sector` |
| Card | `Shift–Return Correlation (1D)` |
| Text box | "n = 20 calls — correlations are illustrative, not statistically significant." |

Optional: **Modeling → New parameter → Fields** with `gap_ret`, `return_1d`, `return_3d`,
`return_5d`; put the parameter on the large scatter's Y axis and add it as a slicer so viewers
can switch return horizons.

### Page 3 — Company Drill-down

| Visual | Fields |
|---|---|
| Matrix | Rows: `Sector Hierarchy` (sector → company → quarter) · Values: `Avg VADER Shift`, `Avg TextBlob Shift`, `Avg 1D Return` — turn on drill-down arrows |
| Line chart | X: `date` · Y: `vader_prepared`, `vader_qa` (Average) · shows prepared vs Q&A tone over time |
| Line chart | X: `date` · Y: `textblob_prepared`, `textblob_qa` (Average) |
| Table | `company`, `date`, `quarter`, `sentiment_score`, `gap_ret`, `return_1d`, `return_5d` |

Enable **drill-through**: on this page, drag `company` into the *Drill through* well. Users can
then right-click a company on page 1 or 2 → *Drill through → Company Drill-down*.

## 4. Publish

1. Save as `dashboard/financial_sentiment.pbix`.
2. **Home → Publish** → choose *My workspace* (free Power BI account works).
3. In the Power BI service: **File → Embed report → Publish to web (public)** if you want a
   link for your resume/portfolio (only if your tenant allows it), otherwise share screenshots.
4. Export screenshots of each page to `dashboard/screenshots/` for the README.
