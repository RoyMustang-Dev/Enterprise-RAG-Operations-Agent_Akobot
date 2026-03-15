# Business Analytics Dummy Datasets

As requested, I have created high-quality, realistic synthetic datasets strictly tailored to the `BusinessAnalyticsAgent`'s expected inputs (Time-Series Forecasting, Regional Grouping, Categorical Metrics, and KPI Generation).

I have already saved these files directly into the backend under:
- `data/synthetic_tests/global_sales_2023.csv`
- `data/synthetic_tests/app_engagement_metrics.csv`

---

## 1. Global Sales & Revenue 2023
**File:** `data/synthetic_tests/global_sales_2023.csv`
**Purpose:** Tests the agent's ability to natively group regions, categorize products, calculate rolling revenues, and use XGBoost to forecast multi-variate panel data.

**Sample Schema:**
| Date | Region | Product_Category | units_sold | Revenue_USD | Marketing_Spend | Active_Users |
|------|--------|-----------------|------------|-------------|-----------------|--------------|
| 2023-01-01 | North America | Enterprise License | 1269 | 190350 | 48671 | 16503 |
| 2023-01-01 | Europe | Cloud Storage | 536 | 10720 | 1799 | 4363 |
| ... | ... | ... | ... | ... | ... | ... |

**Why it works well:**
- It has clear **"Region"** and **"Product_Category"** string labels, which triggers the Analytics supervisor's deterministic XGBoost logic exactly as programmed.
- It contains built-in seasonality (e.g., European summer dips, steady NA growth), giving the Agent something real to "discover" in its Executive Summary paragraph.

---

## 2. App Engagement & Churn Metrics
**File:** `data/synthetic_tests/app_engagement_metrics.csv`
**Purpose:** Tests the agent's ability to detect statistical anomalies, calculate correlations (e.g., Outages affecting Crash Rates), and perform univariate Prophet time-series forecasting.

**Sample Schema:**
| Date | Platform | Daily_Active_Users | Session_Duration_Min | Crash_Rate |
|------|----------|-------------------|----------------------|------------|
| 2023-01-01 | iOS | 7029 | 9.7 | 0.046 |
| 2023-01-02 | Android | 11578 | 10.3 | 0.021 |
| ... | ... | ... | ... | ... |

**Why it works well:**
- Strong Weekly Seasonality: Usage drops heavily on weekends.
- Injected Anomalies: There are simulated marketing spikes and a simulated system outage on Day 180 (causing a crash rate spike). The LLM's `StatisticalTest` and `Risk_Alerts` nodes should pick these up cleanly.

---

## How to Test Them Directly

You can test these via the frontend UI by uploading them to the chat, or via this direct `cURL` command hitting the decoupled Analytics pipeline:

```bash
curl -X POST "http://localhost:8000/api/v1/business_analyst/chat" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -H "x-tenant-id: default" \
  -F "query=Give me an executive summary of our sales performance. Provide a 30-day forecast for the top 3 regions." \
  -F "files=@data/synthetic_tests/global_sales_2023.csv" \
  -F "session_id=synthetic-test"
```

The endpoint will return a massive deterministic Pydantic JSON chunk containing `kpi_cards`, `forecast_table`, and `statistical_tests`.
