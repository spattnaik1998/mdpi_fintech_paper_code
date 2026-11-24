# Is Finance the Frontier of the Exponential Gap?

**Intangible Capital, Productivity and the Labor Share in U.S. Financial Services, 1997–2024**

Code for a paper prepared for MDPI *FinTech*. The project tests whether the build-up of
intellectual-property-products (IPP) capital (software, R&D and data) in U.S. financial-services
industries has separated productivity from pay and lowered the labour share, whether this is stronger in
finance than in other industries, whether it sped up after 2010, and whether machine learning
can forecast and explain these dynamics. The framing draws on Azeem Azhar's *The Exponential Age* (2021).

## Research questions and hypotheses

| | Hypothesis | Notebook |
|---|---|---|
| H1 | IPP capital in finance grows exponentially (H1a), faster than elsewhere and faster after 2010 (H1b), with a structural break (H1c) | 03 |
| H2 | Higher IPP intensity widens the productivity–pay gap | 03 |
| H3 | Higher IPP intensity lowers the labour share, more strongly in finance (FE, first differences, double ML) | 03 |
| H4 | Tree-based ML forecasts labour-share changes better than linear and naïve benchmarks; IPP intensity is a top predictor (SHAP) | 04 |

## Data (all public, via API)

* **BEA** GDP by Industry (value added, components), NIPA Table 6.5D (FTE employees), Fixed Assets (IPP, equipment and total capital by industry)
* **FRED** rates, credit spread, VIX, NASDAQ, PCE prices, e-commerce share, recessions
* **BLS** finance sub-sector employment and earnings; nonfarm labour share
* **Census** Business Formation Statistics

## Setup

1. Python 3.10 or later. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. API keys go in a `.env` file **outside the repository** (the parent folder is searched automatically) or in the project root, which is git-ignored:
   ```
   BEA_API_KEY=...
   FRED_API_KEY=...
   BLS_API_KEY=...
   CENSUS_API_KEY=...
   ```
   Keys are never printed, logged, cached or written to outputs. Error messages are scrubbed of key values.
3. Run the notebooks in `notebooks/` in order: `00` → `06`.

## Repository layout

```
notebooks/
  00_setup_and_environment.ipynb   package, key and folder checks
  01_data_pipeline.ipynb           API pulls -> data/processed/panel.csv
  02_exploratory_analysis.ipynb    Table 1, Figures 1-7
  03_hypothesis_tests.ipynb        H1-H3, Holm-adjusted summary
  04_ml_models.ipynb               H4: expanding-window ML, DM tests, SHAP
  05_robustness.ipynb              alternative measures/samples, placebo, lead test
  06_paper_outputs.ipynb           paper tables/figures -> paper_mdpi_fintech/ (manuscript not in repo)
src/exgap/
  config.py        paths, study parameters, safe key loading
  api_clients.py   cached, key-safe BEA / BLS / FRED / Census clients
  build_panel.py   table discovery, industry matching, variable construction
  stats.py         panel FE, trend/break tests, Diebold-Mariano, Holm, table/figure export
data/raw/          API response cache (git-ignored)
data/processed/    panel.csv, data dictionary, pipeline log
results/tables/    CSV + LaTeX tables
results/figures/   PNG (300 dpi) + PDF figures
```

## Reproducibility notes

* BEA tables are located by description at run time; the choices are logged in `data/processed/pipeline_log.txt`.
* Re-running uses the cached responses in `data/raw/`; set `REFRESH = True` in notebook 01 to pull a new data vintage, and record the run date in the paper.
* Random seeds are fixed (`RANDOM_STATE = 42` in `config.py`).
