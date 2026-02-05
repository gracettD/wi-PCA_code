# Factor Analysis for Large Non-Stationary Panels with Endogenous Missingness and Applications to Causal Inference

This repository contains the replication code for the paper **“Factor Analysis for Large Non-Stationary Panels with Endogenous Missingness and Applications to Causal Inference”**.

## 1. Overview

The main estimator implementations are in `src/Estimator.py`. Scripts that reproduce the paper’s results are:

- **Table 2**: `src/Simulation_Table2.py` (relative MSE of common components for different estimators)
- **Table 3**: `src/Empirical_Table3.py` (synthetic treatment-pattern results; outcome unit: $100/store)
- **Table 4**: `src/Empirical_Table4.py` (ATT estimates for treated states; outcome unit: $100/store)
- **Figure 3**: `src/Empirical_Figure3.py` (observed and estimated control series of per-store beer sales)

Supporting scripts:

- `src/ATT_estimate.py`: computes ATT, bias, and RMSE (used in Table 3)
- `src/data_processing.py`: constructs analysis datasets from the licensed raw data (see Section 2)

## 2. Data availability and provenance

### Licensed data source

The empirical analysis uses **NielsenIQ retail scanner data** provided through the NielsenIQ Datasets at the Kilts Center for Marketing, University of Chicago Booth School of Business. These data are commercial/licensed and cannot be redistributed as part of this replication package.

### How to obtain the licensed data

Researchers with access can obtain the data as follows:

1. Log into the Kilts Center for Marketing Data Center portal: `https://marketingdata.chicagobooth.edu/Login/ResearcherLoginpage`.
2. Download weekly NielsenIQ retail scanner sales data from January 1, 2017 to December 26, 2020.
3. Place the raw extracts into the folder structure expected by `src/data_processing.py`.

### Data construction

Run `src/data_processing.py` to construct the analysis panels used in the empirical analysis. The processing steps include:

- aggregating the target products,
- aggregating beer sales to the state-week level,
- dropping DC and the states AK, HI, CO, WA, OR,
- constructing the outcome panel and treatment panel used in the analysis.

The script outputs:

- `data/beer_sales.csv`: state-by-week outcome panel (beer sales per store)
- `data/treatment.csv`: state-by-week treatment indicator

### Data included in this replication package

To allow users to run the code end-to-end without access to the licensed data, this repository includes:

- `data/beer_sales_contaminated.csv`: a perturbed version of the constructed state-by-week outcome panel with additive Gaussian noise (generated in `src/data_processing.py`)
- `data/treatment.csv`: the constructed treatment panel

**Important note:** `data/beer_sales_contaminated.csv` is provided as a computational example and will generally not reproduce the exact empirical results reported in the paper. Users with access to the licensed data should run `data/data_processing.py` to generate the non-perturbed `data/beer_sales.csv` and then rerun the empirical scripts.

## 3. Running the code

### Software requirements

Install the required Python packages (see `requirements.txt`).

### Reproducing tables and figures

Run the following scripts from the repository root:

- **Table 2**: `python3 src/Simulation_Table2.py`
- **Table 3**: `python3 src/Empirical_Table3.py`
- **Table 4**: `python3 src/Empirical_Table4.py`
- **Figure 3**: `python3 src/Empirical_Figure3.py`

If you do not have access to the licensed data, you can run the empirical scripts using the included perturbed dataset by creating `data/beer_sales.csv` from the contaminated file (so the scripts find `data/beer_sales.csv` without modification), e.g.:

```bash
cp beer_sales_contaminated.csv beer_sales.csv
```

Approximate run-time on a MacBook Pro (Apple M4 Pro, 24 GB memory):

- `src/Simulation_Table2.py`: ~90 minutes
- `src/Empirical_Table3.py`: ~20 minutes
- `src/Empirical_Table4.py`: ~70 hours (with the vast majority of the time spent on variance estimation for the TWFE estimator. If TWFE variance estimation is approximated using wi-PCA with zero factor, the run-time drops to ~50 minutes.)
- `src/Empirical_Figure3.py`: <1 minute
