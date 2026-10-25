# Urban Heat Analysis Across Indian Cities

Daily land surface temperature (LST), NO₂ and wind for the urban and rural areas of **Delhi** and **Bangalore**
(1 Jan 2020 – 30 Dec 2025), with a **Map-First Explorer Dashboard** built in Streamlit.

## Run it on Windows with VS Code

1. Install Python 3.10+ from python.org (tick **Add Python to PATH**). Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Terminal > New Terminal) and create an environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Prepare data and models (only the first time, about 3 minutes):
   ```
   python 01_clean_data.py
   python 02_eda.py
   python 03_train_models.py
   ```
4. Start the dashboard:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501. The map needs internet for its background tiles.
   For an offline demo: `set UHI_MAP_STYLE=offline` before `streamlit run app.py` (markers only, plain background).

Dark theme is the default. For the earlier light look: `set UHI_THEME=light` before `streamlit run app.py`.

## Pages

| Page | What it does |
|---|---|
| Overview | Hero, clickable city map, urban-vs-rural "warming stripes", gap chart |
| Explore Map | Map + filters (city, variable, dates, zones, seasons), trend chart, year clock, heat calendar, NO₂/wind, CSV download, collapsible table |
| Compare Cities | Averages, extremes, trends, seasons, urban–rural gap, comparison table, coverage notes |
| ML Insights | Six-model comparison, accuracy plots, feature importance, prediction tool |
| About | Problem, data, cleaning, models, limitations |

## Backend (written in the style of our practicals)

| File | Steps (practical it follows) |
|---|---|
| `01_clean_data.py` | `read_csv`, `head`, `shape`, `info`, `describe`, `isnull`, `fillna`, `drop` (Practicals 1 and 3); flags 28 outlier LST values |
| `02_eda.py` | `value_counts`, `histplot`, `barplot`, correlation heatmap (Practicals 1 and 3) |
| `03_train_models.py` | Part 1: `LabelEncoder`, `train_test_split`, `LinearRegression`, MAE/MSE/RMSE/R², residual plot, coefficients, `GridSearchCV` (Practical 3). Part 2: `StandardScaler`, `KNeighborsRegressor` (Practical 4). Part 3: comparison of six models (adds Decision Tree, Random Forest, SVR, Gradient Boosting) |

The best model (highest cross-validated R²) is saved and used by the website. Earlier, more complex versions of the code are kept in `archive_old/` (not used).

## Honest notes

- The data contain only two cities and no coordinates. Map markers use approximate city-centre points from `uhi/config.py`;
  this is a city-level map, not a satellite heat map.
- 28 LST values are flagged as outliers (never altered). The dashboard blanks them by default; a toggle shows them.
- The prediction is an estimate, not a measurement.

## Project layout

```
app.py                 dashboard entry point (top navigation)
archive_old/           earlier versions (not used)
views/                 one file per page
dashboard/             theme, charts, analysis helpers, shared Streamlit helpers
uhi/                   data cleaning, features, modelling (unchanged)
01_clean_data.py 02_eda.py 03_train_models.py
tests/                 python -m pytest
```

## Tests

`python -m pytest` runs the original tests plus `tests/test_dashboard.py`.
