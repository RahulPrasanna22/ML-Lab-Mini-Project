# Streamflow prediction mini project

This package contains a Streamlit interface for streamflow prediction and the original Python experiment script.

## Requirements

- Windows with Python 3.11 installed (recommended)
- Internet access for installing the packages listed in `requirements.txt`

## Set up a virtual environment (PowerShell)

Open PowerShell in the extracted `streamflow_project` folder, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start the GUI

With the environment activated:

```powershell
streamlit run app.py
```

Or without activation:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser. The app starts with `data/leaf_river_daily.csv`; you can also upload a CSV. The CSV needs `date`, `rain_in`, and `flow_cfs` columns. The interface supports a historical backtest and a future forecast when flow values are unavailable.

## Run the Python experiment

This runs the script's configured ANN experiments and writes updated plots, grid results, and summary files under `outputs/`:

```powershell
python streamflow_ann.py
```

Or without activation:

```powershell
.\.venv\Scripts\python.exe streamflow_ann.py
```

The original script can take several minutes, depending on the computer. The Streamlit app and experiment script are separate ways to run the project.

## Included files

- `app.py`: interactive prediction and statistics interface
- `streamflow_ann.py`: experiment and evaluation script
- `data/leaf_river_daily.csv`: prepared daily flow and precipitation dataset
- `outputs/summary.json`: saved best configuration defaults used by the interface
- `report/streamflow_project_report.pdf`: project report with results and plots
- `requirements.txt`: Python dependencies

The ZIP does not contain a virtual environment; create a fresh `.venv` after extraction using the steps above. Currently, future flow prediction functionality isnt present, but works with back tracking.
