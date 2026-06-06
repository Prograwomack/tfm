# Cryptobot TFM

Sistema de trading algorítmico para criptomonedas desarrollado como Trabajo Fin de Máster en Inteligencia Artificial y Big Data.

El proyecto tiene como objetivo diseñar, desarrollar y evaluar un sistema completo de trading automático orientado a estrategias de scalping sobre el mercado spot de criptomonedas, utilizando datos históricos reales obtenidos desde la API de Binance.

El sistema integra distintas fases del pipeline de ciencia de datos y machine learning:

- Adquisición y almacenamiento de datos financieros
- Feature engineering basado en indicadores técnicos
- Análisis exploratorio de datos (EDA)
- Entrenamiento y evaluación de modelos supervisados
- Backtesting de estrategias
- Aprendizaje por refuerzo para la toma de decisiones
- Ejecución automatizada
- Dashboard de monitorización

El proyecto se desarrolla principalmente en Python y sigue una estructura modular orientada a reproducibilidad y escalabilidad.

# Project Structure

```text
cryptobot-tfm/
│
├── app/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── results/
│
├── scripts/
│
├── src/
│   ├── backtesting/
│   ├── dashboard/
│   ├── data/
│   ├── execution/
│   ├── features/
│   ├── models/
│   └── rl/
│
├── requirements.txt
├── README.md
└── .env
```

# Features

- Historical cryptocurrency market data acquisition from Binance
- Automated feature engineering pipeline with technical indicators
- Support and resistance feature generation
- Exploratory Data Analysis (EDA)
- Supervised machine learning baseline models
- Multi-version strategy backtesting framework
- Reinforcement learning trading agents
- Binance Spot Testnet integration
- Paper trading execution engine
- Real-time market monitoring dashboard
- Live feature generation and model inference
- Trade execution logging and portfolio tracking
- Modular architecture for experimentation and deployment

# Datasets

Current datasets are based on DOGEUSDT 5-minute candles obtained through Binance API.

Examples:

- `DOGEUSDT_5m_binance_2017_2026.csv`
- `DOGEUSDT_5m_binance_2017_2026_features.csv`

# Models

Current baseline models:

- Logistic Regression
- Random Forest
- XGBoost

Serialized using `joblib`.

# Technologies

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- Jupyter Notebook
- Binance API
- pandas-ta
- Streamlit
- Reinforcement Learning frameworks

# Files

## Root Files

| File | Description |
|---|---|
| `.gitignore` | Git exclusion rules for datasets, cache files and environment variables |
| `README.md` | Main project documentation |
| `requirements.txt` | Python dependency list |
| `launch_dashboard.bat` | Windows launcher for the Streamlit dashboard |
| `launch_live_paper_trader.bat` | Windows launcher for the live paper trading service |

## Application Layer

| File | Description |
|---|---|
| `app/streamlit_dashboard.py` | Interactive dashboard for paper trading monitoring and portfolio visualization |

## Research Notebooks

| Notebook | Description |
|---|---|
| `01_data_acquisition_doge_binance.ipynb` | Binance historical DOGEUSDT data acquisition and validation |
| `02_feature_engineering.ipynb` | Technical indicators, support/resistance features and supervised targets |
| `03_eda.ipynb` | Exploratory analysis, volatility diagnostics and feature inspection |
| `04_baseline_ml.ipynb` | Supervised learning baseline models and evaluation |
| `05_backtesting_ml_v1.ipynb` | Initial ML strategy backtesting framework |
| `06_backtesting_ml_v2.ipynb` | Improved ML backtesting logic and execution constraints |
| `07_backtesting_ml_v3.ipynb` | Advanced ML backtesting experiments and strategy stabilization |
| `08_baseline_RL_v1.ipynb` | Initial Q-learning trading agent |
| `09_baseline_RL_v2.ipynb` | RL trading constraints and portfolio simulation improvements |
| `10_baseline_RL_v3.ipynb` | Advanced RL trading logic with volatility and drawdown controls |
| `11_testnet_paper_trading.ipynb` | Binance Spot Testnet execution and paper trading pipeline |

## Scripts

| File | Description |
|---|---|
| `scripts/create_sample_dashboard_logs.py` | Generates synthetic dashboard execution logs |
| `scripts/run_paper_trading.py` | Runs the paper trading execution pipeline |
| `scripts/run_live_paper_trader.py` | Live paper trading loop using real-time market data and model inference |

## Dashboard Modules

| File | Description |
|---|---|
| `src/dashboard/dashboard_data.py` | Dashboard data loading and portfolio metric utilities |
| `src/dashboard/live_market.py` | Real-time market data acquisition and candle processing |
| `src/dashboard/live_features.py` | Online feature engineering for live model inference |
| `src/dashboard/model_registry.py` | Centralized loading and management of trained ML models |

## Execution Modules

| File | Description |
|---|---|
| `src/execution/binance_spot_testnet.py` | Binance Spot Testnet API wrapper |
| `src/execution/paper_broker.py` | Local paper trading simulator |
| `src/execution/execution_logger.py` | Structured JSONL execution logging system |

## Package Initialization

The project uses modular Python package initialization through:

- `src/__init__.py`
- `src/backtesting/__init__.py`
- `src/dashboard/__init__.py`
- `src/data/__init__.py`
- `src/execution/__init__.py`
- `src/features/__init__.py`
- `src/models/__init__.py`
- `src/rl/__init__.py`

Placeholder `.gitkeep` files are used to preserve empty directory structure within Git.

# Disclaimer

This project is developed exclusively for academic and research purposes.

It does not constitute financial advice or investment recommendation.
