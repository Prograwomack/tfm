# Cryptobot TFM

Sistema de trading algorítmico para criptomonedas desarrollado como Trabajo Fin de Máster en Inteligencia Artificial y Big Data.

El proyecto tiene como objetivo diseñar, desarrollar y evaluar un sistema completo de trading automático orientado a estrategias de scalping sobre el mercado spot de criptomonedas, utilizando datos históricos reales obtenidos desde la API de Binance.

El sistema integra distintas fases del pipeline de ciencia de datos y machine learning:

- adquisición y almacenamiento de datos financieros,
- feature engineering basado en indicadores técnicos,
- análisis exploratorio de datos (EDA),
- entrenamiento y evaluación de modelos supervisados,
- backtesting de estrategias,
- futura integración de aprendizaje por refuerzo,
- ejecución automatizada,
- y dashboard de monitorización.

El proyecto se desarrolla principalmente en Python y sigue una estructura modular orientada a reproducibilidad y escalabilidad.

---

# Project Structure

```text
cryptobot-tfm/
│
├── app/                    # Future dashboard / app layer
├── data/
│   ├── raw/                # Raw Binance market data
│   └── processed/          # Engineered datasets
│
├── models/                 # Trained ML models
├── notebooks/              # Development and experimentation notebooks
├── results/                # Backtesting and evaluation outputs
│
├── src/
│   ├── backtesting/        # Backtesting logic
│   ├── dashboard/          # Monitoring dashboard
│   ├── data/               # Data utilities
│   ├── execution/          # Trading execution modules
│   ├── features/           # Feature engineering
│   ├── models/             # ML model logic
│   └── rl/                 # Reinforcement learning agents
│
├── requirements.txt
├── README.md
└── .env
```

---

# Current Development Status

## Completed

- Historical DOGEUSDT market data acquisition from Binance
- Feature engineering pipeline
- Exploratory Data Analysis (EDA)
- Baseline supervised machine learning models
- Initial backtesting framework

## In Progress

- Strategy refinement and advanced feature engineering
- Reinforcement learning experimentation
- Trading execution pipeline
- Monitoring dashboard

---

# Datasets

Current datasets are based on DOGEUSDT 5-minute candles obtained through Binance API.

Examples:

- `DOGEUSDT_5m_binance_2017_2026.csv`
- `DOGEUSDT_5m_binance_2017_2026_features.csv`

---

# Models

Current baseline models:

- Logistic Regression
- Random Forest
- XGBoost

Serialized using `joblib`.

---

# Technologies

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- Jupyter Notebook
- Binance API
- pandas-ta
- Streamlit (planned)
- Reinforcement Learning frameworks (planned)

---

# Disclaimer

This project is developed exclusively for academic and research purposes.

It does not constitute financial advice or investment recommendation.