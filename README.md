# M5 Forecasting – Data Science Forecasting System

## What this project does
- Loads M5 retail dataset (sales + calendar + prices)
- Builds advanced forecasting features (lags, rolling stats, price, SNAP, events)
- Compares models: Naive baseline vs Prophet vs XGBoost
- Uses walk-forward validation (28-day horizon, multiple folds) to avoid leakage
- Automatically selects best model per series and stores versioned artifacts
- Provides a FastAPI inference API with caching and metrics endpoint

## Run training
```bash
python -m src.train