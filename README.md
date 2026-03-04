# Walmart M5 Demand Forecasting

Demand forecasting for the M5 (Makridakis 5) competition dataset: daily unit sales of retail products across US stores, with calendar, price, and event information. The project trains per-series models, compares a naive last-value baseline to XGBoost, and uses walk-forward cross-validation to pick the best model and report metrics.

---

## Description

- **Data:** M5 sales (wide format), calendar (dates, events, SNAP), and sell prices. Series are store–item combinations (e.g. one product in one state).
- **Features:** Calendar (weekday, month, year), events and SNAP flags, price and price-change, and lag/rolling features (lags 1–28, rolling means and standard deviations) built without future leakage.
- **Models:**  
  - **Naive:** last observed value repeated over the forecast horizon.  
  - **XGBoost:** regression on the feature set above, tuned for a strong baseline.
- **Evaluation:** Walk-forward CV with 28-day horizon and 3 folds; best model per series is chosen by mean MAPE. Trained XGBoost models are saved in a versioned registry for inference.
- **Notebook:** Forecast plot (last 28 days: actual vs predicted) and a diagnostic view (raw sales, stats, and suggestions for noisy or zero-heavy series).

---

## Results

Training was run on the top 20 series by total sales. Cross-validation used 3 folds per series; XGBoost was selected as best for every series.

| Metric | Naive (CV MAPE) | XGBoost (CV MAPE) |
|--------|------------------|-------------------|
| Best series (FOODS_3_586_CA_3) | 0.43 | **0.13** |
| Typical range | 0.30–1.10 | 0.13–0.43 |
| Worst (FOODS_3_541_CA_3) | 1.26 | 1.24 |

Naive baseline MAPE is roughly 30–130% on these series; XGBoost reduces it to about 13–43% on most, with a large gain on the best series. The worst series remains hard for both (high variance / intermittent demand).

Summary report: `reports/model_comparison.csv` (per-series CV MAPE/RMSE, best model, saved version and path).

---

## Plots 

**1. Forecast: last 28 days — actual vs predicted**  
Trained model for the best-performing series; predicted vs actual sales over the last 28 days of the series.

![Forecast: last 28 days](reports/forecast_FOODS_3_586_CA_3_validation.png)

**2. Diagnose: raw sales for a problem series**  
Full history for a high-MAPE series (noisy / many zeros); used to decide on fallbacks or extra features.

![Diagnose: raw sales](reports/diagnose_FOODS_3_541_CA_3_validation.png)

*(If the images above don’t show, generate them from the notebook; it writes these files into `reports/`.)*
