import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

report = pd.read_csv('reports/model_comparison.csv')

report['series_short'] = report['series_id'].str.replace('_validation', '').str[-12:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Walmart M5 Demand Forecasting — Model Comparison', fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0]
x = range(len(report))
width = 0.35
bars1 = ax1.bar([i - width/2 for i in x], report['naive_cv_mape'], width, label='Naive Baseline', color='#d0d0d0')
bars2 = ax1.bar([i + width/2 for i in x], report['xgb_cv_mape'], width, label='XGBoost', color='#4472C4')
ax1.set_xlabel('Series')
ax1.set_ylabel('CV MAPE (lower is better)')
ax1.set_title('MAPE: Naive vs XGBoost per Series')
ax1.set_xticks(list(x))
ax1.set_xticklabels(report['series_short'], rotation=45, ha='right', fontsize=7)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
report['mape_reduction_pct'] = ((report['naive_cv_mape'] - report['xgb_cv_mape']) / report['naive_cv_mape'] * 100).round(1)
colors = ['#4472C4' if v > 0 else '#E05C5C' for v in report['mape_reduction_pct']]
ax2.barh(report['series_short'], report['mape_reduction_pct'], color=colors)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel('MAPE Reduction % (positive = XGBoost wins)')
ax2.set_title('XGBoost Improvement over Naive Baseline')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/forecast_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to reports/forecast_comparison.png")