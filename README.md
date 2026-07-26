## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# after downloading the raw UCI file into data/raw/
python src/data_pipeline.py --raw_path data/raw/LD2011_2014.txt
```

## Status

- [x] Repo scaffolding
- [x] Data pipeline (load, resample to hourly, client filtering, chronological split)
- [x] Metrics module (point + probabilistic)
- [x] Seasonal naive + LightGBM direct multi-horizon baselines (residual-based, regularized, early-stopped)
- [ ] LSTM seq2seq baseline
- [ ] TFT training pipeline
- [ ] Attention-weight interpretability notebook
- [ ] Forecast-drift monitoring
- [ ] FastAPI serving + Docker
- [ ] Streamlit fan-chart demo
- [ ] CI (GitHub Actions)

## Results

### Baselines (test set, 24h horizon)

| Model | MAE | RMSE | SMAPE |
|---|---|---|---|
| Seasonal naive | 247.59 | **1658.38** | 13.74 |
| LightGBM (residual, regularized) | **225.66** | 2475.55 | 14.04 |

**Finding:** seasonal naive is a genuinely strong baseline here due to this
dataset's extreme weekly periodicity -- a naive forecast that simply copies
last week's value at the same hour is hard to beat. A LightGBM model trained
to predict the *residual* against that naive baseline (rather than raw load)
improves typical-case accuracy (better MAE) but is less robust to outlier
events, reflected in a worse RMSE even after regularization and early
stopping to rule out simple overfitting. This points to a real modeling gap
-- most likely missing holiday/anomaly signal and the difficulty of a single
model generalizing across 314 clients with very different consumption
scales -- rather than a tuning artifact. This sets a concrete bar for the
deep learning models below: can LSTM/TFT match LightGBM's typical-case gains
without the robustness trade-off?

### LSTM / TFT

_(populated once trained -- MAE/RMSE/SMAPE per horizon step, pinball loss,
p10-p90 coverage, and full model comparison table will go here)_