# Optimization log

Rules followed: measure first, change one thing at a time, tune on validation only,
keep the test set untouched, fixed seed (42).

## Starting point (measured before any change)
- Persistence baseline, validation: MAE 8.42, RMSE 12.62, category accuracy 61.7%
- LightGBM point model, validation: MAE 7.87, RMSE 11.28, category accuracy 65.3%
- Interval coverage (80% target): 63.9%, mean width 18.2 ug/m3
- Training time: 1.5 s for all three models

## Change 1: conformal calibration of the intervals
- Widened the quantile interval by an offset (2.32 ug/m3) measured on validation.
- Before: coverage 63.9%, width 18.2. After: coverage 80.3%, width 22.9.
- MAE unchanged (7.87). Extra time about 0.1 s. Kept.
- Limit: 80% is by construction on validation; the real check is the test set.
  Coverage varies by city (73% to 87%).

## Change 2: early stopping on validation data
- Each model stops when validation error stops improving (point model used 170 trees, not 800).
- Lighter and faster, less overfitting. Test set never used. Kept.

## Change 3: incremental hourly fetch
- The hourly flow downloads the last 72 hours instead of everything.
- A run takes about 2 minutes versus 20+ minutes for the full fetch. Kept.

## Trade-off
We gained a 6.5% lower MAE than persistence and calibrated intervals at almost no compute cost.
We stopped tuning because further gains were small and the pipeline and RAG agent needed the time.