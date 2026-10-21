# Baseline declaration

Declared before any model was trained or any test result was seen.

- Target: next-day mean PM2.5 (ug/m3) per city (Lahore, Karachi, Islamabad, Peshawar).
- Day = Pakistan time (UTC+5). A day needs at least 18 hourly readings.
- Baseline: persistence (tomorrow's PM2.5 = today's PM2.5).
- Split by time: train = earlier data, validation = 90 days before the last 90 days, test = last 90 days.
- Test set is evaluated once, at the end.
- Metrics: MAE, RMSE, and accuracy of the health category (US EPA PM2.5 bands).
- Success: the model must have lower MAE than persistence on test, without being worse for any city.