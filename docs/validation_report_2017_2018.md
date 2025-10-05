# Validation Report: 2017–2018

- Train window: 2020–2025 (no leakage)
- Test window: 2017–2018 (strictly out-of-sample)

## Summary Metrics

- bri_vix correlation (test, 0-lag): -0.159
- Best lag (BRI leads VIX): -3 days → -0.113
- Data points: 1,002

## Notes on Coverage

- Reddit coverage absent in 2017–2018 via API → sentiment-dependent features neutralized (reduced variance)
- GDELT sample fallback used (column-robust cleaning), still useful for tone/event features

## Visuals

Generate basic BRI test series plot:

```bash
python - << 'PY'
import pandas as pd, json
import matplotlib.pyplot as plt
bri_test = pd.read_csv('output/train_test_from_raw/bri_test.csv', parse_dates=['date'])
plt.figure(figsize=(10,4))
plt.plot(bri_test['date'], bri_test['bri'])
plt.title('BRI (Test 2017–2018)')
plt.ylabel('BRI (0-100)'); plt.xlabel('Date'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('output/train_test_from_raw/bri_test_series.png', dpi=200)
PY
```

## Recommendations

- Backfill Reddit/news archives for 2017–2018 to restore variance
- Add macro proxies for early periods; keep no-leakage policy intact
