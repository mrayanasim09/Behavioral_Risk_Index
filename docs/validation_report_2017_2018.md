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

- BRI Time Series (2017–2018): `/static/bri_test_series.png`
- BRI vs VIX Overlay: `/static/bri_vs_vix_test.png`
- Full PDF: `/static/validation_report_2017_2018.pdf`

## Recommendations

- Backfill Reddit/news archives for 2017–2018 to restore variance
- Add macro proxies for early periods; keep no-leakage policy intact
