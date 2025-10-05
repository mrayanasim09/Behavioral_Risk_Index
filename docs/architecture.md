# BRI Architecture

```mermaid
flowchart TB
  A[Raw Data] -->|Market (Yahoo)| B[DataCollector]
  A -->|News (GDELT)| B
  A -->|Reddit (PRAW)| B
  B --> C[Phase2: Preprocessing]
  C --> D[Phase3: Feature Engineering]
  D --> E[Phase4: BRI Calculation]
  E --> F[Phase5: Validation & Analysis]
  F --> G[Phase6: Visualization]
  G --> H[Phase7: Deliverables]

  subgraph Scripts
    I[process_from_raw.py]
    J[train_test_from_raw.py]
  end

  I --> C
  I --> D
  I --> E
  I --> F
  J --> C
  J --> D
  J --> E
  J --> F
```

- `process_from_raw.py` and `train_test_from_raw.py` operate on saved CSVs to avoid re-downloading.
- Strict no-leakage: parameters fit on training only; applied to test.
