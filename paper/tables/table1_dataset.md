**Table 1.** Dataset composition and event-stratified splits.

| Split | Source event(s) | Patches | Role |
|---|---|---:|---|
| Train | Sikkim South Lhonak 2016 | 3,400 | Model fitting |
| Val | Sikkim South Lhonak 2016 (15% hold-out) | 600 | Selection / early stop |
| Test-temporal | Sikkim South Lhonak 2023 | 4,000 | Temporal generalization (7-yr gap) |
| Test-spatial | UT Chamoli/Chorabari/Kedarnath + HP Gepang Gath | 642 | Spatial (cross-region) generalization |
| **Total labelled** | 6 Himalayan GLOF events | **31,098** | Full corpus |
