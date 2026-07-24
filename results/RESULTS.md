## Experiment 1 — Main Sample (2025-01-01 to 2025-12-31)

Grid: T1=[63], T2=[3], alpha=[0.95], specs=['ACS', 'AES', 'ACI', 'AEI']

### Top-5 specs by Sharpe ratio

| Spec | Sharpe | CumRet | MaxDD | STARR |
|------|--------|--------|-------|-------|
| T1=63_T2=3_a=0.95_ACS | 0.018 | 1.2% | -3.1% | 0.009 |
| T1=63_T2=3_a=0.95_AES | 0.018 | 1.2% | -3.1% | 0.009 |
| T1=63_T2=3_a=0.95_ACI | 0.018 | 1.2% | -3.1% | 0.009 |
| T1=63_T2=3_a=0.95_AEI | 0.018 | 1.2% | -3.1% | 0.009 |

### Benchmarks

- **EW**: Sharpe=0.065, CumRet=7.6%
- **Full-DRMV**: Sharpe=0.044, CumRet=6.5%

### Statistical Tests

- **paired_ttest** (T1=63_T2=3_a=0.95_ACS vs Full-DRMV): p=0.651207979201333, sig@5%=False
- **bootstrap_sharpe** (T1=63_T2=3_a=0.95_ACS vs Full-DRMV): p=0.485, sig@5%=False


## Experiment 2 — Stress Period (2025-10-01 to 2026-03-31)

T1=63 (fixed), T2=[3], alpha=[0.95], specs=['ACS', 'AES', 'ACI', 'AEI']

### Top-5 specs by minimum MaxDrawdown

| Spec | MaxDD | Sharpe | CumRet | STARR |
|------|-------|--------|--------|-------|
| T1=63_T2=3_a=0.95_ACS | -11.4% | -0.146 | -9.8% | -0.060 |
| T1=63_T2=3_a=0.95_AES | -11.4% | -0.146 | -9.8% | -0.060 |
| T1=63_T2=3_a=0.95_ACI | -11.4% | -0.146 | -9.8% | -0.060 |
| T1=63_T2=3_a=0.95_AEI | -11.4% | -0.146 | -9.8% | -0.060 |

### Benchmarks

- **EW**: Sharpe=0.119, MaxDD=-4.4%
- **Full-DRMV**: Sharpe=0.028, MaxDD=-10.7%

### Statistical Tests

- **paired_ttest** (T1=63_T2=3_a=0.95_ACS vs Full-DRMV): p=0.8351626452431882, sig@5%=False
- **bootstrap_sharpe** (T1=63_T2=3_a=0.95_ACS vs Full-DRMV): p=0.491, sig@5%=False
