# X5 formula control validation

Both hosts used 20,000 training customers (9,896 SMS and 10,104 controls) and 20,004 validation customers (9,998 SMS and 10,006 controls). The validation endpoint is incremental buyers per 1,000 SMS among the highest ranked 30%. PTONet values average the three registered seeds.

| Host | Configuration | Validation value | Gain vs Original | Seed SD | Runtime sum, s | GPU time, s | Peak RSS, GiB | Peak GPU reserved, GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| causalpfn | Original | 70.06 | +0.00 | — | 128.5 | 128.2 | 3.56 | 0.17 |
| causalpfn | D | 55.94 | -14.12 | — | 129.2 | 129.0 | 3.91 | 0.19 |
| causalpfn | D_random | 51.11 | -18.95 | — | 129.5 | 129.4 | 3.91 | 0.19 |
| causalpfn | D_shuffled | 53.50 | -16.56 | — | 129.3 | 129.1 | 3.91 | 0.19 |
| ptonet | Original | 79.03 | +0.00 | 4.95 | 22.2 | 0.0 | 0.51 | 0.00 |
| ptonet | D | 70.46 | -8.57 | 3.18 | 20.0 | 0.0 | 0.53 | 0.00 |
| ptonet | D_random | 71.58 | -7.46 | 3.67 | 19.8 | 0.0 | 0.53 | 0.00 |
| ptonet | D_shuffled | 67.41 | -11.62 | 5.59 | 15.4 | 0.0 | 0.53 | 0.00 |
