# Reference rows (balanced accuracy, person mean)

| Dataset | People | Fingerprint top-1 | Pooled | Current (row 2) | Oracle (row 3) |
|---|---|---|---|---|---|
| bnci2014 | 6 | 1.000 | 0.683 | 0.787 | 0.787 |
| bnci2015 | 8 | 0.953 | 0.816 | 0.847 | 0.861 |
| zhou2016 | 3 | 0.747 | 0.796 | 0.851 | 0.853 |
| dreyer_sim2 | 14 | 1.000 | 0.930 | 0.936 | 0.936 |

Current − pooled, 17 dev cross-day people: +0.0612 [+0.0379, +0.0825], 15/2
Oracle − current (routing headroom), 17 dev cross-day people: +0.0070 [+0.0000, +0.0206], 4/1
