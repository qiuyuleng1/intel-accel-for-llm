
H2D
| setting | run | rank | bytes (GB) | in flight (s) | GB/s | waiting in xfer_wait (s) | in flight / run duration |
|---|---|---|---|---|---|---|---|
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 0 | 181.53 | 32.01 | 5.67 | 0.06 | 38.0% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 1 | 181.53 | 32.07 | 5.66 | 0.07 | 38.0% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 2 | 181.53 | 31.97 | 5.68 | 0.07 | 37.9% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 3 | 181.53 | 31.97 | 5.68 | 0.06 | 37.9% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 0 | 181.53 | 36.06 | 5.03 | 0.06 | 44.8% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 1 | 181.53 | 36.06 | 5.03 | 0.06 | 44.8% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 2 | 181.53 | 36.03 | 5.04 | 0.06 | 44.8% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 3 | 181.53 | 36.04 | 5.04 | 0.06 | 44.8% |

D2H
| setting | run | rank | bytes (GB) | in flight (s) | GB/s | waiting in xfer_wait (s) | in flight / run duration |
|---|---|---|---|---|---|---|---|
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 0 | 2.93 | 0.31 | 9.52 | 0.06 | 0.4% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 1 | 2.93 | 0.26 | 11.33 | 0.06 | 0.3% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 2 | 2.93 | 0.24 | 12.29 | 0.05 | 0.3% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 3 | 2.93 | 0.23 | 12.89 | 0.05 | 0.3% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 0 | 2.93 | 0.30 | 9.80 | 0.09 | 0.4% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 1 | 2.93 | 0.30 | 9.82 | 0.09 | 0.4% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 2 | 2.93 | 0.29 | 9.93 | 0.09 | 0.4% |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 3 | 2.93 | 0.29 | 10.17 | 0.08 | 0.4% |

BENCH
| setting | run | duration (s) | req/s | output tok/s | TTFT mean/p50/p95 (ms) | TPOT mean/p50/p95 (ms) |
|---|---|---|---|---|---|---|
| IAXL_DSA_AUTO_FALLBACK_ENABLE=1 | kvshrink_tp4_in32768_con32_hit100 | 84.3 | 3.795 | 485.7 | 1584.4 / 1505.0 / 2933.3 | 53.77 / 53.58 / 62.18 |
| IAXL_DSA_AUTO_FALLBACK_ENABLE=0 | kvshrink_tp4_in32768_con32_hit100 | 80.5 | 3.976 | 509.0 | 1811.8 / 1851.7 / 2911.0 | 49.00 / 48.35 / 58.87 |
