## 1. Compute and retain (mean over seeds, 95% t-interval)

### Group `id` (lengths [4, 8])

| task | regime | seeds | solved ever | first-correct loop (median) | acc @ loop 8 | acc @ final loop (32) | stable-solve | lost after solve | forget hazard / loop |
|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | 0.996 [0.989, 1.002] | 5.167 [3.940, 6.394] | 0.920 [0.856, 0.984] | 0.196 [0.139, 0.254] | 0.007 [-0.009, 0.023] | 0.993 [0.976, 1.009] | 0.268 [0.145, 0.391] |
| cyclic | random_intermediate | 6 | 0.975 [0.956, 0.994] | 1.500 [0.925, 2.075] | 0.944 [0.913, 0.976] | 0.684 [0.604, 0.763] | 0.641 [0.517, 0.766] | 0.342 [0.216, 0.469] | 0.013 [0.008, 0.019] |
| cyclic | random_exit | 6 | 0.982 [0.976, 0.988] | 1.000 [1.000, 1.000] | 0.946 [0.925, 0.967] | 0.566 [0.529, 0.603] | 0.528 [0.496, 0.561] | 0.462 [0.427, 0.497] | 0.019 [0.017, 0.021] |
| dfa | final | 6 | 0.986 [0.983, 0.989] | 2.000 [2.000, 2.000] | 0.986 [0.983, 0.989] | 0.506 [0.291, 0.721] | 0.377 [0.160, 0.594] | 0.618 [0.399, 0.838] | 0.034 [0.016, 0.053] |
| dfa | random_intermediate | 6 | 0.986 [0.983, 0.989] | 1.000 [1.000, 1.000] | 0.986 [0.983, 0.989] | 0.761 [0.702, 0.821] | 0.711 [0.645, 0.777] | 0.279 [0.212, 0.346] | 0.011 [0.008, 0.013] |
| dfa | random_exit | 6 | 0.985 [0.979, 0.991] | 1.000 [1.000, 1.000] | 0.973 [0.958, 0.989] | 0.760 [0.688, 0.832] | 0.734 [0.658, 0.809] | 0.255 [0.180, 0.329] | 0.010 [0.006, 0.013] |
| storyworld | final | 6 | 1.000 [1.000, 1.000] | 3.000 [3.000, 3.000] | 1.000 [1.000, 1.000] | 0.294 [0.198, 0.389] | 0.242 [0.151, 0.334] | 0.758 [0.666, 0.849] | 0.039 [0.031, 0.048] |
| storyworld | random_intermediate | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.858 [0.826, 0.890] | 0.856 [0.824, 0.888] | 0.144 [0.112, 0.176] | 0.005 [0.004, 0.006] |
| storyworld | random_exit | 6 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | 0.715 [0.641, 0.789] | 0.712 [0.639, 0.785] | 0.288 [0.215, 0.361] | 0.010 [0.007, 0.013] |
| sweepweave | final | 6 | 1.000 [1.000, 1.000] | 4.833 [4.405, 5.262] | 0.999 [0.999, 1.000] | 0.096 [0.058, 0.133] | 0.030 [0.014, 0.045] | 0.970 [0.955, 0.986] | 0.090 [0.085, 0.096] |
| sweepweave | random_intermediate | 6 | 0.999 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.993 [0.990, 0.996] | 0.318 [0.236, 0.401] | 0.265 [0.172, 0.358] | 0.735 [0.642, 0.827] | 0.034 [0.028, 0.041] |
| sweepweave | random_exit | 6 | 0.996 [0.995, 0.998] | 1.000 [1.000, 1.000] | 0.981 [0.978, 0.984] | 0.275 [0.200, 0.350] | 0.227 [0.145, 0.309] | 0.772 [0.690, 0.854] | 0.037 [0.031, 0.044] |

### Group `extrap` (lengths [10, 12])

| task | regime | seeds | solved ever | first-correct loop (median) | acc @ loop 8 | acc @ final loop (32) | stable-solve | lost after solve | forget hazard / loop |
|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | 0.977 [0.955, 0.998] | 5.167 [3.940, 6.394] | 0.448 [0.216, 0.680] | 0.201 [0.139, 0.263] | 0.016 [-0.005, 0.037] | 0.984 [0.962, 1.005] | 0.322 [0.140, 0.503] |
| cyclic | random_intermediate | 6 | 0.803 [0.729, 0.877] | 2.000 [2.000, 2.000] | 0.644 [0.538, 0.750] | 0.512 [0.401, 0.624] | 0.449 [0.321, 0.577] | 0.445 [0.329, 0.562] | 0.023 [0.015, 0.030] |
| cyclic | random_exit | 6 | 0.816 [0.762, 0.870] | 2.000 [2.000, 2.000] | 0.628 [0.533, 0.724] | 0.407 [0.363, 0.451] | 0.356 [0.295, 0.416] | 0.562 [0.478, 0.647] | 0.031 [0.024, 0.038] |
| dfa | final | 6 | 0.990 [0.983, 0.997] | 2.000 [2.000, 2.000] | 0.948 [0.928, 0.967] | 0.516 [0.297, 0.736] | 0.355 [0.159, 0.552] | 0.641 [0.442, 0.840] | 0.038 [0.018, 0.057] |
| dfa | random_intermediate | 6 | 0.988 [0.983, 0.994] | 1.000 [1.000, 1.000] | 0.948 [0.930, 0.966] | 0.726 [0.681, 0.770] | 0.645 [0.607, 0.683] | 0.347 [0.309, 0.385] | 0.015 [0.013, 0.016] |
| dfa | random_exit | 6 | 0.963 [0.935, 0.991] | 1.000 [1.000, 1.000] | 0.883 [0.826, 0.939] | 0.678 [0.590, 0.765] | 0.621 [0.540, 0.702] | 0.356 [0.281, 0.431] | 0.015 [0.012, 0.019] |
| storyworld | final | 6 | 0.883 [0.786, 0.980] | 4.333 [3.791, 4.875] | 0.673 [0.504, 0.842] | 0.231 [0.144, 0.317] | 0.176 [0.104, 0.248] | 0.798 [0.711, 0.884] | 0.057 [0.045, 0.069] |
| storyworld | random_intermediate | 6 | 0.619 [0.510, 0.727] | 3.000 [3.000, 3.000] | 0.474 [0.349, 0.600] | 0.488 [0.438, 0.538] | 0.479 [0.429, 0.530] | 0.214 [0.109, 0.319] | 0.009 [0.005, 0.014] |
| storyworld | random_exit | 6 | 0.404 [0.294, 0.513] | 2.833 [1.802, 3.865] | 0.297 [0.242, 0.353] | 0.295 [0.187, 0.403] | 0.286 [0.188, 0.383] | 0.300 [0.224, 0.376] | 0.013 [0.010, 0.017] |
| sweepweave | final | 6 | 0.859 [0.831, 0.887] | 6.500 [5.925, 7.075] | 0.622 [0.593, 0.651] | 0.107 [0.065, 0.150] | 0.050 [0.025, 0.076] | 0.942 [0.914, 0.970] | 0.107 [0.095, 0.119] |
| sweepweave | random_intermediate | 6 | 0.812 [0.756, 0.868] | 3.167 [2.738, 3.595] | 0.636 [0.559, 0.713] | 0.316 [0.187, 0.445] | 0.248 [0.132, 0.363] | 0.697 [0.561, 0.833] | 0.044 [0.030, 0.058] |
| sweepweave | random_exit | 6 | 0.742 [0.660, 0.824] | 3.167 [2.377, 3.957] | 0.523 [0.446, 0.601] | 0.201 [0.166, 0.237] | 0.162 [0.125, 0.198] | 0.780 [0.723, 0.838] | 0.056 [0.047, 0.065] |

Majority-class accuracy by length (chance reference): cyclic: N=4:0.21, N=5:0.20, N=7:0.20, N=8:0.21, N=10:0.20, N=12:0.21; dfa: N=4:0.33, N=5:0.30, N=7:0.30, N=8:0.30, N=10:0.31, N=12:0.30; storyworld: N=4:0.21, N=5:0.19, N=7:0.10, N=8:0.20, N=10:0.20, N=12:0.20; sweepweave: N=4:0.19, N=5:0.16, N=7:0.16, N=8:0.16, N=10:0.11, N=12:0.17

## 2. Reuse after delayed freezing (held-out lengths)

delta = reuse(d) - reuse(0) on the same samples, averaged per seed; 95% t-interval over seeds.

Headline contrast = largest delay vs 0 (Holm over the kind x task x regime rows using the paired-t p; with 6 seeds the exact sign-flip p cannot go below 0.031). Other delays are descriptive.

| reuse kind | task | regime | delay d | reuse(0) | delta(d-0) [95% CI] | t p | sign-flip p | Holm (t) p |
|---|---|---|---|---|---|---|---|---|
| probe | cyclic | final | 1 | 0.954 | -0.017 [-0.030, -0.003] | 0.0247 | 0.0625 | n/a |
| probe | cyclic | final | 2 | 0.954 | -0.034 [-0.052, -0.016] | 0.0046 | 0.0312 | n/a |
| probe | cyclic | final | 4 | 0.954 | -0.077 [-0.123, -0.032] | 0.0073 | 0.0312 | n/a |
| probe | cyclic | final | 8 | 0.954 | -0.138 [-0.217, -0.060] | 0.0062 | 0.0312 | 0.1057 |
| probe | cyclic | random_intermediate | 1 | 0.948 | -0.012 [-0.033, +0.008] | 0.1761 | 0.1875 | n/a |
| probe | cyclic | random_intermediate | 2 | 0.948 | -0.015 [-0.029, -0.002] | 0.0345 | 0.0625 | n/a |
| probe | cyclic | random_intermediate | 4 | 0.948 | -0.021 [-0.035, -0.007] | 0.0109 | 0.0312 | n/a |
| probe | cyclic | random_intermediate | 8 | 0.948 | -0.033 [-0.050, -0.016] | 0.0043 | 0.0312 | 0.0821 |
| probe | cyclic | random_exit | 1 | 0.953 | -0.010 [-0.021, +0.001] | 0.0609 | 0.1250 | n/a |
| probe | cyclic | random_exit | 2 | 0.953 | -0.012 [-0.024, +0.000] | 0.0517 | 0.0938 | n/a |
| probe | cyclic | random_exit | 4 | 0.953 | -0.023 [-0.038, -0.008] | 0.0109 | 0.0312 | n/a |
| probe | cyclic | random_exit | 8 | 0.953 | -0.041 [-0.060, -0.022] | 0.0029 | 0.0312 | 0.0605 |
| probe | dfa | final | 1 | 0.991 | -0.004 [-0.007, -0.001] | 0.0150 | 0.0625 | n/a |
| probe | dfa | final | 2 | 0.991 | -0.005 [-0.007, -0.002] | 0.0060 | 0.0312 | n/a |
| probe | dfa | final | 4 | 0.991 | -0.002 [-0.005, +0.001] | 0.1304 | 0.1562 | n/a |
| probe | dfa | final | 8 | 0.991 | -0.001 [-0.006, +0.004] | 0.6455 | 0.5938 | 1.0000 |
| probe | dfa | random_intermediate | 1 | 0.988 | -0.001 [-0.003, +0.001] | 0.2750 | 0.2812 | n/a |
| probe | dfa | random_intermediate | 2 | 0.988 | +0.001 [-0.001, +0.004] | 0.2521 | 0.2812 | n/a |
| probe | dfa | random_intermediate | 4 | 0.988 | +0.004 [+0.000, +0.007] | 0.0491 | 0.0625 | n/a |
| probe | dfa | random_intermediate | 8 | 0.988 | +0.004 [-0.001, +0.008] | 0.1112 | 0.1250 | 1.0000 |
| probe | dfa | random_exit | 1 | 0.989 | -0.006 [-0.010, -0.002] | 0.0121 | 0.0625 | n/a |
| probe | dfa | random_exit | 2 | 0.989 | -0.006 [-0.013, +0.000] | 0.0615 | 0.0625 | n/a |
| probe | dfa | random_exit | 4 | 0.989 | -0.006 [-0.012, +0.001] | 0.0744 | 0.0625 | n/a |
| probe | dfa | random_exit | 8 | 0.989 | -0.008 [-0.016, +0.001] | 0.0726 | 0.0312 | 0.7256 |
| probe | storyworld | final | 1 | 0.810 | +0.033 [+0.008, +0.058] | 0.0206 | 0.0625 | n/a |
| probe | storyworld | final | 2 | 0.810 | +0.030 [+0.001, +0.060] | 0.0431 | 0.0312 | n/a |
| probe | storyworld | final | 4 | 0.810 | +0.044 [-0.005, +0.093] | 0.0684 | 0.1250 | n/a |
| probe | storyworld | final | 8 | 0.810 | +0.037 [-0.016, +0.090] | 0.1344 | 0.1250 | 1.0000 |
| probe | storyworld | random_intermediate | 1 | 0.755 | +0.096 [+0.074, +0.119] | 0.0001 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 2 | 0.755 | +0.137 [+0.099, +0.175] | 0.0003 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 4 | 0.755 | +0.173 [+0.131, +0.214] | 0.0001 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 8 | 0.755 | +0.185 [+0.143, +0.228] | <1e-4 | 0.0312 | 0.0024 |
| probe | storyworld | random_exit | 1 | 0.722 | +0.084 [+0.011, +0.157] | 0.0321 | 0.0625 | n/a |
| probe | storyworld | random_exit | 2 | 0.722 | +0.110 [+0.031, +0.189] | 0.0157 | 0.0625 | n/a |
| probe | storyworld | random_exit | 4 | 0.722 | +0.142 [+0.072, +0.212] | 0.0034 | 0.0312 | n/a |
| probe | storyworld | random_exit | 8 | 0.722 | +0.151 [+0.085, +0.218] | 0.0021 | 0.0312 | 0.0453 |
| probe | sweepweave | final | 1 | 0.905 | +0.004 [-0.003, +0.010] | 0.1905 | 0.2500 | n/a |
| probe | sweepweave | final | 2 | 0.905 | -0.005 [-0.012, +0.003] | 0.1608 | 0.0625 | n/a |
| probe | sweepweave | final | 4 | 0.905 | -0.007 [-0.030, +0.017] | 0.4847 | 0.5938 | n/a |
| probe | sweepweave | final | 8 | 0.905 | -0.022 [-0.056, +0.011] | 0.1503 | 0.1250 | 1.0000 |
| probe | sweepweave | random_intermediate | 1 | 0.879 | +0.037 [+0.023, +0.051] | 0.0011 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 2 | 0.879 | +0.051 [+0.033, +0.068] | 0.0007 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 4 | 0.879 | +0.050 [+0.021, +0.079] | 0.0065 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 8 | 0.879 | +0.027 [-0.009, +0.064] | 0.1129 | 0.1250 | 1.0000 |
| probe | sweepweave | random_exit | 1 | 0.880 | +0.038 [+0.024, +0.052] | 0.0010 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 2 | 0.880 | +0.041 [+0.029, +0.052] | 0.0003 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 4 | 0.880 | +0.036 [+0.023, +0.049] | 0.0008 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 8 | 0.880 | +0.003 [-0.017, +0.023] | 0.7337 | 0.7188 | 1.0000 |
| native | cyclic | final | 1 | 0.224 | -0.011 [-0.016, -0.006] | 0.0018 | 0.0312 | n/a |
| native | cyclic | final | 2 | 0.224 | -0.019 [-0.028, -0.010] | 0.0033 | 0.0312 | n/a |
| native | cyclic | final | 4 | 0.224 | -0.022 [-0.038, -0.006] | 0.0173 | 0.0625 | n/a |
| native | cyclic | final | 8 | 0.224 | -0.033 [-0.050, -0.015] | 0.0052 | 0.0625 | 0.0931 |
| native | cyclic | random_intermediate | 1 | 0.211 | -0.008 [-0.013, -0.002] | 0.0167 | 0.0312 | n/a |
| native | cyclic | random_intermediate | 2 | 0.211 | -0.006 [-0.013, +0.001] | 0.0684 | 0.0625 | n/a |
| native | cyclic | random_intermediate | 4 | 0.211 | -0.012 [-0.018, -0.006] | 0.0030 | 0.0312 | n/a |
| native | cyclic | random_intermediate | 8 | 0.211 | -0.011 [-0.020, -0.001] | 0.0396 | 0.0625 | 0.4758 |
| native | cyclic | random_exit | 1 | 0.207 | -0.001 [-0.008, +0.005] | 0.6281 | 0.6250 | n/a |
| native | cyclic | random_exit | 2 | 0.207 | -0.003 [-0.012, +0.007] | 0.5205 | 0.5312 | n/a |
| native | cyclic | random_exit | 4 | 0.207 | +0.000 [-0.010, +0.011] | 0.9178 | 0.9062 | n/a |
| native | cyclic | random_exit | 8 | 0.207 | +0.000 [-0.010, +0.010] | 0.9567 | 0.9688 | 1.0000 |
| native | dfa | final | 1 | 0.814 | -0.002 [-0.017, +0.013] | 0.7168 | 0.7188 | n/a |
| native | dfa | final | 2 | 0.814 | -0.010 [-0.030, +0.011] | 0.2697 | 0.2812 | n/a |
| native | dfa | final | 4 | 0.814 | -0.031 [-0.049, -0.013] | 0.0069 | 0.0312 | n/a |
| native | dfa | final | 8 | 0.814 | -0.061 [-0.092, -0.030] | 0.0037 | 0.0312 | 0.0748 |
| native | dfa | random_intermediate | 1 | 0.831 | -0.007 [-0.019, +0.005] | 0.1787 | 0.1875 | n/a |
| native | dfa | random_intermediate | 2 | 0.831 | -0.011 [-0.029, +0.006] | 0.1536 | 0.1562 | n/a |
| native | dfa | random_intermediate | 4 | 0.831 | -0.022 [-0.048, +0.004] | 0.0844 | 0.1250 | n/a |
| native | dfa | random_intermediate | 8 | 0.831 | -0.039 [-0.070, -0.008] | 0.0241 | 0.0625 | 0.3849 |
| native | dfa | random_exit | 1 | 0.728 | -0.008 [-0.025, +0.008] | 0.2561 | 0.2812 | n/a |
| native | dfa | random_exit | 2 | 0.728 | -0.015 [-0.041, +0.010] | 0.1850 | 0.1562 | n/a |
| native | dfa | random_exit | 4 | 0.728 | -0.022 [-0.049, +0.005] | 0.0878 | 0.1250 | n/a |
| native | dfa | random_exit | 8 | 0.728 | -0.029 [-0.055, -0.003] | 0.0334 | 0.0625 | 0.4415 |
| native | storyworld | final | 1 | 0.356 | +0.002 [-0.015, +0.019] | 0.8233 | 0.8438 | n/a |
| native | storyworld | final | 2 | 0.356 | +0.003 [-0.029, +0.035] | 0.8137 | 0.8438 | n/a |
| native | storyworld | final | 4 | 0.356 | +0.001 [-0.041, +0.044] | 0.9316 | 0.8750 | n/a |
| native | storyworld | final | 8 | 0.356 | -0.041 [-0.121, +0.040] | 0.2485 | 0.1562 | 1.0000 |
| native | storyworld | random_intermediate | 1 | 0.362 | +0.001 [-0.018, +0.019] | 0.9371 | 0.9375 | n/a |
| native | storyworld | random_intermediate | 2 | 0.362 | +0.008 [-0.010, +0.027] | 0.2938 | 0.2812 | n/a |
| native | storyworld | random_intermediate | 4 | 0.362 | +0.023 [-0.002, +0.048] | 0.0633 | 0.1250 | n/a |
| native | storyworld | random_intermediate | 8 | 0.362 | +0.035 [-0.022, +0.091] | 0.1752 | 0.1875 | 1.0000 |
| native | storyworld | random_exit | 1 | 0.266 | -0.000 [-0.014, +0.014] | 0.9850 | 1.0000 | n/a |
| native | storyworld | random_exit | 2 | 0.266 | -0.016 [-0.028, -0.004] | 0.0214 | 0.0312 | n/a |
| native | storyworld | random_exit | 4 | 0.266 | -0.029 [-0.057, -0.002] | 0.0422 | 0.0625 | n/a |
| native | storyworld | random_exit | 8 | 0.266 | -0.036 [-0.072, -0.000] | 0.0495 | 0.0625 | 0.5443 |
| native | sweepweave | final | 1 | 0.401 | -0.061 [-0.077, -0.045] | 0.0002 | 0.0312 | n/a |
| native | sweepweave | final | 2 | 0.401 | -0.122 [-0.161, -0.083] | 0.0005 | 0.0312 | n/a |
| native | sweepweave | final | 4 | 0.401 | -0.197 [-0.265, -0.128] | 0.0007 | 0.0312 | n/a |
| native | sweepweave | final | 8 | 0.401 | -0.258 [-0.349, -0.167] | 0.0008 | 0.0312 | 0.0174 |
| native | sweepweave | random_intermediate | 1 | 0.448 | -0.007 [-0.027, +0.012] | 0.3784 | 0.3750 | n/a |
| native | sweepweave | random_intermediate | 2 | 0.448 | -0.012 [-0.046, +0.022] | 0.3970 | 0.4062 | n/a |
| native | sweepweave | random_intermediate | 4 | 0.448 | -0.035 [-0.093, +0.024] | 0.1905 | 0.1875 | n/a |
| native | sweepweave | random_intermediate | 8 | 0.448 | -0.091 [-0.168, -0.013] | 0.0294 | 0.0312 | 0.4415 |
| native | sweepweave | random_exit | 1 | 0.363 | +0.013 [-0.019, +0.044] | 0.3439 | 0.3438 | n/a |
| native | sweepweave | random_exit | 2 | 0.363 | +0.006 [-0.046, +0.057] | 0.7890 | 0.8125 | n/a |
| native | sweepweave | random_exit | 4 | 0.363 | -0.029 [-0.105, +0.048] | 0.3783 | 0.4688 | n/a |
| native | sweepweave | random_exit | 8 | 0.363 | -0.092 [-0.170, -0.014] | 0.0294 | 0.0625 | 0.4415 |

Pooled delta reuse(d=8) - reuse(0), probe: +0.0137 [95% cluster-bootstrap CI -0.0325, +0.0602] over 12 task x regime clusters.

Pooled delta reuse(d=8) - reuse(0), native: -0.0546 [95% cluster-bootstrap CI -0.1009, -0.0206] over 12 task x regime clusters.

## 3. Controller vs fixed-depth recurrence at matched FLOPs

Endpoint per model = mean over the pre-specified budgets kappa in [2, 3, 4, 6, 8, 12] (mean loop-equivalents per sample; controller overhead counted in FLOPs) of [controller accuracy - fixed-depth accuracy at the same mean FLOPs], taking the controller frontier (12 compute prices) and the fixed-depth curve by linear interpolation (a fractional depth = mixture of floor/ceil depth). A model needs >=3 budgets inside the controller's attainable range to count. Unit of inference = independent training seed. Holm correction (paired-t p) across the task x regime tests of each table.

### Group `heldout` - controller vs fixed-depth

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1932 [+0.1543, +0.2321] | 100% | <1e-4 | 0.0312 | 0.0005 | +5.21 |
| cyclic | random_intermediate | 6 | -0.0257 [-0.0372, -0.0142] | 0% | 0.0023 | 0.0312 | 0.0113 | -2.34 |
| cyclic | random_exit | 6 | -0.0305 [-0.0416, -0.0195] | 0% | 0.0009 | 0.0312 | 0.0060 | -2.90 |
| dfa | final | 6 | +0.0615 [+0.0423, +0.0807] | 100% | 0.0004 | 0.0312 | 0.0034 | +3.36 |
| dfa | random_intermediate | 6 | +0.0013 [-0.0089, +0.0114] | 67% | 0.7579 | 0.8125 | 0.7579 | +0.13 |
| dfa | random_exit | 6 | -0.0068 [-0.0145, +0.0010] | 17% | 0.0742 | 0.0625 | 0.2226 | -0.92 |
| storyworld | final | 6 | -0.0295 [-0.0777, +0.0186] | 50% | 0.1759 | 0.2500 | 0.3517 | -0.64 |
| storyworld | random_intermediate | 6 | -0.2111 [-0.2928, -0.1294] | 0% | 0.0012 | 0.0312 | 0.0070 | -2.71 |
| storyworld | random_exit | 6 | -0.1016 [-0.1608, -0.0425] | 0% | 0.0069 | 0.0312 | 0.0277 | -1.80 |
| sweepweave | final | 6 | +0.1030 [+0.0755, +0.1304] | 100% | 0.0002 | 0.0312 | 0.0018 | +3.94 |
| sweepweave | random_intermediate | 6 | -0.0999 [-0.1099, -0.0898] | 0% | <1e-4 | 0.0312 | <1e-4 | -10.47 |
| sweepweave | random_exit | 6 | -0.1419 [-0.1607, -0.1230] | 0% | <1e-4 | 0.0312 | <1e-4 | -7.90 |

Pooled over 12 task x regime clusters: gain = -0.0240 [95% cluster-bootstrap CI -0.0800, +0.0384].

### Group `heldout` - controller vs confidence-threshold early exit

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1959 [+0.1242, +0.2677] | 100% | 0.0009 | 0.0312 | 0.0082 | +2.87 |
| cyclic | random_intermediate | 6 | +0.0265 [-0.0019, +0.0550] | 83% | 0.0617 | 0.0625 | 0.1850 | +0.98 |
| cyclic | random_exit | 6 | +0.0139 [+0.0028, +0.0250] | 83% | 0.0234 | 0.0625 | 0.1170 | +1.32 |
| dfa | final | 6 | +0.0512 [+0.0151, +0.0873] | 100% | 0.0148 | 0.0312 | 0.0890 | +1.49 |
| dfa | random_intermediate | 6 | -0.0128 [-0.0243, -0.0013] | 0% | 0.0355 | 0.0312 | 0.1421 | -1.17 |
| dfa | random_exit | 6 | +0.0045 [-0.0074, +0.0164] | 67% | 0.3768 | 0.3438 | 0.4526 | +0.40 |
| storyworld | final | 6 | -0.0234 [-0.0669, +0.0202] | 33% | 0.2263 | 0.2188 | 0.4526 | -0.56 |
| storyworld | random_intermediate | 6 | -0.1780 [-0.2419, -0.1142] | 0% | 0.0008 | 0.0312 | 0.0082 | -2.93 |
| storyworld | random_exit | 6 | -0.0741 [-0.1179, -0.0303] | 0% | 0.0074 | 0.0312 | 0.0517 | -1.77 |
| sweepweave | final | 6 | +0.1036 [+0.0563, +0.1509] | 100% | 0.0024 | 0.0312 | 0.0195 | +2.30 |
| sweepweave | random_intermediate | 6 | -0.0666 [-0.0801, -0.0530] | 0% | <1e-4 | 0.0312 | 0.0007 | -5.14 |
| sweepweave | random_exit | 6 | -0.0897 [-0.1111, -0.0683] | 0% | 0.0001 | 0.0312 | 0.0013 | -4.40 |

Pooled over 12 task x regime clusters: gain = -0.0041 [95% cluster-bootstrap CI -0.0541, +0.0509].

### Group `id` - controller vs fixed-depth

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.2876 [+0.2181, +0.3571] | 100% | 0.0001 | 0.0312 | 0.0011 | +4.34 |
| cyclic | random_intermediate | 6 | +0.0023 [-0.0027, +0.0073] | 83% | 0.2898 | 0.2812 | 0.8693 | +0.48 |
| cyclic | random_exit | 6 | +0.0006 [-0.0095, +0.0108] | 67% | 0.8767 | 0.8750 | 0.9658 | +0.07 |
| dfa | final | 6 | +0.0811 [+0.0648, +0.0974] | 100% | <1e-4 | 0.0312 | 0.0006 | +5.23 |
| dfa | random_intermediate | 6 | +0.0111 [+0.0081, +0.0141] | 100% | 0.0002 | 0.0312 | 0.0018 | +3.85 |
| dfa | random_exit | 6 | +0.0059 [+0.0012, +0.0106] | 83% | 0.0241 | 0.0625 | 0.0965 | +1.30 |
| storyworld | final | 6 | +0.0986 [+0.0785, +0.1188] | 100% | <1e-4 | 0.0312 | 0.0006 | +5.14 |
| storyworld | random_intermediate | 6 | -0.0027 [-0.0033, -0.0022] | 0% | <1e-4 | 0.0312 | 0.0006 | -5.18 |
| storyworld | random_exit | 6 | -0.0004 [-0.0020, +0.0011] | 50% | 0.4829 | 0.4688 | 0.9658 | -0.31 |
| sweepweave | final | 6 | +0.1655 [+0.1125, +0.2184] | 100% | 0.0005 | 0.0312 | 0.0034 | +3.28 |
| sweepweave | random_intermediate | 6 | -0.0249 [-0.0331, -0.0168] | 0% | 0.0005 | 0.0312 | 0.0034 | -3.20 |
| sweepweave | random_exit | 6 | -0.0331 [-0.0440, -0.0222] | 0% | 0.0006 | 0.0312 | 0.0034 | -3.18 |

Pooled over 12 task x regime clusters: gain = +0.0493 [95% cluster-bootstrap CI +0.0050, +0.1070].

### Group `id` - controller vs confidence-threshold early exit

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.3025 [+0.1901, +0.4148] | 100% | 0.0010 | 0.0312 | 0.0077 | +2.83 |
| cyclic | random_intermediate | 6 | +0.0405 [+0.0237, +0.0572] | 100% | 0.0016 | 0.0312 | 0.0110 | +2.54 |
| cyclic | random_exit | 6 | +0.0405 [+0.0281, +0.0529] | 100% | 0.0004 | 0.0312 | 0.0035 | +3.43 |
| dfa | final | 6 | +0.1213 [+0.0682, +0.1744] | 100% | 0.0020 | 0.0312 | 0.0113 | +2.40 |
| dfa | random_intermediate | 6 | +0.0059 [+0.0004, +0.0113] | 100% | 0.0396 | 0.0312 | 0.1188 | +1.13 |
| dfa | random_exit | 6 | +0.0169 [+0.0051, +0.0286] | 100% | 0.0142 | 0.0312 | 0.0566 | +1.51 |
| storyworld | final | 6 | +0.1291 [+0.0910, +0.1673] | 100% | 0.0003 | 0.0312 | 0.0033 | +3.55 |
| storyworld | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_exit | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | final | 6 | +0.1432 [+0.0816, +0.2048] | 100% | 0.0019 | 0.0312 | 0.0113 | +2.44 |
| sweepweave | random_intermediate | 6 | +0.0036 [-0.0137, +0.0209] | 67% | 0.6153 | 0.6250 | 0.6153 | +0.22 |
| sweepweave | random_exit | 6 | +0.0122 [-0.0091, +0.0335] | 67% | 0.2022 | 0.2188 | 0.4044 | +0.60 |

Pooled over 10 task x regime clusters: gain = +0.0816 [95% cluster-bootstrap CI +0.0325, +0.1438].

### Post-hoc endpoint: budgets spread over each controller's own attainable range

The pre-specified grid leaves some cells empty because a controller that cannot stop before loop ~5 never reaches kappa=2..4. This second endpoint (chosen after seeing that, hence post hoc) uses 6 equally spaced budgets inside each controller's own cost range: wherever the controller can operate, is it better than fixed depth at the same FLOPs?

### Group `heldout` - controller vs fixed-depth (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1616 [+0.1069, +0.2164] | 100% | 0.0006 | 0.0312 | 0.0069 | +3.10 |
| cyclic | random_intermediate | 6 | +0.0155 [-0.0065, +0.0376] | 67% | 0.1301 | 0.1562 | 0.3439 | +0.74 |
| cyclic | random_exit | 6 | -0.0021 [-0.0227, +0.0185] | 50% | 0.8068 | 0.8125 | 0.8068 | -0.11 |
| dfa | final | 6 | +0.0547 [+0.0202, +0.0892] | 100% | 0.0095 | 0.0312 | 0.0477 | +1.67 |
| dfa | random_intermediate | 6 | +0.0512 [+0.0305, +0.0720] | 100% | 0.0014 | 0.0312 | 0.0129 | +2.59 |
| dfa | random_exit | 6 | +0.0270 [+0.0059, +0.0481] | 83% | 0.0216 | 0.0625 | 0.0865 | +1.34 |
| storyworld | final | 6 | -0.0400 [-0.0939, +0.0139] | 17% | 0.1146 | 0.0625 | 0.3439 | -0.78 |
| storyworld | random_intermediate | 6 | -0.2126 [-0.2926, -0.1326] | 0% | 0.0010 | 0.0312 | 0.0103 | -2.79 |
| storyworld | random_exit | 6 | -0.1092 [-0.1715, -0.0468] | 0% | 0.0064 | 0.0312 | 0.0384 | -1.84 |
| sweepweave | final | 6 | +0.1539 [+0.0865, +0.2213] | 100% | 0.0020 | 0.0312 | 0.0163 | +2.40 |
| sweepweave | random_intermediate | 6 | -0.0722 [-0.1105, -0.0339] | 0% | 0.0047 | 0.0312 | 0.0329 | -1.98 |
| sweepweave | random_exit | 6 | -0.1221 [-0.1487, -0.0955] | 0% | <1e-4 | 0.0312 | 0.0009 | -4.82 |

Pooled over 12 task x regime clusters: gain = -0.0078 [95% cluster-bootstrap CI -0.0659, +0.0540].

### Group `heldout` - controller vs confidence-threshold early exit (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1868 [+0.0917, +0.2819] | 100% | 0.0039 | 0.0312 | 0.0275 | +2.06 |
| cyclic | random_intermediate | 5 | +0.0290 [+0.0196, +0.0384] | 100% | 0.0010 | 0.0625 | 0.0092 | +3.83 |
| cyclic | random_exit | 6 | +0.0197 [+0.0084, +0.0310] | 100% | 0.0065 | 0.0312 | 0.0325 | +1.83 |
| dfa | final | 2 | +0.0452 [-0.2873, +0.3777] | 100% | 0.3340 | 0.5000 | 0.4810 | +1.22 |
| dfa | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_exit | 3 | +0.0297 [-0.0069, +0.0663] | 100% | 0.0730 | 0.2500 | 0.2190 | +2.02 |
| storyworld | final | 6 | -0.0253 [-0.0740, +0.0235] | 33% | 0.2405 | 0.2500 | 0.4810 | -0.54 |
| storyworld | random_intermediate | 5 | -0.1611 [-0.2178, -0.1044] | 0% | 0.0014 | 0.0625 | 0.0112 | -3.53 |
| storyworld | random_exit | 4 | -0.0742 [-0.1481, -0.0003] | 0% | 0.0494 | 0.1250 | 0.1977 | -1.60 |
| sweepweave | final | 6 | +0.0939 [+0.0426, +0.1452] | 83% | 0.0053 | 0.0625 | 0.0319 | +1.92 |
| sweepweave | random_intermediate | 6 | -0.0456 [-0.0623, -0.0290] | 0% | 0.0009 | 0.0312 | 0.0089 | -2.88 |
| sweepweave | random_exit | 6 | -0.0760 [-0.0962, -0.0558] | 0% | 0.0002 | 0.0312 | 0.0022 | -3.94 |

Pooled over 11 task x regime clusters: gain = +0.0020 [95% cluster-bootstrap CI -0.0499, +0.0570].

### Group `id` - controller vs fixed-depth (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.2502 [+0.1597, +0.3406] | 100% | 0.0009 | 0.0312 | 0.0094 | +2.90 |
| cyclic | random_intermediate | 6 | +0.0447 [+0.0257, +0.0638] | 100% | 0.0018 | 0.0312 | 0.0126 | +2.46 |
| cyclic | random_exit | 6 | +0.0326 [+0.0103, +0.0549] | 100% | 0.0133 | 0.0312 | 0.0663 | +1.53 |
| dfa | final | 6 | +0.0768 [+0.0415, +0.1120] | 100% | 0.0025 | 0.0312 | 0.0150 | +2.29 |
| dfa | random_intermediate | 6 | +0.0706 [+0.0433, +0.0979] | 100% | 0.0012 | 0.0312 | 0.0102 | +2.71 |
| dfa | random_exit | 6 | +0.0469 [+0.0378, +0.0560] | 100% | <1e-4 | 0.0312 | 0.0005 | +5.39 |
| storyworld | final | 6 | +0.0868 [+0.0542, +0.1195] | 100% | 0.0010 | 0.0312 | 0.0102 | +2.79 |
| storyworld | random_intermediate | 6 | +0.0125 [+0.0025, +0.0226] | 83% | 0.0240 | 0.0625 | 0.0719 | +1.31 |
| storyworld | random_exit | 6 | +0.0387 [+0.0104, +0.0671] | 100% | 0.0171 | 0.0312 | 0.0685 | +1.43 |
| sweepweave | final | 6 | +0.2200 [+0.1361, +0.3040] | 100% | 0.0011 | 0.0312 | 0.0102 | +2.75 |
| sweepweave | random_intermediate | 6 | +0.0384 [-0.0112, +0.0880] | 83% | 0.1035 | 0.1250 | 0.2069 | +0.81 |
| sweepweave | random_exit | 6 | -0.0057 [-0.0295, +0.0181] | 67% | 0.5655 | 0.6250 | 0.5655 | -0.25 |

Pooled over 12 task x regime clusters: gain = +0.0761 [95% cluster-bootstrap CI +0.0383, +0.1252].

### Group `id` - controller vs confidence-threshold early exit (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.2833 [+0.1290, +0.4376] | 100% | 0.0052 | 0.0312 | 0.0262 | +1.93 |
| cyclic | random_intermediate | 4 | +0.0485 [+0.0248, +0.0723] | 100% | 0.0074 | 0.1250 | 0.0296 | +3.25 |
| cyclic | random_exit | 6 | +0.0531 [+0.0349, +0.0713] | 100% | 0.0007 | 0.0312 | 0.0040 | +3.06 |
| dfa | final | 1 | +0.1234 [+nan, +nan] | 100% | n/a | 1.0000 | n/a | +nan |
| dfa | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_exit | 2 | +0.0454 [-0.2100, +0.3008] | 100% | 0.2652 | 0.5000 | 0.2652 | +1.60 |
| storyworld | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_exit | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | final | 4 | +0.1151 [+0.0952, +0.1350] | 100% | 0.0003 | 0.1250 | 0.0024 | +9.21 |
| sweepweave | random_intermediate | 6 | +0.0506 [+0.0115, +0.0898] | 100% | 0.0210 | 0.0312 | 0.0629 | +1.36 |
| sweepweave | random_exit | 6 | +0.0417 [+0.0030, +0.0805] | 83% | 0.0395 | 0.0625 | 0.0790 | +1.13 |

Pooled over 8 task x regime clusters: gain = +0.0952 [95% cluster-bootstrap CI +0.0516, +0.1603].

### Best accuracy within a budget (post hoc)

For a budget B (mean loop-equivalents incl. controller overhead) compare the best accuracy the controller reaches at any of its 12 compute prices with cost <= B against the best fixed depth with cost <= B. Both sides take a maximum over candidates on the same test data, so the selection is symmetric (but optimistic for both). Entries are mean over (task x regime) clusters of the per-cluster seed-mean gain, with a 95% cluster-bootstrap CI.

| lengths | B=3 | B=4 | B=6 | B=8 | B=12 | B=16 |
|---|---|---|---|---|---|---|
| id | +0.038 [-0.006, +0.091] | +0.029 [-0.014, +0.082] | +0.018 [-0.019, +0.078] | -0.014 [-0.022, -0.007] | -0.013 [-0.021, -0.006] | -0.012 [-0.020, -0.006] |
| heldout | -0.031 [-0.084, +0.028] | -0.055 [-0.110, +0.006] | -0.080 [-0.141, -0.011] | -0.094 [-0.141, -0.051] | -0.091 [-0.140, -0.047] | -0.088 [-0.137, -0.045] |

Per task x regime at B=8 (mean over seeds, 95% t-interval; gain = controller best - fixed best):

| task | regime | id gain | held-out gain |
|---|---|---|---|
| cyclic | final | -0.004 [-0.021, 0.014] | -0.022 [-0.044, 0.000] |
| cyclic | random_intermediate | -0.014 [-0.017, -0.010] | -0.042 [-0.055, -0.029] |
| cyclic | random_exit | -0.008 [-0.017, 0.002] | -0.036 [-0.040, -0.033] |
| dfa | final | -0.012 [-0.017, -0.007] | -0.026 [-0.036, -0.017] |
| dfa | random_intermediate | -0.009 [-0.013, -0.004] | -0.027 [-0.046, -0.008] |
| dfa | random_exit | -0.009 [-0.015, -0.003] | -0.025 [-0.034, -0.016] |
| storyworld | final | -0.004 [-0.007, -0.002] | -0.135 [-0.212, -0.058] |
| storyworld | random_intermediate | -0.004 [-0.005, -0.003] | -0.292 [-0.390, -0.194] |
| storyworld | random_exit | -0.001 [-0.002, 0.000] | -0.147 [-0.236, -0.058] |
| sweepweave | final | -0.019 [-0.030, -0.008] | -0.066 [-0.093, -0.039] |
| sweepweave | random_intermediate | -0.039 [-0.048, -0.030] | -0.139 [-0.154, -0.124] |
| sweepweave | random_exit | -0.042 [-0.051, -0.033] | -0.171 [-0.197, -0.145] |

## 4. Controller diagnostics (mean over seeds)

Calibration: ECE / Brier of the controller's own P(correct if halted now) = Q_halt vs the raw softmax max-probability, on the steps the controller visited. Premature = stopped before the first correct loop (among items solvable within the loop cap); lost = returned a wrong answer after having been correct; excess = loops used beyond the first-correct loop when the answer was right; futile = loops spent on items never solved.

### `heldout` lengths, compute price lambda = 0.004

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 10.03 | 0.692 | 0.511 | 0.093 | 0.370 | 0.148 | 0.352 | 0.084 | 0.215 | 4.00 | 15.63 | 0.48/0.42/0.10 |
| cyclic | random_intermediate | 15.67 | 0.768 | 0.754 | 0.111 | 0.177 | 0.140 | 0.190 | 0.040 | 0.099 | 12.80 | 17.47 | 0.36/0.57/0.07 |
| cyclic | random_exit | 10.12 | 0.763 | 0.773 | 0.117 | 0.157 | 0.152 | 0.178 | 0.081 | 0.073 | 8.12 | 11.32 | 0.42/0.48/0.10 |
| dfa | final | 9.48 | 0.947 | 0.967 | 0.032 | 0.114 | 0.085 | 0.136 | 0.020 | 0.026 | 7.15 | 17.20 | 0.43/0.46/0.11 |
| dfa | random_intermediate | 15.95 | 0.951 | 0.891 | 0.026 | 0.039 | 0.050 | 0.055 | 0.012 | 0.029 | 14.52 | 16.72 | 0.44/0.49/0.07 |
| dfa | random_exit | 15.73 | 0.900 | 0.868 | 0.043 | 0.072 | 0.076 | 0.088 | 0.017 | 0.058 | 14.27 | 14.11 | 0.24/0.70/0.07 |
| storyworld | final | 11.15 | 0.631 | 0.730 | 0.235 | 0.319 | 0.250 | 0.309 | 0.157 | 0.156 | 7.32 | 10.15 | 0.45/0.45/0.09 |
| storyworld | random_intermediate | 5.10 | 0.227 | 0.445 | 0.712 | 0.564 | 0.698 | 0.554 | 0.559 | 0.089 | 4.90 | 5.68 | 0.24/0.55/0.21 |
| storyworld | random_exit | 4.31 | 0.147 | 0.224 | 0.828 | 0.760 | 0.823 | 0.745 | 0.581 | 0.050 | 4.00 | 4.59 | 0.22/0.50/0.28 |
| sweepweave | final | 13.09 | 0.756 | 0.529 | 0.103 | 0.363 | 0.158 | 0.333 | 0.041 | 0.145 | 6.95 | 14.20 | 0.40/0.52/0.08 |
| sweepweave | random_intermediate | 8.84 | 0.683 | 0.809 | 0.175 | 0.169 | 0.200 | 0.194 | 0.177 | 0.069 | 7.51 | 7.46 | 0.34/0.54/0.12 |
| sweepweave | random_exit | 6.07 | 0.572 | 0.738 | 0.266 | 0.265 | 0.290 | 0.275 | 0.268 | 0.072 | 4.84 | 6.02 | 0.30/0.53/0.17 |

### `heldout` lengths, compute price lambda = 0.016

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 8.35 | 0.695 | 0.671 | 0.096 | 0.373 | 0.147 | 0.354 | 0.102 | 0.194 | 2.89 | 12.92 | 0.51/0.36/0.12 |
| cyclic | random_intermediate | 6.86 | 0.759 | 0.795 | 0.142 | 0.217 | 0.174 | 0.239 | 0.067 | 0.082 | 3.65 | 9.86 | 0.43/0.42/0.15 |
| cyclic | random_exit | 4.62 | 0.758 | 0.793 | 0.168 | 0.208 | 0.207 | 0.243 | 0.104 | 0.056 | 2.34 | 7.24 | 0.41/0.37/0.22 |
| dfa | final | 5.32 | 0.940 | 0.927 | 0.046 | 0.164 | 0.128 | 0.201 | 0.029 | 0.025 | 2.99 | 11.14 | 0.53/0.28/0.19 |
| dfa | random_intermediate | 6.20 | 0.946 | 0.959 | 0.049 | 0.047 | 0.092 | 0.089 | 0.025 | 0.022 | 4.78 | 6.91 | 0.47/0.34/0.19 |
| dfa | random_exit | 5.25 | 0.899 | 0.921 | 0.077 | 0.087 | 0.120 | 0.123 | 0.034 | 0.042 | 3.72 | 6.10 | 0.35/0.45/0.20 |
| storyworld | final | 7.73 | 0.612 | 0.717 | 0.251 | 0.340 | 0.268 | 0.325 | 0.202 | 0.133 | 3.84 | 6.97 | 0.56/0.31/0.13 |
| storyworld | random_intermediate | 1.33 | 0.174 | 0.200 | 0.785 | 0.656 | 0.776 | 0.631 | 0.718 | 0.012 | 0.35 | 1.40 | 0.12/0.11/0.76 |
| storyworld | random_exit | 1.18 | 0.136 | 0.140 | 0.849 | 0.768 | 0.844 | 0.740 | 0.653 | 0.007 | 0.15 | 1.18 | 0.06/0.08/0.86 |
| sweepweave | final | 10.27 | 0.763 | 0.713 | 0.105 | 0.359 | 0.155 | 0.326 | 0.050 | 0.129 | 4.22 | 11.69 | 0.46/0.44/0.10 |
| sweepweave | random_intermediate | 4.29 | 0.661 | 0.771 | 0.242 | 0.186 | 0.271 | 0.221 | 0.222 | 0.049 | 2.32 | 4.42 | 0.50/0.26/0.24 |
| sweepweave | random_exit | 3.54 | 0.542 | 0.696 | 0.319 | 0.299 | 0.341 | 0.306 | 0.328 | 0.047 | 2.01 | 4.10 | 0.34/0.37/0.29 |

Calibration gain (ECE softmax - ECE Q_halt, lambda=0.004, `heldout` lengths) over all 72 models: +0.0524 [+0.0234, +0.0813], sign-flip p=0.0004.

### `id` lengths, compute price lambda = 0.004

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 9.07 | 0.921 | 0.713 | 0.032 | 0.317 | 0.068 | 0.306 | 0.011 | 0.064 | 3.74 | 21.83 | 0.48/0.41/0.11 |
| cyclic | random_intermediate | 15.22 | 0.930 | 0.906 | 0.029 | 0.060 | 0.047 | 0.085 | 0.010 | 0.036 | 12.86 | 23.68 | 0.34/0.59/0.07 |
| cyclic | random_exit | 10.06 | 0.943 | 0.920 | 0.026 | 0.045 | 0.051 | 0.070 | 0.013 | 0.026 | 8.27 | 19.58 | 0.41/0.49/0.11 |
| dfa | final | 10.06 | 0.975 | 0.982 | 0.050 | 0.114 | 0.059 | 0.127 | 0.004 | 0.007 | 7.55 | 28.42 | 0.40/0.49/0.10 |
| dfa | random_intermediate | 16.81 | 0.981 | 0.896 | 0.022 | 0.017 | 0.021 | 0.027 | 0.002 | 0.004 | 15.46 | 19.68 | 0.44/0.50/0.06 |
| dfa | random_exit | 16.90 | 0.958 | 0.900 | 0.024 | 0.033 | 0.031 | 0.043 | 0.005 | 0.022 | 15.39 | 22.22 | 0.20/0.73/0.06 |
| storyworld | final | 10.59 | 0.994 | 0.993 | 0.031 | 0.103 | 0.023 | 0.118 | 0.002 | 0.004 | 7.59 | nan | 0.39/0.51/0.10 |
| storyworld | random_intermediate | 7.91 | 0.996 | 0.999 | 0.003 | 0.003 | 0.005 | 0.006 | 0.002 | 0.002 | 6.87 | nan | 0.30/0.55/0.15 |
| storyworld | random_exit | 7.52 | 0.998 | 0.998 | 0.004 | 0.003 | 0.004 | 0.005 | 0.001 | 0.001 | 6.50 | 6.00 | 0.22/0.63/0.15 |
| sweepweave | final | 11.71 | 0.963 | 0.773 | 0.050 | 0.284 | 0.054 | 0.264 | 0.003 | 0.033 | 6.71 | 12.00 | 0.39/0.52/0.09 |
| sweepweave | random_intermediate | 8.76 | 0.956 | 0.989 | 0.022 | 0.026 | 0.050 | 0.068 | 0.029 | 0.014 | 7.26 | 12.27 | 0.27/0.61/0.12 |
| sweepweave | random_exit | 6.50 | 0.943 | 0.984 | 0.038 | 0.045 | 0.066 | 0.092 | 0.035 | 0.018 | 5.22 | 10.47 | 0.23/0.61/0.16 |

### `id` lengths, compute price lambda = 0.016

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 7.48 | 0.920 | 0.868 | 0.033 | 0.333 | 0.067 | 0.321 | 0.018 | 0.058 | 2.47 | 16.39 | 0.52/0.35/0.13 |
| cyclic | random_intermediate | 5.70 | 0.930 | 0.943 | 0.047 | 0.095 | 0.081 | 0.145 | 0.020 | 0.026 | 3.32 | 16.30 | 0.40/0.42/0.18 |
| cyclic | random_exit | 3.92 | 0.946 | 0.953 | 0.045 | 0.072 | 0.089 | 0.126 | 0.020 | 0.017 | 2.15 | 13.31 | 0.37/0.36/0.27 |
| dfa | final | 5.05 | 0.973 | 0.949 | 0.067 | 0.170 | 0.097 | 0.200 | 0.007 | 0.007 | 2.60 | 20.16 | 0.52/0.28/0.20 |
| dfa | random_intermediate | 6.53 | 0.977 | 0.977 | 0.031 | 0.021 | 0.045 | 0.049 | 0.006 | 0.003 | 5.24 | 7.90 | 0.47/0.34/0.19 |
| dfa | random_exit | 5.07 | 0.966 | 0.972 | 0.042 | 0.044 | 0.062 | 0.068 | 0.009 | 0.010 | 3.63 | 14.55 | 0.30/0.49/0.20 |
| storyworld | final | 5.44 | 0.995 | 0.969 | 0.042 | 0.158 | 0.041 | 0.182 | 0.004 | 0.002 | 2.48 | nan | 0.55/0.26/0.19 |
| storyworld | random_intermediate | 1.46 | 0.995 | 0.986 | 0.009 | 0.009 | 0.015 | 0.018 | 0.003 | 0.002 | 0.41 | nan | 0.16/0.11/0.73 |
| storyworld | random_exit | 1.17 | 0.998 | 0.992 | 0.009 | 0.005 | 0.011 | 0.012 | 0.002 | 0.000 | 0.15 | 1.00 | 0.06/0.07/0.86 |
| sweepweave | final | 8.52 | 0.971 | 0.974 | 0.048 | 0.289 | 0.057 | 0.267 | 0.004 | 0.024 | 3.58 | 11.00 | 0.48/0.40/0.12 |
| sweepweave | random_intermediate | 3.28 | 0.949 | 0.978 | 0.048 | 0.046 | 0.106 | 0.098 | 0.043 | 0.008 | 1.77 | 4.40 | 0.44/0.25/0.31 |
| sweepweave | random_exit | 3.09 | 0.942 | 0.982 | 0.066 | 0.041 | 0.101 | 0.099 | 0.043 | 0.012 | 1.85 | 7.81 | 0.24/0.43/0.33 |

Calibration gain (ECE softmax - ECE Q_halt, lambda=0.004, `id` lengths) over all 72 models: +0.0600 [+0.0362, +0.0837], sign-flip p=<1e-4.
