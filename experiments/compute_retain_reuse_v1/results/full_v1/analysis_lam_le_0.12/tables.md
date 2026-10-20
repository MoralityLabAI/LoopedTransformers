## 1. Compute and retain (mean over seeds, 95% t-interval)

### Group `id` (lengths [4, 8])

| task | regime | seeds | solved ever | first-correct loop (median) | acc @ loop 8 | acc @ final loop (32) | stable-solve | lost after solve | forget hazard / loop |
|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | 0.996 [0.989, 1.002] | 5.167 [3.940, 6.394] | 0.920 [0.856, 0.984] | 0.196 [0.138, 0.253] | 0.007 [-0.009, 0.023] | 0.993 [0.976, 1.009] | 0.268 [0.145, 0.391] |
| cyclic | random_intermediate | 6 | 0.975 [0.956, 0.994] | 1.500 [0.925, 2.075] | 0.944 [0.913, 0.976] | 0.684 [0.604, 0.763] | 0.641 [0.517, 0.766] | 0.342 [0.216, 0.469] | 0.013 [0.008, 0.019] |
| cyclic | random_exit | 6 | 0.982 [0.976, 0.988] | 1.000 [1.000, 1.000] | 0.946 [0.925, 0.967] | 0.566 [0.529, 0.604] | 0.528 [0.496, 0.561] | 0.462 [0.427, 0.497] | 0.019 [0.017, 0.021] |
| dfa | final | 6 | 0.986 [0.983, 0.989] | 2.000 [2.000, 2.000] | 0.986 [0.983, 0.989] | 0.506 [0.291, 0.721] | 0.377 [0.159, 0.594] | 0.618 [0.399, 0.838] | 0.034 [0.016, 0.053] |
| dfa | random_intermediate | 6 | 0.986 [0.983, 0.989] | 1.000 [1.000, 1.000] | 0.986 [0.983, 0.989] | 0.761 [0.701, 0.821] | 0.711 [0.645, 0.777] | 0.279 [0.212, 0.346] | 0.011 [0.008, 0.013] |
| dfa | random_exit | 6 | 0.985 [0.979, 0.991] | 1.000 [1.000, 1.000] | 0.974 [0.958, 0.989] | 0.760 [0.688, 0.832] | 0.734 [0.658, 0.809] | 0.255 [0.180, 0.329] | 0.010 [0.006, 0.013] |
| storyworld | final | 6 | 1.000 [1.000, 1.000] | 3.000 [3.000, 3.000] | 1.000 [1.000, 1.000] | 0.293 [0.198, 0.389] | 0.242 [0.151, 0.333] | 0.758 [0.667, 0.849] | 0.039 [0.031, 0.048] |
| storyworld | random_intermediate | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.858 [0.826, 0.889] | 0.856 [0.824, 0.888] | 0.144 [0.112, 0.176] | 0.005 [0.004, 0.006] |
| storyworld | random_exit | 6 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | 0.715 [0.641, 0.789] | 0.712 [0.639, 0.785] | 0.288 [0.215, 0.361] | 0.010 [0.007, 0.013] |
| sweepweave | final | 6 | 1.000 [1.000, 1.000] | 4.833 [4.405, 5.262] | 0.999 [0.999, 1.000] | 0.096 [0.059, 0.133] | 0.029 [0.014, 0.045] | 0.971 [0.955, 0.986] | 0.090 [0.085, 0.096] |
| sweepweave | random_intermediate | 6 | 0.999 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.993 [0.990, 0.996] | 0.319 [0.236, 0.402] | 0.265 [0.172, 0.358] | 0.734 [0.641, 0.828] | 0.034 [0.028, 0.041] |
| sweepweave | random_exit | 6 | 0.996 [0.995, 0.998] | 1.000 [1.000, 1.000] | 0.981 [0.978, 0.984] | 0.275 [0.200, 0.350] | 0.227 [0.146, 0.308] | 0.772 [0.691, 0.854] | 0.037 [0.031, 0.044] |

### Group `extrap` (lengths [10, 12])

| task | regime | seeds | solved ever | first-correct loop (median) | acc @ loop 8 | acc @ final loop (32) | stable-solve | lost after solve | forget hazard / loop |
|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | 0.977 [0.955, 0.998] | 5.167 [3.940, 6.394] | 0.448 [0.216, 0.680] | 0.201 [0.139, 0.263] | 0.016 [-0.005, 0.037] | 0.984 [0.962, 1.005] | 0.322 [0.140, 0.503] |
| cyclic | random_intermediate | 6 | 0.803 [0.729, 0.877] | 2.000 [2.000, 2.000] | 0.644 [0.539, 0.750] | 0.512 [0.401, 0.624] | 0.449 [0.320, 0.577] | 0.445 [0.329, 0.562] | 0.023 [0.015, 0.030] |
| cyclic | random_exit | 6 | 0.816 [0.762, 0.870] | 2.000 [2.000, 2.000] | 0.628 [0.533, 0.724] | 0.407 [0.363, 0.451] | 0.356 [0.295, 0.416] | 0.562 [0.478, 0.647] | 0.031 [0.024, 0.038] |
| dfa | final | 6 | 0.990 [0.983, 0.997] | 2.000 [2.000, 2.000] | 0.948 [0.928, 0.967] | 0.516 [0.297, 0.735] | 0.355 [0.159, 0.552] | 0.641 [0.442, 0.840] | 0.038 [0.018, 0.058] |
| dfa | random_intermediate | 6 | 0.988 [0.983, 0.994] | 1.000 [1.000, 1.000] | 0.948 [0.930, 0.966] | 0.726 [0.681, 0.770] | 0.645 [0.608, 0.683] | 0.347 [0.309, 0.385] | 0.015 [0.013, 0.016] |
| dfa | random_exit | 6 | 0.963 [0.935, 0.991] | 1.000 [1.000, 1.000] | 0.883 [0.826, 0.939] | 0.678 [0.590, 0.765] | 0.621 [0.540, 0.702] | 0.356 [0.281, 0.432] | 0.015 [0.012, 0.019] |
| storyworld | final | 6 | 0.883 [0.786, 0.979] | 4.333 [3.791, 4.875] | 0.673 [0.504, 0.842] | 0.231 [0.144, 0.317] | 0.176 [0.104, 0.248] | 0.798 [0.711, 0.884] | 0.057 [0.045, 0.069] |
| storyworld | random_intermediate | 6 | 0.619 [0.510, 0.727] | 3.000 [3.000, 3.000] | 0.474 [0.349, 0.600] | 0.488 [0.438, 0.538] | 0.479 [0.429, 0.530] | 0.214 [0.109, 0.319] | 0.009 [0.005, 0.014] |
| storyworld | random_exit | 6 | 0.404 [0.294, 0.513] | 2.833 [1.802, 3.865] | 0.297 [0.242, 0.353] | 0.295 [0.187, 0.404] | 0.286 [0.188, 0.384] | 0.300 [0.223, 0.376] | 0.013 [0.010, 0.017] |
| sweepweave | final | 6 | 0.859 [0.831, 0.887] | 6.500 [5.925, 7.075] | 0.622 [0.593, 0.651] | 0.107 [0.065, 0.149] | 0.050 [0.025, 0.075] | 0.942 [0.914, 0.970] | 0.107 [0.095, 0.119] |
| sweepweave | random_intermediate | 6 | 0.812 [0.756, 0.868] | 3.167 [2.738, 3.595] | 0.636 [0.559, 0.713] | 0.316 [0.187, 0.445] | 0.248 [0.132, 0.364] | 0.697 [0.561, 0.833] | 0.044 [0.030, 0.058] |
| sweepweave | random_exit | 6 | 0.742 [0.660, 0.824] | 3.167 [2.377, 3.957] | 0.523 [0.445, 0.600] | 0.201 [0.165, 0.237] | 0.161 [0.125, 0.197] | 0.781 [0.724, 0.837] | 0.056 [0.047, 0.065] |

Majority-class accuracy by length (chance reference): cyclic: N=4:0.21, N=5:0.20, N=7:0.20, N=8:0.21, N=10:0.20, N=12:0.21; dfa: N=4:0.33, N=5:0.30, N=7:0.30, N=8:0.30, N=10:0.31, N=12:0.30; storyworld: N=4:0.21, N=5:0.19, N=7:0.10, N=8:0.20, N=10:0.20, N=12:0.20; sweepweave: N=4:0.19, N=5:0.16, N=7:0.16, N=8:0.16, N=10:0.11, N=12:0.17

## 2. Reuse after delayed freezing (held-out lengths)

delta = reuse(d) - reuse(0) on the same samples, averaged per seed; 95% t-interval over seeds.

Headline contrast = largest delay vs 0 (Holm over the kind x task x regime rows using the paired-t p; with 6 seeds the exact sign-flip p cannot go below 0.031). Other delays are descriptive.

| reuse kind | task | regime | delay d | reuse(0) | delta(d-0) [95% CI] | t p | sign-flip p | Holm (t) p |
|---|---|---|---|---|---|---|---|---|
| probe | cyclic | final | 1 | 0.953 | -0.016 [-0.030, -0.002] | 0.0342 | 0.0625 | n/a |
| probe | cyclic | final | 2 | 0.953 | -0.033 [-0.051, -0.015] | 0.0054 | 0.0312 | n/a |
| probe | cyclic | final | 4 | 0.953 | -0.076 [-0.122, -0.031] | 0.0078 | 0.0312 | n/a |
| probe | cyclic | final | 8 | 0.953 | -0.137 [-0.215, -0.059] | 0.0063 | 0.0312 | 0.1079 |
| probe | cyclic | random_intermediate | 1 | 0.948 | -0.011 [-0.031, +0.009] | 0.2157 | 0.2500 | n/a |
| probe | cyclic | random_intermediate | 2 | 0.948 | -0.016 [-0.029, -0.002] | 0.0315 | 0.0625 | n/a |
| probe | cyclic | random_intermediate | 4 | 0.948 | -0.021 [-0.035, -0.008] | 0.0107 | 0.0312 | n/a |
| probe | cyclic | random_intermediate | 8 | 0.948 | -0.033 [-0.050, -0.016] | 0.0040 | 0.0312 | 0.0757 |
| probe | cyclic | random_exit | 1 | 0.951 | -0.009 [-0.019, +0.002] | 0.0928 | 0.1250 | n/a |
| probe | cyclic | random_exit | 2 | 0.951 | -0.009 [-0.022, +0.003] | 0.1126 | 0.1562 | n/a |
| probe | cyclic | random_exit | 4 | 0.951 | -0.021 [-0.036, -0.006] | 0.0139 | 0.0312 | n/a |
| probe | cyclic | random_exit | 8 | 0.951 | -0.039 [-0.058, -0.020] | 0.0034 | 0.0312 | 0.0720 |
| probe | dfa | final | 1 | 0.990 | -0.003 [-0.008, +0.002] | 0.1434 | 0.1875 | n/a |
| probe | dfa | final | 2 | 0.990 | -0.004 [-0.007, -0.000] | 0.0347 | 0.0625 | n/a |
| probe | dfa | final | 4 | 0.990 | -0.001 [-0.005, +0.003] | 0.5358 | 0.5312 | n/a |
| probe | dfa | final | 8 | 0.990 | -0.000 [-0.007, +0.006] | 0.9646 | 0.9688 | 1.0000 |
| probe | dfa | random_intermediate | 1 | 0.988 | -0.001 [-0.003, +0.001] | 0.3104 | 0.2812 | n/a |
| probe | dfa | random_intermediate | 2 | 0.988 | +0.002 [-0.001, +0.004] | 0.1796 | 0.1875 | n/a |
| probe | dfa | random_intermediate | 4 | 0.988 | +0.004 [+0.000, +0.007] | 0.0347 | 0.0625 | n/a |
| probe | dfa | random_intermediate | 8 | 0.988 | +0.004 [-0.001, +0.008] | 0.0827 | 0.0938 | 0.7861 |
| probe | dfa | random_exit | 1 | 0.989 | -0.006 [-0.011, -0.002] | 0.0143 | 0.0625 | n/a |
| probe | dfa | random_exit | 2 | 0.989 | -0.006 [-0.013, +0.001] | 0.0658 | 0.0625 | n/a |
| probe | dfa | random_exit | 4 | 0.989 | -0.006 [-0.013, +0.001] | 0.0814 | 0.0625 | n/a |
| probe | dfa | random_exit | 8 | 0.989 | -0.008 [-0.017, +0.001] | 0.0786 | 0.0312 | 0.7861 |
| probe | storyworld | final | 1 | 0.810 | +0.033 [+0.007, +0.058] | 0.0209 | 0.0625 | n/a |
| probe | storyworld | final | 2 | 0.810 | +0.030 [+0.001, +0.059] | 0.0428 | 0.0312 | n/a |
| probe | storyworld | final | 4 | 0.810 | +0.044 [-0.005, +0.093] | 0.0681 | 0.1250 | n/a |
| probe | storyworld | final | 8 | 0.810 | +0.036 [-0.016, +0.089] | 0.1357 | 0.1250 | 0.9502 |
| probe | storyworld | random_intermediate | 1 | 0.755 | +0.096 [+0.074, +0.118] | 0.0001 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 2 | 0.755 | +0.137 [+0.099, +0.175] | 0.0002 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 4 | 0.755 | +0.173 [+0.131, +0.214] | 0.0001 | 0.0312 | n/a |
| probe | storyworld | random_intermediate | 8 | 0.755 | +0.185 [+0.143, +0.228] | <1e-4 | 0.0312 | 0.0024 |
| probe | storyworld | random_exit | 1 | 0.723 | +0.084 [+0.011, +0.157] | 0.0323 | 0.0625 | n/a |
| probe | storyworld | random_exit | 2 | 0.723 | +0.110 [+0.031, +0.189] | 0.0158 | 0.0625 | n/a |
| probe | storyworld | random_exit | 4 | 0.723 | +0.142 [+0.072, +0.211] | 0.0034 | 0.0312 | n/a |
| probe | storyworld | random_exit | 8 | 0.723 | +0.151 [+0.085, +0.217] | 0.0021 | 0.0312 | 0.0453 |
| probe | sweepweave | final | 1 | 0.905 | +0.004 [-0.003, +0.010] | 0.2202 | 0.2500 | n/a |
| probe | sweepweave | final | 2 | 0.905 | -0.005 [-0.012, +0.002] | 0.1519 | 0.0625 | n/a |
| probe | sweepweave | final | 4 | 0.905 | -0.007 [-0.031, +0.017] | 0.4841 | 0.5938 | n/a |
| probe | sweepweave | final | 8 | 0.905 | -0.022 [-0.056, +0.011] | 0.1495 | 0.1250 | 0.9502 |
| probe | sweepweave | random_intermediate | 1 | 0.879 | +0.037 [+0.024, +0.050] | 0.0007 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 2 | 0.879 | +0.051 [+0.034, +0.068] | 0.0006 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 4 | 0.879 | +0.050 [+0.022, +0.079] | 0.0058 | 0.0312 | n/a |
| probe | sweepweave | random_intermediate | 8 | 0.879 | +0.027 [-0.008, +0.063] | 0.1068 | 0.1250 | 0.8546 |
| probe | sweepweave | random_exit | 1 | 0.881 | +0.037 [+0.024, +0.049] | 0.0007 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 2 | 0.881 | +0.040 [+0.029, +0.050] | 0.0002 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 4 | 0.881 | +0.035 [+0.023, +0.047] | 0.0007 | 0.0312 | n/a |
| probe | sweepweave | random_exit | 8 | 0.881 | +0.002 [-0.018, +0.022] | 0.7897 | 0.7812 | 1.0000 |
| native | cyclic | final | 1 | 0.224 | -0.011 [-0.016, -0.006] | 0.0019 | 0.0312 | n/a |
| native | cyclic | final | 2 | 0.224 | -0.019 [-0.028, -0.010] | 0.0032 | 0.0312 | n/a |
| native | cyclic | final | 4 | 0.224 | -0.022 [-0.038, -0.006] | 0.0181 | 0.0625 | n/a |
| native | cyclic | final | 8 | 0.224 | -0.032 [-0.050, -0.015] | 0.0051 | 0.0625 | 0.0926 |
| native | cyclic | random_intermediate | 1 | 0.212 | -0.008 [-0.014, -0.002] | 0.0177 | 0.0312 | n/a |
| native | cyclic | random_intermediate | 2 | 0.212 | -0.006 [-0.013, +0.001] | 0.0682 | 0.0625 | n/a |
| native | cyclic | random_intermediate | 4 | 0.212 | -0.012 [-0.018, -0.006] | 0.0032 | 0.0312 | n/a |
| native | cyclic | random_intermediate | 8 | 0.212 | -0.011 [-0.021, -0.001] | 0.0403 | 0.0625 | 0.4842 |
| native | cyclic | random_exit | 1 | 0.207 | -0.001 [-0.008, +0.005] | 0.6069 | 0.6250 | n/a |
| native | cyclic | random_exit | 2 | 0.207 | -0.003 [-0.012, +0.007] | 0.5203 | 0.5312 | n/a |
| native | cyclic | random_exit | 4 | 0.207 | +0.000 [-0.010, +0.011] | 0.9383 | 0.9062 | n/a |
| native | cyclic | random_exit | 8 | 0.207 | +0.000 [-0.009, +0.010] | 0.9320 | 0.9688 | 1.0000 |
| native | dfa | final | 1 | 0.814 | -0.002 [-0.017, +0.013] | 0.7168 | 0.7188 | n/a |
| native | dfa | final | 2 | 0.814 | -0.010 [-0.030, +0.011] | 0.2697 | 0.2812 | n/a |
| native | dfa | final | 4 | 0.814 | -0.031 [-0.049, -0.013] | 0.0069 | 0.0312 | n/a |
| native | dfa | final | 8 | 0.814 | -0.061 [-0.092, -0.030] | 0.0038 | 0.0312 | 0.0757 |
| native | dfa | random_intermediate | 1 | 0.831 | -0.007 [-0.019, +0.005] | 0.1787 | 0.1875 | n/a |
| native | dfa | random_intermediate | 2 | 0.831 | -0.011 [-0.029, +0.006] | 0.1568 | 0.1562 | n/a |
| native | dfa | random_intermediate | 4 | 0.831 | -0.022 [-0.047, +0.004] | 0.0806 | 0.1250 | n/a |
| native | dfa | random_intermediate | 8 | 0.831 | -0.039 [-0.070, -0.008] | 0.0234 | 0.0625 | 0.3742 |
| native | dfa | random_exit | 1 | 0.728 | -0.008 [-0.025, +0.008] | 0.2539 | 0.2812 | n/a |
| native | dfa | random_exit | 2 | 0.728 | -0.015 [-0.041, +0.010] | 0.1812 | 0.1562 | n/a |
| native | dfa | random_exit | 4 | 0.728 | -0.022 [-0.048, +0.005] | 0.0894 | 0.1250 | n/a |
| native | dfa | random_exit | 8 | 0.728 | -0.029 [-0.055, -0.003] | 0.0332 | 0.0625 | 0.4313 |
| native | storyworld | final | 1 | 0.356 | +0.002 [-0.015, +0.019] | 0.8281 | 0.8750 | n/a |
| native | storyworld | final | 2 | 0.356 | +0.003 [-0.029, +0.035] | 0.8137 | 0.8438 | n/a |
| native | storyworld | final | 4 | 0.356 | +0.002 [-0.041, +0.044] | 0.9263 | 0.8750 | n/a |
| native | storyworld | final | 8 | 0.356 | -0.041 [-0.121, +0.039] | 0.2474 | 0.1562 | 0.9897 |
| native | storyworld | random_intermediate | 1 | 0.362 | +0.001 [-0.018, +0.019] | 0.9274 | 0.9375 | n/a |
| native | storyworld | random_intermediate | 2 | 0.362 | +0.008 [-0.010, +0.027] | 0.2921 | 0.2812 | n/a |
| native | storyworld | random_intermediate | 4 | 0.362 | +0.023 [-0.002, +0.048] | 0.0661 | 0.1250 | n/a |
| native | storyworld | random_intermediate | 8 | 0.362 | +0.034 [-0.022, +0.091] | 0.1782 | 0.1875 | 0.9502 |
| native | storyworld | random_exit | 1 | 0.266 | -0.000 [-0.014, +0.014] | 0.9850 | 1.0000 | n/a |
| native | storyworld | random_exit | 2 | 0.266 | -0.016 [-0.028, -0.004] | 0.0214 | 0.0312 | n/a |
| native | storyworld | random_exit | 4 | 0.266 | -0.029 [-0.057, -0.002] | 0.0419 | 0.0625 | n/a |
| native | storyworld | random_exit | 8 | 0.266 | -0.036 [-0.073, +0.000] | 0.0502 | 0.0625 | 0.5517 |
| native | sweepweave | final | 1 | 0.401 | -0.061 [-0.077, -0.045] | 0.0002 | 0.0312 | n/a |
| native | sweepweave | final | 2 | 0.401 | -0.122 [-0.161, -0.083] | 0.0005 | 0.0312 | n/a |
| native | sweepweave | final | 4 | 0.401 | -0.197 [-0.265, -0.129] | 0.0007 | 0.0312 | n/a |
| native | sweepweave | final | 8 | 0.401 | -0.258 [-0.349, -0.167] | 0.0008 | 0.0312 | 0.0175 |
| native | sweepweave | random_intermediate | 1 | 0.448 | -0.008 [-0.027, +0.012] | 0.3733 | 0.3750 | n/a |
| native | sweepweave | random_intermediate | 2 | 0.448 | -0.012 [-0.046, +0.022] | 0.3982 | 0.4062 | n/a |
| native | sweepweave | random_intermediate | 4 | 0.448 | -0.035 [-0.093, +0.024] | 0.1882 | 0.1875 | n/a |
| native | sweepweave | random_intermediate | 8 | 0.448 | -0.091 [-0.167, -0.014] | 0.0287 | 0.0312 | 0.4311 |
| native | sweepweave | random_exit | 1 | 0.362 | +0.013 [-0.018, +0.045] | 0.3309 | 0.3125 | n/a |
| native | sweepweave | random_exit | 2 | 0.362 | +0.006 [-0.045, +0.057] | 0.7809 | 0.7812 | n/a |
| native | sweepweave | random_exit | 4 | 0.362 | -0.028 [-0.105, +0.048] | 0.3883 | 0.4688 | n/a |
| native | sweepweave | random_exit | 8 | 0.362 | -0.092 [-0.170, -0.013] | 0.0299 | 0.0625 | 0.4311 |

Pooled delta reuse(d=8) - reuse(0), probe: +0.0139 [95% cluster-bootstrap CI -0.0320, +0.0601] over 12 task x regime clusters.

Pooled delta reuse(d=8) - reuse(0), native: -0.0545 [95% cluster-bootstrap CI -0.1009, -0.0207] over 12 task x regime clusters.

## 3. Controller vs fixed-depth recurrence at matched FLOPs

Endpoint per model = mean over the pre-specified budgets kappa in [2, 3, 4, 6, 8, 12] (mean loop-equivalents per sample; controller overhead counted in FLOPs) of [controller accuracy - fixed-depth accuracy at the same mean FLOPs], taking the controller frontier (10 compute prices) and the fixed-depth curve by linear interpolation (a fractional depth = mixture of floor/ceil depth). A model needs >=3 budgets inside the controller's attainable range to count. Unit of inference = independent training seed. Holm correction (paired-t p) across the task x regime tests of each table.

### Group `heldout` - controller vs fixed-depth

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| cyclic | random_intermediate | 4 | -0.0340 [-0.0481, -0.0200] | 0% | 0.0045 | 0.1250 | 0.0272 | -3.86 |
| cyclic | random_exit | 6 | -0.0328 [-0.0476, -0.0180] | 0% | 0.0023 | 0.0312 | 0.0163 | -2.33 |
| dfa | final | 6 | -0.0129 [-0.0249, -0.0009] | 17% | 0.0392 | 0.0625 | 0.1567 | -1.13 |
| dfa | random_intermediate | 6 | -0.0118 [-0.0236, +0.0000] | 17% | 0.0502 | 0.0938 | 0.1567 | -1.05 |
| dfa | random_exit | 6 | -0.0128 [-0.0254, -0.0003] | 0% | 0.0468 | 0.0312 | 0.1567 | -1.07 |
| storyworld | final | 2 | -0.1031 [-0.4621, +0.2559] | 0% | 0.1703 | 0.5000 | 0.1703 | -2.58 |
| storyworld | random_intermediate | 6 | -0.2111 [-0.3031, -0.1192] | 0% | 0.0020 | 0.0312 | 0.0159 | -2.41 |
| storyworld | random_exit | 6 | -0.1019 [-0.1640, -0.0398] | 0% | 0.0084 | 0.0312 | 0.0418 | -1.72 |
| sweepweave | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | random_intermediate | 5 | -0.0991 [-0.1271, -0.0710] | 0% | 0.0006 | 0.0625 | 0.0061 | -4.39 |
| sweepweave | random_exit | 5 | -0.1150 [-0.1506, -0.0794] | 0% | 0.0009 | 0.0625 | 0.0077 | -4.01 |

Pooled over 10 task x regime clusters: gain = -0.0735 [95% cluster-bootstrap CI -0.1148, -0.0375].

### Group `heldout` - controller vs confidence-threshold early exit

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| cyclic | random_intermediate | 4 | -0.0083 [-0.0401, +0.0234] | 25% | 0.4650 | 0.6250 | 0.9301 | -0.42 |
| cyclic | random_exit | 6 | +0.0085 [-0.0098, +0.0268] | 67% | 0.2878 | 0.2812 | 0.8635 | +0.49 |
| dfa | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_intermediate | 3 | -0.0088 [-0.0540, +0.0365] | 33% | 0.4920 | 0.7500 | 0.9301 | -0.48 |
| dfa | random_exit | 6 | +0.0094 [+0.0018, +0.0171] | 100% | 0.0243 | 0.0312 | 0.0971 | +1.30 |
| storyworld | final | 1 | -0.0172 [+nan, +nan] | 0% | n/a | 1.0000 | n/a | +nan |
| storyworld | random_intermediate | 6 | -0.1794 [-0.2515, -0.1072] | 0% | 0.0014 | 0.0312 | 0.0111 | -2.61 |
| storyworld | random_exit | 6 | -0.0766 [-0.1241, -0.0291] | 0% | 0.0089 | 0.0312 | 0.0447 | -1.69 |
| sweepweave | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | random_intermediate | 5 | -0.0646 [-0.0988, -0.0304] | 0% | 0.0063 | 0.0625 | 0.0441 | -2.35 |
| sweepweave | random_exit | 5 | -0.0555 [-0.0857, -0.0253] | 0% | 0.0070 | 0.0625 | 0.0441 | -2.28 |

Pooled over 9 task x regime clusters: gain = -0.0436 [95% cluster-bootstrap CI -0.0872, -0.0114].

### Group `id` - controller vs fixed-depth

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| cyclic | random_intermediate | 6 | -0.0037 [-0.0140, +0.0065] | 67% | 0.3902 | 0.5000 | 1.0000 | -0.38 |
| cyclic | random_exit | 6 | -0.0071 [-0.0353, +0.0210] | 33% | 0.5432 | 0.7812 | 1.0000 | -0.27 |
| dfa | final | 6 | +0.0018 [-0.0056, +0.0091] | 83% | 0.5677 | 0.5312 | 1.0000 | +0.25 |
| dfa | random_intermediate | 6 | +0.0026 [-0.0053, +0.0104] | 67% | 0.4360 | 0.5000 | 1.0000 | +0.35 |
| dfa | random_exit | 6 | -0.0020 [-0.0143, +0.0103] | 67% | 0.6942 | 0.9062 | 1.0000 | -0.17 |
| storyworld | final | 6 | -0.0024 [-0.0058, +0.0010] | 33% | 0.1335 | 0.1562 | 0.9523 | -0.73 |
| storyworld | random_intermediate | 6 | -0.0022 [-0.0047, +0.0004] | 17% | 0.0783 | 0.0938 | 0.7049 | -0.90 |
| storyworld | random_exit | 6 | -0.0029 [-0.0099, +0.0041] | 50% | 0.3299 | 0.3438 | 1.0000 | -0.44 |
| sweepweave | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | random_intermediate | 6 | -0.0225 [-0.0317, -0.0132] | 0% | 0.0015 | 0.0312 | 0.0154 | -2.55 |
| sweepweave | random_exit | 5 | -0.0137 [-0.0329, +0.0055] | 0% | 0.1190 | 0.0625 | 0.9523 | -0.88 |

Pooled over 10 task x regime clusters: gain = -0.0052 [95% cluster-bootstrap CI -0.0109, -0.0005].

### Group `id` - controller vs confidence-threshold early exit

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| cyclic | random_intermediate | 5 | +0.0292 [+0.0140, +0.0443] | 100% | 0.0059 | 0.0625 | 0.0258 | +2.39 |
| cyclic | random_exit | 5 | +0.0427 [+0.0214, +0.0641] | 100% | 0.0052 | 0.0625 | 0.0258 | +2.48 |
| dfa | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_intermediate | 2 | +0.0064 [-0.0303, +0.0431] | 100% | 0.2707 | 0.5000 | 0.2707 | +1.56 |
| dfa | random_exit | 5 | +0.0159 [-0.0052, +0.0370] | 80% | 0.1039 | 0.1250 | 0.2078 | +0.94 |
| storyworld | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_exit | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | random_intermediate | 6 | +0.0306 [+0.0058, +0.0554] | 83% | 0.0248 | 0.0625 | 0.0743 | +1.30 |
| sweepweave | random_exit | 5 | +0.0551 [+0.0420, +0.0682] | 100% | 0.0003 | 0.0625 | 0.0019 | +5.21 |

Pooled over 6 task x regime clusters: gain = +0.0300 [95% cluster-bootstrap CI +0.0161, +0.0434].

### Post-hoc endpoint: budgets spread over each controller's own attainable range

The pre-specified grid leaves some cells empty because a controller that cannot stop before loop ~5 never reaches kappa=2..4. This second endpoint (chosen after seeing that, hence post hoc) uses 6 equally spaced budgets inside each controller's own cost range: wherever the controller can operate, is it better than fixed depth at the same FLOPs?

### Group `heldout` - controller vs fixed-depth (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1903 [+0.0886, +0.2920] | 100% | 0.0048 | 0.0312 | 0.0484 | +1.96 |
| cyclic | random_intermediate | 6 | +0.0206 [-0.0114, +0.0525] | 67% | 0.1590 | 0.1250 | 0.4769 | +0.68 |
| cyclic | random_exit | 6 | -0.0046 [-0.0248, +0.0157] | 33% | 0.5868 | 0.6250 | 0.6261 | -0.24 |
| dfa | final | 6 | +0.0736 [+0.0158, +0.1315] | 100% | 0.0221 | 0.0312 | 0.1770 | +1.34 |
| dfa | random_intermediate | 6 | +0.0535 [+0.0282, +0.0787] | 100% | 0.0028 | 0.0312 | 0.0339 | +2.23 |
| dfa | random_exit | 6 | +0.0196 [-0.0010, +0.0401] | 83% | 0.0580 | 0.0625 | 0.3478 | +1.00 |
| storyworld | final | 6 | -0.0341 [-0.0745, +0.0062] | 17% | 0.0815 | 0.0625 | 0.4076 | -0.89 |
| storyworld | random_intermediate | 6 | -0.2029 [-0.3002, -0.1057] | 0% | 0.0030 | 0.0312 | 0.0339 | -2.19 |
| storyworld | random_exit | 6 | -0.1023 [-0.1603, -0.0443] | 0% | 0.0062 | 0.0312 | 0.0558 | -1.85 |
| sweepweave | final | 6 | +0.1094 [-0.0376, +0.2563] | 83% | 0.1139 | 0.0938 | 0.4555 | +0.78 |
| sweepweave | random_intermediate | 6 | -0.0328 [-0.1080, +0.0424] | 33% | 0.3130 | 0.3438 | 0.6261 | -0.46 |
| sweepweave | random_exit | 6 | -0.0707 [-0.1358, -0.0055] | 17% | 0.0385 | 0.0938 | 0.2694 | -1.14 |

Pooled over 12 task x regime clusters: gain = +0.0016 [95% cluster-bootstrap CI -0.0538, +0.0606].

### Group `heldout` - controller vs confidence-threshold early exit (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.1460 [+0.0757, +0.2163] | 100% | 0.0031 | 0.0312 | 0.0309 | +2.18 |
| cyclic | random_intermediate | 5 | +0.0208 [+0.0091, +0.0326] | 100% | 0.0079 | 0.0625 | 0.0709 | +2.20 |
| cyclic | random_exit | 6 | +0.0181 [-0.0011, +0.0374] | 83% | 0.0602 | 0.0938 | 0.3150 | +0.99 |
| dfa | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_exit | 3 | +0.0247 [-0.0088, +0.0582] | 100% | 0.0864 | 0.2500 | 0.3457 | +1.83 |
| storyworld | final | 4 | +0.0426 [-0.0189, +0.1042] | 75% | 0.1148 | 0.2500 | 0.3457 | +1.10 |
| storyworld | random_intermediate | 4 | -0.1745 [-0.3003, -0.0488] | 0% | 0.0215 | 0.1250 | 0.1508 | -2.21 |
| storyworld | random_exit | 4 | -0.0764 [-0.1544, +0.0015] | 0% | 0.0525 | 0.1250 | 0.3150 | -1.56 |
| sweepweave | final | 6 | +0.0638 [+0.0195, +0.1080] | 83% | 0.0139 | 0.0625 | 0.1114 | +1.51 |
| sweepweave | random_intermediate | 6 | -0.0214 [-0.0737, +0.0308] | 17% | 0.3393 | 0.3438 | 0.3457 | -0.43 |
| sweepweave | random_exit | 6 | -0.0279 [-0.0619, +0.0062] | 33% | 0.0896 | 0.1250 | 0.3457 | -0.86 |

Pooled over 10 task x regime clusters: gain = +0.0016 [95% cluster-bootstrap CI -0.0519, +0.0529].

### Group `id` - controller vs fixed-depth (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.2204 [+0.0500, +0.3908] | 100% | 0.0209 | 0.0312 | 0.1544 | +1.36 |
| cyclic | random_intermediate | 6 | +0.0483 [+0.0224, +0.0741] | 100% | 0.0049 | 0.0312 | 0.0539 | +1.96 |
| cyclic | random_exit | 6 | +0.0231 [-0.0073, +0.0535] | 100% | 0.1083 | 0.0312 | 0.4675 | +0.80 |
| dfa | final | 6 | +0.0931 [+0.0300, +0.1563] | 100% | 0.0127 | 0.0312 | 0.1258 | +1.55 |
| dfa | random_intermediate | 6 | +0.0703 [+0.0431, +0.0974] | 100% | 0.0012 | 0.0312 | 0.0139 | +2.72 |
| dfa | random_exit | 6 | +0.0339 [+0.0110, +0.0568] | 100% | 0.0126 | 0.0312 | 0.1258 | +1.55 |
| storyworld | final | 6 | +0.0597 [+0.0145, +0.1048] | 100% | 0.0193 | 0.0312 | 0.1544 | +1.39 |
| storyworld | random_intermediate | 6 | +0.0083 [+0.0001, +0.0165] | 83% | 0.0489 | 0.0938 | 0.2933 | +1.06 |
| storyworld | random_exit | 6 | +0.0229 [-0.0056, +0.0514] | 83% | 0.0935 | 0.0938 | 0.4675 | +0.84 |
| sweepweave | final | 6 | +0.1294 [-0.0374, +0.2961] | 100% | 0.1027 | 0.0312 | 0.4675 | +0.81 |
| sweepweave | random_intermediate | 6 | +0.0791 [-0.0260, +0.1842] | 83% | 0.1107 | 0.0938 | 0.4675 | +0.79 |
| sweepweave | random_exit | 6 | +0.0401 [-0.0371, +0.1173] | 67% | 0.2396 | 0.3438 | 0.4675 | +0.54 |

Pooled over 12 task x regime clusters: gain = +0.0690 [95% cluster-bootstrap CI +0.0381, +0.1113].

### Group `id` - controller vs confidence-threshold early exit (post hoc range)

| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Holm (t) p | dz |
|---|---|---|---|---|---|---|---|---|
| cyclic | final | 6 | +0.2102 [+0.0765, +0.3440] | 100% | 0.0099 | 0.0312 | 0.0496 | +1.65 |
| cyclic | random_intermediate | 3 | +0.0464 [+0.0258, +0.0669] | 100% | 0.0105 | 0.2500 | 0.0496 | +5.60 |
| cyclic | random_exit | 5 | +0.0594 [+0.0321, +0.0867] | 100% | 0.0038 | 0.0625 | 0.0227 | +2.70 |
| dfa | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| dfa | random_exit | 2 | +0.0310 [-0.3388, +0.4008] | 100% | 0.4799 | 0.5000 | 0.4799 | +0.75 |
| storyworld | final | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_intermediate | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| storyworld | random_exit | 0 | +nan [+nan, +nan] | nan% | n/a | n/a | n/a | +nan |
| sweepweave | final | 4 | +0.0803 [+0.0661, +0.0945] | 100% | 0.0004 | 0.1250 | 0.0026 | +8.99 |
| sweepweave | random_intermediate | 6 | +0.0731 [+0.0240, +0.1222] | 100% | 0.0123 | 0.0312 | 0.0496 | +1.56 |
| sweepweave | random_exit | 6 | +0.0911 [+0.0313, +0.1509] | 83% | 0.0112 | 0.0625 | 0.0496 | +1.60 |

Pooled over 7 task x regime clusters: gain = +0.0845 [95% cluster-bootstrap CI +0.0497, +0.1349].

### Best accuracy within a budget (post hoc)

For a budget B (mean loop-equivalents incl. controller overhead) compare the best accuracy the controller reaches at any of its 10 compute prices with cost <= B against the best fixed depth with cost <= B. Both sides take a maximum over candidates on the same test data, so the selection is symmetric (but optimistic for both). Entries are mean over (task x regime) clusters of the per-cluster seed-mean gain, with a 95% cluster-bootstrap CI.

| lengths | B=3 | B=4 | B=6 | B=8 | B=12 | B=16 |
|---|---|---|---|---|---|---|
| id | -0.002 [-0.013, +0.009] | -0.003 [-0.015, +0.014] | +0.007 [-0.013, +0.033] | -0.011 [-0.017, -0.006] | -0.012 [-0.018, -0.006] | -0.011 [-0.018, -0.006] |
| heldout | -0.063 [-0.121, -0.015] | -0.084 [-0.133, -0.041] | -0.085 [-0.137, -0.043] | -0.083 [-0.130, -0.043] | -0.082 [-0.128, -0.042] | -0.079 [-0.125, -0.040] |

Per task x regime at B=8 (mean over seeds, 95% t-interval; gain = controller best - fixed best):

| task | regime | id gain | held-out gain |
|---|---|---|---|
| cyclic | final | -0.013 [-0.052, 0.027] | -0.007 [-0.027, 0.014] |
| cyclic | random_intermediate | -0.013 [-0.022, -0.003] | -0.041 [-0.053, -0.029] |
| cyclic | random_exit | -0.008 [-0.016, 0.000] | -0.036 [-0.048, -0.025] |
| dfa | final | -0.010 [-0.016, -0.004] | -0.026 [-0.036, -0.015] |
| dfa | random_intermediate | -0.005 [-0.007, -0.003] | -0.024 [-0.034, -0.014] |
| dfa | random_exit | -0.011 [-0.022, -0.001] | -0.025 [-0.037, -0.012] |
| storyworld | final | -0.004 [-0.007, -0.001] | -0.096 [-0.148, -0.045] |
| storyworld | random_intermediate | -0.002 [-0.004, -0.001] | -0.278 [-0.390, -0.165] |
| storyworld | random_exit | -0.001 [-0.003, 0.000] | -0.152 [-0.245, -0.058] |
| sweepweave | final | -0.011 [-0.014, -0.007] | -0.049 [-0.323, 0.225] |
| sweepweave | random_intermediate | -0.032 [-0.042, -0.021] | -0.131 [-0.173, -0.088] |
| sweepweave | random_exit | -0.026 [-0.038, -0.014] | -0.131 [-0.164, -0.098] |

## 4. Controller diagnostics (mean over seeds)

Calibration: ECE / Brier of the controller's own P(correct if halted now) = Q_halt vs the raw softmax max-probability, on the steps the controller visited. Premature = stopped before the first correct loop (among items solvable within the loop cap); lost = returned a wrong answer after having been correct; excess = loops used beyond the first-correct loop when the answer was right; futile = loops spent on items never solved.

### `heldout` lengths, compute price lambda = 0.004

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 10.01 | 0.693 | 0.618 | 0.083 | 0.389 | 0.144 | 0.368 | 0.089 | 0.208 | 3.62 | 13.65 | 0.53/0.37/0.11 |
| cyclic | random_intermediate | 13.50 | 0.769 | 0.758 | 0.101 | 0.170 | 0.137 | 0.184 | 0.044 | 0.094 | 10.56 | 15.50 | 0.49/0.43/0.08 |
| cyclic | random_exit | 6.56 | 0.751 | 0.786 | 0.148 | 0.192 | 0.187 | 0.220 | 0.094 | 0.073 | 4.06 | 9.00 | 0.48/0.35/0.17 |
| dfa | final | 11.06 | 0.947 | 0.957 | 0.031 | 0.107 | 0.081 | 0.127 | 0.014 | 0.033 | 8.50 | 17.39 | 0.59/0.32/0.09 |
| dfa | random_intermediate | 15.26 | 0.942 | 0.902 | 0.028 | 0.050 | 0.056 | 0.067 | 0.010 | 0.040 | 13.56 | 17.50 | 0.38/0.55/0.07 |
| dfa | random_exit | 7.97 | 0.901 | 0.915 | 0.064 | 0.092 | 0.107 | 0.120 | 0.026 | 0.048 | 6.34 | 8.02 | 0.36/0.50/0.15 |
| storyworld | final | 9.75 | 0.673 | 0.766 | 0.195 | 0.317 | 0.231 | 0.306 | 0.136 | 0.131 | 5.85 | 9.19 | 0.48/0.42/0.10 |
| storyworld | random_intermediate | 3.52 | 0.199 | 0.386 | 0.682 | 0.571 | 0.670 | 0.556 | 0.636 | 0.056 | 3.80 | 3.88 | 0.25/0.45/0.31 |
| storyworld | random_exit | 3.85 | 0.145 | 0.229 | 0.783 | 0.742 | 0.779 | 0.726 | 0.599 | 0.036 | 3.56 | 4.17 | 0.26/0.45/0.29 |
| sweepweave | final | 11.07 | 0.767 | 0.682 | 0.099 | 0.360 | 0.148 | 0.329 | 0.046 | 0.129 | 4.95 | 12.43 | 0.50/0.40/0.10 |
| sweepweave | random_intermediate | 8.55 | 0.693 | 0.779 | 0.182 | 0.184 | 0.209 | 0.209 | 0.156 | 0.079 | 6.61 | 7.68 | 0.36/0.51/0.13 |
| sweepweave | random_exit | 6.88 | 0.609 | 0.716 | 0.227 | 0.268 | 0.256 | 0.279 | 0.224 | 0.073 | 5.37 | 6.62 | 0.36/0.46/0.18 |

### `heldout` lengths, compute price lambda = 0.016

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 8.94 | 0.695 | 0.650 | 0.086 | 0.393 | 0.143 | 0.371 | 0.104 | 0.192 | 2.99 | 11.35 | 0.54/0.34/0.12 |
| cyclic | random_intermediate | 8.51 | 0.760 | 0.787 | 0.122 | 0.201 | 0.161 | 0.220 | 0.063 | 0.085 | 5.32 | 11.63 | 0.48/0.39/0.13 |
| cyclic | random_exit | 4.40 | 0.748 | 0.791 | 0.158 | 0.199 | 0.201 | 0.235 | 0.110 | 0.061 | 2.01 | 6.61 | 0.47/0.27/0.26 |
| dfa | final | 7.42 | 0.946 | 0.969 | 0.033 | 0.132 | 0.104 | 0.159 | 0.020 | 0.027 | 4.99 | 13.30 | 0.60/0.26/0.14 |
| dfa | random_intermediate | 7.64 | 0.947 | 0.963 | 0.044 | 0.058 | 0.087 | 0.091 | 0.017 | 0.029 | 5.97 | 11.05 | 0.43/0.42/0.15 |
| dfa | random_exit | 4.02 | 0.893 | 0.904 | 0.092 | 0.111 | 0.148 | 0.157 | 0.040 | 0.041 | 2.42 | 5.01 | 0.41/0.32/0.27 |
| storyworld | final | 7.91 | 0.648 | 0.727 | 0.204 | 0.330 | 0.238 | 0.316 | 0.171 | 0.123 | 4.15 | 7.11 | 0.52/0.35/0.13 |
| storyworld | random_intermediate | 1.61 | 0.179 | 0.237 | 0.750 | 0.642 | 0.734 | 0.618 | 0.704 | 0.018 | 0.74 | 1.83 | 0.17/0.20/0.63 |
| storyworld | random_exit | 1.54 | 0.139 | 0.156 | 0.838 | 0.777 | 0.833 | 0.754 | 0.641 | 0.010 | 0.54 | 1.63 | 0.18/0.14/0.69 |
| sweepweave | final | 9.58 | 0.767 | 0.762 | 0.105 | 0.361 | 0.145 | 0.327 | 0.055 | 0.120 | 3.51 | 11.10 | 0.52/0.37/0.11 |
| sweepweave | random_intermediate | 5.81 | 0.676 | 0.783 | 0.217 | 0.194 | 0.241 | 0.224 | 0.187 | 0.068 | 3.58 | 5.73 | 0.41/0.39/0.20 |
| sweepweave | random_exit | 4.77 | 0.589 | 0.713 | 0.263 | 0.292 | 0.289 | 0.302 | 0.259 | 0.061 | 2.97 | 5.06 | 0.36/0.41/0.24 |

Calibration gain (ECE softmax - ECE Q_halt, lambda=0.004, `heldout` lengths) over all 72 models: +0.0683 [+0.0399, +0.0966], sign-flip p=<1e-4.

### `id` lengths, compute price lambda = 0.004

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 8.55 | 0.901 | 0.789 | 0.032 | 0.338 | 0.073 | 0.325 | 0.018 | 0.077 | 2.94 | 17.24 | 0.53/0.35/0.12 |
| cyclic | random_intermediate | 12.27 | 0.934 | 0.911 | 0.024 | 0.057 | 0.053 | 0.086 | 0.010 | 0.033 | 9.95 | 20.74 | 0.47/0.44/0.09 |
| cyclic | random_exit | 6.54 | 0.931 | 0.948 | 0.033 | 0.059 | 0.064 | 0.101 | 0.015 | 0.037 | 4.40 | 14.31 | 0.42/0.40/0.17 |
| dfa | final | 11.15 | 0.976 | 0.970 | 0.045 | 0.091 | 0.059 | 0.105 | 0.002 | 0.008 | 8.61 | 27.74 | 0.56/0.35/0.09 |
| dfa | random_intermediate | 15.92 | 0.973 | 0.904 | 0.029 | 0.028 | 0.030 | 0.037 | 0.001 | 0.013 | 14.26 | 29.67 | 0.36/0.57/0.07 |
| dfa | random_exit | 8.26 | 0.960 | 0.956 | 0.042 | 0.044 | 0.056 | 0.061 | 0.009 | 0.016 | 6.67 | 16.61 | 0.34/0.51/0.15 |
| storyworld | final | 9.64 | 0.996 | 0.989 | 0.030 | 0.094 | 0.027 | 0.109 | 0.001 | 0.003 | 6.66 | nan | 0.40/0.49/0.11 |
| storyworld | random_intermediate | 6.05 | 0.997 | 0.999 | 0.007 | 0.005 | 0.007 | 0.010 | 0.001 | 0.002 | 5.00 | nan | 0.23/0.56/0.21 |
| storyworld | random_exit | 7.19 | 0.998 | 0.999 | 0.004 | 0.002 | 0.003 | 0.004 | 0.001 | 0.001 | 6.16 | 1.00 | 0.26/0.59/0.15 |
| sweepweave | final | 9.50 | 0.967 | 0.867 | 0.050 | 0.281 | 0.060 | 0.262 | 0.003 | 0.030 | 4.49 | 13.00 | 0.52/0.37/0.12 |
| sweepweave | random_intermediate | 7.94 | 0.965 | 0.963 | 0.029 | 0.044 | 0.062 | 0.088 | 0.017 | 0.017 | 6.26 | 11.60 | 0.29/0.56/0.15 |
| sweepweave | random_exit | 5.56 | 0.959 | 0.981 | 0.036 | 0.046 | 0.069 | 0.092 | 0.020 | 0.018 | 4.12 | 10.65 | 0.30/0.48/0.22 |

### `id` lengths, compute price lambda = 0.016

| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyclic | final | 7.63 | 0.908 | 0.750 | 0.033 | 0.348 | 0.073 | 0.335 | 0.027 | 0.062 | 2.40 | 14.59 | 0.54/0.32/0.14 |
| cyclic | random_intermediate | 7.31 | 0.934 | 0.939 | 0.035 | 0.080 | 0.076 | 0.126 | 0.015 | 0.027 | 4.89 | 17.76 | 0.44/0.40/0.15 |
| cyclic | random_exit | 4.04 | 0.936 | 0.951 | 0.041 | 0.071 | 0.086 | 0.131 | 0.020 | 0.026 | 2.07 | 12.51 | 0.42/0.30/0.28 |
| dfa | final | 7.11 | 0.976 | 0.984 | 0.054 | 0.125 | 0.081 | 0.147 | 0.005 | 0.005 | 4.67 | 21.68 | 0.57/0.29/0.14 |
| dfa | random_intermediate | 7.65 | 0.979 | 0.982 | 0.048 | 0.040 | 0.056 | 0.060 | 0.002 | 0.005 | 6.06 | 21.89 | 0.41/0.44/0.15 |
| dfa | random_exit | 3.75 | 0.958 | 0.959 | 0.058 | 0.057 | 0.089 | 0.093 | 0.013 | 0.014 | 2.20 | 12.22 | 0.39/0.31/0.30 |
| storyworld | final | 6.72 | 0.996 | 0.988 | 0.037 | 0.130 | 0.038 | 0.150 | 0.002 | 0.002 | 3.77 | nan | 0.46/0.39/0.16 |
| storyworld | random_intermediate | 2.02 | 0.997 | 0.994 | 0.010 | 0.010 | 0.017 | 0.020 | 0.001 | 0.001 | 0.98 | nan | 0.15/0.31/0.54 |
| storyworld | random_exit | 2.19 | 0.998 | 0.996 | 0.007 | 0.003 | 0.007 | 0.007 | 0.001 | 0.001 | 1.17 | 1.00 | 0.22/0.28/0.51 |
| sweepweave | final | 8.13 | 0.970 | 0.939 | 0.055 | 0.292 | 0.061 | 0.271 | 0.004 | 0.026 | 3.14 | 13.00 | 0.54/0.33/0.13 |
| sweepweave | random_intermediate | 4.63 | 0.961 | 0.983 | 0.046 | 0.049 | 0.093 | 0.101 | 0.024 | 0.014 | 2.93 | 10.00 | 0.36/0.39/0.25 |
| sweepweave | random_exit | 3.41 | 0.956 | 0.983 | 0.050 | 0.042 | 0.094 | 0.101 | 0.026 | 0.015 | 1.99 | 6.88 | 0.27/0.41/0.32 |

Calibration gain (ECE softmax - ECE Q_halt, lambda=0.004, `id` lengths) over all 72 models: +0.0606 [+0.0360, +0.0851], sign-flip p=<1e-4.
