### Training receipts

| run | params | val top-1 on choice states (last tick) | steps/s (median) | peak VRAM MiB | thermal/RAM pauses | checkpoint sha256 |
|---|---:|---:|---:|---:|---:|---|
| L_t8_s17b | 30,202,771 | 99.9 | 1.59 | 4532 | 1 (gpu_temp) | `25c0338efa7e` |
| M_t8_s17 | 15,849,875 | 99.9 | 3.03 | 9777 | 5 (gpu_temp) | `9e2b9160dbc7` |
| S_nograph_s17 | 5,186,387 | 49.1 | 7.76 | 5007 | 1 (ram_avail) | `2ad45b6abf7d` |
| S_t1_s17 | 5,186,387 | 49.8 | 29.02 | 1324 | 2 (gpu_temp, ram_avail) | `88e44da26060` |
| S_t2_s17 | 5,186,387 | 97.3 | 20.63 | 1933 | 1 (gpu_temp) | `a2de7d6ebdba` |
| S_t4_s17 | 5,186,387 | 99.4 | 14.57 | 2807 | 1 (gpu_temp) | `a4832c92de38` |
| S_t8_s17b | 5,186,387 | 99.9 | 7.76 | 4979 | 1 (gpu_temp) | `1a1f7fa46af5` |
| S_t8_s29 | 5,186,387 | 99.9 | 8.04 | 4977 | 2 (gpu_temp) | `c1baaccdc023` |
| S_t8_s43 | 5,186,387 | 99.9 | 8.14 | 4986 | 2 (gpu_temp) | `9fe8c69b36a8` |
| Sloop_s17b | 5,278,354 | 99.9 | 4.69 | 6775 | 5 (gpu_temp) | `57098adb27ad` |
| flatL_s17 | 30,236,442 | 99.9 | 4.88 | 6845 | 5 (gpu_temp, ram_avail) | `b641059aa68d` |
| flatM_s17 | 15,891,442 | 99.9 | 6.96 | 5332 | 2 (gpu_temp) | `53185d788ecf` |
| flatS_s17 | 5,358,682 | 99.9 | 10.55 | 3749 | 1 (gpu_temp) | `448b215adcac` |
| flatS_s29 | 5,358,682 | 99.9 | 10.81 | 3744 | 1 (gpu_temp) | `966d64a33a0a` |
| flatS_s43 | 5,358,682 | 99.9 | 10.67 | 3756 | 1 (gpu_temp) | `c76f1730bf09` |

### Policy `best`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalMT16_S_t8_s17b | – | – | – | – | – | 41.1 / 88.3 [360] | – | – | – | – |
| evalMT16_S_t8_s17b_globbug | – | – | – | – | – | 41.4 / 88.3 [360] | – | – | – | – |
| evalMT32_S_t8_s17b | – | – | – | – | – | 39.4 / 87.2 [360] | – | – | – | – |
| evalMT32_S_t8_s17b_globbug | – | – | – | – | – | 38.9 / 87.5 [360] | – | – | – | – |
| evalM_S_t8_s17b | – | – | – | – | – | 50.8 / 89.2 [360] | – | – | – | – |
| evalM_S_t8_s17b_globbug | – | – | – | – | – | 48.9 / 88.9 [360] | – | – | – | – |
| evalM_flatS_s17 | – | – | – | – | – | 58.1 / 86.9 [360] | – | – | – | – |
| evalR_S_t8_s17b | – | – | – | – | 99.7 / 100.0 [300] | – | – | – | – | – |
| evalR_S_t8_s17b_globbug | – | – | – | – | 99.3 / 100.0 [300] | – | – | – | – | – |
| evalR_flatS_s17 | – | – | – | – | 99.3 / 100.0 [300] | – | – | – | – | – |
| evalT16_L_t8_s17b | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 31.4 / 70.6 [360] | – | – | 531.4 | 16.0 |
| evalT16_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 97.0 / 99.7 [300] | 80.6 / 98.1 [360] | – | – | 280.1 | 16.0 |
| evalT16_S_t8_s17b | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 41.1 / 88.3 [360] | – | – | 92.8 | 16.0 |
| evalT16_Sloop_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 85.3 / 98.0 [300] | 60.6 / 92.5 [360] | – | – | 217.9 | 16.0 |
| evalT32_L_t8_s17b | – | 100.0 / 100.0 [360] | – | – | 97.7 / 99.0 [300] | 31.4 / 71.4 [360] | – | – | 1062.7 | 32.0 |
| evalT32_M_t8_s17 | – | 98.9 / 100.0 [360] | – | – | 95.7 / 100.0 [300] | 80.6 / 98.3 [360] | – | – | 560.7 | 32.0 |
| evalT32_S_t8_s17b | – | 91.7 / 100.0 [360] | – | – | 92.7 / 99.7 [300] | 39.4 / 87.2 [360] | – | – | 187.0 | 32.0 |
| evalT32_Sloop_s17b | – | 99.2 / 100.0 [360] | – | – | 89.3 / 98.7 [300] | 65.0 / 93.9 [360] | – | – | 436.0 | 32.0 |
| eval_L_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 29.2 / 70.0 [360] | – | – | 265.7 | 8.0 |
| eval_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 78.9 / 93.3 [360] | – | – | 140.1 | 8.0 |
| eval_S_nograph_s17 | 15.0 / 61.7 [300] | 0.0 / 43.6 [360] | 8.7 / 30.3 [300] | 17.3 / 79.0 [300] | 0.3 / 14.0 [300] | 0.8 / 13.6 [360] | – | – | 105.3 | 8.0 |
| eval_S_t1_s17 | 45.0 / 75.0 [300] | 18.6 / 58.6 [360] | 18.3 / 56.3 [300] | 50.0 / 90.3 [300] | 5.3 / 20.3 [300] | 8.9 / 42.5 [360] | – | – | 15.0 | 1.0 |
| eval_S_t2_s17 | 93.7 / 100.0 [300] | 81.1 / 99.2 [360] | 97.3 / 100.0 [300] | 89.7 / 100.0 [300] | 78.7 / 98.0 [300] | 35.6 / 76.4 [360] | – | – | 14.2 | 2.0 |
| eval_S_t4_s17 | 98.3 / 100.0 [300] | 95.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.0 / 100.0 [300] | 94.7 / 99.7 [300] | 31.1 / 69.4 [360] | – | – | 23.6 | 4.0 |
| eval_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 46.4 | 8.0 |
| eval_S_t8_s17b_globbug | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 46.4 | 8.0 |
| eval_S_t8_s29 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 26.9 / 73.6 [360] | – | – | 46.4 | 8.0 |
| eval_S_t8_s43 | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 61.1 / 93.9 [360] | – | – | 46.4 | 8.0 |
| eval_Sloop_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 48.6 / 80.3 [360] | – | – | 109.0 | 8.0 |
| eval_flatL_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 94.3 / 100.0 [300] | 99.3 / 100.0 [300] | 63.9 / 92.2 [360] | – | – | 71.6 | 8.0 |
| eval_flatM_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 46.4 / 74.2 [360] | – | – | 38.0 | 8.0 |
| eval_flatS_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 13.1 | 8.0 |
| eval_flatS_s29 | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 51.7 / 76.9 [360] | – | – | 13.1 | 8.0 |
| eval_flatS_s43 | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 86.1 / 97.5 [360] | – | – | 13.1 | 8.0 |
| evalcgrt_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 78.9 / 93.3 [360] | – | – | 140.1 | 8.0 |
| evalcgrt_S_t8_s29 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 26.9 / 73.6 [360] | – | – | 46.4 | 8.0 |
| evalcgrt_S_t8_s43 | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 61.1 / 93.9 [360] | – | – | 46.4 | 8.0 |
| evalhalt_L_t8_s17b | 99.3 / 100.0 [300] | 93.1 / 100.0 [360] | 99.7 / 100.0 [300] | 93.0 / 100.0 [300] | 94.0 / 100.0 [300] | 46.7 / 79.2 [360] | – | – | 69.9 | 2.05 |
| evalhalt_M_t8_s17 | 97.7 / 100.0 [300] | 86.7 / 100.0 [360] | 99.0 / 100.0 [300] | 93.0 / 100.0 [300] | 93.3 / 100.0 [300] | 48.1 / 83.9 [360] | – | – | 37.3 | 2.03 |
| evalhalt_S_t8_s17b | 95.3 / 100.0 [300] | 81.4 / 98.9 [360] | 98.7 / 100.0 [300] | 92.7 / 100.0 [300] | 79.3 / 95.0 [300] | 29.4 / 73.6 [360] | – | – | 12.6 | 2.0 |
| evalnohist_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 50.8 / 89.2 [360] | – | – | 46.4 | 8.0 |
| evalnohist_flatS_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 58.1 / 86.9 [360] | – | – | 13.1 | 8.0 |

### Policy `dfs`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalM_S_t8_s17b | – | – | – | – | – | 47.5 / 76.9 [360] | – | – | – | – |
| evalM_S_t8_s17b_globbug | – | – | – | – | – | 44.4 / 75.6 [360] | – | – | – | – |
| evalM_flatS_s17 | – | – | – | – | – | 58.9 / 75.3 [360] | – | – | – | – |
| evalR_S_t8_s17b | – | – | – | – | 99.3 / 99.7 [300] | – | – | – | – | – |
| evalR_S_t8_s17b_globbug | – | – | – | – | 99.0 / 99.7 [300] | – | – | – | – | – |
| evalR_flatS_s17 | – | – | – | – | 99.3 / 99.7 [300] | – | – | – | – | – |
| evalT16_L_t8_s17b | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 99.7 [300] | 28.9 / 61.4 [360] | – | – | 531.4 | 16.0 |
| evalT16_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 97.0 / 99.7 [300] | 83.9 / 91.9 [360] | – | – | 280.1 | 16.0 |
| evalT16_S_t8_s17b | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 98.3 / 100.0 [300] | 99.7 / 99.7 [300] | 38.6 / 73.1 [360] | – | – | 92.8 | 16.0 |
| evalT16_Sloop_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 85.3 / 98.3 [300] | 60.3 / 84.4 [360] | – | – | 217.9 | 16.0 |
| eval_L_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 26.1 / 57.5 [360] | – | – | 265.7 | 8.0 |
| eval_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.0 / 99.7 [300] | 81.4 / 93.3 [360] | – | – | 140.0 | 8.0 |
| eval_S_nograph_s17 | 36.7 / 79.3 [300] | 6.9 / 76.7 [360] | 14.3 / 57.0 [300] | 39.0 / 93.7 [300] | 7.7 / 19.3 [300] | 6.1 / 39.4 [360] | – | – | 69.7 | 8.0 |
| eval_S_t1_s17 | 84.3 / 98.7 [300] | 62.2 / 98.9 [360] | 54.7 / 95.3 [300] | 77.7 / 100.0 [300] | 62.3 / 78.3 [300] | 37.8 / 76.1 [360] | – | – | 7.1 | 1.0 |
| eval_S_t2_s17 | 91.7 / 98.7 [300] | 81.9 / 98.9 [360] | 97.0 / 99.7 [300] | 89.0 / 100.0 [300] | 78.3 / 85.0 [300] | 31.4 / 68.1 [360] | – | – | 14.1 | 2.0 |
| eval_S_t4_s17 | 97.7 / 99.7 [300] | 95.0 / 100.0 [360] | 99.3 / 99.7 [300] | 93.3 / 100.0 [300] | 93.0 / 94.7 [300] | 28.3 / 56.7 [360] | – | – | 23.6 | 4.0 |
| eval_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 46.4 | 8.0 |
| eval_S_t8_s17b_globbug | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 46.4 | 8.0 |
| eval_S_t8_s29 | 100.0 / 100.0 [300] | 99.7 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 17.8 / 53.3 [360] | – | – | 46.4 | 8.0 |
| eval_S_t8_s43 | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 65.6 / 86.7 [360] | – | – | 46.4 | 8.0 |
| eval_Sloop_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 46.4 / 74.2 [360] | – | – | 109.0 | 8.0 |
| eval_flatL_s17 | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 94.3 / 100.0 [300] | 99.3 / 99.7 [300] | 63.1 / 82.5 [360] | – | – | 71.7 | 8.0 |
| eval_flatM_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 46.1 / 67.8 [360] | – | – | 38.0 | 8.0 |
| eval_flatS_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 13.1 | 8.0 |
| eval_flatS_s29 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 52.2 / 71.7 [360] | – | – | 13.1 | 8.0 |
| eval_flatS_s43 | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 85.8 / 92.5 [360] | – | – | 13.1 | 8.0 |
| evalhalt_L_t8_s17b | 99.3 / 100.0 [300] | 91.7 / 100.0 [360] | 99.0 / 100.0 [300] | 93.0 / 100.0 [300] | 96.3 / 97.7 [300] | 46.1 / 81.1 [360] | – | – | 69.6 | 2.05 |
| evalhalt_M_t8_s17 | 98.3 / 100.0 [300] | 89.4 / 100.0 [360] | 99.3 / 100.0 [300] | 92.7 / 100.0 [300] | 96.0 / 97.7 [300] | 43.1 / 75.3 [360] | – | – | 36.8 | 2.04 |
| evalhalt_S_t8_s17b | 93.7 / 98.7 [300] | 81.4 / 98.6 [360] | 98.0 / 99.7 [300] | 92.7 / 100.0 [300] | 79.7 / 87.7 [300] | 36.7 / 67.8 [360] | – | – | 12.4 | 2.0 |
| evalnohist_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 47.5 / 77.2 [360] | – | – | 46.4 | 8.0 |
| evalnohist_flatS_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.7 [300] | 58.9 / 75.3 [360] | – | – | 13.1 | 8.0 |

### Policy `greedy`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalMT16_S_t8_s17b | – | – | – | – | – | 40.3 / 40.3 [360] | – | – | – | – |
| evalMT16_S_t8_s17b_globbug | – | – | – | – | – | 40.6 / 40.6 [360] | – | – | – | – |
| evalMT32_S_t8_s17b | – | – | – | – | – | 38.6 / 38.6 [360] | – | – | – | – |
| evalMT32_S_t8_s17b_globbug | – | – | – | – | – | 39.4 / 39.4 [360] | – | – | – | – |
| evalM_S_t8_s17b | – | – | – | – | – | 47.8 / 47.8 [360] | – | – | – | – |
| evalM_S_t8_s17b_globbug | – | – | – | – | – | 45.8 / 45.8 [360] | – | – | – | – |
| evalM_flatS_s17 | – | – | – | – | – | 59.4 / 59.4 [360] | – | – | – | – |
| evalR_S_t8_s17b | – | – | – | – | 99.7 / 99.7 [300] | – | – | – | – | – |
| evalR_S_t8_s17b_globbug | – | – | – | – | 99.3 / 99.3 [300] | – | – | – | – | – |
| evalR_flatS_s17 | – | – | – | – | 99.3 / 99.3 [300] | – | – | – | – | – |
| evalT32_L_t8_s17b | – | 100.0 / 100.0 [360] | – | – | 97.7 / 97.7 [300] | 29.2 / 29.2 [360] | – | – | 988.1 | 32.0 |
| evalT32_M_t8_s17 | – | 99.4 / 99.4 [360] | – | – | 96.3 / 96.3 [300] | 83.3 / 83.3 [360] | – | – | 519.6 | 32.0 |
| evalT32_S_t8_s17b | – | 93.3 / 93.3 [360] | – | – | 93.0 / 93.0 [300] | 38.6 / 38.6 [360] | – | – | 164.3 | 32.0 |
| evalT32_Sloop_s17b | – | 99.4 / 99.4 [360] | – | – | 90.0 / 90.0 [300] | 65.6 / 65.6 [360] | – | – | 405.1 | 32.0 |
| eval_L_t8_s17b | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 25.8 / 25.8 [360] | – | – | 246.7 | 8.0 |
| eval_M_t8_s17 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 81.1 / 81.1 [360] | – | – | 130.0 | 8.0 |
| eval_S_nograph_s17 | 17.0 / 21.7 [300] | 2.5 / 4.2 [360] | 10.7 / 18.7 [300] | 26.3 / 28.7 [300] | 4.0 / 5.3 [300] | 6.1 / 7.8 [360] | – | – | 19.0 | 8.0 |
| eval_S_t1_s17 | 77.0 / 77.0 [300] | 61.7 / 61.7 [360] | 39.7 / 39.7 [300] | 70.3 / 72.3 [300] | 61.3 / 61.3 [300] | 37.5 / 37.5 [360] | – | – | 4.6 | 1.0 |
| eval_S_t2_s17 | 92.7 / 92.7 [300] | 86.7 / 86.7 [360] | 96.7 / 96.7 [300] | 89.3 / 95.7 [300] | 77.0 / 77.0 [300] | 33.9 / 33.9 [360] | – | – | 11.7 | 2.0 |
| eval_S_t4_s17 | 98.7 / 98.7 [300] | 95.0 / 95.0 [360] | 100.0 / 100.0 [300] | 93.0 / 99.7 [300] | 93.7 / 93.7 [300] | 27.8 / 27.8 [360] | – | – | 21.1 | 4.0 |
| eval_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 43.1 | 8.0 |
| eval_S_t8_s17b_globbug | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 43.1 | 8.0 |
| eval_S_t8_s29 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 99.7 [300] | 17.8 / 17.8 [360] | – | – | 43.1 | 8.0 |
| eval_S_t8_s43 | 100.0 / 100.0 [300] | 99.4 / 99.4 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 99.7 [300] | 65.8 / 65.8 [360] | – | – | 43.0 | 8.0 |
| eval_Sloop_s17b | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 46.7 / 46.7 [360] | – | – | 101.2 | 8.0 |
| eval_flatL_s17 | 100.0 / 100.0 [300] | 99.4 / 99.4 [360] | 100.0 / 100.0 [300] | 94.3 / 100.0 [300] | 99.3 / 99.3 [300] | 63.3 / 63.3 [360] | – | – | 66.4 | 8.0 |
| eval_flatM_s17 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 45.6 / 45.6 [360] | – | – | 35.3 | 8.0 |
| eval_flatS_s17 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | – | – | – | – | 12.2 | 8.0 |
| eval_flatS_s29 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 52.2 / 52.2 [360] | – | – | 12.2 | 8.0 |
| eval_flatS_s43 | 100.0 / 100.0 [300] | 99.2 / 99.2 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 86.4 / 86.4 [360] | – | – | 12.2 | 8.0 |
| evalnohist_S_t8_s17b | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 99.7 [300] | 47.8 / 47.8 [360] | – | – | 43.1 | 8.0 |
| evalnohist_flatS_s17 | 100.0 / 100.0 [300] | 99.7 / 99.7 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 99.3 [300] | 59.4 / 59.4 [360] | – | – | 12.2 | 8.0 |

### Policy `cgrt`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalcgrt_M_t8_s17 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 100.0 / 100.0 [300] | 69.2 / 93.6 [360] | – | – | 560.2 | 10.67 |
| evalcgrt_S_t8_s29 | 100.0 / 100.0 [300] | 100.0 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.3 / 100.0 [300] | 25.0 / 76.4 [360] | – | – | 185.5 | 10.67 |
| evalcgrt_S_t8_s43 | 100.0 / 100.0 [300] | 99.4 / 100.0 [360] | 100.0 / 100.0 [300] | 93.3 / 100.0 [300] | 99.7 / 100.0 [300] | 46.4 / 93.6 [360] | – | – | 185.6 | 10.67 |

### Policy `cgrt_shuf`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalcgrt_M_t8_s17 | 19.7 / 68.3 [300] | 1.1 / 45.3 [360] | 14.3 / 51.3 [300] | 25.3 / 87.0 [300] | 3.7 / 14.3 [300] | 2.8 / 24.7 [360] | – | – | 1187.7 | 10.67 |
| evalcgrt_S_t8_s29 | 21.0 / 69.0 [300] | 1.4 / 43.6 [360] | 14.7 / 53.0 [300] | 26.7 / 87.0 [300] | 4.7 / 14.3 [300] | 3.3 / 24.7 [360] | – | – | 397.1 | 10.67 |
| evalcgrt_S_t8_s43 | 20.3 / 69.3 [300] | 1.1 / 44.2 [360] | 14.0 / 52.0 [300] | 26.0 / 86.7 [300] | 4.0 / 15.0 [300] | 3.6 / 26.4 [360] | – | – | 394.1 | 10.67 |

### Policy `random_dfs`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalM_baselines | – | – | – | – | – | 4.4 / 40.6 [360] | – | – | – | – |
| evalR_baselines | – | – | – | – | 6.0 / 26.0 [300] | – | – | – | – | – |
| eval_baselines | 23.7 / 75.7 [300] | 4.7 / 68.6 [360] | 21.0 / 70.3 [300] | 29.7 / 93.7 [300] | – | – | – | – | – | – |

### Policy `bfs`: certified success % at b1 / b2 (n per split in brackets)

| eval job | test_id | test_depth | test_family | test_dep | test_reason | test_mirror | test_mirror_id | test_mirror_deep | GFLOP/theorem (test_depth) | mean fast steps |
|---|---|---|---|---|---|---|---|---|---:|---:|
| evalM_baselines | – | – | – | – | – | 0.3 / 9.4 [360] | – | – | – | – |
| evalR_baselines | – | – | – | – | 0.0 / 12.0 [300] | – | – | – | – | – |
| eval_baselines | 7.0 / 57.3 [300] | 0.0 / 33.6 [360] | 6.0 / 32.7 [300] | 7.0 / 69.0 [300] | – | – | – | – | – | – |

### Seed spread (3 seeds each)

| model / split / policy / budget | mean % | sd | per seed |
|---|---:|---:|---|
| HRM-S | test_id | dfs | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_id | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_id | best | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_id | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_id | greedy | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_id | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_depth | dfs | b1 | 99.7 | 0.3 | 100.0, 99.7, 99.4 |
| HRM-S | test_depth | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_depth | best | b1 | 99.9 | 0.2 | 100.0, 100.0, 99.7 |
| HRM-S | test_depth | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_depth | greedy | b1 | 99.7 | 0.3 | 100.0, 99.7, 99.4 |
| HRM-S | test_depth | greedy | b2 | 99.7 | 0.3 | 100.0, 99.7, 99.4 |
| HRM-S | test_family | dfs | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_family | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_family | best | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_family | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_family | greedy | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_family | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_dep | dfs | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| HRM-S | test_dep | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_dep | best | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| HRM-S | test_dep | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_dep | greedy | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| HRM-S | test_dep | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_reason | dfs | b1 | 99.3 | 0.0 | 99.3, 99.3, 99.3 |
| HRM-S | test_reason | dfs | b2 | 99.7 | 0.0 | 99.7, 99.7, 99.7 |
| HRM-S | test_reason | best | b1 | 99.7 | 0.0 | 99.7, 99.7, 99.7 |
| HRM-S | test_reason | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| HRM-S | test_reason | greedy | b1 | 99.7 | 0.0 | 99.7, 99.7, 99.7 |
| HRM-S | test_reason | greedy | b2 | 99.7 | 0.0 | 99.7, 99.7, 99.7 |
| HRM-S | test_mirror | dfs | b1 | 43.6 | 24.1 | 47.5, 17.8, 65.6 |
| HRM-S | test_mirror | dfs | b2 | 72.3 | 17.1 | 76.9, 53.3, 86.7 |
| HRM-S | test_mirror | best | b1 | 46.3 | 17.5 | 50.8, 26.9, 61.1 |
| HRM-S | test_mirror | best | b2 | 85.6 | 10.6 | 89.2, 73.6, 93.9 |
| HRM-S | test_mirror | greedy | b1 | 43.8 | 24.3 | 47.8, 17.8, 65.8 |
| HRM-S | test_mirror | greedy | b2 | 43.8 | 24.3 | 47.8, 17.8, 65.8 |
| flat-S | test_id | dfs | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_id | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_id | best | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_id | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_id | greedy | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_id | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_depth | dfs | b1 | 99.8 | 0.3 | 100.0, 100.0, 99.4 |
| flat-S | test_depth | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_depth | best | b1 | 99.7 | 0.3 | 100.0, 99.7, 99.4 |
| flat-S | test_depth | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_depth | greedy | b1 | 99.5 | 0.3 | 99.7, 99.7, 99.2 |
| flat-S | test_depth | greedy | b2 | 99.5 | 0.3 | 99.7, 99.7, 99.2 |
| flat-S | test_family | dfs | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_family | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_family | best | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_family | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_family | greedy | b1 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_family | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_dep | dfs | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| flat-S | test_dep | dfs | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_dep | best | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| flat-S | test_dep | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_dep | greedy | b1 | 93.3 | 0.0 | 93.3, 93.3, 93.3 |
| flat-S | test_dep | greedy | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_reason | dfs | b1 | 99.3 | 0.0 | 99.3, 99.3, 99.3 |
| flat-S | test_reason | dfs | b2 | 99.7 | 0.0 | 99.7, 99.7, 99.7 |
| flat-S | test_reason | best | b1 | 99.3 | 0.0 | 99.3, 99.3, 99.3 |
| flat-S | test_reason | best | b2 | 100.0 | 0.0 | 100.0, 100.0, 100.0 |
| flat-S | test_reason | greedy | b1 | 99.3 | 0.0 | 99.3, 99.3, 99.3 |
| flat-S | test_reason | greedy | b2 | 99.3 | 0.0 | 99.3, 99.3, 99.3 |
| flat-S | test_mirror | dfs | b1 | 65.6 | 17.8 | 58.9, 52.2, 85.8 |
| flat-S | test_mirror | dfs | b2 | 79.8 | 11.1 | 75.3, 71.7, 92.5 |
| flat-S | test_mirror | best | b1 | 65.3 | 18.3 | 58.1, 51.7, 86.1 |
| flat-S | test_mirror | best | b2 | 87.1 | 10.3 | 86.9, 76.9, 97.5 |
| flat-S | test_mirror | greedy | b1 | 66.0 | 18.0 | 59.4, 52.2, 86.4 |
| flat-S | test_mirror | greedy | b2 | 66.0 | 18.0 | 59.4, 52.2, 86.4 |

### Paired comparisons (same theorems; theorem-level bootstrap, 10,000 resamples)

| comparison | difference (points) | 95% interval | n |
|---|---:|---|---:|
| HRM-S vs flat-S | test_id | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_id | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_id | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_id | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_depth | dfs | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-S | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-S | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-S | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-S | test_depth | greedy | b1 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs flat-S | test_depth | greedy | b2 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs flat-S | test_family | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_family | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_dep | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_reason | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_reason | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_reason | best | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-S | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-S | test_reason | greedy | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-S | test_reason | greedy | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-S | test_mirror | dfs | b1 | -11.4 | [-16.1, -6.7] | 360 |
| HRM-S vs flat-S | test_mirror | dfs | b2 | +1.7 | [-1.4, +4.7] | 360 |
| HRM-S vs flat-S | test_mirror | best | b1 | -7.2 | [-11.7, -2.8] | 360 |
| HRM-S vs flat-S | test_mirror | best | b2 | +2.2 | [-1.1, +5.6] | 360 |
| HRM-S vs flat-S | test_mirror | greedy | b1 | -11.7 | [-16.4, -6.9] | 360 |
| HRM-S vs flat-S | test_mirror | greedy | b2 | -11.7 | [-16.4, -6.9] | 360 |
| HRM-M vs flat-M | test_id | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_id | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_id | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_id | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_depth | dfs | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_depth | greedy | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_depth | greedy | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-M vs flat-M | test_family | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_family | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_dep | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | dfs | b1 | -0.3 | [-1.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_reason | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-M vs flat-M | test_mirror | dfs | b1 | +35.3 | [+29.7, +40.6] | 360 |
| HRM-M vs flat-M | test_mirror | dfs | b2 | +25.6 | [+21.1, +30.3] | 360 |
| HRM-M vs flat-M | test_mirror | best | b1 | +32.5 | [+26.9, +37.8] | 360 |
| HRM-M vs flat-M | test_mirror | best | b2 | +19.2 | [+15.0, +23.6] | 360 |
| HRM-M vs flat-M | test_mirror | greedy | b1 | +35.6 | [+30.0, +40.8] | 360 |
| HRM-M vs flat-M | test_mirror | greedy | b2 | +35.6 | [+30.0, +40.8] | 360 |
| HRM-L vs flat-L | test_id | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_id | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_id | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_id | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_depth | dfs | b1 | +0.6 | [+0.0, +1.4] | 360 |
| HRM-L vs flat-L | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-L vs flat-L | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-L vs flat-L | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-L vs flat-L | test_depth | greedy | b1 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-L vs flat-L | test_depth | greedy | b2 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-L vs flat-L | test_family | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_family | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | dfs | b1 | -1.0 | [-2.3, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | best | b1 | -1.0 | [-2.3, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | greedy | b1 | -1.0 | [-2.3, +0.0] | 300 |
| HRM-L vs flat-L | test_dep | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_reason | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-L vs flat-L | test_mirror | dfs | b1 | -36.9 | [-41.9, -31.9] | 360 |
| HRM-L vs flat-L | test_mirror | dfs | b2 | -25.0 | [-29.7, -20.3] | 360 |
| HRM-L vs flat-L | test_mirror | best | b1 | -34.7 | [-39.7, -29.7] | 360 |
| HRM-L vs flat-L | test_mirror | best | b2 | -22.2 | [-26.7, -17.8] | 360 |
| HRM-L vs flat-L | test_mirror | greedy | b1 | -37.5 | [-42.5, -32.5] | 360 |
| HRM-L vs flat-L | test_mirror | greedy | b2 | -37.5 | [-42.5, -32.5] | 360 |
| HRM-S vs single-clock loop | test_id | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_id | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_id | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_id | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_depth | dfs | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs single-clock loop | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs single-clock loop | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs single-clock loop | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs single-clock loop | test_depth | greedy | b1 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs single-clock loop | test_depth | greedy | b2 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs single-clock loop | test_family | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_family | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_dep | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_reason | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_reason | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_reason | best | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs single-clock loop | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs single-clock loop | test_reason | greedy | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs single-clock loop | test_reason | greedy | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs single-clock loop | test_mirror | dfs | b1 | +1.1 | [-3.9, +6.4] | 360 |
| HRM-S vs single-clock loop | test_mirror | dfs | b2 | +2.8 | [-1.1, +6.9] | 360 |
| HRM-S vs single-clock loop | test_mirror | best | b1 | +2.2 | [-2.8, +7.2] | 360 |
| HRM-S vs single-clock loop | test_mirror | best | b2 | +8.9 | [+4.7, +13.3] | 360 |
| HRM-S vs single-clock loop | test_mirror | greedy | b1 | +1.1 | [-4.2, +6.4] | 360 |
| HRM-S vs single-clock loop | test_mirror | greedy | b2 | +1.1 | [-4.2, +6.4] | 360 |
| HRM-S vs no-graph | test_id | dfs | b1 | +63.3 | [+58.0, +68.7] | 300 |
| HRM-S vs no-graph | test_id | dfs | b2 | +20.7 | [+16.0, +25.3] | 300 |
| HRM-S vs no-graph | test_id | best | b1 | +85.0 | [+80.7, +89.0] | 300 |
| HRM-S vs no-graph | test_id | best | b2 | +38.3 | [+33.0, +43.7] | 300 |
| HRM-S vs no-graph | test_id | greedy | b1 | +83.0 | [+78.7, +87.0] | 300 |
| HRM-S vs no-graph | test_id | greedy | b2 | +78.3 | [+73.7, +83.0] | 300 |
| HRM-S vs no-graph | test_depth | dfs | b1 | +93.1 | [+90.3, +95.6] | 360 |
| HRM-S vs no-graph | test_depth | dfs | b2 | +23.3 | [+18.9, +27.8] | 360 |
| HRM-S vs no-graph | test_depth | best | b1 | +100.0 | [+100.0, +100.0] | 360 |
| HRM-S vs no-graph | test_depth | best | b2 | +56.4 | [+51.4, +61.4] | 360 |
| HRM-S vs no-graph | test_depth | greedy | b1 | +97.5 | [+95.8, +98.9] | 360 |
| HRM-S vs no-graph | test_depth | greedy | b2 | +95.8 | [+93.6, +97.8] | 360 |
| HRM-S vs no-graph | test_family | dfs | b1 | +85.7 | [+81.7, +89.7] | 300 |
| HRM-S vs no-graph | test_family | dfs | b2 | +43.0 | [+37.3, +48.7] | 300 |
| HRM-S vs no-graph | test_family | best | b1 | +91.3 | [+88.0, +94.3] | 300 |
| HRM-S vs no-graph | test_family | best | b2 | +69.7 | [+64.3, +74.7] | 300 |
| HRM-S vs no-graph | test_family | greedy | b1 | +89.3 | [+85.7, +92.7] | 300 |
| HRM-S vs no-graph | test_family | greedy | b2 | +81.3 | [+76.7, +85.7] | 300 |
| HRM-S vs no-graph | test_dep | dfs | b1 | +54.3 | [+48.3, +60.0] | 300 |
| HRM-S vs no-graph | test_dep | dfs | b2 | +6.3 | [+3.7, +9.0] | 300 |
| HRM-S vs no-graph | test_dep | best | b1 | +76.0 | [+71.0, +80.7] | 300 |
| HRM-S vs no-graph | test_dep | best | b2 | +21.0 | [+16.7, +25.7] | 300 |
| HRM-S vs no-graph | test_dep | greedy | b1 | +67.0 | [+61.7, +72.0] | 300 |
| HRM-S vs no-graph | test_dep | greedy | b2 | +71.3 | [+66.0, +76.3] | 300 |
| HRM-S vs no-graph | test_reason | dfs | b1 | +91.7 | [+88.3, +94.7] | 300 |
| HRM-S vs no-graph | test_reason | dfs | b2 | +80.3 | [+75.7, +84.7] | 300 |
| HRM-S vs no-graph | test_reason | best | b1 | +99.3 | [+98.3, +100.0] | 300 |
| HRM-S vs no-graph | test_reason | best | b2 | +86.0 | [+82.0, +89.7] | 300 |
| HRM-S vs no-graph | test_reason | greedy | b1 | +95.7 | [+93.3, +97.7] | 300 |
| HRM-S vs no-graph | test_reason | greedy | b2 | +94.3 | [+91.7, +96.7] | 300 |
| HRM-S vs no-graph | test_mirror | dfs | b1 | +41.4 | [+36.1, +46.7] | 360 |
| HRM-S vs no-graph | test_mirror | dfs | b2 | +37.5 | [+31.4, +43.3] | 360 |
| HRM-S vs no-graph | test_mirror | best | b1 | +50.0 | [+44.7, +55.0] | 360 |
| HRM-S vs no-graph | test_mirror | best | b2 | +75.6 | [+71.1, +80.0] | 360 |
| HRM-S vs no-graph | test_mirror | greedy | b1 | +41.7 | [+36.1, +47.2] | 360 |
| HRM-S vs no-graph | test_mirror | greedy | b2 | +40.0 | [+34.2, +45.8] | 360 |
| HRM-S T8 vs T1 | test_id | dfs | b1 | +15.7 | [+11.7, +20.0] | 300 |
| HRM-S T8 vs T1 | test_id | dfs | b2 | +1.3 | [+0.3, +2.7] | 300 |
| HRM-S T8 vs T1 | test_id | best | b1 | +55.0 | [+49.3, +60.7] | 300 |
| HRM-S T8 vs T1 | test_id | best | b2 | +25.0 | [+20.3, +30.0] | 300 |
| HRM-S T8 vs T1 | test_id | greedy | b1 | +23.0 | [+18.3, +27.7] | 300 |
| HRM-S T8 vs T1 | test_id | greedy | b2 | +23.0 | [+18.3, +27.7] | 300 |
| HRM-S T8 vs T1 | test_depth | dfs | b1 | +37.8 | [+32.8, +42.8] | 360 |
| HRM-S T8 vs T1 | test_depth | dfs | b2 | +1.1 | [+0.3, +2.2] | 360 |
| HRM-S T8 vs T1 | test_depth | best | b1 | +81.4 | [+77.2, +85.3] | 360 |
| HRM-S T8 vs T1 | test_depth | best | b2 | +41.4 | [+36.4, +46.4] | 360 |
| HRM-S T8 vs T1 | test_depth | greedy | b1 | +38.3 | [+33.3, +43.3] | 360 |
| HRM-S T8 vs T1 | test_depth | greedy | b2 | +38.3 | [+33.3, +43.3] | 360 |
| HRM-S T8 vs T1 | test_family | dfs | b1 | +45.3 | [+39.7, +51.0] | 300 |
| HRM-S T8 vs T1 | test_family | dfs | b2 | +4.7 | [+2.3, +7.3] | 300 |
| HRM-S T8 vs T1 | test_family | best | b1 | +81.7 | [+77.3, +86.0] | 300 |
| HRM-S T8 vs T1 | test_family | best | b2 | +43.7 | [+38.0, +49.3] | 300 |
| HRM-S T8 vs T1 | test_family | greedy | b1 | +60.3 | [+55.0, +66.0] | 300 |
| HRM-S T8 vs T1 | test_family | greedy | b2 | +60.3 | [+55.0, +66.0] | 300 |
| HRM-S T8 vs T1 | test_dep | dfs | b1 | +15.7 | [+11.0, +20.7] | 300 |
| HRM-S T8 vs T1 | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T1 | test_dep | best | b1 | +43.3 | [+37.7, +49.0] | 300 |
| HRM-S T8 vs T1 | test_dep | best | b2 | +9.7 | [+6.7, +13.0] | 300 |
| HRM-S T8 vs T1 | test_dep | greedy | b1 | +23.0 | [+17.7, +28.3] | 300 |
| HRM-S T8 vs T1 | test_dep | greedy | b2 | +27.7 | [+22.7, +32.7] | 300 |
| HRM-S T8 vs T1 | test_reason | dfs | b1 | +37.0 | [+31.7, +42.3] | 300 |
| HRM-S T8 vs T1 | test_reason | dfs | b2 | +21.3 | [+16.7, +26.0] | 300 |
| HRM-S T8 vs T1 | test_reason | best | b1 | +94.3 | [+91.7, +97.0] | 300 |
| HRM-S T8 vs T1 | test_reason | best | b2 | +79.7 | [+75.0, +84.0] | 300 |
| HRM-S T8 vs T1 | test_reason | greedy | b1 | +38.3 | [+33.0, +43.7] | 300 |
| HRM-S T8 vs T1 | test_reason | greedy | b2 | +38.3 | [+33.0, +43.7] | 300 |
| HRM-S T8 vs T1 | test_mirror | dfs | b1 | +9.7 | [+5.3, +14.4] | 360 |
| HRM-S T8 vs T1 | test_mirror | dfs | b2 | +0.8 | [-3.1, +4.7] | 360 |
| HRM-S T8 vs T1 | test_mirror | best | b1 | +41.9 | [+36.7, +47.5] | 360 |
| HRM-S T8 vs T1 | test_mirror | best | b2 | +46.7 | [+41.1, +52.2] | 360 |
| HRM-S T8 vs T1 | test_mirror | greedy | b1 | +10.3 | [+5.6, +15.0] | 360 |
| HRM-S T8 vs T1 | test_mirror | greedy | b2 | +10.3 | [+5.6, +15.0] | 360 |
| HRM-S T8 vs T2 | test_id | dfs | b1 | +8.3 | [+5.3, +11.7] | 300 |
| HRM-S T8 vs T2 | test_id | dfs | b2 | +1.3 | [+0.3, +2.7] | 300 |
| HRM-S T8 vs T2 | test_id | best | b1 | +6.3 | [+3.7, +9.3] | 300 |
| HRM-S T8 vs T2 | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T2 | test_id | greedy | b1 | +7.3 | [+4.3, +10.3] | 300 |
| HRM-S T8 vs T2 | test_id | greedy | b2 | +7.3 | [+4.3, +10.3] | 300 |
| HRM-S T8 vs T2 | test_depth | dfs | b1 | +18.1 | [+14.2, +21.9] | 360 |
| HRM-S T8 vs T2 | test_depth | dfs | b2 | +1.1 | [+0.3, +2.2] | 360 |
| HRM-S T8 vs T2 | test_depth | best | b1 | +18.9 | [+15.0, +23.1] | 360 |
| HRM-S T8 vs T2 | test_depth | best | b2 | +0.8 | [+0.0, +1.9] | 360 |
| HRM-S T8 vs T2 | test_depth | greedy | b1 | +13.3 | [+10.0, +16.9] | 360 |
| HRM-S T8 vs T2 | test_depth | greedy | b2 | +13.3 | [+10.0, +16.9] | 360 |
| HRM-S T8 vs T2 | test_family | dfs | b1 | +3.0 | [+1.3, +5.0] | 300 |
| HRM-S T8 vs T2 | test_family | dfs | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T2 | test_family | best | b1 | +2.7 | [+1.0, +4.7] | 300 |
| HRM-S T8 vs T2 | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T2 | test_family | greedy | b1 | +3.3 | [+1.3, +5.3] | 300 |
| HRM-S T8 vs T2 | test_family | greedy | b2 | +3.3 | [+1.3, +5.3] | 300 |
| HRM-S T8 vs T2 | test_dep | dfs | b1 | +4.3 | [+2.3, +6.7] | 300 |
| HRM-S T8 vs T2 | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T2 | test_dep | best | b1 | +3.7 | [+1.7, +6.0] | 300 |
| HRM-S T8 vs T2 | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T2 | test_dep | greedy | b1 | +4.0 | [+2.0, +6.3] | 300 |
| HRM-S T8 vs T2 | test_dep | greedy | b2 | +4.3 | [+2.3, +6.7] | 300 |
| HRM-S T8 vs T2 | test_reason | dfs | b1 | +21.0 | [+16.7, +25.7] | 300 |
| HRM-S T8 vs T2 | test_reason | dfs | b2 | +14.7 | [+10.7, +18.7] | 300 |
| HRM-S T8 vs T2 | test_reason | best | b1 | +21.0 | [+16.3, +25.7] | 300 |
| HRM-S T8 vs T2 | test_reason | best | b2 | +2.0 | [+0.7, +3.7] | 300 |
| HRM-S T8 vs T2 | test_reason | greedy | b1 | +22.7 | [+18.0, +27.3] | 300 |
| HRM-S T8 vs T2 | test_reason | greedy | b2 | +22.7 | [+18.0, +27.3] | 300 |
| HRM-S T8 vs T2 | test_mirror | dfs | b1 | +16.1 | [+11.4, +20.8] | 360 |
| HRM-S T8 vs T2 | test_mirror | dfs | b2 | +8.9 | [+4.4, +13.6] | 360 |
| HRM-S T8 vs T2 | test_mirror | best | b1 | +15.3 | [+11.1, +19.7] | 360 |
| HRM-S T8 vs T2 | test_mirror | best | b2 | +12.8 | [+9.2, +16.7] | 360 |
| HRM-S T8 vs T2 | test_mirror | greedy | b1 | +13.9 | [+9.2, +18.6] | 360 |
| HRM-S T8 vs T2 | test_mirror | greedy | b2 | +13.9 | [+9.2, +18.6] | 360 |
| HRM-S T8 vs T4 | test_id | dfs | b1 | +2.3 | [+0.7, +4.3] | 300 |
| HRM-S T8 vs T4 | test_id | dfs | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_id | best | b1 | +1.7 | [+0.3, +3.3] | 300 |
| HRM-S T8 vs T4 | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_id | greedy | b1 | +1.3 | [+0.3, +2.7] | 300 |
| HRM-S T8 vs T4 | test_id | greedy | b2 | +1.3 | [+0.3, +2.7] | 300 |
| HRM-S T8 vs T4 | test_depth | dfs | b1 | +5.0 | [+2.8, +7.5] | 360 |
| HRM-S T8 vs T4 | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S T8 vs T4 | test_depth | best | b1 | +5.0 | [+2.8, +7.5] | 360 |
| HRM-S T8 vs T4 | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S T8 vs T4 | test_depth | greedy | b1 | +5.0 | [+2.8, +7.5] | 360 |
| HRM-S T8 vs T4 | test_depth | greedy | b2 | +5.0 | [+2.8, +7.5] | 360 |
| HRM-S T8 vs T4 | test_family | dfs | b1 | +0.7 | [+0.0, +1.7] | 300 |
| HRM-S T8 vs T4 | test_family | dfs | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_dep | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_dep | best | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S T8 vs T4 | test_dep | greedy | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_dep | greedy | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_reason | dfs | b1 | +6.3 | [+3.7, +9.3] | 300 |
| HRM-S T8 vs T4 | test_reason | dfs | b2 | +5.0 | [+2.7, +7.7] | 300 |
| HRM-S T8 vs T4 | test_reason | best | b1 | +5.0 | [+2.7, +7.7] | 300 |
| HRM-S T8 vs T4 | test_reason | best | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S T8 vs T4 | test_reason | greedy | b1 | +6.0 | [+3.3, +8.7] | 300 |
| HRM-S T8 vs T4 | test_reason | greedy | b2 | +6.0 | [+3.3, +8.7] | 300 |
| HRM-S T8 vs T4 | test_mirror | dfs | b1 | +19.2 | [+12.8, +25.6] | 360 |
| HRM-S T8 vs T4 | test_mirror | dfs | b2 | +20.3 | [+14.7, +25.8] | 360 |
| HRM-S T8 vs T4 | test_mirror | best | b1 | +19.7 | [+14.2, +25.6] | 360 |
| HRM-S T8 vs T4 | test_mirror | best | b2 | +19.7 | [+15.0, +24.4] | 360 |
| HRM-S T8 vs T4 | test_mirror | greedy | b1 | +20.0 | [+13.6, +26.4] | 360 |
| HRM-S T8 vs T4 | test_mirror | greedy | b2 | +20.0 | [+13.6, +26.4] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_id | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_id | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_id | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_id | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | dfs | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | dfs | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | greedy | b1 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_depth | greedy | b2 | +0.3 | [+0.0, +0.8] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_family | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_family | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_family | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_family | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | greedy | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_dep | greedy | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | dfs | b1 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | dfs | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | best | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | greedy | b1 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_reason | greedy | b2 | +0.3 | [+0.0, +1.0] | 300 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | dfs | b1 | +1.4 | [-3.3, +6.1] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | dfs | b2 | +9.2 | [+5.6, +12.8] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | best | b1 | +4.4 | [+0.3, +8.6] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | best | b2 | +15.0 | [+10.8, +19.4] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | greedy | b1 | +2.2 | [-2.5, +6.9] | 360 |
| HRM-S vs flat-M (FLOP-matched) | test_mirror | greedy | b2 | +2.2 | [-2.5, +6.9] | 360 |
| CGRT vs shuffled (M_t8_s17) | test_id | b1 | +80.3 | [+75.7, +84.7] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_id | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_id | b2 | +31.7 | [+26.7, +37.0] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_id | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_depth | b1 | +98.9 | [+97.8, +99.7] | 360 |
| CGRT vs best-first 1-context (M_t8_s17) | test_depth | b1 | +0.0 | [+0.0, +0.0] | 360 |
| CGRT vs shuffled (M_t8_s17) | test_depth | b2 | +54.7 | [+49.4, +60.0] | 360 |
| CGRT vs best-first 1-context (M_t8_s17) | test_depth | b2 | +0.0 | [+0.0, +0.0] | 360 |
| CGRT vs shuffled (M_t8_s17) | test_family | b1 | +85.7 | [+81.7, +89.3] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_family | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_family | b2 | +48.7 | [+43.0, +54.3] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_family | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_dep | b1 | +68.0 | [+62.7, +73.3] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_dep | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_dep | b2 | +13.0 | [+9.3, +17.0] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_dep | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_reason | b1 | +96.3 | [+94.0, +98.3] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_reason | b1 | +0.7 | [+0.0, +1.7] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_reason | b2 | +85.7 | [+81.7, +89.7] | 300 |
| CGRT vs best-first 1-context (M_t8_s17) | test_reason | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (M_t8_s17) | test_mirror | b1 | +66.4 | [+61.4, +71.4] | 360 |
| CGRT vs best-first 1-context (M_t8_s17) | test_mirror | b1 | -9.7 | [-13.3, -6.1] | 360 |
| CGRT vs shuffled (M_t8_s17) | test_mirror | b2 | +68.9 | [+63.9, +73.9] | 360 |
| CGRT vs best-first 1-context (M_t8_s17) | test_mirror | b2 | +0.3 | [-0.8, +1.7] | 360 |
| CGRT vs shuffled (S_t8_s29) | test_id | b1 | +79.0 | [+74.3, +83.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_id | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_id | b2 | +31.0 | [+26.0, +36.3] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_id | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_depth | b1 | +98.6 | [+97.2, +99.7] | 360 |
| CGRT vs best-first 1-context (S_t8_s29) | test_depth | b1 | +0.0 | [+0.0, +0.0] | 360 |
| CGRT vs shuffled (S_t8_s29) | test_depth | b2 | +56.4 | [+51.4, +61.4] | 360 |
| CGRT vs best-first 1-context (S_t8_s29) | test_depth | b2 | +0.0 | [+0.0, +0.0] | 360 |
| CGRT vs shuffled (S_t8_s29) | test_family | b1 | +85.3 | [+81.3, +89.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_family | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_family | b2 | +47.0 | [+41.3, +52.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_family | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_dep | b1 | +66.7 | [+61.0, +72.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_dep | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_dep | b2 | +13.0 | [+9.3, +17.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_dep | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_reason | b1 | +94.7 | [+92.0, +97.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_reason | b1 | -0.3 | [-1.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_reason | b2 | +85.7 | [+81.7, +89.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s29) | test_reason | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s29) | test_mirror | b1 | +21.7 | [+16.9, +26.4] | 360 |
| CGRT vs best-first 1-context (S_t8_s29) | test_mirror | b1 | -1.9 | [-3.9, +0.0] | 360 |
| CGRT vs shuffled (S_t8_s29) | test_mirror | b2 | +51.7 | [+46.1, +57.0] | 360 |
| CGRT vs best-first 1-context (S_t8_s29) | test_mirror | b2 | +2.8 | [+0.8, +5.0] | 360 |
| CGRT vs shuffled (S_t8_s43) | test_id | b1 | +79.7 | [+75.0, +84.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_id | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_id | b2 | +30.7 | [+25.3, +36.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_id | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_depth | b1 | +98.3 | [+96.9, +99.4] | 360 |
| CGRT vs best-first 1-context (S_t8_s43) | test_depth | b1 | -0.3 | [-0.8, +0.0] | 360 |
| CGRT vs shuffled (S_t8_s43) | test_depth | b2 | +55.8 | [+50.8, +61.1] | 360 |
| CGRT vs best-first 1-context (S_t8_s43) | test_depth | b2 | +0.0 | [+0.0, +0.0] | 360 |
| CGRT vs shuffled (S_t8_s43) | test_family | b1 | +86.0 | [+82.0, +89.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_family | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_family | b2 | +48.0 | [+42.3, +53.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_family | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_dep | b1 | +67.3 | [+61.7, +72.7] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_dep | b1 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_dep | b2 | +13.3 | [+9.7, +17.3] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_dep | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_reason | b1 | +95.7 | [+93.3, +98.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_reason | b1 | +0.0 | [-1.0, +1.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_reason | b2 | +85.0 | [+80.7, +89.0] | 300 |
| CGRT vs best-first 1-context (S_t8_s43) | test_reason | b2 | +0.0 | [+0.0, +0.0] | 300 |
| CGRT vs shuffled (S_t8_s43) | test_mirror | b1 | +42.8 | [+37.5, +48.1] | 360 |
| CGRT vs best-first 1-context (S_t8_s43) | test_mirror | b1 | -14.7 | [-18.6, -11.1] | 360 |
| CGRT vs shuffled (S_t8_s43) | test_mirror | b2 | +67.2 | [+62.2, +72.2] | 360 |
| CGRT vs best-first 1-context (S_t8_s43) | test_mirror | b2 | -0.3 | [-1.9, +1.4] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_depth | best | b1 | -0.6 | [-1.4, +0.0] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_reason | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| infer T16 vs T8 (S_t8_s17b) | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| infer T16 vs T8 (S_t8_s17b) | test_mirror | best | b1 | -9.7 | [-13.1, -6.7] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_mirror | best | b2 | -0.8 | [-3.6, +1.9] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_mirror | greedy | b1 | -7.5 | [-11.1, -4.2] | 360 |
| infer T16 vs T8 (S_t8_s17b) | test_mirror | greedy | b2 | -7.5 | [-11.1, -4.2] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_depth | best | b1 | -8.3 | [-11.4, -5.6] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_depth | greedy | b1 | -6.7 | [-9.4, -4.2] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_depth | greedy | b2 | -6.7 | [-9.4, -4.2] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_reason | best | b1 | -7.0 | [-10.0, -4.3] | 300 |
| infer T32 vs T8 (S_t8_s17b) | test_reason | best | b2 | -0.3 | [-1.0, +0.0] | 300 |
| infer T32 vs T8 (S_t8_s17b) | test_reason | greedy | b1 | -6.7 | [-9.7, -3.7] | 300 |
| infer T32 vs T8 (S_t8_s17b) | test_reason | greedy | b2 | -6.7 | [-9.7, -3.7] | 300 |
| infer T32 vs T8 (S_t8_s17b) | test_mirror | best | b1 | -11.4 | [-15.3, -7.8] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_mirror | best | b2 | -1.9 | [-5.0, +1.1] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_mirror | greedy | b1 | -9.2 | [-13.1, -5.6] | 360 |
| infer T32 vs T8 (S_t8_s17b) | test_mirror | greedy | b2 | -9.2 | [-13.1, -5.6] | 360 |
| history visible vs hidden (S_t8_s17b) | test_id | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_id | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| history visible vs hidden (S_t8_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| history visible vs hidden (S_t8_s17b) | test_family | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_family | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_dep | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_dep | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_reason | best | b1 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| history visible vs hidden (S_t8_s17b) | test_mirror | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| history visible vs hidden (S_t8_s17b) | test_mirror | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (M_t8_s17) | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (M_t8_s17) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (M_t8_s17) | test_reason | best | b1 | -2.3 | [-4.7, -0.3] | 300 |
| infer T16 vs T8 (M_t8_s17) | test_reason | best | b2 | -0.3 | [-1.0, +0.0] | 300 |
| infer T16 vs T8 (M_t8_s17) | test_mirror | best | b1 | +1.7 | [-1.7, +5.0] | 360 |
| infer T16 vs T8 (M_t8_s17) | test_mirror | best | b2 | +4.7 | [+2.5, +6.9] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_depth | best | b1 | -1.1 | [-2.2, -0.3] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_depth | greedy | b1 | -0.3 | [-1.4, +0.6] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_depth | greedy | b2 | -0.3 | [-1.4, +0.6] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_reason | best | b1 | -3.7 | [-6.0, -1.3] | 300 |
| infer T32 vs T8 (M_t8_s17) | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| infer T32 vs T8 (M_t8_s17) | test_reason | greedy | b1 | -3.0 | [-5.3, -1.0] | 300 |
| infer T32 vs T8 (M_t8_s17) | test_reason | greedy | b2 | -3.0 | [-5.3, -1.0] | 300 |
| infer T32 vs T8 (M_t8_s17) | test_mirror | best | b1 | +1.7 | [-1.7, +5.0] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_mirror | best | b2 | +5.0 | [+2.8, +7.5] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_mirror | greedy | b1 | +2.2 | [-1.1, +5.8] | 360 |
| infer T32 vs T8 (M_t8_s17) | test_mirror | greedy | b2 | +2.2 | [-1.1, +5.8] | 360 |
| infer T16 vs T8 (L_t8_s17b) | test_depth | best | b1 | -0.3 | [-0.8, +0.0] | 360 |
| infer T16 vs T8 (L_t8_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (L_t8_s17b) | test_reason | best | b1 | +0.3 | [+0.0, +1.0] | 300 |
| infer T16 vs T8 (L_t8_s17b) | test_reason | best | b2 | +0.0 | [+0.0, +0.0] | 300 |
| infer T16 vs T8 (L_t8_s17b) | test_mirror | best | b1 | +2.2 | [+0.6, +4.2] | 360 |
| infer T16 vs T8 (L_t8_s17b) | test_mirror | best | b2 | +0.6 | [-2.2, +3.3] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_depth | greedy | b1 | +0.3 | [+0.0, +0.8] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_depth | greedy | b2 | +0.3 | [+0.0, +0.8] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_reason | best | b1 | -1.7 | [-3.3, -0.3] | 300 |
| infer T32 vs T8 (L_t8_s17b) | test_reason | best | b2 | -1.0 | [-2.3, +0.0] | 300 |
| infer T32 vs T8 (L_t8_s17b) | test_reason | greedy | b1 | -1.7 | [-3.7, +0.0] | 300 |
| infer T32 vs T8 (L_t8_s17b) | test_reason | greedy | b2 | -1.7 | [-3.7, +0.0] | 300 |
| infer T32 vs T8 (L_t8_s17b) | test_mirror | best | b1 | +2.2 | [+0.0, +4.4] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_mirror | best | b2 | +1.4 | [-1.4, +4.4] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_mirror | greedy | b1 | +3.3 | [+0.6, +6.4] | 360 |
| infer T32 vs T8 (L_t8_s17b) | test_mirror | greedy | b2 | +3.3 | [+0.6, +6.4] | 360 |
| infer T16 vs T8 (Sloop_s17b) | test_depth | best | b1 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (Sloop_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T16 vs T8 (Sloop_s17b) | test_reason | best | b1 | -14.0 | [-18.0, -10.0] | 300 |
| infer T16 vs T8 (Sloop_s17b) | test_reason | best | b2 | -2.0 | [-3.7, -0.7] | 300 |
| infer T16 vs T8 (Sloop_s17b) | test_mirror | best | b1 | +11.9 | [+7.5, +16.7] | 360 |
| infer T16 vs T8 (Sloop_s17b) | test_mirror | best | b2 | +12.2 | [+8.9, +15.8] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_depth | best | b1 | -0.8 | [-1.9, +0.0] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_depth | best | b2 | +0.0 | [+0.0, +0.0] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_depth | greedy | b1 | -0.3 | [-1.1, +0.6] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_depth | greedy | b2 | -0.3 | [-1.1, +0.6] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_reason | best | b1 | -10.0 | [-13.7, -6.7] | 300 |
| infer T32 vs T8 (Sloop_s17b) | test_reason | best | b2 | -1.3 | [-2.7, -0.3] | 300 |
| infer T32 vs T8 (Sloop_s17b) | test_reason | greedy | b1 | -9.3 | [-13.0, -6.0] | 300 |
| infer T32 vs T8 (Sloop_s17b) | test_reason | greedy | b2 | -9.3 | [-13.0, -6.0] | 300 |
| infer T32 vs T8 (Sloop_s17b) | test_mirror | best | b1 | +16.4 | [+11.7, +21.4] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_mirror | best | b2 | +13.6 | [+10.0, +17.5] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_mirror | greedy | b1 | +18.9 | [+13.9, +24.2] | 360 |
| infer T32 vs T8 (Sloop_s17b) | test_mirror | greedy | b2 | +18.9 | [+13.9, +24.2] | 360 |

### CGRT branch scorer (fit on training theorems, checked on held-aside training theorems)

| job | rows | check AUC learned | check AUC prior | check AUC mean log p |
|---|---:|---:|---:|---:|
| branch_L_t8_s17b | 4619 | 1.0 | 1.0 | 1.0 |
| branch_M_t8_s17 | 4619 | 1.0 | 1.0 | 1.0 |
| branch_S_t8_s17b | 4619 | 1.0 | 1.0 | 1.0 |
| branch_S_t8_s29 | 4619 | 1.0 | 1.0 | 1.0 |
| branch_S_t8_s43 | 4619 | 1.0 | 1.0 | 1.0 |
