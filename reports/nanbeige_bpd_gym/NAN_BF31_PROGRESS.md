# NAN-BF-31: storage-backed BF16 control and matched RAM28 NF4

Full scope: 1,549 baseline trajectories and 216 genuine B2 clean/intervention pairs per precision. BF16 is a fresh source-frozen RAM condition; NF4 stays the separately frozen RAM28 condition. No neural response or original receipt is copied or synthesized.

BF16 uses four resident decoder layers, one reusable GPU stage, a resident BF16 head and file-backed CPU embeddings/reference weight. Other decoder parameters remain meta. All offloaded-layer stage bytes are checked against the pinned source before any forward; source files and logical/resident parameters are verified again at normal termination. The unused full-gym observer list is cleared after authoritative exports. This is a new engineering condition; no old failed gate is repaired.

The identical source/token/parser/budget roster, two native loops, full-vocabulary greedy decode, bounded MLP method, fixed SAE/random towers and independently qualified BPD factors are retained. Native and actual edited A/B rows share one decoder batch. A/B has no serial fallback. Required B1x9216 and padded B2x4608 admission precedes any scored episode.

Both full native trajectories are graded only after closure. NF4 keeps its original canonical host audit. BF16 has a separate post-model host audit; substantial host analyses are serialized. Shared geometry requires identical complete audited decoder-batch inputs, including companion-row histories, modes, order, lengths and padding. Missing/unclosed companions and unequal tails are explicitly excluded. Within-root means precede aggregation; accuracy differences can include different admitted batch shapes.

All registered task/control/difficulty strata remain visible with missing-outcome bounds. Story format, final/checkpoint/path/joint endpoints remain in the private audited records. No outcome-based selection, neural retry, orphan rescue, new fitting or donor transfer. Generated traces do not establish faithful reasoning. Causal/sham floors and power remain uncalibrated; formal comparisons 0, REDESIGN_OR_PILOT_ONLY. The original 168-case test stays sealed and the continuous goal stays active.

Qualification: BF31_not_started_or_waiting_RAM28_actual_termination. Resource admission never selects cases by correctness.

NF4: unrun_or_live_no_terminal_claim. Closed baseline unknown/1549; A/B unknown/216.
Timing of closed generation batches: {"CUDA_event_over_forward_wall_ratio": 0.9995784217717143, "CUDA_event_seconds": 12912.916899414062, "CUDA_peak_allocated_bytes": 3647102976, "closed_generated_prefix_batches": 599, "forward_wall_seconds": 12918.362999999954, "generated_prefixes_without_closed_episode_audit": 1114, "generated_tokens": 1114, "generated_tokens_per_end_to_end_second": null, "independently_audited_prefixes": 0, "two_row_batches": 515}. CUDA events measure forward intervals including synchronization gaps; this is not a kernel occupancy measurement.
BF16: unrun_or_live_no_terminal_claim. Closed baseline unknown/1549; A/B unknown/216.

All registered task/control strata are shown. Missing outcomes are unknown; zero correct among zero observed replies is not a zero accuracy.

|Backend|Stage|Split|Lane|Mode|Stratum|Closed|Roots|Format|Correct|Prefixes|Missing|Correct bounds / registered|
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
|bf16|nan_comp14|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|AB|native|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|AB|native|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|baseline|native|{"arm": "copy"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|baseline|native|{"arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|baseline|native|{"arm": "focused_erased"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|baseline|native|{"arm": "legacy"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|calibration|baseline|native|{"arm": "legacy_erased"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_comp14|development|AB|component_only|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|AB|native|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|AB|native|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|baseline|native|{"arm": "copy"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|baseline|native|{"arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|baseline|native|{"arm": "focused_erased"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|baseline|native|{"arm": "legacy"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_comp14|development|baseline|native|{"arm": "legacy_erased"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|bf16|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|bf16|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|calibration|baseline|native|{"arm": "main", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|calibration|baseline|native|{"arm": "copy", "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc1"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc2"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "routing"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "storyworld"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "sudoku"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc1"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc2"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "suite": "routing"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "suite": "storyworld"}|0|0|0|0|0|16|[0, 16]/16|
|bf16|nan_gym05|development|baseline|native|{"arm": "main", "suite": "sudoku"}|0|0|0|0|0|3|[0, 3]/3|
|bf16|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|bf16|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_comp14|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|AB|native|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|AB|native|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "copy"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "focused"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "focused_erased"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "legacy"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "legacy_erased"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_comp14|development|AB|component_only|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "copy"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "focused_erased"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "legacy"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "legacy_erased"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|nf4|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|nf4|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|1|[0, 1]/1|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "join"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx08|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle", "task": "lookup"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|calibration|baseline|native|{"arm": "main", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|8|[0, 8]/8|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "copy", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "erased", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 4096, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "early"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "late"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_ctx21|development|baseline|native|{"arm": "main", "context_budget": 8192, "position": "middle"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "copy", "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc1"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc2"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "routing"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "storyworld"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "sudoku"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc1"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc2"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "routing"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "storyworld"}|0|0|0|0|0|16|[0, 16]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "sudoku"}|0|0|0|0|0|3|[0, 3]/3|
|nf4|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|component_only|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "component_only", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|0|0|0|0|0|2|[0, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|0|0|0|0|0|2|[0, 2]/2|
