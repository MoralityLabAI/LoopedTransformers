# NAN-MLP-27: full development trajectories and batched A/B

Conditional new capacity condition: gate/up token chunks256, one full physical down projection, padded reached-state capacity probes. FULL26 remains separate with all failures and grades preserved. Scope: 1,549 baseline contexts per backend: GYM05 322, COMP14 240, CTX08 291, CTX21 624, STORY23 72. After full baseline generation: 216 clean/intervention pairs per backend (all 24 story worlds and 48 focused competence worlds × three routes).

Native full-prefix generation, two recurrent loops, full vocabulary greedy sampling. Native clean row A and edited row B share one real decoder batch with bounded gate/up activation workspaces; one model is resident. Capacity-only preflight governs batch two versus one for baseline contexts. A/B requires batch two; no serial substitution. All 40 physical vectors are saved at every generated prefix, with exact BF16 storage. Fixed SAE/random controls and independent BPD mask/write audits stream on one CPU thread.

This is a new bounded-MLP and batched execution condition on the identical frozen FULL26 prompts. Earlier one-token scores are not its matched control. Cross-precision accuracy differences include batch-shape differences wherever capacity admission differs; geometry requires identical histories and shapes. Generated prefixes without a closed audited episode retain unknown grades and are not replayed. All 168 original test cases remain sealed. Repeated roots, tokens, routes and features are correlated measures. No transfer, equivalence or causal confirmation is claimed; effect floors, dose/sham calibration and power remain uncalibrated. Verdict: REDESIGN_OR_PILOT_ONLY.

Qualification: pending_FULL26_actual_termination. No task correctness selects admission; a resource failure runs BOTH matched precisions fresh. No old reply is replayed.

NF4: unrun_or_live_no_terminal_claim. Closed baseline unknown/1549; A/B unknown/216.
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
