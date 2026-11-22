# NAN-RAM-28: GPU-resident NF4 full gym with bounded host RAM

New source-frozen resource condition after the user requested less host-memory pressure. It uses file-backed BF16 checkpoint tensors, quantizes decoder layers with the original NF4 conversion, and retains the GPU head. It does not run the old conditional MLP27 registration or claim its qualification.

Scope remains all 1,549 baseline trajectories and 216 true batched clean/intervention A/B pairs for NF4. MLP gate/up workspaces use 256-token chunks and one physical full-token down projection. Existing padded capacity/closure checks precede scored episodes. All 40 states, native reached outputs and masks are retained at every generated prefix. Fixed SAE/random controls and independent BPD audits use the original numerical methods after GPU termination.

BF16 is deferred under the host-RAM policy; its missing results remain unknown. The interrupted CTX21 BF16 run, original failed terminal audit and all closed files remain preserved. No original receipt was synthesized and no reply was replayed.

Host limits: start with at least 4 GiB available; check for at least 2 GiB available and at most 6 GiB private worker commitment between captures. A separate supervisor terminates only this authenticated worker if available RAM falls below 1 GiB or private commitment exceeds 6.5 GiB. The later host audit has a 1.5 GiB private budget and 2 GiB system reserve. One CPU thread, Below Normal priority; no concurrent substantial host audit. The math process is untouched.

These resource thresholds are operating budgets, not scientific competence gates. The original 168-case test stays sealed. Repeated sources/tokens/routes/features are not independent samples; no transfer, equivalence or causal confirmation is claimed. The continuous goal remains active.

NF4: capacity_stopped; closed baseline 636/1549; A/B 0/216.
BF16: deferred_host_RAM_policy; closed baseline unknown/1549; A/B unknown/216.

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
|nf4|nan_comp14|calibration|baseline|native|{"arm": "copy"}|16|8|0|0|292|0|[0, 0]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "focused"}|16|8|0|0|539|0|[0, 0]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "focused_erased"}|16|8|0|0|375|0|[0, 0]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "legacy"}|16|8|0|0|559|0|[0, 0]/16|
|nf4|nan_comp14|calibration|baseline|native|{"arm": "legacy_erased"}|16|8|0|0|511|0|[0, 0]/16|
|nf4|nan_comp14|development|AB|component_only|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "component_only", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|native|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|reached_state|{"AB_condition": "reached_state", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|AB|static_CAL_mean|{"AB_condition": "static_CAL_mean", "arm": "focused"}|0|0|0|0|0|32|[0, 32]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "copy"}|32|16|0|0|584|0|[0, 0]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "focused"}|32|16|0|0|1395|0|[0, 0]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "focused_erased"}|32|16|0|0|1022|0|[0, 0]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "legacy"}|32|16|0|0|1329|0|[0, 0]/32|
|nf4|nan_comp14|development|baseline|native|{"arm": "legacy_erased"}|32|16|0|0|742|0|[0, 0]/32|
|nf4|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 1024, "position": "middle", "task": "lookup"}|1|1|0|0|43|0|[0, 0]/1|
|nf4|nan_ctx08|calibration|baseline|native|{"arm": "main", "context_budget": 2048, "position": "middle", "task": "lookup"}|1|1|0|0|50|0|[0, 0]/1|
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
|nf4|nan_gym05|calibration|baseline|native|{"arm": "copy", "suite": "pointer"}|32|32|0|0|588|0|[0, 0]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|32|32|0|0|990|0|[0, 0]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|32|32|0|0|608|0|[0, 0]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|32|32|0|0|898|0|[0, 0]/32|
|nf4|nan_gym05|calibration|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|32|32|0|0|1151|0|[0, 0]/32|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc1"}|3|3|0|0|468|0|[0, 0]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "arc2"}|3|3|0|0|316|0|[0, 0]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "pointer"}|16|16|0|0|295|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "routing"}|16|16|0|0|392|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "storyworld"}|16|16|0|0|486|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "copy", "suite": "sudoku"}|3|3|0|0|291|0|[0, 0]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 1, "suite": "pointer"}|16|16|0|0|751|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 2, "suite": "pointer"}|16|16|0|0|415|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 3, "suite": "pointer"}|16|16|0|0|473|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "hops": 4, "suite": "pointer"}|16|16|0|0|627|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc1"}|3|3|0|0|175|0|[0, 0]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "arc2"}|3|3|0|0|407|0|[0, 0]/3|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "routing"}|16|16|0|0|867|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "storyworld"}|16|16|0|0|1647|0|[0, 0]/16|
|nf4|nan_gym05|development|baseline|native|{"arm": "main", "suite": "sudoku"}|3|3|0|0|586|0|[0, 0]/3|
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
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|2|1|2|1|106|0|[1, 1]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|2|1|2|0|104|0|[0, 0]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|2|1|2|0|97|0|[0, 0]/2|
|nf4|nan_story23|calibration|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|2|1|2|0|104|0|[0, 0]/2|
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
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "actual"}|2|1|2|2|104|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "belief"}|2|1|2|0|104|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 32, "query_type": "nested_belief"}|2|1|2|2|103|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "actual"}|2|1|2|1|106|0|[1, 1]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "belief"}|2|1|2|1|106|0|[1, 1]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 64, "query_type": "nested_belief"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "actual"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "belief"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "copy", "event_count": 96, "query_type": "nested_belief"}|2|1|2|2|106|0|[2, 2]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "actual"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 32, "query_type": "nested_belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "actual"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 64, "query_type": "nested_belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "actual"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "erased", "event_count": 96, "query_type": "nested_belief"}|2|1|0|0|128|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "actual"}|2|1|2|0|94|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "belief"}|2|1|2|1|104|0|[1, 1]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 32, "query_type": "nested_belief"}|2|1|2|0|103|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "actual"}|2|1|2|0|104|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "belief"}|2|1|2|0|93|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 64, "query_type": "nested_belief"}|2|1|2|0|96|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "actual"}|2|1|2|0|80|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "belief"}|2|1|2|0|106|0|[0, 0]/2|
|nf4|nan_story23|development|baseline|native|{"arm": "main", "event_count": 96, "query_type": "nested_belief"}|2|1|2|0|104|0|[0, 0]/2|


[Resource protocol](../../experiments/nan_ram28/PROTOCOL.md) · [Complete aggregate](nan_ram28_progress.json)
