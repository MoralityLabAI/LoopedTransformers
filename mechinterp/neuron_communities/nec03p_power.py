"""NEC-03P finite-sample and source recruitment arithmetic (stdlib only).

All rates are sensitivity parameters, never estimated causal efficacy.
The root is an independently generated five-input Boolean world. The
95/95 sample-maximum tolerance bound is conditional on independent,
exchangeable *eligible* roots from a fixed target population.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "NEC-03P/source-eligibility-and-fsi-planning/v1"
SOURCE_ROOTS = tuple(range(8, 96))
EARLIER_SOURCE_ROOTS = tuple(range(8))
EMBARGOED_ROOTS = tuple(range(96, 128))
PRIOR_PAIRED_ELIGIBLE = (6, 8)
TARGET_SHAM_TAIL = 0.95
TARGET_CONFIDENCE = 0.95
MINIMUM_MEANINGFUL_CAUSAL_EFFECT = 0.02
FSI_ALPHA = 0.05


def binomial_tail(trials: int, minimum: int, p: float) -> float:
    if not (0 <= minimum <= trials and 0 <= p <= 1):
        raise ValueError("binomial arguments out of range")
    return math.fsum(math.comb(trials, k)*p**k*(1-p)**(trials-k)
                     for k in range(minimum, trials+1))


def binomial_cdf(trials: int, maximum: int, p: float) -> float:
    if maximum < 0:
        return 0.
    if maximum >= trials:
        return 1.
    return math.fsum(math.comb(trials,k)*p**k*(1-p)**(trials-k)
                     for k in range(maximum+1))


def exact_cp_interval(successes: int, trials: int, alpha: float=.05):
    """Two-sided equal-tail Clopper-Pearson interval by beta/binomial inversion."""
    if not 0 <= successes <= trials or trials <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid exact binomial interval arguments")
    if successes == 0:
        low = 0.
    else:
        l, h = 0., 1.
        for _ in range(80):
            mid = (l+h)/2
            if binomial_tail(trials,successes,mid)<alpha/2:
                l=mid
            else:
                h=mid
        low=h
    if successes == trials:
        upper=1.
    else:
        l,h=0.,1.
        for _ in range(80):
            mid=(l+h)/2
            if binomial_cdf(trials,successes,mid)>alpha/2:
                l=mid
            else:
                h=mid
        upper=l
    return low,upper


def minimum_tolerance_roots(q:float=TARGET_SHAM_TAIL,
                            confidence:float=TARGET_CONFIDENCE):
    if not (0<q<1 and 0<confidence<1):
        raise ValueError("bad tolerance target")
    return math.ceil(math.log(1-confidence)/math.log(q))


def fsi_effect_margin(roots:int, comparisons:int=1,alpha:float=FSI_ALPHA):
    if roots<=0 or comparisons<1 or not (0<alpha<1):
        raise ValueError("bad root-level effect bounds")
    return math.sqrt(2*math.log(comparisons/alpha)/roots)


def plan():
    # Registry remains grounded in original full five-input and confirmation seals.
    from mechinterp.neuron_communities.nec03l_five_input_worlds import (
        registration,worlds,
    )
    from mechinterp.neuron_communities.nec03g_worlds import registry as four
    original=registration()
    if (original["registry_sha256"] !=
        "6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217"
        or four()["confirmation_full_spec_sha256"] !=
        "670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39"):
        raise AssertionError("original source or confirmation family registry changed")
    worlds_all=worlds()
    if (len(SOURCE_ROOTS)!=88 or len(EARLIER_SOURCE_ROOTS)!=8
        or set(SOURCE_ROOTS)&set(EARLIER_SOURCE_ROOTS)
        or any(worlds_all[i]["cohort"]!="development" for i in SOURCE_ROOTS)
        or any(worlds_all[i]["cohort"]!="embargoed" for i in EMBARGOED_ROOTS)):
        raise AssertionError("NEC03P root partition overlaps source or embargo")
    eligible,seen=PRIOR_PAIRED_ELIGIBLE
    lower,upper=exact_cp_interval(eligible,seen)
    tolerance=minimum_tolerance_roots()
    scenarios = [
        {"assumed_eligibility_probability":p,
         "expected_eligible_roots":len(SOURCE_ROOTS)*p,
         "probability_at_least_59_eligible_roots":binomial_tail(len(SOURCE_ROOTS),tolerance,p)}
        for p in (0.55,0.60,0.65,0.70,0.75,0.80,0.85)
    ]
    return {
        "schema":SCHEMA,
        "root_unit":"one registered independently seeded five-input Boolean world",
        "old_source_development_root_indices":list(EARLIER_SOURCE_ROOTS),
        "new_source_recruitment_root_indices":list(SOURCE_ROOTS),
        "embargoed_root_indices_by_number_only":list(EMBARGOED_ROOTS),
        "source_recruitment_root_count":len(SOURCE_ROOTS),
        "two_native_arms_nested_per_root":True,
        "previous_paired_source_eligible_roots":eligible,
        "previous_source_roots_attempted":seen,
        "previous_paired_eligibility_exact_95pct_interval":[lower,upper],
        "sample_maximum_95quantile_95confidence_minimum_independent_eligible_roots":tolerance,
        "sham_95_95_tail_requires_independent_causal_sham_measurements":True,
        "source_eligibility_is_not_sham_outcome_or_causal_efficacy":True,
        "probability_scenarios_are_iid_model_sensitivities_only":scenarios,
        "minimum_meaningful_causal_effect":MINIMUM_MEANINGFUL_CAUSAL_EFFECT,
        "fsi_cluster_bounds":[
            {"independent_roots":n,
             "single_comparison_hoeffding_margin":fsi_effect_margin(n),
             "minimum_observed_mean_effect_needed_to_exceed_2pp":fsi_effect_margin(n)+MINIMUM_MEANINGFUL_CAUSAL_EFFECT}
            for n in (8,32,59,88,96)
        ],
        "source_registry_sha256":original["registry_sha256"],
        "four_input_confirmation_sha256":four()["confirmation_full_spec_sha256"],
        "source_training_groups_per_root":80,
        "focal_masks_per_source_group":32,
        "sae_fit_groups":40,"sae_rank_groups":40,
        "source_sham_geometry_groups":8,
        "five_input_development_evaluation_groups_scored":0,
        "five_input_reserved_groups_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "causal_results_measured":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    info=plan()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    print("NEC03P_SOURCE_RECRUITMENT_AND_FINITE_SAMPLE_PREFLIGHT_PASS")


if __name__=="__main__":
    main()
