"""Use held-out gold only on the host, after WRITER77's physical gates."""

from collections import Counter
from statistics import fmean

from experiments.nan_writer77.host import qualify, score_roots
from experiments.nan_writer77.pipeline import MODES

from .heads import INVALID, LABELS, margin, verify_head


CONTRASTS = (("active", "clean"), ("active", "matched_seeded"),
             ("active", "original_seeded"), ("active", "active_k_rescue"))


def annotated(maxima, native, classes, bare, answer, foil):
    top_class = verify_head(maxima, native, classes, bare)
    out = margin(maxima, answer)
    out["paired_margin"] = native["labels"][answer] - native["labels"][foil]
    out["correct"] = top_class == LABELS.index(answer)
    out["foil_class"] = "invalid" if classes[out["foil_token"]] == INVALID else "other_label"
    if (out["margin"] != min(out["valid_margin"], out["invalid_margin"])
            or (out["margin"] > 0 and not out["correct"])
            or (out["margin"] < 0 and out["correct"])):
        raise ValueError("Inconsistent full-vocabulary correctness")
    return out


def effects(candidate, reference):
    gold = candidate["gold_score"] - reference["gold_score"]
    foil = candidate["foil_score"] - reference["foil_score"]
    full = candidate["margin"] - reference["margin"]
    if abs(full - (gold - foil)) > 1e-10:
        raise ValueError("Gold and foil deltas do not reconstruct the margin change")
    a, b = reference["correct"], candidate["correct"]
    return {"full_margin": full, "paired_margin": candidate["paired_margin"] - reference["paired_margin"],
            "valid_margin": candidate["valid_margin"] - reference["valid_margin"],
            "invalid_margin": candidate["invalid_margin"] - reference["invalid_margin"],
            "gold_logit": gold, "foil_logit": foil, "foil_token_switched": (
                candidate["foil_token"] != reference["foil_token"]),
            "competitor_class_switched": candidate["foil_class"] != reference["foil_class"],
            "correctness": "repair" if b and not a else "regression" if a and not b else "tie"}


def reduce_contrast(rows, total):
    metric = ("full_margin", "paired_margin", "valid_margin", "invalid_margin", "gold_logit", "foil_logit")
    count = Counter(world["correctness"] for root in rows for world in root)
    return {"registered_roots": total, "complete_roots": len(rows), "unknown_worlds": 2 * (total-len(rows)),
            "repairs": count["repair"], "regressions": count["regression"], "ties": count["tie"],
            "root_macro_delta": {name: fmean(fmean(world[name] for world in root) for root in rows)
                                 if rows else None for name in metric},
            "foil_token_switches": sum(w["foil_token_switched"] for root in rows for w in root),
            "competitor_class_switches": sum(w["competitor_class_switched"] for root in rows for w in root),
            "paired_and_full_opposite_sign_worlds": sum(w["paired_margin"] * w["full_margin"] < 0
                                                         for root in rows for w in root)}


def score(source, enriched, episodes, classes, tokenizer, config):
    """Unpack the full-vocabulary sidecar and reconcile the frozen host gates."""
    bare = source["tokens"]["bare_label_token_ids"]
    baseline = [{k:v for k,v in row.items() if k != "class_maxima"} for row in enriched]
    gate, observed = qualify(source, baseline, bare, tokenizer, len(classes), config)
    full = {row["id"]: row["class_maxima"] for row in enriched}
    for key, maxima in full.items():
        for reduced, native in zip(maxima, observed[key]["phases"], strict=True):
            verify_head(reduced, native, classes, bare)
    focused = {}
    for row in source["gold"]:
        if row["arm"] == "focused":
            focused.setdefault(row["root"], {})[row["world"]] = row
    lookup, answers = {}, {}
    for root, pair in focused.items():
        rows = [pair[k] for k in ("A", "B")]
        answers[root] = [(row["id"], row["answer"], rows[1-i]["answer"]) for i, row in enumerate(rows)]
        lookup[root] = [(key, target, foil, observed[key]) for key, target, foil in answers[root]]
    legacy = score_roots(episodes, gate, lookup, bare, tokenizer, len(classes))
    private = []
    for episode in episodes:
        if episode["status"] != "closed":
            continue
        root = episode["root"]
        worlds = []
        for edited, (key, target, foil) in zip(episode["worlds"], answers[root], strict=True):
            maxima = edited["class_maxima"]
            if set(maxima) != set(MODES) or maxima["active"][0] != maxima["active_k_rescue"][0]:
                raise ValueError("Class observations omit an arm or changed the rescue first pass")
            reads = {"clean": [annotated(r, n, classes, bare, target, foil) for r,n in zip(
                full[key], observed[key]["phases"], strict=True)]}
            for mode in MODES:
                reads[mode] = [annotated(r, n, classes, bare, target, foil) for r,n in zip(
                    maxima[mode], edited["arms"][mode], strict=True)]
            worlds.append(reads)
        private.append((root, worlds))
    contrasts = {}
    for candidate, reference in CONTRASTS:
        contrasts[f"{candidate}_minus_{reference}"] = {
            f"pass{p+1}": reduce_contrast([[effects(w[candidate][p], w[reference][p]) for w in worlds]
                                            for _, worlds in private], len(lookup)) for p in range(2)}
    return {"legacy": legacy, "contrasts": contrasts, "native_gate": gate["baseline_counts"],
            "formal_comparisons": 0}, private
