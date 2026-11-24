"""Compare fixed-budget PercInterp selectors at the full Nanbeige head."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from math import isclose
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_degree48.run import PRIOR_ARCHIVE
from experiments.nan_delay39.graph_host import graph_cells
from experiments.nan_delay39.instrument import prepare
from experiments.nan_gauge42.run import extract_pair, read_archive
from experiments.nan_gauge42.transport import fit, head_gradient, orthonormal_bank
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_select47.select import mask_sets
from experiments.nan_vocab81.run import LABELS, native_controls, reduce_tiles, stream_tiles, token_classes


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "experiments/nan_readpath37/registration.json"
GAUGE = REPO / "experiments/nan_gauge42/registration.json"
DOSE = REPO / "experiments/nan_delay39/registration.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
RANK = 8
CUTOFF = .5
QUOTA = 4
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30,
          "minimum_available_RAM_bytes": 4 * 2**30, "maximum_private_bytes": 768 * 2**20,
          "maximum_seconds": 600}


def seal():
    paths = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py")]
    paths += [BASE, GAUGE, DOSE, *[REPO / n for n in (
        "experiments/nan_gauge42/SOURCE_MANIFEST.json", "experiments/nan_gauge42/run.py",
        "experiments/nan_gauge42/transport.py", "experiments/nan_degree48/run.py",
        "experiments/nan_vocab81/run.py", "experiments/nan_vocab81/SOURCE_MANIFEST.json",
        "experiments/nan_delay39/graph_host.py", "experiments/nan_delay39/instrument.py",
        "experiments/nan_dose38/run.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_select47/select.py", "experiments/nan_select47/SOURCE_MANIFEST.json",
        "experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py",
    )]]
    return {"experiment": "NAN-SIEVE-84", "cal_cutoff": CUTOFF, "receiver_quota": QUOTA,
            "gauge42_closed_private_sha256": PRIOR_ARCHIVE, "resources": BUDGET,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def readout(head, states, classes, canonical, gold, foil, check, np):
    rows = len(states)
    peak, bare, maximum, top, normalizer = reduce_tiles(
        stream_tiles(head, np.stack(states), check, np), classes, canonical, rows, np)
    output = []
    for i, (target, paired) in enumerate(zip(gold, foil, strict=True)):
        competitors = peak[i].copy()
        competitors[target] = -np.inf
        invalid = competitors[-1]
        valid = competitors[:-1].max()
        full = peak[i,target] - max(invalid, valid)
        correct = int(classes[top[i]]) == target
        assert (full <= 0 or correct) and (full >= 0 or not correct)
        output.append({"full_margin":float(full), "paired_margin":float(bare[i,target]-bare[i,paired]),
                       "gold_score":float(peak[i,target]), "invalid_score":float(invalid),
                       "other_label_score":float(valid), "correct":correct, "top":int(top[i]),
                       "peak":float(maximum[i]), "logz":float(normalizer[i]),
                       "foil_kind":"tie" if invalid == valid else "invalid" if invalid > valid else "other_label"})
    return output


def paired_tape(selected, twins, stage):
    positions = defaultdict(dict)
    for i, row in enumerate(selected.values()):
        positions[row["split"], row["arm"], row["root"]][row["world"]] = i
    states, labels, paired = [None]*len(selected), [None]*len(selected), [None]*len(selected)
    for key, worlds in twins.items():
        if set(positions[key]) != {"A","B"} or set(worlds) != {"A","B"}:
            raise ValueError("Missing archived world")
        for a,b in (("A","B"),("B","A")):
            i = positions[key][a]
            label, physical, _ = worlds[a]
            states[i] = physical[2,stage]
            labels[i] = LABELS.index(label)
            paired[i] = LABELS.index(worlds[b][0])
    return positions, states, labels, paired


def pick(cal, middle, inverse, writes, norms, head, native_states, tokens, positions, np):
    """Fit receiver masks on CAL only using two definitions of the sink."""
    paired, full = [], []
    native_bank = writes / norms[:,None]
    for key, pair in cal.items():
        begin, end = pair
        source = (begin["coefficients"] @ inverse) * norms
        phase2 = (end["coefficients"] @ inverse) * norms
        old_sink = native_bank @ end["gradient"]
        earlier = native_bank @ begin["gradient"]
        old, new = [], []
        for sign, w in ((1.,"A"),(-1.,"B")):
            i = positions[key][w]
            state = native_states[i].astype(np.float64)
            # Request the exact token IDs for the active gold spelling and competitor from the
            # restricted CAL token witness. It is derived from the full vocabulary scan.
            target, rival = tokens[i]
            direction = head.gamma.astype(np.float64) * (
                head.row(target).astype(np.float64) - head.row(rival).astype(np.float64))
            gradient = native_bank @ head_gradient(state, direction, head.eps, np)
            shared = {"source":(sign*source).tolist(), "middle":middle,
                      "activity1":source.tolist(), "activity2":phase2.tolist(),
                      "isolated1":earlier.tolist()}
            old.append(shared | {"sink":(sign*old_sink).tolist(), "isolated2":old_sink.tolist()})
            new.append(shared | {"sink":gradient.tolist(), "isolated2":gradient.tolist()})
        paired.append(old)
        full.append(new)
    cfg = {"calibration_cutoff":CUTOFF, "source_components":list(range(RANK)), "source_quota":3,
           "receiver_components":list(range(RANK)), "receiver_quota":QUOTA, "minimum_complete_CAL_roots":8}
    old = mask_sets(paired, cfg)
    new = mask_sets(full, cfg)
    if old["status"] != "frozen" or new["status"] != "frozen":
        raise ValueError("The fixed CAL graph roster did not close")
    methods = {name:tuple(result["masks"][selector]["receiver_observer"]) for name,result,selector in (
        ("paired_sieve", old, "sieve"), ("full_sieve", new, "sieve"),
        ("local_degree", new, "local_degree"), ("local_weighted_degree", new, "local_weighted_degree"),
        ("local_head_sensitivity", new, "isolated_margin_response"),
        ("phase2_activity", new, "fixed_bank_activity"),
    )}
    if any(len(indices) != QUOTA for indices in methods.values()):
        raise ValueError("An intervention mask did not keep the equal receiver budget")
    return methods


def clean_and_cal_witness(head, states, classes, canonical, gold, foil, cal_idx, check, np):
    # This reducer returns class maxima but only the global token ID. Keep all real token
    # maxima at just the 16 CAL states, via a second pass over the same frozen head tiles.
    clean = readout(head, states, classes, canonical, gold, foil, check, np)
    peaks = np.full((len(cal_idx), 2), -np.inf)
    leaders = np.full((len(cal_idx), 2), -1, dtype=np.int32)
    offset = 0
    targets = np.asarray([gold[i] for i in cal_idx])
    for scores in stream_tiles(head, np.stack([states[i] for i in cal_idx]), check, np):
        same = classes[offset:offset+scores.shape[1]][None,:] == targets[:,None]
        for c, selector in enumerate((same, ~same)):
            kept = np.where(selector, scores, -np.inf)
            local = kept.argmax(axis=1)
            value = kept[np.arange(len(cal_idx)),local]
            update = value > peaks[:,c]
            peaks[update,c] = value[update]
            leaders[update,c] = offset + local[update]
        offset += scores.shape[1]
    if offset != head.vocab or (leaders < 0).any():
        raise ValueError("Full-vocabulary CAL token witnesses did not close")
    index = {}
    for i,(ids,values) in zip(cal_idx,zip(leaders,peaks,strict=True),strict=True):
        record = clean[i]
        if values[0] != record["gold_score"] or values[1] != max(record["invalid_score"],record["other_label_score"]):
            raise ValueError("CAL active token witnesses disagree with class maxima")
        index[i] = [int(value) for value in ids]
    return clean,index


def interventions(dev, positions, world_states, bank, masks, cfg, np):
    """Build all signed edits before scoring, with one joint same-dose gate per root."""
    unique = sorted(set(masks.values()))
    bases = {key:np.linalg.qr(bank[list(key)].T,mode="reduced")[0] for key in unique}
    roster, report = [], {}
    for key, worlds in sorted(dev.items()):
        root = key[2]
        delta = (worlds["A"][1][2,"down_proj_input"].astype(np.float64)
                 - worlds["B"][1][2,"down_proj_input"].astype(np.float64))
        write_delta = delta @ cfg["reads"] @ bank
        per_world = []
        for sign,w in ((1.,"A"),(-1.,"B")):
            i = positions[key][w]
            state = world_states[i]
            arms = {}
            for mask,basis in bases.items():
                direction = sign * basis @ (basis.T @ write_delta)
                arms[mask] = [prepare(state,s*direction,cfg["phase2_readout_dose_fraction"],
                                      cfg["dose_relative_tolerance"],cfg["minimum_direction_cosine"],np)
                              for s in (1.,-1.)]
            all_edits = [patch for group in arms.values() for patch in group]
            requested = all_edits[0].requested_l2
            assert all(p.requested_l2 == requested for p in all_edits)
            gap = (max(p.realized_l2 for p in all_edits)-min(p.realized_l2 for p in all_edits))/requested
            eligible = all(p.eligible for p in all_edits) and gap <= cfg["max_contrast_dose_gap_fraction"]
            per_world.append((i,arms,gap,eligible))
        report[root] = {"both_world_dose_qualified":all(v[3] for v in per_world),
                        "worlds": [{"index":i,"dose_gap":gap,"qualified":eligible,
                                    "min_cosine":min(p.cosine for group in arms.values() for p in group)}
                                   for i,arms,gap,eligible in per_world]}
        if report[root]["both_world_dose_qualified"]:
            roster.extend((root,i,mask,p,r) for i,arms,_,_ in per_world
                          for mask,patches in arms.items() for p,r in zip((1,-1),patches,strict=True))
    return roster,report


def compare(outcomes, masks, known, clean):
    results = {}
    for name, mask in masks.items():
        groups = defaultdict(list)
        for root, by_index in known.items():
            if not by_index["both_world_dose_qualified"]:
                continue
            for meta in by_index["worlds"]:
                i = meta["index"]
                p,m = (outcomes[root,i,mask,s] for s in (1,-1))
                groups[root].append({"half_full":(p["full_margin"]-m["full_margin"])/2,
                    "half_paired":(p["paired_margin"]-m["paired_margin"])/2,
                    "toward_clean_correct":p["correct"], "clean_correct":clean[i]["correct"],
                    "foil_class_switched":p["foil_kind"] != m["foil_kind"]})
        worlds = [w for group in groups.values() for w in group]
        n = len(worlds)
        def macro(key):
            return fmean(fmean(w[key] for w in group) for group in groups.values()) if n else None
        results[name] = {"complete_roots":len(groups), "unknown_worlds":2*len(known)-n,
            "full_half_difference":macro("half_full"), "paired_half_difference":macro("half_paired"),
            "directional_sign_disagreement_worlds":sum(w["half_full"]*w["half_paired"] < 0 for w in worlds),
            "competitor_class_switch_worlds":sum(w["foil_class_switched"] for w in worlds),
            "repairs_toward_vs_clean":sum(w["toward_clean_correct"] and not w["clean_correct"] for w in worlds),
            "regressions_toward_vs_clean":sum(not w["toward_clean_correct"] and w["clean_correct"] for w in worlds)}
    return results


def backend_eval(backend, base, gauge, dose, selected, head, tok, classes, canonical, work, check, np):
    twins, reads, writes, max_logit = read_archive(backend, base, selected, head, check, np)
    root, inverse, white = orthonormal_bank(writes, np)
    labels = dict(zip(LABELS, canonical, strict=True))
    pairs = {k:extract_pair(k,v,labels,head,reads,root,white,np) for k,v in sorted(twins.items())}
    cal = {k:p for k,p in pairs.items() if k[0]=="calibration"}
    dev = {k:v for k,v in twins.items() if k[0]=="development"}
    if (len(cal),len(dev)) != (8,16):
        raise ValueError("Expected the registered independent CAL and DEV roots")
    c1,c2 = (np.stack([v[i]["coefficients"] for v in cal.values()]) for i in (0,1))
    transfer = fit(c1,c2,gauge["ridge_factor"],np)
    norms = np.linalg.norm(writes,axis=1)
    middle = ((root @ transfer @ inverse) * (norms[None,:]/norms[:,None])).tolist()
    archive = REPO / "runs/nan_gauge42/attempt01" / f"{backend}_private.json"
    old = {(r["split"],r["arm"],r["root"]):r for r in json.loads(bound(archive,PRIOR_ARCHIVE[backend]))}
    for key,(p1,p2) in pairs.items():
        source = ((p1["coefficients"] @ inverse)*norms).tolist()
        sink = ((writes @ p2["gradient"])/norms).tolist()
        observed = graph_cells(source,middle,sink)[1]
        saved = old[key]["graphs"]["native"][1]
        assert observed["occupancy_rank_fraction"]==saved["occupancy_rank_fraction"]==CUTOFF
        assert observed["backbone_nodes"]==saved["backbone_nodes"]
        assert isclose(observed["signed_path_product_sum"],saved["signed_path_product_sum"],abs_tol=1e-10,rel_tol=1e-9)
    positions, states, gold, foil = paired_tape(selected,twins,"residual_after_layer")
    cal_idx = sorted(i for key in cal for i in positions[key].values())
    clean, tokens = clean_and_cal_witness(head,states,classes,canonical,gold,foil,cal_idx,check,np)
    verified = native_controls(backend,base,selected,tok,check,np)
    for saved, row in zip(verified,clean,strict=True):
        token, peak, logz, correct = saved[1]
        if (row["top"] != token or row["peak"] != peak or row["correct"] != bool(correct)
                or abs(row["logz"]-logz)>base["native_logit_abs_tol"]):
            raise ValueError("Complete native head or task oracle did not close")
    masks = pick(cal,middle,inverse,writes,norms,head,states,tokens,positions,np)
    roster, known = interventions(dev,positions,states,writes,masks,dose | {"reads":reads},np)
    records = {}
    if roster:
        edited = readout(head,[p.after for _,_,_,_,p in roster],classes,canonical,
                         [gold[i] for _,i,_,_,_ in roster],[foil[i] for _,i,_,_,_ in roster],check,np)
        records = {(k,i,m,s):row for (k,i,m,s,_),row in zip(roster,edited,strict=True)}
    kept = Counter()
    for name,mask in masks.items():
        kept[mask] += 1
    outcomes = compare(records,masks,known,clean)
    write_new(work/f"{backend}_private.json",{"qualification":known,"clean":clean,
              "edits":[{"root":k,"world_index":i,"mask":list(m),"sign":s,**row}
                       for (k,i,m,s),row in records.items()]})
    return {"masks":masks,"distinct_receiver_masks":len(kept),
            "alias_groups":[[name for name,m in masks.items() if m==mask] for mask in sorted(kept)],
            "clean_head_exact_rows":len(clean),"prior_bottleneck_closure_roots":len(pairs),
            "native_selected_logit_max_error":max_logit,"signed_edit_head_rows":len(records),
            "jointly_qualified_worlds":sum(v["qualified"] for r in known.values() for v in r["worlds"]),
            "complete_roots":sum(r["both_world_dose_qualified"] for r in known.values()),
            "outcomes":outcomes}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args = cli.parse_args()
    frozen = seal()
    if args.freeze:
        write_new(MANIFEST,frozen)
        return
    if not __debug__ or not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes()) != frozen:
        cli.error("Require the source freeze, pinned snapshot and fresh result directory")
    base,gauge,dose = (json.loads(p.read_bytes()) for p in (BASE,GAUGE,DOSE))
    if args.snapshot.name != base["revision"]:
        cli.error("Wrong checkpoint revision")
    with lease(BUDGET) as (check,samples):
        if (version("numpy"),version("tokenizers")) != ("2.5.5","0.20.3"):
            raise RuntimeError("Pinned replay packages differ")
        import numpy as np
        from tokenizers import Tokenizer

        tokenizer_file = args.snapshot/"tokenizer.json"
        bound(tokenizer_file,base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(tokenizer_file))
        ids = [tok.encode(c,add_special_tokens=False).ids for c in LABELS]
        if any(len(x)!=1 for x in ids) or len(set(tuple(x) for x in ids))!=len(LABELS):
            raise ValueError("Bare label IDs differ")
        canonical = [x[0] for x in ids]
        roster,_=joined(base)
        selected={i:row for i,row in roster.items() if row["arm"]=="focused"}
        args.work.mkdir(parents=True,exist_ok=False)
        results={}
        with (args.snapshot/SHARD).open("rb") as stream:
            head = FixedHead(stream,args.snapshot,base,np,check)
            classes = token_classes(tok,head.vocab,np,check)
            if classes[canonical].tolist()!=list(range(len(LABELS))):
                raise ValueError("Pinned bare label class mapping differs")
            for name in base["backends"]:
                results[name]=backend_eval(name,base,gauge,dose,selected,head,tok,classes,
                                           canonical,args.work,check,np)
                print(name,results[name]["masks"],"closed roots",results[name]["complete_roots"],flush=True)
    assert frozen==seal() and "torch" not in sys.modules
    write_new(args.work/"aggregate.json",{"experiment":frozen["experiment"],"status":"passed",
               "manifest_sha256":digest(MANIFEST),"unit":"focused COMP14 pointer root","roots":base["roots"],
               "per_backend":results,"pretrained_decoder_forwards":0,"formal_comparisons":0,
               "sealed_test_accessed":False,
               "resources":{"max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
               "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
               "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)}})


if __name__=="__main__":
    main()
