"""CAL audit of the frozen complementary-ray BF16 dose repair."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import sys

from experiments.nan_attnspan67.run import SITES, make_pair, read_archive, read_banks
from experiments.nan_cidose74.geometry import ARMS, COSINE, DOSE, DOSE_TOLERANCE, directions
from experiments.nan_cidose74.run import ATTN, BASE, angle, summarize
from experiments.nan_cigate73.run import controller
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .repair import compensate, verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PRIOR = REPO / "reports/nanbeige_bpd_gym/nan_cidose74_results.json"


def seal():
    parent = REPO / "experiments/nan_cidose74/SOURCE_MANIFEST.json"
    files = [HERE/name for name in (".gitattributes", "__init__.py", "repair.py", "run.py", "PROTOCOL.md")]
    return {"experiment": "NAN-CIBRACKET-75", "files_sha256": json.loads(parent.read_bytes())["files_sha256"] |
            {p.relative_to(REPO).as_posix(): digest(p) for p in [*files, parent, PRIOR]}}


def match(new, native, direction, np):
    patches = []
    rec = {"status": "zero_native_write", "arms": {}, "effects": {}}
    if not new:
        return rec
    radius = np.linalg.norm(native, axis=1)
    nominal = DOSE * radius
    axis = native / radius[:, None]
    for arm in ARMS:
        pair = [compensate(h, d, np) for h,d in zip(native,new[arm],strict=True)]
        reports = []
        for w, (patch, method, jumps) in enumerate(pair):
            delta = patch.after - native[w]
            dose = float(np.linalg.norm(delta))
            direction_cosine = float(delta @ new[arm][w]) / (dose*np.linalg.norm(new[arm][w])) if dose else 0.
            if patch.eligible:
                np.testing.assert_allclose(dose, patch.realized_l2, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(direction_cosine, patch.cosine, rtol=1e-13, atol=1e-13)
                assert abs(dose/nominal[w] - 1) <= DOSE_TOLERANCE and direction_cosine >= COSINE
            reports.append({"dose": dose, "cosine": direction_cosine, "eligible": patch.eligible,
                            "method": method, "jump_coordinates": jumps,
                            "signed_radial_share": float(delta @ axis[w] / dose) if dose else None,
                            "relative_norm_drift": float(np.linalg.norm(patch.after)/radius[w] - 1)})
        rec["arms"][arm] = reports
        patches.append([patch for patch, _, _ in pair])
    spread = np.ptp([[p.realized_l2 for p in pair] for pair in patches], axis=0)
    rec.update(nominal_l2=nominal.tolist(), absolute_dose_spread=spread.tolist(), status="matched")
    if not all(p.eligible for pair in patches for p in pair):
        return rec | {"status": "physical_arm_unqualified"}
    if (spread > DOSE_TOLERANCE * nominal).any():
        return rec | {"status": "joint_dose_band_unqualified"}
    for arm, pair in zip(ARMS, patches, strict=True):
        terms = [float((p.after - p.before) @ direction) for p in pair]
        scaled = [t*g/p.realized_l2 for t,g,p in zip(terms, nominal, pair, strict=True)]
        rec["effects"][arm] = {"physical": terms[0]-terms[1], "exact_dose_linear": scaled[0]-scaled[1]}
    return rec


def evaluate(worlds, comparison, fit, root, erased, masks, np):
    bank, ci, static, _ = fit
    a, b = (worlds[k][2] for k in ("A", "B"))
    rows = []
    for p, state in enumerate(comparison["phase"], start=1):
        x, y = (np.stack([world[p,21,site] for world in (a,b)]).astype(np.float64) for site in SITES[1:])
        gate = masks(x, bank, ci)
        proposed, parity = directions(x, y, gate, np.broadcast_to(static[p-1], gate.shape), bank, p, root, np)
        if erased and any(not np.array_equal(d[0],d[1]) for d in proposed.values()):
            raise RuntimeError("Erased prompts do not agree at all seven intended directions")
        point = match(proposed, y, state["direction"], np)
        point.update(native_attention_readout=state["direct_write"], verified_reference_endpoints=parity,
                     direction_cosines={
                        f"{left}_{right}": [angle(v,w,np) for v,w in zip(proposed[left],proposed[right],strict=True)]
                        for left,right in (("dynamic_on","static_on"),("dynamic_off","static_off"),
                                           ("dynamic_on","dynamic_radial"))} if proposed else {})
        if erased and any(effect != 0 for values in point["effects"].values() for effect in values.values()):
            raise RuntimeError("Equal erased prompts produced different matched write readouts")
        rows.append(point)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Enable Python assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    frozen = seal()
    if (not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes()) != frozen
            or any(digest(REPO/p) != h for p,h in frozen["files_sha256"].items())):
        parser.error("Supply the frozen source, pinned snapshot and fresh output")
    prior = json.loads(PRIOR.read_bytes())
    cfg, base = (json.loads(p.read_bytes()) for p in (ATTN,BASE))
    if (args.snapshot.name != base["revision"] or cfg["revision"] != base["revision"]
            or prior["dose_fraction"] != DOSE or prior["dose_tolerance"] != DOSE_TOLERANCE):
        raise ValueError("Source and dose contract changed")
    with lease(cfg) as (check, samples):
        assert (version("numpy"),version("tokenizers")) == ("2.5.5","0.20.3")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        fixture = verify(np)
        config = json.loads(bound(args.snapshot/"config.json", base["snapshot_pins"]["config.json"]))
        assert not config["attention_bias"]
        bound(args.snapshot/"tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot/"tokenizer.json"))
        labels = {s: tokenizer.encode(s,add_special_tokens=False).ids for s in "ABCDEFGHIJKLMNOP"}
        assert all(len(v)==1 for v in labels.values()) and len({v[0] for v in labels.values()})==16
        labels = {s: v[0] for s,v in labels.items()}
        roster,_ = joined(base)
        cal = {i:r for i,r in roster.items() if r["split"]=="calibration"}
        assert len(cal)==48 and len({row["root"] for row in cal.values()})==8
        banks = read_banks("bf16",21,cfg,base,np)
        fit = controller("t2_layer21_o_proj_rank8", cfg["ranksite_audits"]["2"], banks["attn_bpd_t2"],np,masks)
        assert fit[-1]==prior["controller"]
        args.work.mkdir(parents=True,exist_ok=False)
        rows = []
        groups = defaultdict(list)
        with (args.snapshot/SHARD).open("rb") as stream:
            head = FixedHead(stream,args.snapshot,base,np,check)
            archive,error = read_archive("bf16",(21,),base,cal,head,check,np)
            for (split,arm,root), worlds in sorted(archive.items()):
                check()
                assert split=="calibration"
                erased = arm=="focused_erased"
                comparison = make_pair(worlds,21,labels,head,erased,np)
                result = evaluate(worlds,comparison,fit,root,erased,masks,np)
                rows.append({"split":split,"arm":arm,"root":root,"phases":result})
                groups[arm].append(result)
        assert "torch" not in sys.modules
    assert seal()==frozen
    write_new(args.work/"private.json",rows)
    result = {}
    for arm,pairs in groups.items():
        totals = Counter(w["method"] for pair in pairs for p in pair for r in p["arms"].values() for w in r)
        outcome = summarize(pairs)
        outcome["arm_world_methods"] = dict(totals)
        outcome["original_ray_roots_qualified"] = prior["groups"][f"calibration/{arm}"]["roots_qualified"]
        result[arm] = outcome
    write_new(args.work/"aggregate.json",{
        "experiment":"NAN-CIBRACKET-75","status":"passed","source_manifest_sha256":digest(MANIFEST),
        "prior_ray_aggregate_sha256":digest(PRIOR),"archive_pins":prior["archive_pins"],"controller":fit[-1],
        "head_replay_max_abs_error":error,"fixture":fixture,"independent_CAL_roots":8,
        "DEV_capture_files_opened":0,"pretrained_forwards":0,"formal_comparisons":0,"sealed_test_accessed":False,
        "groups":result,"max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
        "max_sampled_resident_bytes":max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples),
    })
    print(result["focused"],flush=True)


if __name__=="__main__":
    main()
