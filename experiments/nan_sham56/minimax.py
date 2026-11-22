"""Minimize a continuous full-vocabulary peak proxy, then qualify BF16 shams."""
import argparse
import json
from pathlib import Path
from statistics import median

from experiments.nan_dark44.head import full_head
from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import next_neighbors, route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, decode_bf16, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters

from .geometry import basis, drive, remove, span_fraction
from .run import ACTIVE_AGG, HERE, MANIFEST, physical_head_closure, summarize


LOCK = HERE / "MINIMAX_MANIFEST.json"
NAME = "ambient_peak_minimax"
WIDTH = 24
STEP_SIZES = (.14, .07, .035, .0175)


def seal():
    files = [Path(__file__), HERE / "MINIMAX.md", MANIFEST, HERE / "run.py", HERE / "geometry.py"]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in files}


def optimize(matrix, np):
    q = np.zeros(matrix.shape[1], dtype=np.float32)
    q[0] = 1.
    z = matrix @ q
    before = peak = float(abs(z).max())
    for step in STEP_SIZES:
        for _ in range(8):
            idx = np.argpartition(abs(z), -64)[-64:]
            weights = np.exp((abs(z[idx]) - peak) / .003) * np.sign(z[idx])
            slope = weights @ matrix[idx]
            slope -= float(slope @ q) * q
            length = float(np.linalg.norm(slope))
            if length < 1e-10:
                break
            changed = q - step / length * slope
            changed /= np.linalg.norm(changed)
            predicted = matrix @ changed
            proposed_peak = float(abs(predicted).max())
            if proposed_peak >= peak:
                break
            q, z, peak = changed, predicted, proposed_peak
    return q, before, peak


def form_basis(h, raw, gradient, reached, radius, tag, cfg, np):
    lo, hi = next_neighbors(h.astype(np.float32), np)
    gaps = np.maximum(h - lo.astype(np.float64), hi.astype(np.float64) - h)
    fine = gaps <= .25 * radius / np.sqrt(len(h))
    fixed = np.column_stack((raw, gradient, *reached))[fine]
    length = np.linalg.norm(fixed, axis=0)
    fixed = fixed[:, length > 1e-12] / length[length > 1e-12]
    first = drive(tag, cfg["sham_namespace"], len(h), np)[fine]
    noise = drive(tag, "NAN-SHAM-56/minimax-basis", len(h)*WIDTH, np).reshape(len(h), WIDTH)[fine]
    partial, r = np.linalg.qr(remove(np.column_stack((first, noise[:, 1:])), fixed, np), mode="reduced")
    if partial.shape[1] != WIDTH or min(abs(r.diagonal())) < 1e-8 or np.max(abs(fixed.T @ partial)) > 1e-9:
        raise ValueError("Fine-supported null basis lost rank or closure")
    full = np.zeros((len(h), WIDTH))
    full[fine] = partial
    return full, int(fine.sum())


def derivative_columns(head, h, axes, radius, np):
    reciprocal = (np.mean(h**2) + head.eps)**-.5
    pushed = reciprocal * axes - h[:, None] * ((h @ axes)[None, :] * reciprocal**3 / len(h))
    return (head.gamma.astype(np.float64)[:, None] * pushed * radius).astype(np.float32)


def atlas(head, directions, rows, check, np):
    columns = np.concatenate(directions, axis=1)
    effects = np.empty((head.vocab, columns.shape[1]), dtype=np.float32)
    head.stream.seek(head.base + head.weight_offset)
    stride = 2*head.width
    for start in range(0, head.vocab, rows):
        check()
        n = min(rows, head.vocab - start)
        packet = head.stream.read(n*stride)
        if len(packet) != n*stride:
            raise ValueError("Missing full head derivative block")
        weights = decode_bf16(packet, np).reshape(n, head.width)
        effects[start:start+n] = weights @ columns
    return effects


def replay(source, head, tokens, base, cfg, round_cfg, dose, bank_pack, parent, check, masks, np):
    bank, controller, eps, anchor = bank_pack
    wbasis = basis(bank["U"].T.astype(np.float64), np)
    folder = ARCHIVE / "nf4"
    audit = json.loads(bound(ARCHIVE / "audits/nf4.json", base["backends"]["nf4"]["audit"]))
    sites = {(r["root"], r["world"]): slot for slot, r in source.items()}
    packets, columns = [], []
    for row, (h, x, y) in route_states("nf4", base, source, check, np):
        h = h.astype(np.float64)
        root, world = row["root"], row["world"]
        foil = source[sites[root, "B" if world == "A" else "A"]]
        pair = [tokens[foil["answer"]], tokens[row["answer"]]]
        slot = sites[root, world]
        case = json.loads(bound(folder / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
        phase = case["initial_phases"][0]
        if phase["loop"] != 1 or pair[0] == pair[1]:
            raise ValueError("Wrong native readout scope")
        vectors, _ = directions(x, y, masks(x[None], bank, controller)[0], anchor, bank, eps, np)
        raw = vectors["learned"] - vectors["static"]
        g = head.gamma.astype(np.float64) * (head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
        g = head_gradient(h, g, head.eps, np)
        radius = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
        tag = f"nf4/{root}/{world}"
        dense, original, _ = make_pair(h, raw, g, radius, None, tag, round_cfg, np)
        active, metrics, qualifies = patches(h, {"contrast": dense["contrast"]}, dose, np)
        expected = parent[root, world]["recipes"]["0.0075/dense"]["arm_dose"]
        if not qualifies or any(any(expected[k][field] != value for field, value in v.items()) for k, v in metrics.items()):
            raise ValueError("The fixed active edit changed")
        span, support = form_basis(h, original, g, [p.after-p.before for p in active.values()], radius, tag, cfg, np)
        columns.append(derivative_columns(head, h, span, radius, np))
        packets.append((row, h, pair, phase["score"], dense["contrast"], original, span, support, active))
    matrix = atlas(head, columns, dose["block_rows"], check, np)
    states, refs, labels, rows = [], [], [], []

    def append(value, pair, ref=None):
        idx = len(states)
        refs.append(idx if ref is None else ref)
        states.append(value)
        labels.append(pair)
        return idx

    for i, (row, h, pair, score, contrast, original, span, support, active) in enumerate(packets):
        check()
        q, old, peak = optimize(matrix[:, i*WIDTH:(i+1)*WIDTH].copy(), np)
        sham = span @ q.astype(np.float64)
        clean = append(h, pair)
        indexed = {k: append(p.after, pair, clean) for k, p in active.items()}
        edits, result = match(h, {"contrast": contrast, "sham": sham}, original, round_cfg, dose, np)
        result.update(sham_support_size=support, predicted_peak_before=old, predicted_peak_after=peak,
                      sham_energy_in_original_span={
                          s: span_fraction(edits[f"sham/{s}"].after - h, wbasis, np) for s in SIGNS})
        if result["physical_gate"]:
            result["head_indices"] = indexed | {k: append(p.after, pair, clean) for k, p in edits.items() if k not in indexed}
        rows.append({"root": row["root"], "world": row["world"], "clean": clean, "citation": score,
                     "fixed_active_head_indices": indexed, "recipes": {NAME: result}})
    del matrix
    live = full_head(head, np.stack(states), np.asarray(refs), np.asarray(labels), dose["block_rows"], check, np)
    custody = physical_head_closure(rows, live, parent, dose, base)
    summary = summarize(rows, [(NAME, "ambient", .25)])[NAME]
    metric = [r["recipes"][NAME] for r in rows]
    summary["median_predicted_peak_before"] = median(r["predicted_peak_before"] for r in metric)
    summary["median_predicted_peak_after"] = median(r["predicted_peak_after"] for r in metric)
    peaks = [r["arm_dose"][f"sham/{s}"]["max_abs_logit_delta"] for r in metric if r["physical_gate"] for s in SIGNS]
    summary["median_reached_sham_peak"] = median(peaks) if peaks else None
    return rows, summary | {"custody": custody}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--active", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(LOCK, seal())
        return
    if not __debug__ or json.loads(LOCK.read_bytes()) != seal():
        raise ValueError("Minimax source freeze changed")
    if not (args.snapshot and args.active and args.output):
        cli.error("--snapshot, --active, and --output required")
    if digest(args.active / "aggregate.json") != digest(ACTIVE_AGG):
        raise ValueError("FRONTIER55 aggregate changed")
    parent = json.loads(ACTIVE_AGG.read_bytes())
    parent_rows = json.loads(bound(args.active / "private.json", parent["private_sha256"]))
    cfg = json.loads((HERE / "registration.json").read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes()) | {"phase1_dose_fraction": cfg["phase1_dose_fraction"]}
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Frozen checkpoint revision changed")
    args.output.mkdir(parents=True, exist_ok=False)
    with lease(cfg) as (check, samples):
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        known = np.tile(np.array([[2., .2], [1., .01]], dtype=np.float32), (64, 1))
        _, initial, reduced = optimize(known, np)
        assert initial == 2. and reduced < initial
        bank = calibrations(base, masks, np)["nf4"]
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(path))
        encoded = {c: tokenizer.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in encoded.values()) or len({v[0] for v in encoded.values()}) != 16:
            raise ValueError("Pinned bare labels changed")
        tokens = {c: v[0] for c, v in encoded.items()}
        outcomes = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for panel, roster in zip(("fit", "check"), rosters(base)):
                receipt = {(r["root"], r["world"]): r for r in parent_rows if r["panel"] == panel}
                rows, summary = replay(roster, head, tokens, base, cfg, round_cfg, dose, bank, receipt, check, masks, np)
                private = args.output / f"{panel}_private.json"
                write_new(private, rows)
                outcomes[panel] = summary | {"private_sha256": digest(private)}
                print(panel, summary, flush=True)
        check()
    write_new(args.output / "aggregate.json", {"experiment": "NAN-SHAM-56-MINIMAX", "panels": outcomes,
        "source_sha256": digest(LOCK), "frontier_active_sha256": digest(ACTIVE_AGG), "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False})


if __name__ == "__main__":
    main()
