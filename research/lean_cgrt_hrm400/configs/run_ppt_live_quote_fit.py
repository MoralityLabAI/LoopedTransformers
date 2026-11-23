"""Fit and score late-detour quote policies without a held transition table."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from configs.freeze_ppt_live_quote_fragments import DETOUR_SCHEMA
from configs.freeze_ppt_live_quote_targets import SCHEMA as TARGET_SCHEMA
from configs.freeze_ppt_native_progress import advance
from configs.freeze_ppt_priced_fragments import checked_root
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_quote_policy import summarize
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.bound_quotes import REGIMES, outcomes, public_view
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import candidate_menu
from ppt_gym.live_quote_fragments import NativeQuoteBackup
from ppt_gym.live_quote_meter import fit_meter, predict_billing, public_graph
from ppt_gym.live_quote_policy import LiveQuoteTRM, PublicQuoteFeatures, TwoSlotQuoteActor, public_agenda
from ppt_gym.private_quotes import PENALTY, QUOTES
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.live-quote-policy-fit.v1"
SEED = 1081
UPDATES = 640
BATCH = 32


def registered(args):
    bank = read_receipt(args.bank)
    source = {case["module"] for case in bank["source"]}
    held = {case["module"] for case in bank["held"]}
    pins = {key: sha256_file(getattr(args, key)) for key in ("bank", "targets", "warm", "prior")}
    if (bank["schema"] != DETOUR_SCHEMA or len(source) != 48 or len(held) != 40 or source & held
            or bank["lean"] != sha256_file(LEAN_EXE) or bank["repl"] != sha256_file(REPL_EXE)):
        raise ValueError("the frozen live-quote cohorts or toolchain changed")
    return bank, pins


@torch.no_grad()
def warm_values(prior, public, menu):
    if not menu:
        return {}
    scores = prior.curves(public, menu, []).tolist()
    return dict(zip(menu, scores, strict=True))


def pick_rows(case, ledger):
    """Sample public depths at fixed quotas without inspecting any target."""
    root = case["native"]["root"]
    groups = ([], [], [])
    for row in ledger:
        group = 0 if row["state"] == root else (1 if row["ticks"] >= 4 else 2)
        groups[group].append(row)
    for group, count in zip(groups, (56, 56, 56), strict=True):
        yield from sorted(group, key=lambda row: digest([SEED, case["module"], row["state"], row["regime"],
                                                          row["charges"], row["quotes"]]))[:count]


def prepare(source, targets, prior, estimated):
    features, intervals, suffixes, group_ids = [], [], [], []
    ledgers = {root["root"]: root["rows"] for root in targets}
    if ledgers.keys() != {case["module"] for case in source}:
        raise ValueError("a non-source theorem entered the source target ledger")
    for group, case in enumerate(source):
        encoder = PublicQuoteFeatures(case, estimated)
        warm = {}
        for row in pick_rows(case, ledgers[case["module"]]):
            key = row["state"]
            state = case["native"]["states"][key]
            public, tactics = state["public"], state["menu"]
            if key not in warm:
                warm[key] = warm_values(prior, public, tactics)
            view = case["views"][row["regime"]]
            menu, packed = encoder.encode(public, tactics, view, row["remaining"], row["ticks"],
                                          row["quotes"], row["charges"], warm[key])
            if set(menu) != row["targets"].keys():
                raise ValueError("a source decision changed its recorded action menu")
            features.append(packed)
            for action in menu:
                target = row["targets"][action]
                intervals.append(target["bellman"])
                observed = target["suffix"]["mean"]
                suffixes.append(float("nan") if observed is None else observed)
                group_ids.append(group % 6)
    return torch.cat(features), torch.tensor(intervals), torch.tensor(suffixes), torch.tensor(group_ids)


def train(model, features, bands, batches, *, rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    for chosen in batches:
        target = bands[chosen]
        known = target.isfinite().all(dim=-1)
        if not known.any():
            continue
        scores = model(features[chosen][known])
        lower, upper = target[known].unbind(-1)
        loss = ((lower - scores).relu().square() + (scores - upper).relu().square()).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def models_for():
    candidate = TwoSlotQuoteActor()
    return {"bellman_two_slot": candidate, "suffix_two_slot": deepcopy(candidate), "bellman_trm": LiveQuoteTRM()}


def fit(args, tree):
    bank, pins = registered(args)
    target = read_receipt(args.targets)
    if target["schema"] != TARGET_SCHEMA or target["bank_sha256"] != pins["bank"]:
        raise ValueError("the matched source targets changed")
    prior = MixedSourcePrior(args)
    meter_model = fit_meter(bank["source"])
    features, bands, empirical, folds = prepare(bank["source"], target["source"], prior, meter_model)
    batches = torch.randint(len(features), (UPDATES, BATCH), generator=torch.Generator().manual_seed(SEED))
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["bellman_two_slot"] != initial["suffix_two_slot"]:
        raise ValueError("the Bellman and suffix copies have different initialization")
    for arm, model in models.items():
        labels = torch.stack((empirical, empirical), -1) if arm == "suffix_two_slot" else bands
        train(model, features, labels, batches, rate=.002)
        print("fitted", arm, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    # Ridge calibrates exact observed backups. Censored intervals stay outside its objective.
    exact = bands[:, 0] == bands[:, 1]
    frozen = ridge(features[exact].flatten(1), bands[exact, 0], folds[exact])
    weights = {arm: model.state_dict() for arm, model in models.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "meter": meter_model, "weights": weights, "frozen": frozen,
                    "initial": initial, "weights_sha256": {arm: tensor_hash(value) for arm, value in weights.items()},
                    "batch_sha256": tensor_hash({"batches": batches}), "source_actions": len(features),
                    "source_suffix_targets": int(empirical.isfinite().sum()), "prior_sources": prior.counts,
                    "memory": tree.stats()}, handle)
    print("sealed fit", sha256_file(args.fit), flush=True)


class LiveRoom:
    def __init__(self, repl, case, prior):
        self.repl, self.prior = repl, prior
        state, public = checked_root(repl, case)
        self.root = digest(public)
        if self.root != case["visible_sha256"]:
            raise ValueError("the declared Lean proof root changed")
        self.states = {self.root: (state, public)}
        self.transition = {}
        self.menu = {}
        self.warm = {}

    def visible(self, key):
        state, public = self.states[key]
        if key not in self.menu:
            self.menu[key] = candidate_menu(self.repl, state)
        return public, self.menu[key]

    def evidence(self, key):
        if key not in self.warm:
            public, menu = self.visible(key)
            self.warm[key] = warm_values(self.prior, public, menu)
        return self.warm[key]

    def act(self, key, move):
        marker = key, move
        if marker not in self.transition:
            state, public = self.states[key]
            status, child, after = advance(self.repl, state, public, move)
            following = digest(after) if status == "progress" else None
            if following is not None:
                self.states.setdefault(following, (child, after))
            self.transition[marker] = status, following
        return self.transition[marker]


def symbolic_policy(case, encoder, view, estimated):
    graph = public_graph(case)
    moves = {move for outgoing in graph.edges.values() for move in outgoing}
    ledger = [{**world, "prices": {move: world["prices"].get(move, 1) for move in moves}}
              for world in predict_billing(case, estimated)]
    solver = NativeQuoteBackup(graph, view["fees"], ledger)

    def select(public, tactics, left, ticks, quotes, charges, *_):
        agenda = digest(public_agenda(public))
        belief = tuple(encoder.posterior(quotes, charges))
        bought = sum(1 << QUOTES.index(move) for move, _ in quotes)
        left_quotes = len(QUOTES) - len(quotes)
        menu = solver.menu(agenda, left, ticks, left_quotes, bought)
        if set(tactics) != {action for action in menu if action not in QUOTES}:
            raise ValueError("the symbolic meter grammar disagrees with Lean")

        def score(action):
            return solver.action(agenda, left, ticks, belief, left_quotes, bought, action)[0]

        best = max(menu, key=score)
        return best if score(best) > 0 else None

    return select, solver


def replay(room, select, view, prices, bits, *, patch=None):
    key, left, ticks = room.root, view["budget"], 8
    quotes, charges, script, trace = [], [], [], []
    proof = False
    patched_at = None
    while left and ticks:
        public, tactics = room.visible(key)
        if not tactics:
            break
        use_patch = patch is not None and patched_at is None and patch <= set(tactics)
        cache = room.evidence(key)
        move = select(public, tactics, left, ticks, quotes, charges, cache, use_patch)
        if use_patch:
            patched_at = len(trace)
        if move is None:
            break
        if move in QUOTES:
            charge = view["fees"][move]
            if charge > left or move in dict(quotes):
                raise ValueError("a live policy requested an unavailable quote")
            quotes.append((move, bits[move]))
            trace.append({"state": key, "quote": move, "charge": charge, "bit": bits[move]})
            left -= charge
            continue
        if move not in tactics:
            raise ValueError("the learned action is outside the live tactic menu")
        charge = prices.get(move, 1)
        if charge > left:
            trace.append({"state": key, "action": move, "charge": 0, "status": "unaffordable"})
            break
        status, following = room.act(key, move)
        trace.append({"state": key, "action": move, "charge": charge, "status": status})
        left -= charge
        ticks -= 1
        if status not in ("progress", "closed"):
            break
        script.append(move)
        if status == "closed":
            proof = True
            break
        charges.append((move, charge))
        key = following
    paid = sum(event["charge"] for event in trace if "quote" in event)
    return {"proof": proof, "script": script, "trace": trace, "budget": view["budget"],
            "quote_effort": paid, "proof_effort": view["budget"] - left - paid,
            "utility": int(proof) - paid * PENALTY, "patched_at": patched_at}


def swapped_binding(case):
    changed = deepcopy(case)
    late = changed["late_link"]
    channel = changed["channels"][late["quote"]]
    original, other = channel["rule"], late["counterpart"]
    channel["rule"], late["counterpart"] = other, original
    return changed, {original, other}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins = registered(args)
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or pins != checkpoint["pins"]:
        raise ValueError("the sealed fit no longer matches its inputs")
    models = models_for()
    for arm, model in models.items():
        state = checkpoint["weights"][arm]
        if tensor_hash(state) != checkpoint["weights_sha256"][arm]:
            raise ValueError("the sealed live actor changed")
        model.load_state_dict(state)
        model.eval()
    records, proofs, checked_steps = defaultdict(list), {}, 0
    prior = MixedSourcePrior(args)
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected Lean version")
        for index, case in enumerate(bank["held"], start=1):
            room = LiveRoom(repl, case, prior)
            bound = PublicQuoteFeatures(case, checkpoint["meter"])
            changed, swap = swapped_binding(case)
            swapped = PublicQuoteFeatures(changed, checkpoint["meter"])
            for regime in REGIMES:
                view = public_view(case, regime)
                metered, solver = symbolic_policy(case, bound, view, checkpoint["meter"])
                for arm in (*models, "frozen_calibration", "supervised_meter_planner"):
                    predictions = {}

                    def select(public, tactics, left, ticks, quotes, charges, warm, altered):
                        key = digest(public), left, ticks, tuple(quotes), tuple(charges), altered
                        if key not in predictions:
                            encoder = swapped if altered else bound
                            menu, fields = encoder.encode(public, tactics, view, left, ticks, quotes, charges, warm)
                            if arm == "frozen_calibration":
                                scores = apply_ridge(fields.flatten(1), checkpoint["frozen"])
                            else:
                                scores = models[arm](fields)
                            predictions[key] = menu[int(scores.argmax())]
                        return predictions[key]

                    policy = metered if arm == "supervised_meter_planner" else select
                    for world, (prices, bits) in enumerate(outcomes(case)):
                        tags = {"root": case["module"], "regime": regime, "world": world, "family": case["family"]}
                        row = replay(room, policy, view, prices, bits)
                        records[arm].append({**tags, **row})
                        if row["proof"]:
                            proofs[case["module"], tuple(row["script"])] = case
                        if arm == "bellman_two_slot":
                            donor = replay(room, policy, view, prices, bits, patch=swap)
                            position = donor["patched_at"]
                            if position is not None:
                                if row["trace"][:position] != donor["trace"][:position]:
                                    raise ValueError("the late binding patch altered its paid prefix")

                                def identity(public, tactics, left, ticks, quotes, charges, warm, _altered):
                                    return policy(public, tactics, left, ticks, quotes, charges, warm, False)

                                sham = replay(room, identity, view, prices, bits, patch=swap)
                                if sham != {**row, "patched_at": position}:
                                    raise ValueError("the identity binding changed a natural trace")
                            records["late_binding_swap"].append({**tags, **donor, "identity_sham": position is not None})
                            if donor["proof"]:
                                proofs[case["module"], tuple(donor["script"])] = case
                solver.action.cache_clear()
                solver.value.cache_clear()
            checked_steps += len(room.transition)
            if index % 10 == 0:
                print("scored roots", index, flush=True)
    with TemporaryDirectory(prefix="ppt-live-quote-fit-") as directory:
        independent = verify_independent(Path(directory), lean_env(), (), [(case, key[1]) for key, case in proofs.items()])
    summary = {arm: summarize(rows) for arm, rows in records.items()}
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "results": records, "summary": summary, "independent": independent,
                                 "unique_native_transitions": checked_steps, "memory": tree.stats()})
    for arm, value in summary.items():
        print(arm, value, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "targets", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
