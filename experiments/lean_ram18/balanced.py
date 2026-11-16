"""RAM-18 family -> module -> theorem -> script balanced verified-positive sampler.

Every curriculum example must carry its independent three-source-compile receipt.
Counts stay auditable at all four levels. Missing families are a failed gate,
not a reason to invent replacement labels or recycle exposed Gramian roots.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import random

FAMILIES = ("nat_induction", "simp_prior", "rewrite_prior")


class VerifiedCurriculum:
    def __init__(self, positives: list[dict], seed: int):
        self.rng = random.Random(seed)
        raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        seen: set[tuple[str, str, str]] = set()
        for row in positives:
            if row.get("role") != "train" or row.get("label") != 3 or row.get("source_compiles") != 3:
                raise ValueError("curriculum row lacks independent TRAIN triple-source certification")
            fam = row.get("family")
            src = row.get("source_file")
            root = row.get("theorem")
            script = row.get("script")
            if fam not in FAMILIES or not isinstance(src, str) or not src.startswith("MathCan/") or not src.endswith(".lean"):
                raise ValueError("invalid source family or module")
            if src == "MathCan/GramianTransfer.lean":
                raise ValueError("previously exposed RAM-14 Gramian cannot enter RAM-18")
            if not isinstance(root, str) or not root or not isinstance(script, str) or not script:
                raise ValueError("missing theorem root or action")
            key = (src, root, script)
            if key in seen:
                raise ValueError("duplicate independently certified action")
            seen.add(key)
            raw[fam][src][root].append(row)
        if set(raw) != set(FAMILIES):
            raise ValueError("registered all-family compiler admission gate failed")
        self.families = FAMILIES
        self.modules: dict = {}
        self.roots: dict = {}
        self.scripts: dict = {}
        for fam in self.families:
            modules = sorted(raw[fam])
            self.rng.shuffle(modules)
            self.modules[fam] = modules
            for module in modules:
                roots = sorted(raw[fam][module])
                self.rng.shuffle(roots)
                self.roots[fam, module] = roots
                for root in roots:
                    scripts = sorted(raw[fam][module][root], key=lambda row: row["script"])
                    self.rng.shuffle(scripts)
                    self.scripts[fam, module, root] = scripts
        self.next_family = 0
        self.next_module: Counter = Counter()
        self.next_root: Counter = Counter()
        self.next_script: Counter = Counter()
        self.family_counts: Counter = Counter()
        self.module_counts: Counter = Counter()
        self.root_counts: Counter = Counter()

    def sample(self) -> dict:
        fam = self.families[self.next_family % len(self.families)]
        self.next_family += 1
        mod_list = self.modules[fam]
        module = mod_list[self.next_module[fam] % len(mod_list)]
        self.next_module[fam] += 1
        roots = self.roots[fam, module]
        root = roots[self.next_root[fam, module] % len(roots)]
        self.next_root[fam, module] += 1
        scripts = self.scripts[fam, module, root]
        row = scripts[self.next_script[fam, module, root] % len(scripts)]
        self.next_script[fam, module, root] += 1
        self.family_counts[fam] += 1
        self.module_counts[fam, module] += 1
        self.root_counts[fam, module, root] += 1
        return dict(before=row["before"], family=fam, script=row["script"],
                    label=3, delta=1, source_file=module,
                    theorem=root, development=False)

    def receipt(self) -> dict:
        counts = [self.family_counts[fam] for fam in self.families]
        if max(counts) - min(counts) > 1:
            raise AssertionError("macro sampler broke exact family balancing")
        return {
            "strategy": "family_module_root_script_round_robin_seeded",
            "total_draws": sum(counts),
            "by_family": {fam:self.family_counts[fam] for fam in self.families},
            "by_module": {fam:{m:self.module_counts[fam,m] for m in self.modules[fam]}
                          for fam in self.families},
            "by_root": {fam:{m:{root:self.root_counts[fam,m,root] for root in self.roots[fam,m]}
                                  for m in self.modules[fam]}
                          for fam in self.families},
            "verified_positive_only": True
        }


def draw_batch(task: list[dict], sampler: VerifiedCurriculum, step: int,
               mode: str, rng: random.Random) -> tuple[list[dict], int]:
    if not task:
        raise ValueError("no original audited RAM-08 task labels")
    if mode not in ("task_only", "task_plus_structural"):
        raise ValueError("unregistered objective")
    if step < 0:
        raise ValueError("step must start at zero")
    n_macro = (6 + int(step % 5 in (0, 2))) if mode == "task_plus_structural" else 0
    batch = [task[rng.randrange(len(task))] for _ in range(64-n_macro)]
    batch.extend(sampler.sample() for _ in range(n_macro))
    rng.shuffle(batch)
    return batch, n_macro
