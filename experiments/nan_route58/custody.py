"""ROUTE58 source custody shared by the model lane and the independent host."""
import json
from pathlib import Path

from experiments.nan_readout34.tokenizer_check import sha

from .source import CONFIG, HERE, REPO, digest, inputs, seal, tokenizer


SOURCE = HERE / "SOURCE_MANIFEST.json"
SOURCE_CHECK = HERE / "SOURCE_CHECK.json"
RUNTIME = HERE / "RUNTIME_MANIFEST.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeRoute58-20261122")
FIELDS = {"public": "public_sha256", "gold": "gold_sha256", "tokens": "token_bindings_sha256"}


def bound_source(folder, key):
    receipt = json.loads(SOURCE_CHECK.read_bytes())
    if receipt["source_manifest_sha256"] != sha(SOURCE) or json.loads(SOURCE.read_bytes()) != seal():
        raise ValueError("Frozen ROUTE58 source bytes differ")
    value = json.loads((folder / f"{key}.json").read_bytes())
    if digest(value) != receipt[FIELDS[key]]:
        raise ValueError(f"Frozen {key} input changed")
    return value


def load_model_inputs(folder, snapshot, check):
    """No gold path is opened by the model-worker input loader."""
    public, old = (bound_source(folder, key) for key in ("public", "tokens"))
    tok, pins = tokenizer(snapshot)
    if inputs(tok, public, snapshot, check) != old or pins != json.loads(SOURCE_CHECK.read_bytes())["tokenizer_pins"]:
        raise ValueError("Pinned prompt token binding changed")
    data = {(r["root"], r["arm"], r["world"]): (r["id"], old["row_bindings"][r["id"]]["ids"]) for r in public}
    if len(data) != len(public):
        raise ValueError("Identical registered source slots")
    return data, old["bare_label_token_ids"], tok.tok


def settings():
    from .source import SHAM

    def read(path):
        return json.loads(path.read_bytes())

    cfg = read(CONFIG)
    round_cfg = read(REPO / "experiments/nan_round54/registration.json")
    dose = read(REPO / "experiments/nan_dose51/registration.json")
    return cfg, read(SHAM / "registration.json"), round_cfg, dose | {"phase1_dose_fraction": cfg["phase1_dose_fraction"]}
