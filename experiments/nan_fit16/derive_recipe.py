"""Derive the unchanged RECOMP09 optimizer recipe with sixteen CAL worlds."""
import ast
import hashlib
from .common import ROOT, write
from pathlib import Path

REPLACEMENTS = {
    'import gc\n': 'import gc\nimport hashlib\n',
    'assert len(cal)==8': 'assert len(cal)==16',
    "/8 for t in range(passes)": "/16 for t in range(passes)",
    "receipt['calibration_unit_count']=8": "receipt['calibration_unit_count']=16",
    "receipt['initial_effective_operator_closure']=operator_closure(core,cfg)":
        "effective=core.W_target.detach().float().cpu().contiguous().numpy()\n        assert hashlib.sha256(memoryview(effective)).hexdigest()==source['effective_operator_sha256'];del effective\n        receipt['initial_effective_operator_closure']=operator_closure(core,cfg)",
}


def derive(text):
    for old, new in REPLACEMENTS.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def main():
    old = Path('experiments/nan_recomp09/fit.py').read_text(encoding='utf-8')
    new = derive(old)
    for a, b in REPLACEMENTS.items():
        new = new.replace(b, a)
    assert ast.dump(ast.parse(new)) == ast.dump(ast.parse(old))
    generated = derive(old)
    (ROOT / 'fit.py').write_text(generated, encoding='utf-8', newline='\n')
    write(ROOT / 'derivation_receipt.json', dict(upstream='experiments/nan_recomp09/fit.py',
        upstream_normalized_SHA256=hashlib.sha256(old.encode()).hexdigest(),
        generated_SHA256=hashlib.sha256(generated.encode()).hexdigest(), replacements=REPLACEMENTS,
        optimizer_loss_masks_controller_and_diagnostics_unchanged=True,
        added_effective_operator_SHA_check=True,
        calibration_unit='16 world-prefixes from 8 CAL roots, not 16 independent roots'))


if __name__ == '__main__':
    main()
