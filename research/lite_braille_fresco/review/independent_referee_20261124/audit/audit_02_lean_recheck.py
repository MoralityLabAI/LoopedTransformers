"""Audit 02 - Independent Lean 4.19.0 recheck of lean/LBF17AbstractInterp.lean.

1. Recompiles the registered file unmodified, checks all 15 `#print axioms` receipts and
   compares source / compiler-log hashes with results/lbf17/lean_theory_certificates.
2. Appends LBF17AuditAppendix.lean (kernel-checked counterexample to the soundness of the
   registered executable non-backtracking provenance interpreter) to a scratch copy and
   compiles the concatenation.

Usage: python audit_02_lean_recheck.py --repo <repo> --lean <path/to/lean> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EXPECTED = {
    "abstract_iter_soundness", "best_transformer_soundness", "galois_pos1", "galois_adj2",
    "galois_sub3", "adj2_refines_pos1", "sub3_refines_adj2", "provenance_fibers_refine_merged",
    "repaired_lbf16_horizon_support_soundness", "nonbacktracking_provenance_soundness",
    "cex1_concrete_component_exact", "cex1_pos1_pass1_exact_pass2_spurious",
    "cex1_relational_and_provenance_exact", "cex2_adj2_spurious_sub3_and_prov_exact",
    "cex3_sub3_loses_global_consistency_prov_exact",
}
AUDIT_EXPECTED = {
    "audit_concrete_component", "audit_nonbacktrack_prov_pos1_loses_reachable_word",
    "audit_nonbacktrack_prov_pos1_symbol_false_negative", "audit_base_domains_sound_on_instance",
    "audit_word_filter_model_is_sound_on_instance",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lean", required=True, help="path to the Lean 4.19.0 `lean` executable")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    src = common.lbf_dir(repo) / "lean" / "LBF17AbstractInterp.lean"
    receipt = json.loads((common.lbf_dir(repo) / "results" / "lbf17" / "lean_theory_certificates" / "manifest.json").read_text())

    version = subprocess.run([args.lean, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    res = subprocess.run([args.lean, str(src)], capture_output=True, text=True, timeout=600)
    log = res.stdout + res.stderr
    passed = set(re.findall(r"'LBF17AbstractInterp\.([A-Za-z0-9_]+)' does not depend on any axioms", log))
    log_lf = hashlib.sha256(log.encode()).hexdigest()
    log_crlf = hashlib.sha256(log.replace("\n", "\r\n").encode()).hexdigest()

    text = src.read_text(encoding="utf-8")
    appendix = (Path(__file__).resolve().parent / "LBF17AuditAppendix.lean").read_text(encoding="utf-8")
    combo = scratch / "LBF17AbstractInterp_plus_audit.lean"
    combo.write_text(text + "\n" + appendix, encoding="utf-8")
    res2 = subprocess.run([args.lean, str(combo)], capture_output=True, text=True, timeout=600)
    log2 = res2.stdout + res2.stderr
    passed2 = set(re.findall(r"'LBF17Audit\.([A-Za-z0-9_]+)' does not depend on any axioms", log2))

    # Which mathematical claims of the specification have NO counterpart in the Lean file.
    decls = set(re.findall(r"^(?:private )?theorem\s+([A-Za-z0-9_]+)", text, flags=re.M))
    out = {
        "lean_version": version,
        "registered_lean_version": receipt["lean_version"],
        "source_sha256": common.sha256_file(src),
        "source_sha256_matches_receipt": common.sha256_file(src) == receipt["source_sha256"],
        "returncode": res.returncode,
        "axiom_free_theorems": sorted(passed),
        "all_15_registered_theorems_axiom_free": passed == EXPECTED and res.returncode == 0,
        "compiler_log_sha256_lf": log_lf,
        "compiler_log_sha256_crlf": log_crlf,
        "compiler_log_matches_receipt_under_crlf": log_crlf == receipt["log_sha256"],
        "theorem_declarations_in_file": sorted(decls),
        "sorry_or_axiom_keywords": len(re.findall(r"\bsorry\b|^\s*axiom\b|\bnative_decide\b", text, flags=re.M)),
        "audit_appendix": {
            "returncode": res2.returncode,
            "axiom_free_audit_theorems": sorted(passed2),
            "all_audit_counterexamples_kernel_checked": passed2 == AUDIT_EXPECTED and res2.returncode == 0,
            "compiler_log_tail": log2[-1500:],
        },
    }
    common.write_json(Path(args.out), out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("theorem_declarations_in_file", "axiom_free_theorems")}, indent=2))


if __name__ == "__main__":
    main()
