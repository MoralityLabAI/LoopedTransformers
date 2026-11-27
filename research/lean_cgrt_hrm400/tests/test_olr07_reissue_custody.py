"""Custody checks on synthetic archives: an intact reissue authenticates, tampering fails."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
import olr07_reissue_custody as C  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(x):
    return (json.dumps(x, sort_keys=True) + "\n").encode()


class CustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = (C.PINS, C.DIGESTS)
        self.roots = [{"id": f"r{i}", "role": role, "file": "F.lean", "decl": "d", "goal": "g", "line": i, "col": 2}
                      for i, role in enumerate(("train", "calibration", "evaluation"))]
        reg = dump(self.roots)
        files = {"registry.json": sha(reg)}
        man = dump({"files": files})
        self.orig = self.tmp / "orig.zip"
        with zipfile.ZipFile(self.orig, "w") as z:
            z.writestr("audit_manifest.json", man)
            z.writestr("registry.json", reg)
        C.PINS = {"07b": {**self.saved[0]["07b"], "manifest_sha": sha(man), "registry_sha": files["registry.json"],
                          "count": 3}}
        self.custody = {"archive_sha256": sha(self.orig.read_bytes()), "verified_files": 1,
                        "manifest_sha256": sha(man), "registry_sha256": files["registry.json"]}

    def tearDown(self):
        C.PINS, C.DIGESTS = self.saved

    def build(self, tamper=None):
        rows = [{**r, "scored": r["role"] != "evaluation", "human_closed": True, "scope_version": C.VERSION,
                 "env_scope": "before", "decl_in_env_before": False} for r in self.roots]
        rows[2] = {**self.roots[2], "scored": False, "reason": "not_portable_before_command"}
        census = {r["role"]: {"selected": 1, "scored": int(r["role"] != "evaluation"),
                              "human_closed": int(r["role"] != "evaluation"),
                              "exclusions": {} if r["role"] != "evaluation" else {"not_portable_before_command": 1}}
                  for r in self.roots}
        contract = {"kind": "07b", "scope_version": C.VERSION, "source_commit": C.SOURCE, "custody": self.custody,
                    "candidate_hb": 500, "human_hb": 20000, "tracer_sha256": "t", "runner_sha256": "u",
                    "lean_sha256": "l", "repl_sha256": "p"}
        passed = dump({"scope_version": C.VERSION, **{k: True for k in C.SENTINEL_KEYS}})
        body = {"contract.json": dump(contract), "frozen_registry.json": dump(self.roots),
                "receipts.json": dump(rows), "sentinel/PASS.json": passed}
        if tamper == "registry":
            body["frozen_registry.json"] = dump(self.roots[:2])
        manifest = {"schema": C.SCHEMA, "scope_version": C.VERSION, "kind": "07b", "old_receipts_superseded": True,
                    "model_training_updates": 0, "architecture_promotion_eligible": False, "census": census,
                    "custody": self.custody, "native_valid": False, "files": {k: sha(v) for k, v in body.items()}}
        if tamper == "hash":
            body["receipts.json"] = body["receipts.json"] + b" "
        path = self.tmp / f"art-{tamper}.zip"
        with zipfile.ZipFile(path, "w") as z:
            for k, v in body.items():
                z.writestr("scope-reissue-07b/" + k, v)
            z.writestr("scope-reissue-07b/manifest.json", dump(manifest))
            z.writestr("scope-sentinel-07b/PASS.json", passed)
            if tamper == "extra":
                z.writestr("scope-reissue-07b/unlisted.json", b"{}")
        C.DIGESTS = {"07b": sha(path.read_bytes())}
        return path

    def check(self, path):
        roots, oc = C.original(self.orig, "07b")
        return C.reissue(path, "07b", roots, oc, None, "t", "u")

    def test_intact_reissue_authenticates(self):
        out = self.check(self.build())
        self.assertEqual(out["listed_files_verified"], 4)
        self.assertTrue(all(out["sentinels"]["standalone"].values()))

    def test_tampering_is_rejected(self):
        for tamper in ("registry", "hash", "extra"):
            with self.subTest(tamper=tamper), self.assertRaises(C.CustodyError):
                self.check(self.build(tamper))

    def test_wrong_digest_and_runner_rejected(self):
        path = self.build()
        roots, oc = C.original(self.orig, "07b")
        with self.assertRaises(C.CustodyError):
            C.reissue(path, "07b", roots, oc, None, "t", "other-runner")
        C.DIGESTS = {"07b": "0" * 64}
        with self.assertRaises(C.CustodyError):
            C.reissue(path, "07b", roots, oc, None, "t", "u")


if __name__ == "__main__":
    unittest.main()
