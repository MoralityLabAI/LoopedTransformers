"""RAM-17 TRAIN-only candidate freeze must fail closed on paired-model errors."""
from __future__ import annotations
import json,hashlib,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pretest_freeze as f

def make_fixture(base:Path):
    fits=base/"fits";third=base/"third";third.mkdir()
    heldout={f"MathCan/{name}.lean" for name in f.CONFIG["source_modules"]["test"]}
    positive_sha="c"*64
    gate=dict(status="passed_compiler_replay_gate_admitted",gate_admitted=True,
              source_heldout_test_accessed=False,validated_positive_data_sha256=positive_sha)
    (third/"independent_gate.json").write_text(json.dumps(gate))
    rows=[]
    for arch in f.ARCHS:
        for seed in f.SEEDS:
            for obj in f.OBJECTIVES:
                name=f"{arch}_{obj}_s{seed}"
                folder=fits/name;folder.mkdir(parents=True)
                shards={}
                for label in ("best","last"):
                    file=folder/(label+".pt");file.write_bytes((name+label).encode())
                    shards[label]=hashlib.sha256(file.read_bytes()).hexdigest()
                structural=obj=="task_plus_structural"
                counts={"nat_induction":854 if structural else 0,
                        "simp_prior":853 if structural else 0,
                        "rewrite_prior":853 if structural else 0}
                usage=dict(verified_positive_only=True,
                    total_draws=2560 if structural else 0,
                    by_family=counts,
                    by_module={fam:{"MathCan/BasinCodes.lean":n} for fam,n in counts.items()})
                rows.append(dict(architecture=arch,seed=seed,objective=obj,
                    training_steps=400,test_cases_scored=0,
                    starting_state_sha256=hashlib.sha256((arch+str(seed)).encode()).hexdigest(),
                    macro_examples_sampled=2560 if structural else 0,
                    macro_balance_receipt=usage,parameters=207365 if arch=="loop64" else 208390,
                    best_checkpoint=f"fits/{name}/best.pt",
                    best_checkpoint_sha256=shards["best"],last_checkpoint_sha256=shards["last"],
                    best_step=200,best_validation_loss=.5))
    report=dict(status="complete",fits=12,whole_module_TEST_touched=False,
        heldout_proof_cases_scored=0,seeds=list(f.SEEDS),optimizer_updates_each=400,
        source_split=dict(source_train_positive_sha256=positive_sha,
                          prospective_test_modules_excluded=sorted(heldout)),model_fits=rows)
    (base/"training_report.json").write_text(json.dumps(report))
    return base,third,report

class FreezeTest(unittest.TestCase):
    def test_synthetic_twelve_fit_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,th,r=make_fixture(Path(tmp))
            result=f.candidate(tr,th)
            self.assertEqual(len(result["all_twelve_models_sha256"]),12)
            self.assertEqual(result["test_cases_scored"],0)
            self.assertFalse(result["checkpoint_lock_committed"])
    def test_unpaired_initial_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,th,r=make_fixture(Path(tmp))
            r["model_fits"][1]["starting_state_sha256"]="0"*64
            (tr/"training_report.json").write_text(json.dumps(r))
            with self.assertRaises(ValueError):f.candidate(tr,th)
    def test_test_source_legacy_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,th,r=make_fixture(Path(tmp))
            r["source_split"]["prospective_test_modules_excluded"]=[]
            (tr/"training_report.json").write_text(json.dumps(r))
            with self.assertRaises(ValueError):f.candidate(tr,th)
    def test_checkpoint_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,th,r=make_fixture(Path(tmp))
            target=r["model_fits"][0];(tr/"fits"/f"{target['architecture']}_{target['objective']}_s{target['seed']}"/"best.pt").write_bytes(b"bad")
            with self.assertRaises(ValueError):f.candidate(tr,th)
    def test_unverified_macro_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,th,r=make_fixture(Path(tmp))
            r["model_fits"][1]["macro_balance_receipt"]["verified_positive_only"]=False
            (tr/"training_report.json").write_text(json.dumps(r))
            with self.assertRaises(ValueError):f.candidate(tr,th)
if __name__=="__main__":unittest.main()
