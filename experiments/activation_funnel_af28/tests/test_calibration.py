"""AF-28 deterministic source-only STOP calibrator tests."""
import unittest
from experiments.activation_funnel_af28 import calibration as cal


def example_rows(split,positive=80,negative=112):
    rows=[]
    for i in range(positive+negative):
        truth=0 if i<positive else 1
        margin=-2. if truth==0 else 2.
        features=[
            margin,float(truth)*2-1,0.6 if truth else -0.3,
            margin/3,0.5,-0.5
        ]
        rows.append({
            "split":split,
            "case_id":f"{split}-{i:04d}",
            "stop_target":truth,
            "features":features,
        })
    return rows


class CalibratorTests(unittest.TestCase):
    def test_train_only_full_and_scalar_fit(self):
        train=example_rows("train")
        dev=example_rows("dev")
        models=cal.make_models(train,dev)
        self.assertEqual(models["CAL"]["feature_indices"],list(range(6)))
        self.assertEqual(models["THR"]["feature_indices"],[0])
        for arm in ("CAL","THR"):
            x=models[arm]
            self.assertEqual(x["updates"],600)
            self.assertEqual(x["train_rows"],192)
            self.assertEqual(x["train_label_positive"],112)
            self.assertEqual(x["train_label_negative"],80)
            self.assertIn(x["dev"]["chosen_threshold"],cal.THRESHOLDS)
            self.assertGreater(
                cal.predict(x,dev[-1]["features"]),
                cal.predict(x,dev[0]["features"])
            )

    def test_fit_deterministic(self):
        data=example_rows("train")
        a=cal.fit(data,tuple(range(6)))
        b=cal.fit(data,tuple(range(6)))
        self.assertEqual(a,b)

    def test_dev_threshold_uses_root_only(self):
        t=cal.fit(example_rows("train"),(0,))
        dev=example_rows("dev")
        r=cal.select_threshold(t,dev)
        self.assertTrue(r["used_eligible_dev_floor"])
        self.assertGreaterEqual(r["dev_chosen_positive_continue"],72)
        self.assertEqual(len(r["dev_grid_receipt"]),19)
        with self.assertRaises(ValueError):
            cal.select_threshold(t,example_rows("train"))
        with self.assertRaises(ValueError):
            cal.select_threshold(t,dev[:150])

    def test_reject_test_split_as_train(self):
        with self.assertRaises(ValueError):
            cal.fit(example_rows("held_depth5"),(0,))
        with self.assertRaises(ValueError):
            cal.fit(example_rows("train")[:5],())

    def test_native_feature_extraction_uses_public_only(self):
        source={
            "scores":{"direct":[1,2]+[-1e4]*12+[0.5],
                      "vi2":[-3,-2]+[-1e4]*12+[-1]},
            "trace":{"nodes":[{
                "state":[0,1,2,3],
                "reach_logit":-2,
                "distance_logit":5,
            }]}
        }
        v=cal.feature_row(source,frozenset((0,1,2,3)),(0,1))
        self.assertEqual(len(v),6)
        self.assertEqual(v[0],-1.5)
        self.assertEqual(v[1],-.5)
        self.assertEqual(v[2],1.25)
        self.assertEqual(v[3],.125)
        self.assertEqual(v[4],-.5)
        self.assertEqual(v[5],0)
        with self.assertRaises(ValueError):
            cal.feature_row(source,frozenset((0,1,2,3)),())

    def test_stopping_probability_and_legal_gate(self):
        self.assertFalse(cal.choose(.49,.5))
        self.assertTrue(cal.choose(.5,.5))
        with self.assertRaises(ValueError):
            cal.choose(1.2,.5)
        with self.assertRaises(ValueError):
            cal.choose(.4,.123)


if __name__=="__main__":
    unittest.main()
