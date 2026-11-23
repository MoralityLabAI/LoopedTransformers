import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbf_gym as m


class GlyphCodec(unittest.TestCase):
    def test_injective_round_trip(self):
        characters = []
        for role in m.ROLES:
            for frame in m.FRAMES:
                for polarity in m.POLARITIES:
                    char = m.glyph(role, frame, polarity)
                    self.assertEqual((role, frame, polarity), m.unglyph(char))
                    characters.append(char)
        self.assertEqual(144, len(set(characters)))
        for mask in (0, 7 << 3, 192):
            with self.assertRaises(ValueError):
                m.unglyph(chr(0x2800 + mask))

    def test_quantization_and_nonfinite_rejected(self):
        self.assertEqual([0,1,2,3,3], [m.light_level(x) for x in (0, .25, .5, .75, 1)])
        for x in (float('nan'),float('inf'),-0.1,1.01):
            with self.assertRaises(ValueError):
                m.light_level(x)

    def test_verifier_leak_and_shape_rejected(self):
        raw, _ = m.fixture(1, 7)
        row = copy.deepcopy(raw[0])
        row['verdict'] = 'success'
        with self.assertRaises(ValueError):
            m.encode(row)
        row = copy.deepcopy(raw[0])
        row['visits'][0]['tiles'][0]['certificate'] = 'passed'
        with self.assertRaises(ValueError):
            m.encode(row)
        row = copy.deepcopy(raw[0])
        row['visits'][1]['pass'] = 1
        with self.assertRaises(ValueError):
            m.encode(row)

    def test_encode_render_svg_and_agent_observation(self):
        raw, labels = m.fixture(1, 9)
        view = m.encode(raw[0])
        self.assertNotIn('activation',m.canonical(view))
        self.assertNotIn('outcomes',m.canonical(view))
        ET.fromstring(m.fresco_svg(view))
        env = m.FrescoEnv(view,labels[0])
        self.assertEqual(1,len(env.observation()['visits']))
        self.assertNotIn('outcomes',m.canonical(env.observation()))
        cont = env.step('continue')
        self.assertEqual(-.02,cont['reward'])
        self.assertEqual(2,len(env.observation()['visits']))
        chosen = m.choose_candidate(view,'glyph')
        result = env.step('select',chosen)
        self.assertEqual('success',result['status'])
        self.assertAlmostEqual(.98,result['total_reward'])
        with self.assertRaises(ValueError):
            env.step('continue')

    def test_mismatched_labels_and_censoring(self):
        raw,labels = m.fixture(2,4)
        encoded=[m.encode(r) for r in raw]
        corrupt=copy.deepcopy(labels)
        corrupt[0]['lineage']='wrong'
        with self.assertRaises(ValueError):
            m.score(encoded,corrupt)
        corrupt=copy.deepcopy(labels)
        corrupt[0]['outcomes']['t0']='censored'
        result=m.score(encoded,corrupt)
        self.assertEqual(1,result['roots_censored'])
        self.assertEqual(1,result['roots_eligible'])

    def test_controls_are_representation_only(self):
        raw, labels = m.fixture(60, 20261123)
        x=[m.encode(r) for r in raw]
        summary=m.score(x,labels)
        self.assertEqual(60,summary['roots_eligible'])
        self.assertEqual(60,summary['results']['intact']['glyph']['correct'])
        self.assertLess(summary['results']['roll_roles']['glyph']['correct'],60)
        self.assertLess(summary['results']['intact']['led']['correct'],60)
        self.assertFalse(summary['scientific_eligible'])
        self.assertEqual(m.canonical(x[0]),m.canonical(m.intervene(x[0],'intact')))

    def test_deterministic_fixture(self):
        self.assertEqual(m.digest(m.fixture(10,2026)),m.digest(m.fixture(10,2026)))
        self.assertNotEqual(m.digest(m.fixture(10,2026)),m.digest(m.fixture(10,2027)))


if __name__ == '__main__':
    unittest.main()
