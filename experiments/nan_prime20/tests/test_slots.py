import unittest
from experiments.nan_prime20.common import public_rows
from experiments.nan_prime20.wire import OrderedSlots


def request(row):
    return dict(model='nanbeige-451ed48c-nf4', messages=[dict(role='user', content=row['prompt'])], temperature=0, top_p=1, max_tokens=1)


class SlotFence(unittest.TestCase):
    def test_entire_roster_keeps_duplicate_erased_sources(self):
        _, public = public_rows(); fence = OrderedSlots(public); hashes = []
        for i, row in enumerate(public):
            ordinal, key, found = fence.claim(request(row), 'nf4', {'request_max_tokens': 1})
            self.assertEqual(ordinal, i); self.assertEqual(found['id'], row['id']); hashes.append(key)
            with self.assertRaises(ValueError):
                fence.claim(request(row), 'nf4', {'request_max_tokens': 1})
            fence.commit(row['id'])
            with self.assertRaises(ValueError):
                fence.claim(request(row), 'nf4', {'request_max_tokens': 1})
        self.assertEqual(fence.next_index, 96); self.assertEqual(len(fence.attempted), 96)
        self.assertLess(len(set(hashes)), 96)

    def test_adjacent_duplicate_sources_require_a_different_transport_design(self):
        _, public = public_rows(); a = public[0]; b = dict(a, id='different-source')
        with self.assertRaises(AssertionError):
            OrderedSlots([a, b])


if __name__ == '__main__':
    unittest.main()
