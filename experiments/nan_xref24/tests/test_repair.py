import ast
from pathlib import Path
import unittest
from experiments.nan_xref24.repair import decoder_cast,repaired_tree


class Matrix:
    def __init__(self,rows):self.rows=rows
    def astype(self,dtype):return self
    @property
    def T(self):return Matrix(list(map(list,zip(*self.rows))))


def product(code,matrix):
    assert len(code)==len(matrix.rows),'rectangular decoder orientation mismatch'
    return [sum(code[i]*matrix.rows[i][j] for i in range(len(code))) for j in range(len(matrix.rows[0]))]


class Repair(unittest.TestCase):
    def test_three_reversible_changes_in_both_frozen_auditors(self):
        for source in ('nan_xref19','nan_xref22'):
            tree,changed=repaired_tree((Path('experiments')/source/'audit.py').read_text())
            self.assertEqual([x[1] for x in changed],['sae','random','sae'])
            nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr=='T' and decoder_cast(n.value)]
            self.assertEqual(len(nodes),3)
            namespace={name:{'decoder':Matrix([[1,2],[3,4],[5,6]])} for name in ('sae','random')}
            for node in nodes:
                # Evaluate the EXACT mechanically changed expression, not a
                # separately reimplemented decoder orientation convention.
                value=eval(compile(ast.Expression(node),'rectangular_fixture','eval'),namespace)
                normalized=product([1,2],value);self.assertEqual(normalized,[5,11,17])
                with self.assertRaises(AssertionError):
                    product([1,2],eval(compile(ast.Expression(node.value),'old_fixture','eval'),namespace))
                reconstructed=[m+s*(b+x) for m,s,b,x in zip([10,20,30],[2,3,4],[.1,.2,.3],normalized)]
                for a,b in zip(reconstructed,[20.2,53.6,99.2]):self.assertAlmostEqual(a,b)
                physical=[[x*s for x,s in zip(row,[2,3,4])] for row in value.rows]
                self.assertEqual(physical,[[2,9,20],[4,12,24]])

    def test_missing_or_extra_decoder_cast_is_rejected(self):
        source=Path('experiments/nan_xref19/audit.py').read_text()
        for altered in (source.replace("code @ sae['decoder'].astype(float)","code @ other_decoder",1),
            source+"\nextra = sae['decoder'].astype(float)\n"):
            with self.assertRaises(AssertionError):repaired_tree(altered)


if __name__=='__main__':unittest.main()
