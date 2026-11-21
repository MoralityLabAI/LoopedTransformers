"""Match the worker's physical-geometry float64 operand before strict replay."""
import ast
from .common import read,sha
from experiments.nan_xref24.repair import repaired_tree as decoder_tree


def repaired_tree(source):
    tree,changes=decoder_tree(source);before=ast.dump(tree)
    class Promote(ast.NodeTransformer):
        changed=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='U':
                assert isinstance(node.value,ast.IfExp),'Only the frozen geometry-direction selection may change'
                self.changed+=1
                node.value=ast.Call(func=ast.Attribute(value=node.value,attr='astype',ctx=ast.Load()),args=[ast.Name(id='float',ctx=ast.Load())],keywords=[])
            return self.generic_visit(node)
    promote=Promote();tree=promote.visit(tree);assert promote.changed==1
    class Reverse(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='U':
                assert isinstance(node.value,ast.Call) and node.value.func.attr=='astype'
                node.value=node.value.func.value
            return self.generic_visit(node)
    assert ast.dump(Reverse().visit(ast.parse(ast.unparse(tree))))==before
    return ast.fix_missing_locations(tree),dict(decoder_transposes=changes,physical_geometry_float64_direction_casts=1)
