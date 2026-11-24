"""Exactly three decoder transposes, mechanically reversible on frozen auditors."""
import ast


def decoder_cast(node):
    return (isinstance(node,ast.Call) and len(node.args)==1 and not node.keywords
        and isinstance(node.args[0],ast.Name) and node.args[0].id=='float'
        and isinstance(node.func,ast.Attribute) and node.func.attr=='astype'
        and isinstance(node.func.value,ast.Subscript) and isinstance(node.func.value.value,ast.Name)
        and node.func.value.value.id in ('sae','random')
        and isinstance(node.func.value.slice,ast.Constant) and node.func.value.slice.value=='decoder')


def repaired_tree(source):
    original=ast.parse(source);before=ast.dump(original)
    class Transpose(ast.NodeTransformer):
        def __init__(self):self.changed=[]
        def visit_Call(self,node):
            if decoder_cast(node):
                self.changed.append((node.lineno,node.func.value.value.id))
                return ast.copy_location(ast.Attribute(value=node,attr='T',ctx=ast.Load()),node)
            return self.generic_visit(node)
    change=Transpose();tree=change.visit(original)
    assert len(change.changed)==3 and [v[1] for v in change.changed]==['sae','random','sae'],change.changed
    # Reverse only decoder-specific .T wrappers; all thresholds, assertions,
    # variable names, root averaging and independent equations must match.
    class Reverse(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if node.attr=='T' and decoder_cast(node.value):return node.value
            return self.generic_visit(node)
    restored=Reverse().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(restored)==before,'More than the three orientation expressions changed'
    return ast.fix_missing_locations(tree),change.changed
