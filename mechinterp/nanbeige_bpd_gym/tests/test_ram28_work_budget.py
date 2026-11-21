"""Regression: attention/MLP input widths differ from residual width."""
from types import SimpleNamespace
import unittest
from mechinterp.nanbeige_bpd_gym.ram28_work_budget import capture_layout


def spans(specs):
    result={}
    for layer,(hidden,intermediate,qwidth) in enumerate(specs):
        prefix=f'model.layers.{layer}.'
        for name,shape in [('mlp.down_proj.weight',(hidden,intermediate)),('self_attn.o_proj.weight',(hidden,qwidth)),
                           ('post_attention_layernorm.weight',(hidden,))]:
            result[prefix+name]=SimpleNamespace(dtype='BF16',shape=shape)
    return result


class WorkBudgetTests(unittest.TestCase):
    def test_pinned_unequal_site_widths_regression(self):
        data=spans([(3072,10752,6144)]*4);layout,payload=capture_layout(data,range(4))
        self.assertEqual(payload,417792)
        self.assertEqual(layout[0],dict(residual_after_layer=3072,down_proj_input=10752,down_proj_output=3072,o_proj_input=6144,o_proj_output=3072))
        self.assertNotEqual(payload,40*3072*2)
        self.assertEqual(payload*242048,101125718016)
    def test_each_layer_keeps_its_own_physical_input_widths(self):
        layout,payload=capture_layout(spans([(8,12,16),(8,16,8)]),[0,1])
        self.assertEqual(payload,400);self.assertEqual(layout[1]['down_proj_input'],16)
        self.assertEqual(layout[0]['o_proj_input'],16)
    def test_non_BF16_metadata_is_refused(self):
        data=spans([(8,12,16)]);data['model.layers.0.self_attn.o_proj.weight'].dtype='F32'
        with self.assertRaisesRegex(ValueError,'BF16'):capture_layout(data,[0])
    def test_incompatible_residual_or_projection_shape_is_refused(self):
        data=spans([(8,12,16)]);data['model.layers.0.self_attn.o_proj.weight'].shape=(9,16)
        with self.assertRaisesRegex(ValueError,'residual'):capture_layout(data,[0])
        data['model.layers.0.self_attn.o_proj.weight'].shape=(8,16,2)
        with self.assertRaisesRegex(ValueError,'matrix'):capture_layout(data,[0])


if __name__=='__main__':unittest.main()
