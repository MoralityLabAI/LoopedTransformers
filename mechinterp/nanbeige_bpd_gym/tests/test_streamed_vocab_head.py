"""Tiny CPU matrices: all vocabulary rows, B1/B2, stage reuse and source bytes."""
import hashlib
import unittest
import torch
from torch.nn import functional as F
from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead


def digest(tensor):return hashlib.sha256(tensor.detach().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class StreamedHeadTests(unittest.TestCase):
    def test_all_rows_match_same_chunk_arithmetic_with_one_reused_stage(self):
        torch.manual_seed(2232)
        for dtype in (torch.float64,torch.bfloat16):
            weight=torch.randn(137,64,dtype=dtype);before=digest(weight)
            head=StreamedVocabularyHead(weight,'cpu',7);pointer=head.stage.data_ptr()
            for B in (1,2):
                hidden=torch.randn(B,64,dtype=dtype)
                expected=torch.cat([F.linear(hidden,weight[s:s+7]).float() for s in range(0,len(weight),7)],dim=-1)
                actual=head(hidden)
                self.assertEqual(actual.shape,(B,137));self.assertTrue(torch.equal(actual,expected))
                self.assertEqual(head.stage.data_ptr(),pointer);self.assertEqual(head.staging_bytes,7*64*weight.element_size())
                self.assertEqual(head.last_chunks,20);self.assertEqual(head.last_copied_bytes,weight.numel()*weight.element_size())
                self.assertEqual(digest(weight),before)
            head.close();self.assertEqual(head.staging_bytes,0)
            with self.assertRaises(RuntimeError):head(hidden)
    def test_invalid_target_or_boundary_is_refused(self):
        with self.assertRaises(ValueError):StreamedVocabularyHead(torch.ones(4,3,requires_grad=True),'cpu')
        with self.assertRaises(ValueError):StreamedVocabularyHead(torch.ones(4,3),'cpu',0)
        head=StreamedVocabularyHead(torch.ones(4,3),'cpu',2)
        try:
            for hidden in (torch.ones(1,2),torch.ones(1,3,dtype=torch.float64),torch.ones(1,3,requires_grad=True)):
                with self.assertRaises(ValueError):head(hidden)
        finally:head.close()


if __name__=='__main__':unittest.main()
