"""Instrument the fixed-loop Transformers reader without mutating shared state."""

from contextlib import ExitStack

import numpy as np

from .core import ARMS, HF_READER_LAYER, TARGET_COORD, WIDTH

CORE_START = 6
CORE_END = 18
WRITER = CORE_END - 1


def reader_call(pass_number: int, layer: int = HF_READER_LAYER) -> int:
    """Zero-based message_input visit. Pass 1 has no previous-pass reader."""
    if pass_number not in (2, 3) or not CORE_START <= layer < CORE_END:
        raise ValueError("reader must be a core layer in pass 2 or 3")
    return (pass_number - 2) * (CORE_END - CORE_START) + layer - CORE_START


def apply_native_patch(bank3, arm: str, sham: int, target: int = TARGET_COORD):
    """Apply a registered edit to a private copy of the final-token bank."""
    import torch

    current = bank3[:, -1, :]
    original = current.clone()
    patched = original.clone()
    if arm in ("zero_2027", "reader_rescue", "norm_preserving_zero"):
        patched[..., target] = 0
        if arm == "norm_preserving_zero":
            original_norm = torch.linalg.vector_norm(original.float(), dim=-1, keepdim=True)
            reduced_norm = torch.linalg.vector_norm(patched.float(), dim=-1, keepdim=True)
            if (reduced_norm == 0).any():
                raise ValueError("cannot preserve norm of an all-target vector")
            patched = patched.float() * (original_norm / reduced_norm)
    elif arm == "l2_sham":
        patched[..., sham] = original[..., sham] - original[..., target]
    elif arm == "swap":
        patched[..., target] = original[..., sham]
        patched[..., sham] = original[..., target]
    else:
        raise ValueError(f"unexpected arm {arm}")
    bank3[:, -1, :] = patched.to(current.dtype)
    return original


def as_numpy(tensor):
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float64).numpy()


class ReaderTrace:
    """Run one clean lane and one lane per intervention/pass in the same forward."""

    def __init__(self, decoder, passes, sham):
        self.decoder = decoder
        self.sham = sham
        self.calls = {}
        self.lanes = {(None, "clean"): 0}
        for pass_number in passes:
            self.calls[reader_call(pass_number)] = pass_number
            for arm in ARMS:
                self.lanes[pass_number, arm] = len(self.lanes)
        self.writers = []
        self.readers = {}
        self.visits = 0
        self.original_bank = None
        self.handles = ExitStack()

    def __enter__(self):
        reader = self.decoder.message_input
        if tuple(reader.query.shape) != (CORE_END - CORE_START, 4, WIDTH):
            raise RuntimeError("unexpected message reader query layout")
        if reader.query.is_meta:
            raise RuntimeError("message_input must be resident for query attribution")
        try:
            self.handles.callback(self.decoder.layers[WRITER].register_forward_hook(self.capture_writer).remove)
            self.handles.callback(reader.register_forward_pre_hook(self.before_reader).remove)
            self.handles.callback(reader.register_forward_hook(self.after_reader).remove)
        except BaseException:
            self.handles.close()
            raise
        return self

    def __exit__(self, *exc):
        self.handles.close()

    def capture_writer(self, module, args, output):
        self.writers.append(as_numpy(output[0, -1]))

    def before_reader(self, module, args):
        import torch

        block2, block3, local_index = args
        call = self.visits
        self.visits += 1
        if local_index != call % (CORE_END - CORE_START):
            raise RuntimeError(f"unexpected reader order at call {call}: local layer {local_index}")
        if self.original_bank is not None:
            original_object, original_value, pass_number = self.original_bank
            if block3 is not original_object or not torch.equal(block3, original_value):
                raise RuntimeError(f"block-3 bank changed before reader {call}")
            self.readers[pass_number]["restored"] = as_numpy(block3[:, -1])
            self.original_bank = None
        if call not in self.calls:
            return None
        pass_number = self.calls[call]
        expected = (len(self.lanes), block3.shape[1], WIDTH)
        if tuple(block3.shape) != expected or tuple(block2.shape) != (*expected[:2], 1, WIDTH):
            raise RuntimeError("expected one completed block-2 state and one partial block-3 state")
        self.original_bank = (block3, block3.clone(), pass_number)
        patched = block3.clone()
        for arm in ARMS:
            lane = self.lanes[pass_number, arm]
            apply_native_patch(patched[lane:lane + 1], arm, self.sham)
        self.readers[pass_number] = {
            "bank2": as_numpy(block2[:, -1, 0]),
            "original": as_numpy(block3[:, -1]),
            "bank3": as_numpy(patched[:, -1]),
            "query": as_numpy(module.query[local_index]),
        }
        return block2, patched, local_index

    def after_reader(self, module, args, output):
        pass_number = self.calls.get(self.visits - 1)
        if pass_number is None:
            return None
        if tuple(output.shape) != (*args[1].shape[:2], 4, WIDTH):
            raise RuntimeError("unexpected native Q/K/V/MoE reader output shape")
        result = self.readers[pass_number]
        result["message"] = as_numpy(output[:, -1])
        output = output.clone()
        output[self.lanes[pass_number, "reader_rescue"], -1] = output[0, -1]
        result["delivered"] = as_numpy(output[:, -1])
        return output

    def verify(self):
        if self.visits != 2 * (CORE_END - CORE_START) or len(self.writers) != 3 or self.original_bank is not None:
            raise RuntimeError(f"incomplete fixed-loop trace: {self.visits} readers, {len(self.writers)} writers")
        for pass_number, reader in self.readers.items():
            if not np.array_equal(reader["original"][0], self.writers[pass_number - 2]):
                raise RuntimeError(f"writer does not match previous-pass block-3 bank for pass {pass_number}")
            lanes = [0, *(lane for (reader_pass, _), lane in self.lanes.items() if reader_pass == pass_number)]
            for key in ("bank2", "original"):
                if not np.array_equal(reader[key][lanes], np.broadcast_to(reader[key][0], (len(lanes), WIDTH))):
                    raise RuntimeError(f"intervention lanes diverged before pass {pass_number} {key}")
