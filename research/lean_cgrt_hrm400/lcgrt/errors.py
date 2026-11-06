"""Classified failures. Integration boundaries raise these, never bare exceptions."""
from __future__ import annotations

CODES = {
    "lean_unavailable", "lean_timeout", "lean_crashed", "lean_request_error",
    "generation_error", "extraction_mismatch", "certificate_rejected", "split_leak",
    "checkpoint_corrupt", "config_mismatch", "hardware_pause", "nonfinite_loss",
}


class GymError(RuntimeError):
    def __init__(self, code: str, message: str):
        assert code in CODES, code
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    def __reduce__(self):
        # Picklable, so a worker process can hand it back; an unpicklable exception makes
        # multiprocessing's result handler die and the pool hang instead of reporting.
        return (type(self), (self.code, self.message))


class LeanError(GymError):
    pass
