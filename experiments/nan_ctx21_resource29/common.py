"""Streaming verification and a self-only CPU memory watchdog; no Torch."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_ctx21_resource29')
OLD = Path('experiments/nan_ctx21')
OLD_RUNS = Path('runs/nan_ctx21')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def no_torch():
    assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)


def verify():
    seal = read(ROOT/'SOURCE_MANIFEST.json')
    for name, digest in seal['critical_files_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in seal['additional_dependencies_sha256'].items():
        assert sha(name) == digest, name
    assert sha(OLD/'SOURCE_MANIFEST.json') == seal['original_source_manifest_sha256']
    original = read(OLD/'SOURCE_MANIFEST.json')
    for key in ('critical_files_sha256', 'dependencies_sha256', 'host_files_sha256'):
        for name, digest in original[key].items():
            path = Path(name) if key == 'dependencies_sha256' else OLD/name
            assert sha(path) == digest, str(path)
    for name, digest in read(ROOT/'registration.json')['administrative_artifacts_sha256'].items():
        assert sha(name) == digest, name
    return seal


def select_prefix(journal, queue, outcome, receipt_exists):
    """All and only native case_closed ordinals; a failed gate stays failed."""
    assert not receipt_exists, 'A new normal receipt requires a separate review'
    assert outcome['BF16_worker_ended'] is True
    assert outcome['original_terminal_audit'] == 'failed_missing_normal_receipt'
    assert outcome['original_receipt_synthesized'] is False
    count = outcome['closed_native_cases_preserved']
    assert count == 378
    terminal = [v for v in queue if v.get('backend') == 'bf16' and v['event'] == 'worker_terminal']
    audit = [v for v in queue if v.get('backend') == 'bf16' and v['event'] == 'audit_terminal']
    assert terminal and terminal[-1]['exit_code'] == outcome['original_worker_exit_code'] == 4294967295
    assert audit and audit[-1]['exit_code'] != 0
    assert not any(v['event'] == 'worker_receipt_closed' for v in journal)
    closed = [v for v in journal if v['event'] == 'case_closed']
    assert [v['ordinal'] for v in closed] == list(range(count))
    assert [v['completed_contexts'] for v in closed] == list(range(1, count+1))
    assert sum(v['context_budget'] == 4096 for v in closed) == outcome['closed_4K'] == 312
    assert sum(v['context_budget'] == 8192 for v in closed) == outcome['closed_8K'] == 66
    return count


class Guard:
    def __enter__(self):
        for name in THREADS:
            os.environ[name] = '1'
        os.environ.update(TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1',
                          USE_TORCH='0', USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
        os.environ.pop('TRANSFORMERS_CACHE', None)
        from experiments.nan_ranksite13.resources import below_normal
        from experiments.nan_ram28.resources import memory
        below_normal(); no_torch()
        self.memory = memory; self.cfg = read(ROOT/'registration.json')
        self.stop = threading.Event(); self.samples = []; self.stage = 'admission'
        sample = self.check()
        assert sample['available_physical_bytes'] >= self.cfg['minimum_start_available_RAM_bytes']
        self.thread = threading.Thread(target=self.watch, daemon=True); self.thread.start()
        return self

    def check(self):
        no_torch(); sample = self.memory()
        sample.update(UTC=datetime.now(timezone.utc).isoformat(), stage=self.stage)
        assert sample['private_bytes'] <= self.cfg['maximum_private_bytes'], sample
        assert sample['available_physical_bytes'] >= self.cfg['minimum_available_RAM_bytes'], sample
        self.samples.append(sample)
        return sample

    def watch(self):
        while not self.stop.wait(2):
            try:
                self.check()
            except BaseException as error:
                try:
                    write(RUNS/'resource_failure.json', dict(status='self_only_resource_stop', error=str(error), samples=self.samples))
                finally:
                    os._exit(3)

    def __exit__(self, kind, error, traceback):
        self.stop.set(); self.thread.join(); self.check()
        write(RUNS/'resources.json', dict(status='passed' if kind is None else 'failed', samples=self.samples,
            CPU_threads=1, priority='BelowNormal', pretrained_forwards=0, Torch_imported=False,
            model_loaded=False, sealed_test_accessed=False))
