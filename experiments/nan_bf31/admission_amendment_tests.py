"""Actual worker entry/admission with tiny metadata fixtures; no Torch import."""
import ast
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from .common import ROOT
from .amendment import assert_worker_equivalence


def candidate():return Path(os.environ.get('BF31_RUNTIME_ADMISSION_CANDIDATE',str(ROOT/'worker.py')))


def probe_entry(path,scenario):
    from unittest.mock import patch
    from . import admission
    from experiments.nan_ram28.common import write,read,sha
    import experiments.nan_ranksite13.resources as resources
    name='experiments.nan_bf31.worker_admission_probe'
    spec=importlib.util.spec_from_loader(name,importlib.machinery.SourceFileLoader(name,str(path)))
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    class AdmissionReached(Exception):pass
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);runs=root/'runtime';runs.mkdir();write(root/'SOURCE_MANIFEST.json',dict(MOCK_ONLY=True))
        write(root/'config.json',read(ROOT/'config.json'));seal=sha(root/'SOURCE_MANIFEST.json')
        if scenario!='missing_predecessor':
            write(runs/'predecessor_lane_closed.json',dict(source_manifest_sha256=seal,all_authenticated_predecessors_ended=True,known_model_host_audit_inventory_clear=True))
        write(runs/'source_publication.json',dict(source_manifest_sha256='WRONG' if scenario=='wrong_publication' else seal,remote_commit='MOCK_ONLY'))
        actual=admission.verify_worker;actual_read=admission.read
        def checked_read(path):
            value=actual_read(path)
            if scenario=='wrong_pid' and Path(path).name=='bf16_runtime.json':value['pid']=-1
            return value
        def gate():actual();raise AdmissionReached()
        args=['worker','--backend','bf16','--snapshot','unused','--compactification-root','unused','--bpd-root','unused','--artifact-root',str(root/'artifacts')]
        with (patch.object(worker,'ROOT',root),patch.object(worker,'RUNS',runs),patch.object(worker,'verify',return_value={'MOCK_ONLY':True}),
             patch.object(admission,'ROOT',root),patch.object(admission,'RUNS',runs),patch.object(admission,'read',checked_read),
             patch.object(admission,'verify_worker',gate),patch.object(resources,'below_normal'),patch.object(sys,'argv',args)):
            try:worker.main()
            except AdmissionReached:assert scenario=='passed'
            except (FileNotFoundError,AssertionError):assert scenario!='passed'
            else:raise AssertionError('Worker bypassed admission')
        runtime=read(runs/'bf16_runtime.json')
        assert runtime['backend']=='bf16' and runtime['pid']==os.getpid() and runtime['condition']=='NAN-BF-31'
        assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)


class RuntimeAmendmentTests(unittest.TestCase):
    def test_exact_metadata_only_worker_change(self):
        result=assert_worker_equivalence(candidate(),ROOT/'worker_before_runtime_admission.py.txt')
        self.assertTrue(result['all_other_worker_AST_exact'])
    def test_actual_entry_records_bf16_and_passes_existing_admission_before_Torch(self):
        code='from experiments.nan_bf31.admission_amendment_tests import probe_entry; probe_entry('+repr(str(candidate()))+',"passed")'
        subprocess.run([sys.executable,'-B','-S','-c',code],check=True)
    def test_actual_entry_rejects_missing_lane_wrong_pid_and_publication_before_Torch(self):
        for scenario in ('missing_predecessor','wrong_pid','wrong_publication'):
            code='from experiments.nan_bf31.admission_amendment_tests import probe_entry; probe_entry('+repr(str(candidate()))+','+repr(scenario)+')'
            subprocess.run([sys.executable,'-B','-S','-c',code],check=True)
    def test_original_runtime_identity_would_disagree_with_existing_gate(self):
        source=(ROOT/'worker_before_runtime_admission.py.txt').read_text()
        self.assertIn("backend='nf4'",source);self.assertIn("runtime['backend']=='bf16'",(ROOT/'admission.py').read_text())
    def test_nonmetadata_worker_change_is_rejected(self):
        source=candidate().read_text();changed=source.replace('baseline_registered=1549','baseline_registered=1548')
        self.assertNotEqual(changed,source)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.py';path.write_text(changed)
            with self.assertRaises(AssertionError):assert_worker_equivalence(path,ROOT/'worker_before_runtime_admission.py.txt')


if __name__=='__main__':unittest.main()
