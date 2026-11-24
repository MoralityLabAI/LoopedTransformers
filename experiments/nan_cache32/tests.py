"""Full-scope synthetic schedule, reporting bounds and cold admission contracts."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from .plan import STAGES,build,digest


def source_fixture():
    source=[];bindings={}
    for stage,N in STAGES.items():
        for i in range(N):
            index=len(source);name=stage+f'/MOCK_{i:04d}'
            arm='main' if stage=='nan_story23' and i<24 else 'focused' if stage=='nan_comp14' and i<48 else 'copy'
            r=dict(id=name,source_id=name,root=f'MOCK_ROOT_{index%150}',stage=stage,split='CAL' if index%4==0 else 'DEV',arm=arm,
                schema=dict(type='label',allowed=['A','B'],max_characters=64),prompt=name)
            if stage=='nan_story23':r['schema'].update(checkpoints=[1,2,3],nodes=['E00','P0','R0','R1','R2'])
            source.append(r);bindings[name]=dict(tokens=4,ids_sha256='a'*64)
    return source,bindings


class Contracts(unittest.TestCase):
    def test_full_scope_static_plan_and_source_adjacent_AB(self):
        source,bindings=source_fixture();p=build(source,bindings)
        self.assertEqual(p['trajectory_slots_per_backend'],1981)
        self.assertEqual(sum(j['lane']=='AB' for j in p['jobs']),216)
        self.assertEqual([m['source_index'] for j in p['jobs'] if j['lane']=='baseline' for m in j['members']],list(range(1549)))
        for i,j in enumerate(p['jobs']):
            if j['lane']=='baseline':continue
            prior=next(x for x in reversed(p['jobs'][:i]) if x['lane']=='baseline')
            self.assertIn(j['members'][0]['source_id'],[m['source_id'] for m in prior['members']])
            self.assertEqual([m['mode'] for m in j['members']],['native',('component_only','static_CAL_mean','reached_state')[j['ordinal']//72]])
        self.assertEqual(digest(p),digest(build(copy.deepcopy(source),copy.deepcopy(bindings))))
    def test_scope_missing_binding_and_unadmitted_AB_cannot_be_silently_reduced(self):
        source,bindings=source_fixture()
        with self.assertRaises(AssertionError):build(source[:-1],bindings)
        changed=copy.deepcopy(bindings);changed[source[0]['id']]['tokens']=4608
        with self.assertRaises(AssertionError):build(source,changed)
        del bindings[source[10]['id']]
        with self.assertRaises(KeyError):build(source,bindings)
    def test_actual_worker_scheduler_visits_every_planned_row_once(self):
        from .worker import execute_schedule
        source,bindings=source_fixture();plan=build(source,bindings);calls=[];events=[];completed=[]
        def generate(rows,modes,ordinal,lane):calls.append((lane,ordinal,[r['id'] for r in rows],modes));return [{} for r in rows]
        engine=SimpleNamespace(gym=SimpleNamespace(bytes=0,generate=generate),budget=lambda *a:None)
        execute_schedule(engine,source,plan,completed,events.append)
        self.assertEqual(completed,list(range(len(plan['jobs']))));self.assertEqual(sum(len(c[2]) for c in calls),1981)
        self.assertEqual(len(events),len(plan['jobs']))
        calls.clear();completed.clear()
        def fail(rows,modes,ordinal,lane):
            calls.append((lane,ordinal))
            if len(calls)==4:raise RuntimeError('MOCK_ONLY failure; no retry')
            return [{} for r in rows]
        engine.gym.generate=fail
        with self.assertRaises(RuntimeError):execute_schedule(engine,source,plan,completed,events.append)
        self.assertEqual(len(calls),4);self.assertEqual(completed,[0,1,2])
    def test_all_missing_report_keeps_3962_slots_and_every_story_endpoint(self):
        from .report import collect
        source,bindings=source_fixture();data=collect(build(source,bindings),source,{r['id']:{} for r in source},{})
        self.assertEqual(sum(r['registered'] for r in data['accuracy']),3962)
        self.assertEqual(sum(r['unknown'] for r in data['accuracy']),3962)
        for r in data['accuracy']:
            self.assertEqual(r['world_accuracy_bounds'],[0.,1.]);self.assertEqual(r['root_macro_accuracy_bounds'],[0.,1.])
            if r['stage']=='nan_story23':
                self.assertEqual(set(r['endpoint_counts']),{'format_valid','final_correct','checkpoints_correct','path_correct','joint_correct'})
        self.assertTrue(all(r['root_macro_difference_bounds']==[-1.,1.] for r in data['paired_precision_accuracy']+data['AB_behavior']))
    def test_no_source_freeze_loader_or_entry_fails_before_torch(self):
        code="""from experiments.nan_cache32.loader import load
from experiments.nan_cache32 import common
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import sys,tempfile
with tempfile.TemporaryDirectory() as folder,patch.object(common,'ROOT',Path(folder)):
 try:load(SimpleNamespace(backend='nf4'),{})
 except FileNotFoundError:assert 'torch' not in sys.modules
 else:raise AssertionError('Unfrozen load admitted')
"""
        subprocess.run([sys.executable,'-B','-S','-c',code],check=True)
    def test_entry_has_no_limit_or_subset_switch(self):
        args=[sys.executable,'-B','-S','-m','experiments.nan_cache32.worker','--backend','nf4','--snapshot','unused','--compactification-root','unused',
              '--bpd-root','unused','--artifact-root','unused','--limit','1']
        result=subprocess.run(args,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('unrecognized arguments: --limit 1',result.stderr)
    def test_model_admission_needs_own_backend_permit_and_actual_pid(self):
        from . import common
        def read(path):
            if path.name=='predecessor_lane_closed.json':return dict(source_manifest_sha256='MOCK',all_authenticated_predecessors_ended=True,known_model_host_audit_inventory_clear=True)
            if path.name=='source_publication.json':return dict(source_manifest_sha256='MOCK',remote_commit='MOCK')
            if path.name=='runtime.json':return dict(pid=__import__('os').getpid(),backend='bf16',condition='NAN-CACHE-32',source_manifest_sha256='MOCK')
            raise FileNotFoundError('No BF16 permit before actual NF4 exit')
        with patch.object(common,'read',side_effect=read),patch.object(common,'sha',return_value='MOCK'):
            with self.assertRaises(FileNotFoundError):common.verify_worker('bf16')
    def test_process_GPU_allocation_counter_validation(self):
        from .resources import summarize_gpu,ResourceAdmissionError
        valid=dict(Name='pid_7_luid_MOCK',DedicatedUsage=123,SharedUsage=4,TotalCommitted=127)
        self.assertEqual(summarize_gpu(7,[valid])['shared_bytes'],4)
        for rows in ([],[dict(valid,Name='pid_8_wrong')],[dict(valid,SharedUsage=-1)],[dict(valid,SharedUsage=True)]):
            with self.assertRaises(ResourceAdmissionError):summarize_gpu(7,rows)
    def test_preflight_uses_actual_calibration_split_spelling(self):
        from .preflight import run
        class ReachedCalibration(Exception):pass
        def ids(row):raise ReachedCalibration()
        engine=SimpleNamespace(t=None,ids=ids)
        with self.assertRaises(ReachedCalibration):run(engine,dict(split='calibration'))
        for split in ('development','DEV','CAL'):
            with self.assertRaises(AssertionError):run(engine,dict(split=split))


if __name__=='__main__':unittest.main()
