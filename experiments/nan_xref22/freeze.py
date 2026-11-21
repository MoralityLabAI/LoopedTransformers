"""One-time future-cache binding and analysis lock before CTX21 inference."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import subprocess
from .common import ROOT,RUNS,read,sha,write
from .derive import transform,WORKER_RULES,AUDIT_RULES,check_coverage_source


def main():
    assert not RUNS.exists();cfg=read(ROOT/'config.json');old=read('experiments/nan_xref19/SOURCE_MANIFEST.json')
    assert sha('experiments/nan_ctx21/SOURCE_MANIFEST.json')==cfg['CTX21_seal']
    assert sha('experiments/nan_fit16_io18/SOURCE_MANIFEST.json')==cfg['FIT16_seal']
    journal=Path('runs/nan_ctx21/queue_events.jsonl');raw=journal.read_bytes() if journal.exists() else b''
    complete=raw[:raw.rfind(b'\n')+1]
    import json
    events=[json.loads(v) for v in complete.splitlines()]
    assert not any(v['event']=='worker_started' for v in events),'Freeze prospective analysis before any CTX21 model outcome'
    assert not any((Path('runs/nan_ctx21')/b/'manifest.json').exists() for b in cfg['backends'])
    write(ROOT/'PRE_FREEZE_EXPOSURE.json',dict(CTX21_pretrained_inference_started=False,CTX21_complete_queue_event_lines=len(events),
        complete_queue_prefix_bytes=len(complete),complete_queue_prefix_sha256=hashlib.sha256(complete).hexdigest(),
        prior_COMP14_CTX08_IO18_aggregates_exposed=True,new_cache_vectors_read=0,sealed_test_accessed=False))
    for name,rules in [('worker.py',WORKER_RULES),('audit.py',AUDIT_RULES)]:
        expected=transform((Path('experiments/nan_xref19')/name).read_text(encoding='utf-8'),rules)
        if name=='audit.py':
            expected=expected.replace('\n\nif __name__',check_coverage_source()+'\n\nif __name__')
        assert (ROOT/name).read_text(encoding='utf-8')==expected
    dependencies=set(old['dependencies'])|{'experiments/nan_xref19/'+n for n in ('SOURCE_MANIFEST.json','worker.py','audit.py','cache.py','common.py','CACHE_INVENTORY.json')}
    dependencies|={'experiments/nan_ctx21/'+n for n in ('SOURCE_MANIFEST.json','config.json','fixtures/public.jsonl','fixtures/tokenizer_receipt.json')}
    bindings={}
    for name in sorted(dependencies):
        raw=Path(name).read_bytes();blob=subprocess.check_output(['git','show','HEAD:'+name])
        assert raw.replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'),name
        bindings[name]=dict(runtime_sha256=sha(name),git_sha256=hashlib.sha256(blob).hexdigest())
    critical={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py','.json','.md') and p.name!='SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes']=sha(ROOT/'.gitattributes')
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment=cfg['experiment'],frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical,dependencies=bindings,SAE_tower_manifests=old['SAE_tower_manifests'],
        exposure='Prior COMP14/CTX08/IO18 aggregates known;CTX21 pretrained inference and XREF22 numerical cache analysis have not started.',
        prefreeze_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        byte_policy='Exact observed runtime bytes;CI exact registered Git bytes;freezer permits only whole-file LF/CRLF differences between views.',
        new_model_calls=0,new_grades=0,optimizer_steps=0,formal_inferential_comparisons=0,sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),critical_files=len(critical),dependencies=len(bindings)))


if __name__=='__main__':
    main()
