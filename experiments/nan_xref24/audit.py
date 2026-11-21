"""Run the repaired independent auditor against original closed host archives."""
import argparse
import importlib
import os
from pathlib import Path
from .common import ROOT,RUNS,SOURCES,read,sha,verify,write
from .repair import repaired_tree


def run(source,backend):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();seal=verify();module=SOURCES[source];upstream=importlib.import_module(module+'.common')
    receipt=upstream.RUNS/backend/'receipt.json';terminal=read(receipt)
    assert terminal['status']=='completed' and terminal['source_manifest_sha256']==sha(upstream.ROOT/'SOURCE_MANIFEST.json')
    if source=='xref19':assert sha(receipt)==seal['closed_xref19_inputs'][backend]['receipt_sha256']
    sourcefile=upstream.ROOT/'audit.py';tree,changed=repaired_tree(sourcefile.read_text(encoding='utf-8'))
    namespace={'__name__':module+'.audit_xref24_runtime','__package__':module,'__file__':str(sourcefile.resolve())}
    exec(compile(tree,str(sourcefile)+':three_decoder_transposes','exec'),namespace)
    original_output=upstream.RUNS/'audits'/(backend+'.json');proofs=[]
    def redirected(path,value):
        assert Path(path)==original_output and not proofs,'Only the final audit receipt may be redirected'
        assert value['status']=='passed' and value['source_manifest_sha256']==sha(upstream.ROOT/'SOURCE_MANIFEST.json')
        proofs.append(value)
    namespace['write']=redirected
    try:
        namespace['run'](backend);assert len(proofs)==1
        result=dict(status='passed',source=source,backend=backend,original_arithmetic_audit=proofs[0],
            upstream_receipt_sha256=sha(receipt),upstream_auditor_sha256=sha(sourcefile),three_transposed_expression_lines=changed,
            source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),original_failed_attempt_unchanged=True,
            new_model_calls=0,new_optimizer_steps=0,new_grades=0,sealed_test_accessed=False)
    except Exception as error:
        result=dict(status='failed_repaired_host_audit',source=source,backend=backend,error_type=type(error).__name__,error_message=str(error),
            upstream_receipt_sha256=sha(receipt),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            original_failed_attempt_unchanged=True,new_model_calls=0,sealed_test_accessed=False)
        write(RUNS/'audits'/(source+'_'+backend+'.json'),result);raise
    write(RUNS/'audits'/(source+'_'+backend+'.json'),result)
    print(dict(source=source,backend=backend,status=result['status'],new_model_calls=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=tuple(SOURCES),required=True);p.add_argument('--backend',choices=('nf4','bf16'),required=True);a=p.parse_args()
    run(a.source,a.backend)
