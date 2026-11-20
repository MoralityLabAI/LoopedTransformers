"""Seal the separate Prime condition before loading its pretrained decoder."""
from pathlib import Path
from experiments.nan_gym05.common import read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('A registered Prime condition cannot be silently resealed')
    previous=read('experiments/nan_gym05/SOURCE_MANIFEST.json');dependencies=dict(previous['model_dependencies_sha256'])
    for name in ('experiments/nan_gym05/common.py','experiments/nan_gym05/observe.py','experiments/nan_gym05/parser.py',
                 'experiments/nan_gym05/grade.py','mechinterp/nanbeige_bpd_gym/suite_rows.py',
                 'mechinterp/nanbeige_bpd_gym/prime_suite.py','nanbeige_bpd_suite.py'):
        dependencies[name]=sha(name)
    files=['__init__.py','config.json','wire.py','server.py','freeze.py','queue.py','run_cli.sh','PROTOCOL.md','.gitattributes','tests/test_transport.py','receipts/mock_transport.json']
    write(target,{'experiment':'NAN-PRIME-06','evidence_class':'fresh local SDK model rollouts on already exposed development sources',
        'files_sha256':{name:sha(root/name) for name in files},'dependencies_sha256':dependencies,
        'native_source_manifest_sha256':sha('experiments/nan_gym05/SOURCE_MANIFEST.json'),
        'prompt_cohort_sha256':previous['combined_public_sha256'],'host_grading_sha256':previous['combined_host_sha256'],
        'scope':'162 registered DEV contexts per backend; no CAL or sealed test inference',
        'pretrained_Prime_condition_inference_started':False,'formal_inferential_comparisons':0,'BPD_transfer_enabled':False})
    print({'manifest_sha256':sha(target),'files':len(files),'dependencies':len(dependencies)})


if __name__=='__main__':main()
