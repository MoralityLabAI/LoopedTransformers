"""Record an isolated persistent Linux SDK and completed gold/noop evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/prime_runtime_persistent');p.add_argument('--out',default='reports/nanbeige_bpd_gym/prime_persistent_runtime.json');a=p.parse_args()
    folder=Path(a.runs);discovery=json.loads((folder/'discovery.json').read_text())
    assert sys.platform=='linux' and '/.venvs/nanbeige-prime-ed70053/' in sys.executable and not sys.executable.startswith('/tmp/')
    distribution=metadata.distribution('verifiers');direct=json.loads(distribution.read_text('direct_url.json'))
    commit='ed70053b6a7a4079d6976e03bdcd7d420356d594';assert direct['vcs_info']['commit_id']==commit
    assert discovery['status']=='passed' and discovery['sdk_commit']==commit and discovery['official_runtime_import_and_plugin_discovery']=='passed'
    assert discovery['development_main_rows']==89 and discovery['paid_API_calls']==0 and not discovery['sealed_test_accessed']
    validation=folder/'validation/nanbeige-persistent-engineering/summary.json';raw=json.loads(validation.read_text())
    assert raw['total']==raw['terminal']==raw['recorded']==raw['outcomes']['valid']==2 and raw['owed']==0
    for mode in ('gold','noop'):
        assert raw['checks'][mode]['valid']==2 and sum(raw['checks'][mode].values())==2
    lock='\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in metadata.distributions()))+'\n'
    (folder/'packages.lock.txt').write_text(lock,encoding='utf-8')
    # Preserve the official CLI summary verbatim; consumers can inspect its gold/noop coverage.
    result=dict(status='passed',updated_UTC=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),sdk_version=distribution.version,sdk_commit=commit,
        persistent_executable='~/.venvs/nanbeige-prime-ed70053/bin/vf-run',persistent_Python='~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12',
        package_lock_sha256=hashlib.sha256(lock.encode()).hexdigest(),official_discovery=discovery,official_validation=raw,
        official_validation_file_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
        local_validation_scope='two software gold/noop episodes; not model evaluation; no replay/retry of old PRIME06 tail',
        CUDA_environment_modified=False,pretrained_weights_downloaded=False,model_rollouts=0,paid_API_calls=0,sealed_test_accessed=False)
    target=Path(a.out);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print({'status':'passed','persistent_runtime':True,'sdk_commit':commit,'model_rollouts':0})


if __name__=='__main__':main()
