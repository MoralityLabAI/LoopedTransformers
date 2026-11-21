"""Hash-bound NAN-GYM-05 interchange; never accepts an arbitrary test roster."""
from pathlib import Path
from experiments.nan_gym05.common import read,rows,sha
from experiments.nan_gym05.parser import parse
from experiments.nan_gym05.grade import score_value

SOURCE_SEAL='96294903b1be52021725f2b839a7ad79fa635fd75aa4d5752cb3d53af0d132b0'
REGISTRATION='18fb185dcda884473d51fe9c420654418c6a9f49'
SUITES={'pointer','arc1','arc2','sudoku','storyworld','routing'}

def load_suite_rows(repository_root=None,*,scope='public',split='development',suite='all',include_copy_controls=False):
    if split not in ('calibration','development'):raise ValueError('Only registered CAL/DEV splits')
    if suite!='all' and suite not in SUITES:raise ValueError('Unknown native suite')
    repo=Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]
    manifest=repo/'experiments/nan_gym05/SOURCE_MANIFEST.json'
    if sha(manifest)!=SOURCE_SEAL:raise ValueError('Unknown source registration; refuse before opening cohort data')
    seal=read(manifest)
    if scope=='public':
        folder=repo/'experiments/nan_gym05/fixtures';expected=(seal['public_repo_fixture_sha256'],seal['public_repo_gold_sha256'])
    elif scope=='local':
        folder=repo/'runs/nan_gym05/cohort';expected=(seal['combined_public_sha256'],seal['combined_host_sha256'])
    else:raise ValueError('Only registered public or local source scopes')
    pub,host=folder/'public.jsonl',folder/'host_labels.jsonl'
    if (sha(pub),sha(host))!=expected:raise ValueError('Registered task-source hash mismatch')
    labels={r['id']:r for r in rows(host)};out=[]
    for r in rows(pub):
        if r['split']!=split or (suite!='all' and r['suite']!=suite):continue
        label=labels[r['id']]
        if label.get('copy_control') and not include_copy_controls:continue
        out.append({'id':r['id'],'root_id':r['root'],'suite':r['suite'],'prompt':r['prompt'],
            'schema':r['schema'],'private_label':label,'registration':REGISTRATION,'source_manifest_sha256':SOURCE_SEAL,
            'endpoint':'complete_final_reply; separate from native first-complete-prefix generation'})
    return out

def score_suite_reply(text,schema,private_label):
    parsed=parse(text,schema,final=True)
    return score_value(parsed['value'],private_label)|{'format_valid':parsed['status']=='accepted'}
