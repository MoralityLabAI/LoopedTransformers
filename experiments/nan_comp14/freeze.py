"""One-time source/tokenizer/parser/criteria lock before pretrained inference."""
from datetime import datetime, timezone
import re
import subprocess
from .common import ROOT, RUNS, read, rows, sha, write_new


def main():
    assert not RUNS.exists(), 'Freeze must precede the registered model run'
    cfg = read(ROOT / 'config.json'); source = rows(ROOT / 'fixtures/public.jsonl')
    assert len(source) == 240 and len({r['root'] for r in source}) == 24
    assert read(ROOT / 'fixtures/tokenizer_receipt.json')['maximum_prompt_tokens'] <= cfg['maximum_prompt_tokens']
    def graph(text):
        return tuple(sorted(dict(re.findall(r'([A-P])->([A-P])', text.split('\n\n')[-1])).items()))
    new = {graph(r['prompt']) for r in source if r['arm'] in cfg['templates']}
    assert len(new) == 48 and all(len(v) == 16 for v in new)
    exclusions = read(ROOT / 'fixtures/source_receipt.json')['exposed_exclusion_files_sha256']
    old = set()
    for file, expected in exclusions.items():
        assert sha(file) == expected
        old.update(v for r in rows(file) if len(v := graph(r['prompt'])) == 16)
    assert not new.intersection(old)
    audit = ROOT / 'fixtures/independent_source_audit.json'
    write_new(audit, dict(status='passed', fresh_query_graphs=48, named_exposed_query_graphs=len(old),
        query_graph_overlap=0, oracle='Independent textual edge extraction, bijection and graph-overlap checks; no model outcomes.',
        sealed_test_accessed_or_used_as_exclusion=False))
    deps = read('experiments/nan_ctx08/SOURCE_MANIFEST.json')['model_dependencies_sha256']
    names = set(deps) | {'experiments/nan_recomp09/common.py', 'experiments/nan_ranksite13/resources.py'}
    host = ['fixtures/host_labels.jsonl']
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.md', '.json', '.jsonl') and p.name != 'SOURCE_MANIFEST.json'
        and '__pycache__' not in p.parts and str(p.relative_to(ROOT)).replace('\\', '/') not in host}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    seal = dict(experiment=cfg['experiment'], frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, host_files_sha256={name: sha(ROOT / name) for name in host},
        dependencies_sha256={name: sha(name) for name in sorted(names)}, checkpoint_revision=cfg['checkpoint_revision'],
        compactification_adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456',
        prefreeze_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        exposure='Prior NF4 FIT07 DEV5/8, native GYM05 and context outcomes visible. COMP14 source-only design and all two-template outcomes frozen before any COMP14 pretrained inference.',
        pretrained_inference_started=False, neural_retries=0, formal_inferential_comparisons=0,
        changes_old_gates=False, sealed_test_accessed=False)
    write_new(ROOT / 'SOURCE_MANIFEST.json', seal)
    print({'source_manifest_sha256': sha(ROOT / 'SOURCE_MANIFEST.json'), 'critical_files': len(critical), 'dependencies': len(names)})


if __name__ == '__main__':
    main()
