"""Fresh graph roots; standalone host materializer, never imported by worker."""
from pathlib import Path
from experiments.nan_gym05.sources import pointer_root,NODES
from .io import read,rows,sha,write,write_rows


def materialize(cfg):
    public=[];host=[]
    for split,first in (('calibration',cfg['calibration_first_seed']),('development',cfg['development_first_seed'])):
        for index in range(cfg['roots_per_split']):
            seed=first+index;_,prompt,answer=pointer_root(seed,[1])[0];root=f'fit07:{split}:{seed}'
            for copy in (False,True):
                key=root+(':copy' if copy else ':main')
                text='Copy the following verified answer exactly, with no explanation:\n'+answer if copy else prompt
                public.append({'id':key,'root':root,'split':split,'suite':'pointer','prompt':text,
                    'schema':{'type':'label','allowed':list(NODES),'max_characters':64},'copy_control':copy})
                host.append({'id':key,'root':root,'split':split,'answer':answer,'copy_control':copy,'hops':1})
    return public,host


def main():
    root=Path(__file__).parent;folder=root/'fixtures'
    if folder.exists():raise FileExistsError('Fresh source cohort must not be silently regenerated')
    cfg=read(root/'config.json');public,host=materialize(cfg)
    old={r['prompt'] for r in rows('experiments/nan_gym05/fixtures/public.jsonl') if not r['id'].endswith(':copy')}
    main=[r['prompt'] for r in public if not r['copy_control']]
    if len(main)!=len(set(main)) or old.intersection(main):raise ValueError('Fresh main prompts overlap older public contexts')
    write_rows(folder/'public.jsonl',public);write_rows(folder/'host_labels.jsonl',host)
    write(folder/'source_receipt.json',{'source_only':True,'pretrained_inference_used_for_selection':False,'source_roots':16,
        'calibration_main':8,'development_main':8,'explicit_copy_controls':16,'contexts':32,
        'pointer_generator_sha256':sha('experiments/nan_gym05/sources.py'),'old_public_fixture_sha256':sha('experiments/nan_gym05/fixtures/public.jsonl'),
        'main_prompt_overlap_with_old_public_cohort':0,'sealed_test_accessed_or_used_as_exclusion':False,
        'oracle_scope':'exact graph walk is a structural ceiling, not a neural result'})
    print({'contexts':len(public),'fresh_main_roots':len(main),'old_main_prompt_overlap':0})


if __name__=='__main__':main()
