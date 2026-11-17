"""Build an immutable, self-contained ROC-02C public adapter/receipt archive."""
from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parent
src=root/'results_corrected';out=root/'public_corrected';out.mkdir(exist_ok=True)
files=[]
for f in sorted(src.glob('roc02c_R*.json')):files.append(('receipts/'+f.name,f))
for f in sorted((src/'adapters').glob('roc02c_adapter_R*.pt')):files.append(('adapters/'+f.name,f))
for fn in ('roc02c_equal_budget_rank8.py','roc02c_summarize.py','test_roc02c_public_receipts.py','test_roc02c_equal_budget_rank8.py','ROC02C_EQUAL_BUDGET_CALIBRATION.md','README_REPLAY.md'):
 files.append(('source/'+fn,root/fn))
name=out/'roc02c_receipts_adapters.zip'
with zipfile.ZipFile(name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for arc,srcpath in files:
  zi=zipfile.ZipInfo(arc,date_time=(1980,1,1,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16
  z.writestr(zi,srcpath.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=8)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={
 'study_id':'ROC-02C-rank8-equal-budget-null-normalized-calibration-v1',
 'registered_commit':'006cdce9de6eaf529fc03fd37b71b448cbe4b063',
 'source_roc02b_branch':'research/roc02b-calibration-observer-card-20261116',
 'source_model_depths':[8,16],
 'model_seeds':[2026111601,2026111603,2026111607,2026111609,2026111613,2026111619],
 'source_train_sha256':sha(Path('/mnt/data/roc02_prospective/data/train_inputs.jsonl')),
 'source_calibration_sha256':sha(Path('/mnt/data/roc02_prospective/data/calibration_inputs.jsonl')),
 'sealed_test_opened':False,
 'archive':{'name':name.name,'sha256':sha(name),'size_bytes':name.stat().st_size,
            'entries':len(files),'n_receipts':12,'n_adapters':12},
 'public_data_sha256':{f.name:sha(f) for f in [*out.glob('roc02c_*.csv'),out/'roc02c_summary.json']},
}
(out/'roc02c_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('zip',name.stat().st_size,'hash',manifest['archive']['sha256'],'entries',len(files))
