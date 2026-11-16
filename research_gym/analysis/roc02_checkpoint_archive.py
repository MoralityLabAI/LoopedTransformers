"""Pure-stdlib verification/unpacking of ROC-02's twelve training checkpoints."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(archive_dir: Path, index_path: Path, summary_path: Path,
           destination: Path | None = None) -> dict:
    index = json.loads(index_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (index['study'] != 'ROC-02-delayed-rule-source-implementation-v1' or
            summary['source_lock'] != index['source_lock_commit']):
        raise ValueError('source registration and model lock differ')
    if index['training_depths'] != [8,16] or len(index['archives']) != 6:
        raise ValueError('wrong model depths or package count')
    if len(summary['rows']) != 12:
        raise ValueError('missing seed/depth model receipts')
    digest_by_key={(int(r['depth']),int(r['seed'])):r['checkpoint_sha256'] for r in summary['rows']}
    if len(digest_by_key)!=12:
        raise ValueError('duplicate model receipts')
    seen=set()
    for a in index['archives']:
        p=archive_dir/a['name']
        if sha(p)!=a['sha256']:
            raise RuntimeError('archive SHA mismatch: '+p.name)
        with ZipFile(p) as z:
            if z.testzip() is not None or len(z.namelist())!=a['members']:
                raise RuntimeError('ZIP corrupt or wrong member count')
            for info in z.infolist():
                name=info.filename
                pure=PurePosixPath(name)
                if (info.is_dir() or pure.is_absolute() or '..' in pure.parts or
                    name in seen or '\\' in name):
                    raise ValueError('unsafe/duplicate source package path')
                seen.add(name)
                raw=z.read(info)
                if destination is not None:
                    target=destination.joinpath(*pure.parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():raise FileExistsError(target)
                    target.write_bytes(raw)
    if len(seen)!=24:
        raise AssertionError('twelve checkpoints and twelve receipts required')
    if destination is not None:
        for r in summary['rows']:
            d,s=int(r['depth']),int(r['seed'])
            ck=destination/f'checkpoints/roc02_R{d}_S{s}_step240.pt'
            receipt=json.loads((destination/f'receipts/roc02_R{d}_S{s}.json').read_text())
            if sha(ck)!=digest_by_key[(d,s)] or receipt['checkpoint_sha256']!=digest_by_key[(d,s)]:
                raise RuntimeError('original checkpoint receipt hash mismatch')
            if receipt['steps']!=240 or receipt['seed']!=s or receipt['depth']!=d:
                raise AssertionError('model ID/optimizer budget changed')
            if receipt['calibration']!=r['calibration'] or receipt['train']!=r['train']:
                raise AssertionError('train/calibration score receipt moved')
    return {'verified_archives':6,'verified_members':24,'model_receipts':12,
            'unique_models':len(digest_by_key),'sealed_test_evaluated':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.archive_dir,a.index,a.summary,a.destination),indent=2))

if __name__=='__main__':main()
