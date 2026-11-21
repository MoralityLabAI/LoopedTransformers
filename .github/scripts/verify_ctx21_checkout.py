"""Check only exact pre-registered runtime/Git byte views; no runtime relaxation."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('experiments/nan_ctx21'); seal=json.loads((root/'SOURCE_MANIFEST.json').read_bytes()); checked=0
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for field in ('critical_files_sha256','host_files_sha256'):
        for name,h in seal[field].items():
            assert digest(root/name)==h,str(root/name); checked+=1
    for name,h in seal['validation_files_sha256'].items():
        assert digest(Path(name))==h,name; checked+=1
    conversions=[]
    for name,h in seal['dependencies_sha256'].items():
        actual=digest(Path(name)); assert actual in (h,seal['dependencies_git_blob_sha256'][name]),name
        if actual!=h:
            conversions.append(name)
        checked+=1
    assert not seal['pretrained_inference_started'] and not seal['sealed_test_accessed']
    assert seal['neural_retries']==seal['formal_inferential_comparisons']==0
    print(dict(registered_files_checked=checked,exact_registered_Git_view_paths=conversions,runtime_byte_check_unchanged=True))


if __name__=='__main__':
    main()
