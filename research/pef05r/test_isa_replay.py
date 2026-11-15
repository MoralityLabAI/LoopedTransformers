"""Static pinned ISA reference inventory and unchanged PEF-05 source guard."""
import re
from research.pef05r.isa_replay import REFERENCE, SOURCE_SCRIPT_GIT_SHA1, check_source_blob
from research.pef05.pef05 import SEEDS


def test_reference_has_all_four_seeds_and_bytewise_distinct_sources():
    assert set(REFERENCE)=={"avx2","avx512"}
    assert set(REFERENCE["avx2"])==set(SEEDS)
    assert set(REFERENCE["avx512"])==set(SEEDS)
    for seed in SEEDS:
        a=REFERENCE["avx2"][seed]
        b=REFERENCE["avx512"][seed]
        assert len(a)==len(b)==3
        assert all(re.fullmatch(r"[0-9a-f]{64}",h) for h in a+b)
        assert a[0]==b[0]
        assert a[1]!=b[1]
        assert a[2]!=b[2]


def test_script_is_exactly_original_registered_git_blob():
    assert check_source_blob()==SOURCE_SCRIPT_GIT_SHA1
