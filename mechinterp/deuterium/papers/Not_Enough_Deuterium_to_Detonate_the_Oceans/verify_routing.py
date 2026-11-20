"""Verify displayed routing counts; optionally compare them to the pinned full BPD JSON.

Run:
    python verify_routing.py
    python verify_routing.py --source-json /path/to/08_autonomous_routing_study.json

Without --source-json this only validates the committed figure transcription and
headline arithmetic. With the complete receipt it independently checks each root.
"""
import argparse, json
from pathlib import Path

def verify(source=None):
    record=json.loads(Path(__file__).with_name('routing_root_values.json').read_text())
    rows=record['unit_rows']; assert len(rows)==8
    assert [r['root'] for r in rows] == list(range(2605900,2605908))
    n=record['n_cases_per_root']; assert n==64
    for r in rows:
        for k in ('dense','online','fixed','teacher_scheduled'):
            assert isinstance(r[k],int) and 0<=r[k]<=n
    sums={k:sum(r[k] for r in rows) for k in ('dense','online','fixed','teacher_scheduled')}
    assert sums=={'dense':239,'online':140,'fixed':8,'teacher_scheduled':197},sums
    competent=[r for r in rows if r['dense']/n>=0.5]
    other=[r for r in rows if r not in competent]
    assert len(competent)==3 and len(other)==5
    competent_sums={k:sum(r[k] for r in competent) for k in sums}
    assert competent_sums=={'dense':171,'online':113,'fixed':8,'teacher_scheduled':157},competent_sums
    assert {r['root'] for r in competent}=={2605900,2605906,2605907}
    assert sum(r['fixed'] for r in other)==0
    for r in rows:
        if r['root'] in {2605903,2605905}:
            assert all(r[k]==0 for k in sums),r
        if r['root']==2605904:
            assert r['online']==8 and r['teacher_scheduled']==7,r
    assert [sum(r[k] for r in other) for k in ('dense','online','fixed')]==[68,27,0]
    if source:
        raw=json.loads(Path(source).read_text())
        by_id={u['unit_id']:u for u in raw['units']}
        assert len(by_id)==8
        for r in rows:
            u=by_id[f'excluded_pilot_root_{r["root"]}']
            assert u['n_cases']==n
            for k in ('dense','online','fixed','teacher_scheduled'):
                value = (u['dense_teacher'] if k=='dense' else u['arms'][k])['by_depth']['8']['exact_board_accuracy']
                assert abs(value*n-r[k])<1e-9,(r['root'],k,value)
    print('PASS:',sums,'competent:',competent_sums,'all 8 roots;',len(competent),'competent and',len(other),'other roots; full-source comparison:',bool(source))

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-json')
    verify(ap.parse_args().source_json)
