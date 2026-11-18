"""Recompute exact LRM-REV-03 finite theorem and population-risk receipt.

This script performs deterministic algebra, no model training, no sampling and
no statistical test. Its --check mode compares a committed receipt bytewise.
"""
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

from lrm_rev03_exact import count_fiber_bound, exact_population_regret

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if HERE.name=='revision_memory_gym' else HERE
SOURCE_FILES=('lrm_rev03_exact.py','test_lrm_rev03_exact.py',
              'benchmark_lrm_rev03.py','PROTOCOL_LRM_REV03.md')


def compact_decimal(q:F)->str:
    with localcontext() as ctx:
        ctx.prec=64
        return format(Decimal(q.numerator)/Decimal(q.denominator),'.15f')


def fraction_digest(q:F)->str:
    return hashlib.sha256(f'{q.numerator}/{q.denominator}'.encode('ascii')).hexdigest()


def receipt():
    bound=count_fiber_bound()
    curve=[]
    for r in range(7):
        candidate=exact_population_regret(r,'corrected')
        random=exact_population_regret(r,'random')
        curve.append({'r':r,'corrected_count_oracle_excess':compact_decimal(candidate),
                      'random_count_oracle_excess':compact_decimal(random),
                      'random_minus_corrected':compact_decimal(random-candidate),
                      'corrected_fraction_sha256':fraction_digest(candidate),
                      'random_fraction_sha256':fraction_digest(random)})
    return {'schema_version':1,'experiment_id':'LRM-REV-03',
            'evidence_class':'exact_finite_theorem_and_population_calculation',
            'confirmation_population_allocated':False,'model_training_performed':False,
            'scientific_mechanism_claim':False,
            'contract':{'n':12,'j':6,'prior':['9/20','7/20','1/5'],
                        'match_probability':['19/20','13/20','3/20'],
                        'policy':['all_corrected_old_values','independent_uniform_six_slot_set'],
                        'loss':'squared_L2_to_full_named_history_Bayes'},
            'continuation_partition':bound,'exact_population_curve':curve,
            'source_sha256':{p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in SOURCE_FILES}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'results'/'lrm_rev03_exact_v1.json')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    data=json.dumps(receipt(),sort_keys=True,indent=2)+'\n'
    if args.check:
        if args.out.read_text(encoding='utf-8')!=data:
            raise SystemExit('LRM-REV-03 receipt differs from exact source replay')
        print('PASS: exact finite proof, risk curve and frozen source hashes')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data,encoding='utf-8')
        print('WROTE',args.out)


if __name__=='__main__':
    main()
