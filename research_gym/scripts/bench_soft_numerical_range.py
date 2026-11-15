"""Emit NRF-02D bounded-depth exact-rational soft-route receipt."""
import json
from fractions import Fraction as Q
from research_gym.core.numerical_range_certificate import sqdist
from research_gym.core.soft_numerical_range import SoftJordanMixer

def main():
    m=SoftJordanMixer()
    a,b=(Q(1),Q(-1)),(Q(-1),Q(1))
    n=5
    x=m.iterate(a,n);y=m.iterate(b,n)
    print(json.dumps({
        "experiment":"NRF-02D",
        "type":"exact-rational smooth routing",
        "alpha_fixed":str(m.fixed_probability_rate),
        "alpha_soft":str(m.soft_rate),
        "expanding_branch_rate":str(m.expansive.gain),
        "repair_branch_rate":str(m.repair.gain),
        "soft_probability_range":["3/16","5/16"],
        "total_variation_lipschitz":"1/8",
        "box_diameter_upper":"3",
        "steps":n,
        "final_pair_distance_sq":str(sqdist(x,y)),
        "certified_pair_distance_sq":str(m.distance_bound_sq(a,b,n)),
        "posterior_fixed_point_radius_sq":str(m.fixed_point_tail_bound_sq(a,n)),
        "interpretation":"nonlinear mixture map; no claim for randomly sampled routes",
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
