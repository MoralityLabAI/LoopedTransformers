"""Generate an exact-rational NRF-02 graph and routing receipt."""
import json
from fractions import Fraction as Q
from research_gym.core.graph_numerical_range import (
    example_graph,example_policy,guarded_trace,potential,certify_path,
    choose,replay_perturbed,
)
from research_gym.core.numerical_range_certificate import sqdist

def run()->dict:
    graph=example_graph()
    policies=example_policy()
    alpha=Q(3,4)
    weights=potential(graph,alpha)
    start=(Q(1,2),Q(1,2))
    delta=Q(1,20000)
    trace=guarded_trace(graph,policies,"idle",start,Q(1,1000),(delta,)*12)
    perturbed=replay_perturbed(
        graph,policies,"idle",(start[0]+Q(1,1000),start[1]),
        (delta,Q(0)),12)
    x=(Q(1,1000),Q(1,2)); y=(Q(-1,1000),Q(1,2))
    pos=choose(graph,policies,"idle",x)
    neg=choose(graph,policies,"idle",y)
    ratio_sq=sqdist(pos.phase.step(x),neg.phase.step(y))/sqdist(x,y)
    path=("sample","review","audit_loop","close","hold")
    target,exact,bound=certify_path(graph,weights,alpha,"idle",path)
    return {
        "experiment":"NRF-02",
        "source":"exact-rational graph toy",
        "alpha":str(alpha),
        "weights":{v:str(w) for v,w in weights.items()},
        "edge_rates":{e.name:str(e.rate) for e in graph.edges},
        "path":list(path),"path_target":target,
        "path_gain":str(exact),"path_weighted_bound":str(bound),
        "policy_lock":{
            "nominal_edges":[r.edge for r in trace],
            "observed_edges":list(perturbed[2]),
            "terminal_error_sq":str(sqdist(perturbed[1],trace[-1].nominal_after)),
            "terminal_bound_sq":str(trace[-1].error_after**2),
            "all_guards_separated":True,
        },
        "branch_discontinuity":{
            "left_edge":neg.name,"right_edge":pos.name,
            "gain_ratio_squared":str(ratio_sq),
        },
        "interpretation":"shared-path contractivity; strict guard checks only on this nominal trace",
    }

def main()->None:
    print(json.dumps(run(),indent=2,sort_keys=True))

if __name__=="__main__":
    main()
