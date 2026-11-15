"""NRF-02: exact graph potentials and guarded policy certificates.

This module certifies shared admissible schedules. State-selected routes
require strict guard separation. All values use rational arithmetic.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Union
from research_gym.core.numerical_range_certificate import Phase, Vector, project_box, sqdist

@dataclass(frozen=True)
class Edge:
    name: str
    source: str
    target: str
    phase: Phase

    @property
    def rate(self) -> Q:
        return self.phase.gain

@dataclass(frozen=True)
class Graph:
    vertices: tuple[str,...]
    edges: tuple[Edge,...]

    def __post_init__(self):
        if not self.vertices or len(set(self.vertices))!=len(self.vertices):
            raise ValueError("distinct nonempty vertices required")
        if not self.edges or len(set(e.name for e in self.edges))!=len(self.edges):
            raise ValueError("distinct nonempty edge names required")
        for e in self.edges:
            if e.source not in self.vertices or e.target not in self.vertices:
                raise ValueError("unknown endpoint")
            if e.rate<0:
                raise ValueError("negative Lipschitz bound")

    def edge(self,name:str)->Edge:
        for e in self.edges:
            if e.name==name: return e
        raise ValueError(f"unknown edge: {name}")

def potential(graph:Graph,alpha:Q)->dict[str,Q]:
    """Exact max-product Bellman-Ford; require all edge certificates."""
    if not Q(0)<alpha<Q(1):
        raise ValueError("alpha must lie in (0,1)")
    weights={v:Q(1) for v in graph.vertices}
    for _ in range(len(graph.vertices)-1):
        new=weights.copy()
        for e in graph.edges:
            candidate=e.rate*weights[e.source]/alpha
            new[e.target]=max(new[e.target],candidate)
        weights=new
    for e in graph.edges:
        if e.rate*weights[e.source]>alpha*weights[e.target]:
            raise ValueError("no positive potential: excessive directed cycle")
    return weights

def certify_path(graph:Graph,weights:dict[str,Q],alpha:Q,
                 start:str,path:tuple[str,...])->tuple[str,Q,Q]:
    if start not in graph.vertices or set(weights)!=set(graph.vertices):
        raise ValueError("bad start or weights")
    if any(w<=0 for w in weights.values()):
        raise ValueError("weights must be positive")
    if not Q(0)<alpha<Q(1): raise ValueError("invalid alpha")
    for e in graph.edges:
        if e.rate*weights[e.source]>alpha*weights[e.target]:
            raise ValueError("invalid potential")
    vertex=start; exact=Q(1)
    for name in path:
        edge=graph.edge(name)
        if edge.source!=vertex: raise ValueError("noncomposable edge path")
        exact*=edge.rate
        vertex=edge.target
    bound=alpha**len(path)*weights[vertex]/weights[start]
    if exact>bound: raise AssertionError("broken path certificate")
    return vertex,exact,bound

@dataclass(frozen=True)
class Guard:
    normal: Vector
    threshold: Q
    nonnegative_edge: str
    negative_edge: str

    def __post_init__(self):
        if self.normal==(Q(0),Q(0)):
            raise ValueError("guard normal cannot vanish")

    def score(self,x:Vector)->Q:
        return sum((a*b for a,b in zip(self.normal,x)),Q(0))-self.threshold

    def locked(self,x:Vector,radius:Q)->bool:
        """Strict separation via squared rational Cauchy-Schwarz."""
        if radius<0: raise ValueError("negative radius")
        s=self.score(x)
        normal_sq=sum((n*n for n in self.normal),Q(0))
        return s*s>normal_sq*radius*radius

Policy=Union[str,Guard]
class UncertifiedGuard(ValueError):
    pass

def choose(graph:Graph,policies:dict[str,Policy],vertex:str,x:Vector)->Edge:
    rule=policies[vertex]
    name=(rule.nonnegative_edge if rule.score(x)>=0 else rule.negative_edge) \
        if isinstance(rule,Guard) else rule
    edge=graph.edge(name)
    if edge.source!=vertex:
        raise ValueError("policy selects an edge at a different source")
    return edge

@dataclass(frozen=True)
class Receipt:
    vertex: str
    edge: str
    next_vertex: str
    nominal_before: Vector
    nominal_after: Vector
    error_before: Q
    error_after: Q
    guard_score: Q | None

def guarded_trace(graph:Graph,policies:dict[str,Policy],
                  start:str,seed:Vector,initial_error:Q,
                  disturbances:tuple[Q,...])->tuple[Receipt,...]:
    """Certify all nearby traces select identical edges for this finite horizon."""
    if start not in graph.vertices or initial_error<0 or any(d<0 for d in disturbances):
        raise ValueError("invalid trace inputs")
    vertex=start; x=seed; error=initial_error; results=[]
    for delta in disturbances:
        rule=policies[vertex]
        if isinstance(rule,Guard) and not rule.locked(x,error):
            raise UncertifiedGuard(f"branch margin exhausted at {vertex}")
        edge=choose(graph,policies,vertex,x)
        after=edge.phase.step(x)
        next_error=edge.rate*error+delta
        results.append(Receipt(vertex,edge.name,edge.target,x,after,error,
                               next_error,rule.score(x) if isinstance(rule,Guard) else None))
        vertex=edge.target; x=after; error=next_error
    return tuple(results)

def replay_perturbed(graph:Graph,policies:dict[str,Policy],start:str,
                     seed:Vector,per_step_noise:Vector,count:int):
    """A concrete perturbed trajectory projected back into the box."""
    vertex=start; x=seed; events=[]
    for _ in range(count):
        edge=choose(graph,policies,vertex,x)
        y=edge.phase.step(x)
        x=project_box((y[0]+per_step_noise[0],y[1]+per_step_noise[1]))
        events.append(edge.name)
        vertex=edge.target
    return vertex,x,tuple(events)

def example_graph()->Graph:
    def phase(eta:Q,gate:str="identity",coef=(Q(0),),
              bias=(Q(0),Q(0)))->Phase:
        return Phase(Q(0),Q(2),coef,eta,gate,bias)
    return Graph(("idle","probe","audit"),(
        Edge("hold","idle","idle",phase(Q(1,2))),
        Edge("sample","idle","probe",phase(Q(1),
             coef=(Q(0),Q(3,4)),bias=(Q(0),Q(1,3)))),
        Edge("approve","probe","idle",phase(Q(3,4))),
        Edge("review","probe","audit",phase(Q(1,2),"negate")),
        Edge("close","audit","idle",phase(Q(3,4))),
        Edge("audit_loop","audit","audit",phase(Q(1,2),"rot90")),
    ))

def example_policy()->dict[str,Policy]:
    return {"idle":Guard((Q(1),Q(0)),Q(0),"sample","hold"),
            "probe":"approve","audit":"close"}
