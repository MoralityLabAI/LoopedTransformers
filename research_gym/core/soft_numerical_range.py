"""NRF-02D: exact rational state-dependent soft mixture for Jordan loopers."""
from fractions import Fraction as Q
from research_gym.core.numerical_range_certificate import Phase, Vector, sqdist

class SoftJordanMixer:
    def __init__(self) -> None:
        self.expansive=Phase(Q(0),Q(2),(Q(0),Q(3,4)),Q(1),"identity",
                             (Q(0),Q(1,3)))
        self.repair=Phase(Q(0),Q(2),(Q(0),),Q(3,4),"identity")

    def high_probability(self,x:Vector)->Q:
        if len(x)!=2 or any(not -1<=v<=1 for v in x):
            raise ValueError("state outside convex box")
        p=Q(1,4)+x[0]/16
        assert Q(3,16)<=p<=Q(5,16)
        return p

    def fixed_probability_step(self,x:Vector)->Vector:
        p=Q(1,4)
        a=self.expansive.step(x);b=self.repair.step(x)
        return tuple(p*a[i]+(1-p)*b[i] for i in range(2))

    def step(self,x:Vector)->Vector:
        p=self.high_probability(x)
        a=self.expansive.step(x);b=self.repair.step(x)
        return tuple(p*a[i]+(1-p)*b[i] for i in range(2))

    @property
    def fixed_probability_rate(self)->Q:
        return Q(1,4)*self.expansive.gain+Q(3,4)*self.repair.gain

    @property
    def soft_rate(self)->Q:
        # p_high <= 5/16, edge rates 3/2 and 1/4.
        # ||pi(x)-pi(y)||_1 <= ||x-y||/8.
        # Diameter of [-1,1]^2 = 2sqrt(2) <= 3.
        return (Q(5,16)*self.expansive.gain
                +Q(11,16)*self.repair.gain
                +Q(3)*Q(1,8)/2)

    def iterate(self,x:Vector,n:int,soft:bool=True)->Vector:
        if n<0: raise ValueError("negative iterations")
        op=self.step if soft else self.fixed_probability_step
        for _ in range(n): x=op(x)
        return x

    def distance_bound_sq(self,x:Vector,y:Vector,steps:int)->Q:
        return self.soft_rate**(2*steps)*sqdist(x,y)

    def fixed_point_tail_bound_sq(self,x:Vector,steps:int)->Q:
        """A posteriori squared distance to the unique fixed point."""
        r=self.soft_rate
        if r>=1: raise ValueError("no contraction certificate")
        return r**(2*steps)*sqdist(self.step(x),x)/(1-r)**2
