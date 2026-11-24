"""Construct learned, static and seeded writes with matched realized geometry."""

from hashlib import sha256
from itertools import product
from math import acos, asin, pi, sin, sqrt

from experiments.nan_cidose74.geometry import COSINE, DOSE, DOSE_TOLERANCE
from experiments.nan_delay39.instrument import prepare
from experiments.nan_radial76.geometry import MAX_NORM_GAP, MAX_RADIAL_FRACTION, MAX_TANGENT_COSINE
from experiments.nan_readpath37.run import bf16


ARMS = ("learned", "cal_static", "seeded")
RADII = (0., -.0025, .0025, -.005, .005, -.01, .01, -.02, .02, -.04, .04)
SCALES = (1., .995, 1.005, .99, 1.01, .98, 1.02, .97, 1.03)
GRID = tuple(product(RADII, SCALES))


def latitude_interval(radial, cosine=COSINE):
    latitude = asin(max(-1., min(1., float(radial))))
    half_angle = acos(cosine)
    return sin(max(-pi/2, latitude-half_angle)), sin(min(pi/2, latitude+half_angle))


def relaxed_radial_interval(interval):
    """Possible r / nominal L2 under independent allowed lengths."""
    endpoints = [q*length for q in interval for length in (1-DOSE_TOLERANCE,1+DOSE_TOLERANCE)]
    return min(endpoints), max(endpoints)


def span_orthogonal_seed(tangents, axis, seed, np):
    matrix = np.stack((axis,*tangents))
    _,singular,basis = np.linalg.svd(matrix,full_matrices=False)
    basis = basis[singular>1e-10]
    random = np.random.default_rng(seed).standard_normal(len(axis))
    random -= (basis @ random) @ basis
    length = np.linalg.norm(random)
    if length<=1e-12:
        raise ValueError("Seeded tangent space is empty")
    return random/length


def candidates(native, axis, tangent, aim, q, nominal, np, lock=None):
    radius = np.linalg.norm(native)
    tangential = sqrt(max(0.,1-q*q))
    radii, factors = np.asarray(GRID).T
    after = bf16(native+(q+radii[:,None])*nominal*axis+
                 factors[:,None]*nominal*tangential*tangent,np).astype(np.float64)
    if lock is not None:
        after[:,lock]=native[lock]
    delta = after-native
    dose = np.linalg.norm(delta,axis=1)
    radial = delta@axis
    norms = np.linalg.norm(after,axis=1)
    alignment = np.divide(delta@aim,dose,out=np.zeros_like(dose),where=dose>0)
    eligible = (dose>0)&(abs(dose/nominal-1)<=DOSE_TOLERANCE)&(alignment>=COSINE)
    result = []
    for i in np.flatnonzero(eligible):
        l,r = float(dose[i]),float(radial[i])
        # The native norm identity also verifies the sign of the reached radial term.
        if not np.isclose(norms[i]**2,radius**2+2*radius*r+l*l,rtol=1e-12,atol=1e-10):
            raise ValueError("Reached write norm does not close")
        result.append((after[i],l,r,float(norms[i]),float(alignment[i]),int(i)))
    return result


def joint(values, nominal, radius):
    dose=[v[1] for v in values]
    radial=[v[2] for v in values]
    norms=[v[3] for v in values]
    return ((max(dose)-min(dose))/nominal,(max(radial)-min(radial))/nominal,
            (max(norms)-min(norms))/radius)


def pair_cost(a,b,nominal,radius,np):
    widths=np.array((nominal*DOSE_TOLERANCE,nominal*MAX_RADIAL_FRACTION,radius*MAX_NORM_GAP))
    return np.max(abs(a[:,None,:3]-b[None,:,:3])/widths,axis=2)


def match_options(options,nominal,radius,np):
    if not all(options):
        return None
    a,b,c=(np.asarray([row[1:] for row in group]) for group in options)
    ab,ac,bc=(pair_cost(x,y,nominal,radius,np) for x,y in ((a,b),(a,c),(b,c)))
    candidate_errors=[abs(x[:,0]/nominal-1) for x in (a,b,c)]
    last, best = None, None
    for i in range(len(a)):
        cost=np.maximum(bc,np.maximum(ab[i,:,None],ac[i,None,:]))
        j,k=np.where(cost<=1.)
        if not len(j):
            continue
        sums=candidate_errors[0][i]+candidate_errors[1][j]+candidate_errors[2][k]
        cosine=-np.minimum(a[i,3],np.minimum(b[j,3],c[k,3]))
        order=np.lexsort((c[k,4],b[j,4],cosine,sums,cost[j,k]))
        top=order[0]
        u,v=int(j[top]),int(k[top])
        key=(float(cost[u,v]),float(sums[top]),float(cosine[top]),(int(a[i,4]),int(b[u,4]),int(c[v,4])))
        if best is None or key<best:
            best,last=key,(options[0][i],options[1][u],options[2][v])
    return last

def resolve(native, axis, original, tangent, random, q, mode, np, lock=None):
    radius = float(np.linalg.norm(native))
    nominal = DOSE*radius
    intended = original if mode!="tangent" else tangent
    seeded_aim = q*axis+sqrt(max(0.,1-q*q))*random
    options = [candidates(native,axis,t,aim,q,nominal,np,lock)
               for t,aim in zip((*tangent,random),(*intended,seeded_aim),strict=True)]
    counts = dict(zip(ARMS,map(len,options),strict=True))
    last=match_options(options,nominal,radius,np)
    if last is None:
        return {}, {"status":"rounded_unqualified","eligible_candidates":counts,"target_radial_share":q}
    dose_gap,radial_gap,norm_gap=joint(last,nominal,radius)
    outputs, receipt = {}, {}
    for i,(name,chosen) in enumerate(zip(ARMS,last,strict=True)):
        state, dose, radial, norm, cosine, ix=chosen
        delta=state-native
        outputs[name]=state
        receipt[name]={"realized_l2":dose,"radial_l2":radial,"relative_reached_norm":norm/radius,
                       "planned_direction_cosine":cosine,"grid":list(GRID[ix]),
                       "original_direction_cosine":float(delta@original[i]/dose) if i<2 else None}
    alias=np.array_equal(outputs["learned"],outputs["cal_static"])
    return outputs,{"status":"matched","eligible_candidates":counts,"target_radial_share":q,
                   "measured":receipt,"same_learned_and_static_bytes":bool(alias),
                   "nominal_l2":nominal,"dose_gap_fraction":dose_gap,"radial_gap_fraction":radial_gap,
                   "relative_norm_gap":norm_gap}


def shell_candidates(native, axis, local, rest, aim, pivot, active, radial, lattice, np):
    """Hold one BF16 coordinate fixed and solve the two remaining shell equations."""
    norm=float(np.linalg.norm(native))
    nominal=DOSE*norm
    pivot_values=lattice[np.searchsorted(lattice,native[pivot]-(1+DOSE_TOLERANCE)*nominal):
                         np.searchsorted(lattice,native[pivot]+(1+DOSE_TOLERANCE)*nominal,side="right")]
    remaining_axis=axis.copy()
    remaining_axis[pivot]=0.
    axis_share=float(np.linalg.norm(remaining_axis))
    if axis_share<=1e-10:
        return []
    correction,scales=np.asarray(GRID).T
    out=[]
    for value in pivot_values:
        delta_k=float(value-native[pivot])
        # An orthogonal remainder to the clean axis still has one radial
        # component in the subspace that excludes the fixed coordinate.
        alpha=(radial+correction*nominal-axis[pivot]*delta_k)/axis_share
        available=active**2-delta_k**2-alpha**2
        valid=available>0
        if not valid.any():
            continue
        states=native+alpha[valid,None]*rest+(scales[valid]*np.sqrt(available[valid]))[:,None]*local
        states[:,pivot]=value
        after=bf16(states,np).astype(np.float64)
        shift=after-native
        l2=np.linalg.norm(shift,axis=1)
        reached_r=shift@axis
        reached_norm=np.linalg.norm(after,axis=1)
        if aim is None:
            intended=states-native
            denom=np.linalg.norm(intended,axis=1)*l2
            angle=np.divide(np.sum(shift*intended,axis=1),denom,out=np.zeros_like(l2),where=denom>0)
        else:
            angle=np.divide(shift@aim,l2,out=np.zeros_like(l2),where=l2>0)
        eligible=(abs(l2/nominal-1)<=DOSE_TOLERANCE)&(angle>=COSINE)
        for k in np.flatnonzero(eligible):
            out.append((after[k],float(l2[k]),float(reached_r[k]),float(reached_norm[k]),float(angle[k]),
                        int(np.flatnonzero(valid)[k])))
    return out


def anchored_tangent(native, axis, tangent, seed, np):
    """Match the reached learned tangent's shell while keeping the natural gate geometry fixed."""
    from experiments.nan_inverse65.lattice import lattice

    radius=float(np.linalg.norm(native))
    nominal=DOSE*radius
    learned=prepare(native,tangent[0],DOSE,DOSE_TOLERANCE,COSINE,np)
    if not learned.eligible:
        return {}, {"status":"learned_tangent_unqualified"}
    reached=learned.after-native
    r=float(reached@axis)
    a=(learned.after,learned.realized_l2,r,float(np.linalg.norm(learned.after)),learned.cosine,0)

    pivot=int(np.argmax(abs(native)))
    rest=axis.copy()
    rest[pivot]=0.
    rest_norm=np.linalg.norm(rest)
    if rest_norm<=1e-10:
        return {}, {"status":"degenerate_remainder"}
    rest/=rest_norm
    local=tangent.copy()
    local[:,pivot]=0.
    local-=(local@rest)[:,None]*rest
    lengths=np.linalg.norm(local,axis=1)
    if not np.all(lengths>1e-10):
        return {}, {"status":"degenerate_local_tangent"}
    local/=lengths[:,None]
    e=np.zeros_like(axis)
    e[pivot]=1.
    seeded=span_orthogonal_seed((*local,e),rest,seed,np)
    radial_share=r/a[1]
    grid=lattice(np).astype(np.float64)
    static=shell_candidates(native,axis,local[1],rest,tangent[1],pivot,a[1],r,grid,np)
    sham=shell_candidates(native,axis,seeded,rest,None,pivot,a[1],r,grid,np)
    viable=match_options(((a,),static,sham),nominal,radius,np)
    counts={"learned":1,"cal_static":len(static),"seeded":len(sham)}
    if viable is None:
        return {},{"status":"anchored_shell_unqualified","eligible_candidates":counts,
                   "target_radial_share":radial_share}
    gap,radial_gap,norm_gap=joint(viable,nominal,radius)
    outputs={arm:v[0] for arm,v in zip(ARMS,viable,strict=True)}
    reached_tangents=[]
    for arm in ARMS:
        delta=outputs[arm]-native
        delta[pivot]=0.
        delta-=float(delta@rest)*rest
        length=np.linalg.norm(delta)
        if length<=1e-12:
            return {},{"status":"degenerate_reached_tangent","eligible_candidates":counts}
        reached_tangents.append(delta/length)
    seed_cosines=[float(reached_tangents[-1]@t) for t in reached_tangents[:2]]
    if max(map(abs,seed_cosines))>MAX_TANGENT_COSINE:
        return {},{"status":"seed_orientation_unqualified","eligible_candidates":counts}
    measured={arm:{"realized_l2":l,"radial_l2":b,"relative_reached_norm":n/radius,
                   "planned_direction_cosine":angle,"original_direction_cosine":angle if arm!="seeded" else None}
              for arm,(_,l,b,n,angle,_) in zip(ARMS,viable,strict=True)}
    return outputs,{"status":"matched","measured":measured,"eligible_candidates":counts,
                   "target_radial_share":radial_share,"nominal_l2":nominal,
                   "dose_gap_fraction":gap,"radial_gap_fraction":radial_gap,"relative_norm_gap":norm_gap,
                   "realized_sham_remainder_cosines":seed_cosines,
                   "same_learned_and_static_bytes":bool(np.array_equal(outputs["learned"],outputs["cal_static"]))}

def plan(native, learned, static, key, np):
    native=np.asarray(native,dtype=np.float64)
    radius=np.linalg.norm(native)
    directions=np.asarray([learned,static],dtype=np.float64)
    lengths=np.linalg.norm(directions,axis=1)
    if radius<=0 or not np.all(lengths>1e-12):
        return {}, {"status":"degenerate_direction"}
    original=directions/lengths[:,None]
    axis=native/radius
    shares=original@axis
    tangent=original-shares[:,None]*axis
    lengths=np.linalg.norm(tangent,axis=1)
    if not np.all(lengths>1e-10):
        return {}, {"status":"degenerate_tangent"}
    tangent/=lengths[:,None]
    limits=[latitude_interval(q) for q in shares]
    lo,hi=max(a for a,_ in limits),min(b for _,b in limits)
    relaxed=sorted([relaxed_radial_interval(interval) for interval in limits])
    gap=max(0.,relaxed[1][0]-relaxed[0][1])
    key_hash=sha256(f"NAN-ROUTEMATCH-86|{key}".encode()).digest()[:8]
    random=span_orthogonal_seed(tangent,axis,int.from_bytes(key_hash,"big"),np)
    rays=[prepare(native,d,DOSE,DOSE_TOLERANCE,COSINE,np) for d in original]
    nominal=DOSE*radius
    ray_geometry=[(p.after,p.realized_l2,float((p.after-native)@axis),float(np.linalg.norm(p.after)))
                  for p in rays]
    ray_match=all(r.eligible for r in rays) and all(a<=b for a,b in zip(
        joint(ray_geometry,nominal,radius),(DOSE_TOLERANCE,MAX_RADIAL_FRACTION,MAX_NORM_GAP),strict=True))
    if lo<=hi:
        q=float(np.clip(shares.mean(),lo,hi))
        natural,nr=resolve(native,axis,original,tangent,random,q,"natural",np)
    else:
        natural,nr={},{"status":"no_exact_common_radial_share"}
    tangential,tr=resolve(native,axis,original,tangent,random,0.,"tangent",np)
    pivot=int(np.argmax(abs(native)))
    pivot_basis=np.zeros_like(axis)
    pivot_basis[pivot]=1.
    subaxis=axis.copy()
    subaxis[pivot]=0.
    subaxis_norm=np.linalg.norm(subaxis)
    locked={}
    if subaxis_norm>1e-10:
        subaxis/=subaxis_norm
        directions=tangent.copy()
        directions[:,pivot]=0.
        directions-=(directions@subaxis)[:,None]*subaxis
        n=np.linalg.norm(directions,axis=1)
        if np.all(n>1e-10):
            directions/=n[:,None]
            s=span_orthogonal_seed((*directions,pivot_basis),subaxis,int.from_bytes(key_hash,"big"),np)
            locked,lr=resolve(native,axis,tangent,directions,s,0.,"locked_tangent",np,lock=pivot)
        else:
            lr={"status":"degenerate_locked_tangent"}
    else:
        lr={"status":"degenerate_locked_tangent"}
    for reached in locked.values():
        if reached[pivot]!=native[pivot]:
            raise ValueError("A fixed BF16 coordinate changed")
    if lr["status"]=="matched":
        for i, arm in enumerate(ARMS[:2]):
            delta=locked[arm]-native
            lr["measured"][arm]["natural_direction_cosine"]=float(delta@original[i]/np.linalg.norm(delta))
    anchor,ar=anchored_tangent(native,axis,tangent,int.from_bytes(key_hash,"big"),np)
    if ar["status"]=="matched":
        for i,arm in enumerate(ARMS[:2]):
            delta=anchor[arm]-native
            ar["measured"][arm]["natural_direction_cosine"]=float(delta@original[i]/np.linalg.norm(delta))
    meta={"status":"ok","natural_cosine":float(original[0]@original[1]),
          "radial_shares":shares.tolist(),"continuous_radial_intersection":bool(lo<=hi),
          "relaxed_radial_gap_fraction":gap,"original_angle_impossible":gap>MAX_RADIAL_FRACTION+1e-10,
          "raw_rays_match":bool(ray_match),"natural":nr,"tangent":tr,"locked_tangent":lr,"anchored_tangent":ar,"pivot_abs_share":float(abs(axis[pivot])),
          "commanded_tangent_orthogonalities":(tangent@random).tolist()}
    if meta["original_angle_impossible"] and (natural or ray_match):
        raise ValueError("A natural edit contradicts the continuous obstruction")
    return {"natural":natural,"tangent":tangential,"locked_tangent":locked,"anchored_tangent":anchor},meta


def self_test(np):
    near=latitude_interval(1.)
    far=latitude_interval(-1.)
    assert np.isclose(near[0],COSINE) and np.isclose(far[1],-COSINE)
    assert relaxed_radial_interval(far)[1]+MAX_RADIAL_FRACTION < relaxed_radial_interval(near)[0]
    rng=np.random.default_rng(86023)
    y=bf16(rng.normal(size=768),np).astype(np.float64)
    unit=y/np.linalg.norm(y)
    a,b=rng.normal(size=(2,len(y)))
    a-=float(a@unit)*unit
    b-=float(b@unit)*unit
    found, report=plan(y,a,b,"fixture",np)
    assert report["natural"]["status"] == report["tangent"]["status"] == "matched"
    assert report["anchored_tangent"]["status"]=="matched"
    assert len(found["natural"])==3 and not report["natural"]["same_learned_and_static_bytes"]
    assert max(abs(q) for q in report["commanded_tangent_orthogonalities"])<1e-10
    narrow, obstruction=plan(y,unit+.01*a/np.linalg.norm(a),-unit+.01*b/np.linalg.norm(b),"obstruction",np)
    assert not narrow["natural"] and obstruction["original_angle_impossible"]
    return {"rounded_three_arm_shell":True,"continuous_obstruction":True,"orthogonal_seed":True}
