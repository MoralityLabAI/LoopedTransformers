"""Conservative enclosure of the frozen CPU BF16 RMSNorm and head."""

from experiments.nan_readpath37.run import bf16, decode_bf16
from experiments.nan_inverse65.lattice import add


def residual_box(grid, h, attention, native_write, edited_write, np):
    from experiments.nan_inverse65.lattice import invert

    lo, hi, c_lo, c_hi = invert(grid,h,attention,native_write,np)
    if (lo>hi).any() or not np.array_equal(add(c_lo,native_write,np),h) or not np.array_equal(
            add(c_hi,native_write,np),h):
        raise ValueError("The archived BF16 residual has no compatible physical carry")
    return add(c_lo,edited_write,np),add(c_hi,edited_write,np),c_lo,c_hi


def norm_box(low, high, gamma, eps, np):
    """Enclose the registered float32 reductions plus both BF16 roundings."""
    low,high=(np.asarray(v,dtype=np.float32) for v in (low,high))
    if (low>high).any() or not all(np.isfinite(x).all() for x in (low,high,gamma)):
        raise ValueError("Require an ordered finite residual and finite RMSNorm weight")
    minabs=np.where((low<=0)&(high>=0),0.,np.minimum(abs(low),abs(high)))
    maxabs=np.maximum(abs(low),abs(high))
    # With the same shape, the float32 positive reduction is monotone in
    # each term. Compute its two outward extreme inputs using that reduction.
    rms_bounds=[np.reciprocal(np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+eps))
                for x in (maxabs,minabs)]
    corners=np.stack([h*s for h in (low,high) for s in rms_bounds])
    u,v=bf16(corners.min(axis=0),np)*gamma,bf16(corners.max(axis=0),np)*gamma
    return bf16(np.minimum(u,v),np),bf16(np.maximum(u,v),np)


def logit_box(center,radius,weights,np,*,center_logits=None):
    middle=center@weights.T if center_logits is None else center_logits
    spread=radius@abs(weights).T
    # A conservative allowance for length-3072 float64 dot products and
    # center/radius construction, including cancellation.
    slack=1e-11*(np.sum(abs(center)+radius,axis=1)[:,None]*np.max(abs(weights),axis=1))+1e-12
    return bf16(middle-spread-slack,np),bf16(middle+spread+slack,np)


def calculate(head, clean, edits, owners, targets, foils, native_top, classes, canonical, check, np):
    clean_n=head.norm(clean).astype(np.float64)
    lower,upper=norm_box(edits[0],edits[1],head.gamma,head.eps,np)
    if not np.isfinite([lower,upper]).all() or (lower>upper).any():
        raise ValueError("Nonfinite or invalid normalized envelope")
    center=(lower.astype(np.float64)+upper)/2
    radii=(upper.astype(np.float64)-lower)/2
    main=np.concatenate((clean_n,center),axis=0)
    n=len(clean)
    k=len(center)
    target=targets[owners]
    clean_labels=np.empty((n,len(canonical)),dtype=np.float64)
    edit_lo,edit_hi=(np.empty((k,len(canonical)),dtype=np.float64) for _ in range(2))
    top=np.full(n,-1,dtype=np.int32)
    top_value=np.full(n,-np.inf)
    logz=np.full(n,-np.inf)
    drift_lo,drift_hi=(np.zeros(k,dtype=np.float64) for _ in range(2))
    wp,cp,gp,bp=(np.full((k,2),-np.inf) for _ in range(4))
    head.stream.seek(head.base+head.weight_offset)
    for start in range(0,head.vocab,768):
        check()
        stop=min(start+768,head.vocab)
        size=(stop-start)*head.width
        raw=head.stream.read(2*size)
        if len(raw)!=2*size:
            raise ValueError("Truncated native BF16 head tile")
        W=decode_bf16(raw,np).reshape(stop-start,head.width).astype(np.float64)
        middle=main@W.T
        baseline=bf16(middle[:n],np).astype(np.float64)
        bound=logit_box(center,radii,W,np,center_logits=middle[n:])
        local=baseline.argmax(axis=1)
        score=baseline[np.arange(n),local]
        update=score>top_value
        top[update]=start+local[update]
        top_value[update]=score[update]
        logz=np.logaddexp(logz,np.logaddexp.reduce(baseline,axis=1))
        clean_rows=baseline[owners]
        low,high=bound
        worst=np.maximum(abs(low-clean_rows),abs(high-clean_rows))
        must=np.maximum(np.maximum(low-clean_rows,clean_rows-high),0.)
        drift_hi=np.maximum(drift_hi,worst.max(axis=1))
        drift_lo=np.maximum(drift_lo,must.max(axis=1))
        ids=np.arange(start,stop)[None,:]
        native=ids==native_top[owners,None]
        gold=classes[start:stop][None,:]==target[:,None]
        for j,values in enumerate(bound):
            for result,mask in ((wp,native),(cp,~native),(gp,gold),(bp,~gold)):
                result[:,j]=np.maximum(result[:,j],np.where(mask,values,-np.inf).max(axis=1))
        for j,token in enumerate(canonical):
            if start<=token<stop:
                clean_labels[:,j]=baseline[:,token-start]
                edit_lo[:,j],edit_hi[:,j]=low[:,token-start],high[:,token-start]
    if not np.isfinite([wp,cp,gp,bp]).all() or not np.isfinite([edit_lo,edit_hi]).all():
        raise ValueError("Vocabulary bound omitted an archived winner or grammar class")
    rows=np.arange(k)
    foils=foils[owners]
    paired=np.stack((edit_lo[rows,target]-edit_hi[rows,foils],edit_hi[rows,target]-edit_lo[rows,foils]),axis=1)
    paired-= (clean_labels[owners,target]-clean_labels[owners,foils])[:,None]
    grammar=np.stack((gp[:,0]-bp[:,1],gp[:,1]-bp[:,0]),axis=1)
    return {"winner":top,"peak":top_value,"logz":logz,"winner_value":wp,"competitor":cp,
            "gold_margin":grammar,"paired_margin_change":paired,"peak_drift":np.stack((drift_lo,drift_hi),axis=1),
            "normalized_unique":(lower==upper).sum(axis=1)}


def fixture(np):
    from itertools import product
    from experiments.nan_inverse65.lattice import lattice

    grid=lattice(np)
    b=np.array([-.125,.25,1.,-.5],dtype=np.float32)
    high=grid[np.searchsorted(grid,b)+2]
    gamma=np.array([1.,-.5,.75,-1.125],dtype=np.float32)
    eps=np.float32(1e-5)
    lower,upper=norm_box(b[None],high[None],gamma,eps,np)
    W=np.array([[1.,-.5,.25,.625],[-.5,.75,1.,.125]],dtype=np.float64)
    center=(lower.astype(np.float64)+upper)/2
    radius=(upper.astype(np.float64)-lower)/2
    lo,hi=logit_box(center,radius,W,np)
    count=0
    for hit in product(*(grid[np.searchsorted(grid,l):np.searchsorted(grid,r,side="right")]
                         for l,r in zip(b,high,strict=True))):
        h=np.array(hit,dtype=np.float32)
        variance=np.mean(h*h,dtype=np.float32)
        s=bf16(bf16(h*np.reciprocal(np.sqrt(variance+eps)),np)*gamma,np)
        out=bf16(s.astype(np.float64)@W.T,np)
        assert (s>=lower[0]).all() and (s<=upper[0]).all()
        assert (out>=lo[0]).all() and (out<=hi[0]).all()
        count+=1
    assert count==81
    return {"exhaustive_BF16_head_inputs":count,"signed_norm_weights":True,"finite_head_enclosed":True}
