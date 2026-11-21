"""Source-locked observational SAE/VPD/BPD geometry; never edits a model."""
import re
import numpy as np
from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary,fingerprint_arrays

REVISION='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
FIELDS=('checkpoint_revision','backend','layer_zero_based','loop','site','width')

def validate_site(meta):
    if meta.get('checkpoint_revision')!=REVISION:raise ValueError('Checkpoint mismatch')
    if meta.get('backend') not in ('bf16','nf4'):raise ValueError('Backend must be a distinct condition')
    if type(meta.get('layer_zero_based')) is not int or not 0<=meta['layer_zero_based']<22:raise ValueError('Layer outside native model')
    if meta.get('loop') not in (1,2):raise ValueError('Native loops only')
    if meta.get('site')!='residual_after_layer':raise ValueError('A different site requires an explicit coordinate/Jacobian bridge')
    if meta.get('width')!=3072:raise ValueError('Pinned Nanbeige physical state width is 3072')
    if not re.fullmatch(r'[0-9a-f]{64}',meta.get('artifact_sha256','')):raise ValueError('Artifact file-byte SHA required')

def require_same_site(left,right):
    validate_site(left);validate_site(right)
    if any(left[k]!=right[k] for k in FIELDS):raise ValueError('Checkpoint/backend/layer/loop/site/width mismatch')

def probe_and_crossref(hidden,sae,vpd_output_vectors,*,state_meta,sae_meta,vpd_meta,bpd_vectors=None,bpd_meta=None):
    """Vectors are in physical residual coordinates; alignment is not causality.

    VPD right/input factors are NOT accepted as residual output directions.
    Caller must supply already mapped output directions from the stated site.
    """
    require_same_site(state_meta,sae_meta);require_same_site(state_meta,vpd_meta)
    if sae_meta.get('fit_split')!='calibration':raise ValueError('SAE dictionary must be frozen from calibration only')
    if not re.fullmatch(r'[0-9a-f]{64}',sae_meta.get('fit_manifest_sha256','')):raise ValueError('Calibration-fit manifest SHA required')
    if vpd_meta.get('factor_role')!='physical_output_directions':raise ValueError('VPD input factors are not hidden-state directions')
    h=np.asarray(hidden,dtype=np.float64);width=state_meta['width']
    if h.shape!=(width,) or not np.isfinite(h).all():raise ValueError('Bad hidden vector')
    E,b,D=to_physical_dictionary(sae['encoder'],sae['bias'],sae['decoder'],sae['mean'],sae['scale'])
    V=np.asarray(vpd_output_vectors,dtype=np.float64)
    if E.shape[1]!=width or V.ndim!=2 or V.shape[1]!=width or not np.isfinite(V).all():raise ValueError('Physical dimension mismatch')
    pre=E@h+b;top=np.argsort(-pre,kind='stable')[:min(4,len(pre))];z=np.zeros(len(pre));z[top]=np.maximum(pre[top],0)
    mean=np.asarray(sae['mean'],dtype=np.float64);scale=np.asarray(sae['scale'],dtype=np.float64);dbias=np.asarray(sae.get('decoder_bias',np.zeros(width)),dtype=np.float64)
    if dbias.shape!=(width,) or not np.isfinite(dbias).all():raise ValueError('Bad normalized decoder bias')
    reconstruction=mean+scale*dbias+z@D
    def cos(x,y):
        denom=np.linalg.norm(x,axis=1)[:,None]*np.linalg.norm(y,axis=1)[None,:]
        return np.divide(x@y.T,denom,out=np.zeros_like(denom),where=denom>0)
    result={'evidence_class':'observational dictionary geometry only','state_site':{k:state_meta[k] for k in FIELDS},'top4_indices':top.tolist(),'codes':z.tolist(),
        'encoder_vpd_cosine':cos(E,V).tolist(),'decoder_vpd_cosine':cos(D,V).tolist(),'encoder_vpd_preactivation_sensitivity':(E@V.T).tolist(),
        'dictionary_array_sha256':fingerprint_arrays(E,b,D),'state_array_sha256':fingerprint_arrays(h),'VPD_array_sha256':fingerprint_arrays(V),
        'dictionary_file_sha256':sae_meta['artifact_sha256'],'state_file_sha256':state_meta['artifact_sha256'],'VPD_file_sha256':vpd_meta['artifact_sha256'],
        'reconstruction_l2':float(np.linalg.norm(h-reconstruction)),
        'normalized_reconstruction_l2':float(np.linalg.norm((h-reconstruction)/scale)),
        'normalization_arrays_sha256':fingerprint_arrays(mean,scale,dbias),
        'fit_manifest_sha256':sae_meta['fit_manifest_sha256'],
        'zero_norm_counts':{'encoder_rows':int(np.sum(np.linalg.norm(E,axis=1)==0)),'decoder_rows':int(np.sum(np.linalg.norm(D,axis=1)==0)),'VPD_vectors':int(np.sum(np.linalg.norm(V,axis=1)==0))},
        'model_edit_performed':False,'BPD_transfer_attempted':False}
    if bpd_vectors is not None:
        if bpd_meta is None:raise ValueError('BPD source metadata required')
        require_same_site(state_meta,bpd_meta)
        B=np.asarray(bpd_vectors,dtype=np.float64)
        if B.ndim!=2 or B.shape[1]!=width or not np.isfinite(B).all():raise ValueError('BPD dimension mismatch')
        result|={'encoder_bpd_cosine':cos(E,B).tolist(),'decoder_bpd_cosine':cos(D,B).tolist(),'bpd_vpd_cosine':cos(B,V).tolist(),
            'BPD_array_sha256':fingerprint_arrays(B),'BPD_file_sha256':bpd_meta['artifact_sha256'],'BPD_note':'geometry only; no task fidelity or transfer inference'}
    return result
