"""Resource-limited launcher and portable handoff. Python 3.11+, CPU torch."""
import os,sys,platform,time,threading,json,zipfile,shutil,argparse,ctypes
from pathlib import Path
HOME=Path(__file__).resolve().parent

def limit_memory(gib):
    n=int(gib*1024**3)
    if not n:return None,'explicit uncapped smoke'
    if sys.platform=='win32':
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('user',ctypes.c_longlong),('job',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('aff',ctypes.c_size_t),('pri',w.DWORD),('sched',w.DWORD)]
        class IO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('read','write','other','rb','wb','ob')]
        class Ext(ctypes.Structure):_fields_=[('basic',Basic),('io',IO),('proc',ctypes.c_size_t),('job',ctypes.c_size_t),('peakproc',ctypes.c_size_t),('peakjob',ctypes.c_size_t)]
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateJobObjectW.restype=w.HANDLE;k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.GetCurrentProcess.restype=w.HANDLE
        k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
        h=k.CreateJobObjectW(None,None);v=Ext();v.basic.flags=0x200|0x2000;v.job=n
        if not h or not k.SetInformationJobObject(h,9,ctypes.byref(v),ctypes.sizeof(v)) or not k.AssignProcessToJobObject(h,k.GetCurrentProcess()):raise OSError(ctypes.get_last_error(),'Windows hard memory job assignment failed')
        def peak():
            x=Ext()
            if not k.QueryInformationJobObject(h,9,ctypes.byref(x),ctypes.sizeof(x),None):raise OSError('job memory query failed')
            return x.peakjob
        return peak,'Windows Job Object hard process-tree commit limit'
    if sys.platform.startswith('linux'):
        rel=next((s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')),None)
        if rel is not None:
            d=Path('/sys/fs/cgroup')/rel.lstrip('/')
            for q in (d,*d.parents):
                p=q/'memory.max'
                if p.exists():
                    v=p.read_text().strip()
                    if v!='max' and int(v)<=n:return None,'verified cgroup memory.max='+v
        raise SystemExit('Run inside systemd-run --user --scope -p MemoryMax='+str(gib)+'G ...')
    raise SystemExit('Hard cap unsupported on this platform; uncapped smoke only')

def main():
    # Import run08 only after environment and job limits are in effect.
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--ram-gib',type=float,default=12);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    from run08 import cli
    args=cli()
    if not 1<=args.workers<=args.threads<=64 or a.ram_gib<0 or (args.profile=='full' and a.ram_gib==0):raise SystemExit('invalid CPU/memory configuration')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),LEAN_NUM_THREADS=str(max(1,args.threads//args.workers)))
    peak,kind=limit_memory(a.ram_gib)
    import psutil
    proc=psutil.Process()
    if hasattr(proc,'cpu_affinity'):proc.cpu_affinity(proc.cpu_affinity()[:args.threads])
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    stats={'os':platform.platform(),'python':sys.version,'memory_limit':kind,'requested_gib':a.ram_gib,'threads':args.threads,'workers':args.workers,'peak_sampled_tree_rss':0,'completed':False,'cpu_only':True}
    done=threading.Event()
    def sample():
        while not done.is_set():
            try:stats['peak_sampled_tree_rss']=max(stats['peak_sampled_tree_rss'],sum(p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            done.wait(.5)
    t=threading.Thread(target=sample,daemon=True);t.start();start=time.perf_counter()
    from run08 import main as run,sha,write
    try:run(args);stats['completed']=True
    except BaseException as e:stats['error']=repr(e);raise
    finally:
        done.set();t.join();stats['seconds_this_invocation']=time.perf_counter()-start
        if peak:stats['peak_job_commit']=peak()
        write(out/'resource.json',stats)
    code=out/'code';code.mkdir(exist_ok=True)
    for p in HOME.rglob('*'):
        if p.is_file() and (p.suffix in ('.py','.md','.lean','.toml') or p.name in ('lean-toolchain','spent.json','train_registry.json','package_manifest.json')) and '.lake' not in p.parts and '__pycache__' not in p.parts:
            dst=code/p.relative_to(HOME);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for k,p in [('07a',args.reissue_a),('07b',args.reissue_b)]:
        if p:shutil.copy2(p,out/f'reissue_{k}_manifest.json')
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='audit_manifest.json'};write(out/'audit_manifest.json',{'files':files})
    full=out.with_name(out.name+'_AUDIT.zip');small=out.with_name(out.name+'_FABLE.zip')
    for zipname,names in [(full,[*files,'audit_manifest.json']),(small,[n for n in files if n.startswith('code/') or n in ('summary.json','contract.json','registry_census.json','resource.json','frozen_policy.json','prior.json') or n.startswith('reissue_')]+['audit_manifest.json'])]:
        with zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED) as z:
            for n in names:z.write(out/n,n)
        print(zipname,sha(zipname),flush=True)
if __name__=='__main__':main()
