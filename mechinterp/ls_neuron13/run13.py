from repair13 import *
from concurrent.futures import ProcessPoolExecutor
import experiment as e

def job(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);seed,dose,arm=a
    if arm in ('frozen','full'):
        e.SEEDS=SEEDS;e.train(dose,arm,Path('baseline'))
    else:run(seed,dose,arm,Path('repairs'))
if __name__=='__main__':
    jobs=[(0,d,a) for d in (15,5) for a in ('frozen','full')]+[(s,d,a) for d in (15,5) for s in SEEDS for a in ARMS]
    with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(job,jobs,chunksize=1))
