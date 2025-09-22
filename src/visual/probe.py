import os, time
import webdataset as wds
from torch.utils.data import get_worker_info
import torch, numpy as np, pickle
from PIL import Image

def obj_nbytes(x):
    if isinstance(x, torch.Tensor):
        # CPU/GPU 都行；这里统计张量本体大小
        return x.element_size() * x.nelement()
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, (bytes, bytearray, memoryview)):
        return len(x)
    if isinstance(x, Image.Image):
        # 解码到原始像素再算；注意这会额外开一份临时内存
        return len(x.tobytes())
    # 其它 Python 对象用 pickle 近似（不精确，但有参考价值）
    try:
        return len(pickle.dumps(x, protocol=4))
    except Exception:
        return 0

def sample_nbytes(sample):
    # sample 可能是 dict / tuple：把里面每个字段都算上
    if isinstance(sample, dict):
        return sum(obj_nbytes(v) for v in sample.values())
    if isinstance(sample, (list, tuple)):
        return sum(obj_nbytes(v) for v in sample)
    return obj_nbytes(sample)


def probe_map(sample, every=10):
    # 统计与抽样打印
    info = get_worker_info()
    wid  = info.id if info is not None else 0
    probe_map.counts.setdefault(wid, 0)
    probe_map.totals.setdefault(wid, 0)

    sz = sample_nbytes(sample)
    probe_map.counts[wid] += 1
    probe_map.totals[wid] += sz

    if probe_map.counts[wid] % every == 0:
        avg = probe_map.totals[wid] / probe_map.counts[wid]
        print(f"[probe][pid={os.getpid()}][worker={wid}] "
              f"samples={probe_map.counts[wid]} avg={avg/1024/1024:.3f} MB "
              f"last={sz/1024/1024:.3f} MB at {time.strftime('%H:%M:%S')}")

    return sample
probe_map.counts = {}
probe_map.totals = {}
