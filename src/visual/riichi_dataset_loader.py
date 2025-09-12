# make_splits.py
import os, json, numpy as np, torch, zarr
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data import DataLoader


NUM_TILES = 34 
NUM_FEATURES = 29

# Dataset class
class RiichiDatasetZarr(Dataset):
    """
    读取 zarr 分块数据，支持 (images, labels, masks, filenames)
    indices: 可传 sklearn 产出的子集索引
    """
    def __init__(self, root, indices=None, transform=None, target_transform=None, return_mask=True, to_float32=True):
        self.root_path = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_mask = return_mask
        self.to_float32 = to_float32

        # 延迟打开（每个 worker 各自打开一次，避免句柄共享问题）
        self._g = None
        self._imgs = self._lbls = self._msks = None

        # 预读 N 以便 __len__
        g = zarr.open_group(root, mode="r")
        self.N = g["labels"].shape[0]
        print(self.N)
        g.store.close() if hasattr(g.store, "close") else None

        if indices is None:
            self.indices = np.arange(self.N, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        # 读 meta（可选）
        self.meta = None
        meta_p = os.path.join(root, "meta.json")
        if os.path.exists(meta_p):
            try:
                self.meta = json.load(open(meta_p, "r", encoding="utf-8"))
            except Exception:
                pass

    def _ensure_open(self):
        if self._g is None:
            self._g = zarr.open_group(self.root_path, mode="r")
            self._imgs = self._g["images"]
            self._lbls = self._g["labels"]
            self._msks = self._g["masks"]
            self._fns  = self._g["filenames"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        gi = int(self.indices[idx])

        x = self._imgs[gi]               # (C,H)  numpy
        y = int(self._lbls[gi])
        m = self._msks[gi]               # (H,)

        if self.to_float32 and x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        x = torch.from_numpy(x).unsqueeze(-1).expand(NUM_FEATURES, NUM_TILES, NUM_TILES).contiguous()  # (C,H,W)
        m = torch.from_numpy(m.astype(np.float32, copy=False))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return (x, y, m) if self.return_mask else (x, y)


# 示例：
# g = torch.Generator().manual_seed(42)  # 可复现
# train_ds = ZarrMultiChannel("dataset_zarr", indices=train_idx)
''' 
# 可复现shuffle
import numpy as np, random
def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True, generator=g,
    worker_init_fn=seed_worker, num_workers=4, pin_memory=True, persistent_workers=True
)

# 分布式训练
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(train_ds, shuffle=True, seed=42)
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, shuffle=False)
'''
def build_loaders_example(root, splits_json=None, batch_size=64, num_workers=4, pin_memory=True):
    if splits_json:
        splits = json.load(open(splits_json, "r"))
        idx_tr, idx_va, idx_te = splits["train"], splits["val"], splits["test"]
    else:
        idx_tr = idx_va = idx_te = None

    train_ds = RiichiDatasetZarr(root, indices=idx_tr, return_mask=True)
    #val_ds   = RiichiDatasetZarr(root, indices=idx_va, return_mask=True)
    test_ds  = RiichiDatasetZarr(root, indices=idx_te, return_mask=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), prefetch_factor=2
    )
    #val_loader = DataLoader(
    #    val_ds, batch_size=batch_size*2, shuffle=False,
    #    num_workers=num_workers, pin_memory=pin_memory,
    #    persistent_workers=(num_workers > 0)
    #)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size*2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader

if __name__ == "__main__":
    pass