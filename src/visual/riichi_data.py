from __future__ import annotations

import math
import os
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

from riichi_dataset_loader import RiichiDatasetZarr


class ChunkedRandomSampler(Sampler[int]):
    """Memory efficient sampler for very large map-style datasets.

    ``torch.utils.data.RandomSampler`` generates a full random permutation of
    dataset indices when ``shuffle=True``.  For Riichi we often deal with tens
    of millions of samples which would require gigabytes of memory just to
    materialise that permutation.  ``ChunkedRandomSampler`` avoids this by
    shuffling data in manageable chunks, keeping the memory footprint bounded
    while still visiting every element exactly once per epoch.
    """

    _DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # ≈4 MiB worth of indices

    def __init__(
        self,
        data_source: Sized,
        *,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("data_source must implement __len__")

        self.data_source = data_source
        self.generator = generator
        self._length = len(data_source)

        if self._length == 0:
            self._chunk_size = 0
            self._dtype = torch.int64
            return

        # Prefer int32 when possible to halve the memory used for indices.
        self._dtype = torch.int32 if self._length < 2 ** 31 else torch.int64

        if chunk_size is None:
            element_size = torch.empty((), dtype=self._dtype).element_size()
            approx = max(1, self._DEFAULT_CHUNK_BYTES // element_size)
            # Avoid tiny chunks which would negatively impact shuffling quality.
            chunk_size = max(32_768, approx)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self._chunk_size = min(chunk_size, self._length)

    def __iter__(self) -> Iterator[int]:
        if self._length == 0:
            return iter(())
        return self._generate_indices()

    def _randperm(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=self._dtype)
        if self.generator is None:
            return torch.randperm(n, dtype=self._dtype)
        return torch.randperm(n, generator=self.generator, dtype=self._dtype)

    def _generate_indices(self) -> Iterator[int]:
        chunk_size = self._chunk_size
        total = self._length
        num_chunks = math.ceil(total / chunk_size)

        for chunk_idx in self._randperm(num_chunks).tolist():
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)
            span = end - start
            if span <= 0:
                continue
            for offset in self._randperm(span):
                yield start + int(offset.item())

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return self._length

# DataLoader workers on some systems may encounter "bus error" crashes when
# the default shared memory strategy is used.  To avoid relying on `/dev/shm`,
# switch to the more robust file-system based sharing strategy.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    # If the strategy cannot be set (e.g. unsupported backend) we simply
    # proceed with the default behaviour.
    pass

def build_riichi_dataloader(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    class_conditional: bool,
    img_size: int,
    cf_guidance_p: float,
):
    """Return a dataloader for the Riichi dataset.

    The signature mirrors :func:`build_cifar10_dataloader` for convenience even
    though most of the arguments are unused for this dataset.  ``RiichiDatasetZarr``
    already yields properly normalised tensors so no additional transform is
    required.  Depending on ``class_conditional`` the dataloader will either
    return only images (and masks if available) or image/label pairs.

    Parameters
    ----------
    root: str
        Path to the zarr dataset.
    batch_size: int
        Batch size for the dataloader.
    num_workers: int
        Number of dataloader workers.
    download: bool
        Present for API compatibility.  No downloading is performed.
    class_conditional: bool
        Whether to return labels alongside images.
    img_size: int
        Unused, kept for API compatibility.
    cf_guidance_p: float
        Unused, kept for API compatibility.

    Returns
    -------
    torch.utils.data.DataLoader
        A dataloader yielding batches from ``RiichiDatasetZarr``.
    """
    tfm = transforms.Compose([transforms.Resize(img_size)])

    # The dataset stores pre-computed tensors.  ``download``/``img_size`` and
    # ``cf_guidance_p`` are not required but kept to match the interface of the
    # other builders.
    ds = RiichiDatasetZarr(os.path.join(root, "training"), transform=tfm, return_mask=True)
    ds_tst = RiichiDatasetZarr(os.path.join(root, "test"), transform=tfm, return_mask=True)

    if class_conditional:
        # Each sample is (image, label, mask).  We want to stack the images and
        # masks while converting labels to a tensor.
        def collate(batch):
            xs, ys, ms = zip(*batch)
            x = torch.stack(xs, dim=0)
            y = torch.tensor(ys, dtype=torch.long)
            m = torch.stack(ms, dim=0)
            return x, y, m
    else:
        # Drop the labels but keep the masks if provided.
        def collate(batch):
            xs, _, ms = zip(*batch)
            x = torch.stack(xs, dim=0)
            m = torch.stack(ms, dim=0)
            return x, m

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_sampler = ChunkedRandomSampler(
        ds,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )

    train_loader = DataLoader(
        ds,
        sampler=train_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    eval_loader_kwargs = loader_kwargs.copy()
    eval_loader_kwargs.pop("prefetch_factor", None)
    # Evaluation data does not need shuffling, keep worker persistence for
    # throughput when applicable.
    eval_loader = DataLoader(
        ds_tst,
        shuffle=False,
        **eval_loader_kwargs,
    )

    return train_loader, eval_loader
