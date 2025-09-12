from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from riichi_dataset_loader import RiichiDatasetZarr

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

    # The dataset stores pre-computed tensors.  ``download``/``img_size`` and
    # ``cf_guidance_p`` are not required but kept to match the interface of the
    # other builders.
    ds = RiichiDatasetZarr(os.path.join(root, "training"), transform=tfm, return_mask=True)
    ds_tst = RiichiDatasetZarr(os.path.join(root, "test"), transform=tfm, return_mask=True)
    tfm = transforms.Compose([
        transforms.Resize(img_size)
    ])

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

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
    ), DataLoader(
        ds_tst,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
    )