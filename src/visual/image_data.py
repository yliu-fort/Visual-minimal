from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_cifar10_dataloader(root: str, batch_size: int, num_workers: int, download: bool, class_conditional: bool, img_size: int, cf_guidance_p: float) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    ds = datasets.CIFAR10(root=root, train=True, transform=tfm, target_transform=None, download=download)
    ds_tst = datasets.CIFAR10(root=root, train=False, transform=tfm, target_transform=None, download=download)
    if class_conditional:
        def collate(batch):
            xs, ys = zip(*batch)
            x = torch.stack(xs, dim=0)
            y = torch.tensor(ys, dtype=torch.long)
            return x, y
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate), \
               DataLoader(ds_tst, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate)
    else:
        def collate(batch):
            xs, _ = zip(*batch)
            x = torch.stack(xs, dim=0)
            return x
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate), \
               DataLoader(ds_tst, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate)

def build_mnist_dataloader(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    class_conditional: bool,
    img_size: int,
    cf_guidance_p: float
) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # grayscale, so single channel
    ])
    
    ds = datasets.MNIST(
        root=root,
        train=True,
        transform=tfm,
        target_transform=None,
        download=download
    )

    if class_conditional:
        def collate(batch):
            xs, ys = zip(*batch)
            x = torch.stack(xs, dim=0)
            y = torch.tensor(ys, dtype=torch.long)
            return x, y
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate
        )
    else:
        def collate(batch):
            xs, _ = zip(*batch)
            x = torch.stack(xs, dim=0)
            return x
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate
        )