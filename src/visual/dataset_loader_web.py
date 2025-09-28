import json
import os, struct, numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import torch
import torch.nn.functional as F
from torchvision import transforms
import webdataset as wds
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple, List, Optional
from probe import probe_map

from mahjong_features import RiichiResNetFeatures, RiichiState, PlayerPublic, NUM_TILES, RIVER_LEN, HAND_LEN, DORA_MAX


NUM_FEATURES = 128
# Resize targets are anisotropic; historically the height 224 pairs with width 65.
RESIZE_BASE_HEIGHT = 224
RESIZE_BASE_WIDTH = 65
DEFAULT_OUTPUT_DIR = os.path.join("output", "webdataset")


# states_iter 需要你提供一个迭代器/生成器，产生 dict 形式的 RiichiState（或直接改成你的对象字段）
# 写完后，用 webdataset 的 make_index 生成 .idx 文件（命令行或 python 调用均可）

# The second decode function
def decode_record(raw: bytes)->Tuple[RiichiState, int, List]:
    v = np.frombuffer(raw, dtype=np.uint8)
    off = 0

    def take(n):
        nonlocal off
        a = v[off:off+n]
        off += n
        return a

    # Ego
    hand = take(NUM_TILES).astype(np.uint8)
    meld_self = take(NUM_TILES).astype(np.uint8)
    riichi_self = bool(take(1)[0])
    turn = int(take(1)[0])
    honba = int(take(1)[0])
    sticks = int(take(1)[0])
    dealer = bool(take(1)[0])

    round_wind = int(take(1)[0])
    seat_wind = int(take(1)[0])

    river_self_raw = take(RIVER_LEN)
    river_self = [int(x) for x in river_self_raw.tolist() if x != 255]

    # Opponents
    opps = []
    for _ in range(3):
        rv_raw = take(RIVER_LEN)
        rv = [int(x) for x in rv_raw.tolist() if x != 255]
        rflag = bool(take(1)[0])
        rturn = int(take(1)[0])
        meld = take(NUM_TILES).astype(np.uint8)
        opps.append((rv, rflag, rturn, meld))

    dora_raw = take(DORA_MAX)
    dora_indicators = [int(x) for x in dora_raw.tolist() if x != 255]
    aka_flags = int(take(1)[0])

    # Legal mask: 5 bytes (little-endian 40-bit), take lower 34 bits
    legal_bytes = bytes(take(5).tolist())
    bits = int.from_bytes(legal_bytes, "little")
    legal_mask = [(bits >> i) & 1 for i in range(NUM_TILES)]

    # Legal actions mask: 32 bytes (253 little-endian bits)
    legal_actions_bytes = take(32)
    legal_actions_mask_bits = np.unpackbits(legal_actions_bytes, bitorder="little")
    legal_actions_mask = legal_actions_mask_bits[:253].astype(int).tolist()

    # Last tiles (uint16 little-endian with -1 offset)
    u16 = v[off:off+6].view(dtype="<u2")
    last_draw = int(u16[0]) - 1
    last_disc = int(u16[1]) - 1
    last_disr = int(u16[2]) - 1
    off += 6

    # Build RiichiState
    left = PlayerPublic(
        river=opps[0][0],
        meld_counts=opps[0][3].astype(int).tolist(),
        riichi=opps[0][1],
        riichi_turn=opps[0][2],
    )
    across = PlayerPublic(
        river=opps[1][0],
        meld_counts=opps[1][3].astype(int).tolist(),
        riichi=opps[1][1],
        riichi_turn=opps[1][2],
    )
    right = PlayerPublic(
        river=opps[2][0],
        meld_counts=opps[2][3].astype(int).tolist(),
        riichi=opps[2][1],
        riichi_turn=opps[2][2],
    )

    visible_counts = take(NUM_TILES).tolist()
    remaining_counts = take(NUM_TILES).tolist()
    shantens = take(NUM_TILES).tolist()
    ukeires = take(NUM_TILES).tolist()

    label = int(take(1)[0])

    state = RiichiState(
        hand_counts=hand.astype(int).tolist(),
        meld_counts_self=meld_self.astype(int).tolist(),
        riichi=riichi_self,
        river_self=river_self,
        left=left,
        across=across,
        right=right,
        round_wind=round_wind,
        seat_wind_self=seat_wind,
        dealer_self=dealer,
        turn_number=turn,
        honba=honba,
        riichi_sticks=sticks,
        dora_indicators=dora_indicators,
        aka5m=bool(aka_flags & 0x1),
        aka5p=bool(aka_flags & 0x2),
        aka5s=bool(aka_flags & 0x4),
        legal_discards_mask=legal_mask,
        legal_actions_mask=legal_actions_mask,
        last_draw_136=last_draw,
        last_discarded_tile_136=last_disc,
        last_discarder=last_disr,
        visible_counts=visible_counts,
        remaining_counts=remaining_counts,
        shantens=shantens,
        ukeires=ukeires,
    )

    return state, label, legal_actions_mask



class DecodeHelper:
    _extractor: Optional[RiichiResNetFeatures] = None
    _transform = None
    _target_transform = None

    @staticmethod
    def ensure_initialized() -> RiichiResNetFeatures:
        if DecodeHelper._extractor is None:
            extractor = RiichiResNetFeatures()
            extractor.eval()
            DecodeHelper._extractor = extractor
        return DecodeHelper._extractor

    @staticmethod
    @torch.inference_mode()
    def _extract(state: RiichiState) -> torch.Tensor:
        extractor = DecodeHelper.ensure_initialized()
        features = extractor(state)["x"]
        return features.detach()[..., :1]

    @staticmethod
    def apply(sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        state, label, _ = decode_record(sample["bin"])

        x = DecodeHelper._extract(state)
        y = torch.as_tensor(label, dtype=torch.long)

        if DecodeHelper._transform:
            x = DecodeHelper._transform(x)
        if DecodeHelper._target_transform:
            y = DecodeHelper._target_transform(y)

        return x, y

    @staticmethod
    def apply_with_mask(sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, label, mask = decode_record(sample["bin"])

        x = DecodeHelper._extract(state)
        y = torch.as_tensor(label, dtype=torch.long)
        m = torch.as_tensor(mask, dtype=torch.long)

        if DecodeHelper._transform:
            x = DecodeHelper._transform(x)
        if DecodeHelper._target_transform:
            y = DecodeHelper._target_transform(y)

        return x, y, m
    
    @staticmethod
    def resize_batch(batch, size=RESIZE_BASE_HEIGHT):
        """Resize helper retained for backward compatibility."""
        xs, ys, *rest = batch                        # [B,C,H,W]
        x = resize_batch_on_device(xs, compute_resize_shape(size))
        if rest:
            return x, ys, *rest
        return x, ys


def compute_resize_shape(target_height: int) -> Tuple[int, int]:
    """Return the (height, width) pair used when upscaling feature maps."""
    width = int(round(target_height * RESIZE_BASE_WIDTH / RESIZE_BASE_HEIGHT))
    width = max(1, width)
    return target_height, width


def resize_batch_on_device(images: torch.Tensor, target_shape: Optional[Tuple[int, int]], mode: str = "nearest") -> torch.Tensor:
    """Resize a tensor batch on the caller's device if needed.

    Args:
        images: Tensor with shape ``(B, C, H, W)``.
        target_shape: ``(height, width)`` tuple. If ``None`` the tensor is returned as-is.
        mode: Interpolation mode (defaults to ``"nearest"`` to match previous behaviour).
    """

    if target_shape is None:
        return images

    height, width = target_shape
    if height <= 0 or width <= 0:
        raise ValueError("target_shape must contain positive dimensions")

    if images.dim() != 4:
        raise ValueError("images tensor must be 4D (B, C, H, W)")

    if images.shape[-2:] == (height, width):
        return images

    return F.interpolate(images, size=(height, width), mode=mode)


def _worker_init_fn(worker_id: int) -> None:
    """Initialise heavy objects once per worker and tame threading."""
    DecodeHelper.ensure_initialized()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
def make_loader(pattern, batch_size, num_workers=4, shard_shuffle=True, class_conditional=True, prefetch_factor=1, seed=42):
    # 在旧版 webdataset 兼容接口下，ResampledShards 需要通过 resampled=True 打开，否则会因类型断言失败
    webdataset_kwargs = {"seed": seed}
    if shard_shuffle:
        webdataset_kwargs["resampled"] = True
    else:
        webdataset_kwargs["shardshuffle"] = False
    # NOTE:
    #   The feature tensor for each sample is roughly 24.6 MB in float32.
    #   Using a massive shuffle buffer (100k) as before balloons memory
    #   usage to tens of GB per worker, which leads to the dataloader
    #   workers being OOM-killed.  Keep a reasonably large buffer for
    #   stochasticity, but cap it to something that scales with the
    #   batch size.
    sample_shuffle = min(65536, max(512, int(batch_size) * 32))

    ds = (
        wds.WebDataset(pattern, **webdataset_kwargs)
        .shuffle(2000)  # 轻度预热，先打散样本键, 单条样本非常便宜
        .decode()       # 我们自己解码，不用自动解码器
        .map(DecodeHelper.apply_with_mask if class_conditional else DecodeHelper.apply)
        #.map(probe_map)                    # ← 在这里测“单样本解码后”的体积～0.017 MB
        .shuffle(sample_shuffle)  # 片内大缓冲区乱序（关键！）
        .batched(batch_size, partial=False)
    )

    loader_kwargs = dict(batch_size=None, pin_memory=True)
    if torch.cuda.is_available():
        loader_kwargs["pin_memory_device"] = "cuda"

    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            worker_init_fn=_worker_init_fn,
        )
    else:
        loader_kwargs.update(num_workers=0)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
    return loader

def build_riichi_dataloader(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    class_conditional: bool,
    img_size: int,
    cf_guidance_p: float,
):
    ds = make_loader(
        os.path.join(root, "webdataset/train/discard/riichi-{000000..004035}.tar"),
        batch_size=batch_size,
        num_workers=num_workers,
        shard_shuffle=False,
        class_conditional = class_conditional,
        prefetch_factor=4
    )
    ds_tst = make_loader(
        os.path.join(root, "webdataset/test/discard/riichi-{000000..000004}.tar"),
        batch_size=batch_size,
        num_workers=1,
        shard_shuffle=False,
        class_conditional = class_conditional,
        prefetch_factor=1
    )
    return ds, ds_tst

if __name__ == "__main__":
    loader = make_loader(
        "../MajhongEnv/output/webdataset/train/discard/riichi-{000000..000007}.tar",
        batch_size=16,
        num_workers=1,
        shard_shuffle=True,
        class_conditional = True,
        prefetch_factor=2
    )
    for i, batch in enumerate(loader):
        x, y, m = batch
        print(i, x.shape, y.shape, m.shape)