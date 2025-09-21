import json
import os, struct, numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import torch
from torchvision import transforms
import webdataset as wds
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple

from mahjong_features import RiichiResNetFeatures, RiichiState, PlayerPublic, NUM_TILES, RIVER_LEN, HAND_LEN, DORA_MAX


NUM_FEATURES = 128
DEFAULT_OUTPUT_DIR = os.path.join("output", "webdataset")


# states_iter 需要你提供一个迭代器/生成器，产生 dict 形式的 RiichiState（或直接改成你的对象字段）
# 写完后，用 webdataset 的 make_index 生成 .idx 文件（命令行或 python 调用均可）

# The second decode function
def decode_record(raw: bytes)->Tuple[RiichiState, int]:
    import numpy as np
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

    # Last tiles (uint16 little-endian with -1 offset)
    u16 = v[off:off+4].view(dtype="<u2")
    last_draw = int(u16[0]) - 1
    last_disc = int(u16[1]) - 1
    off += 4

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
        last_draw_136=last_draw,
        last_discarded_tile_136=last_disc,
        visible_counts=visible_counts,
        remaining_counts=remaining_counts,
        shantens=shantens,
        ukeires=ukeires,
    )

    return state, label

class DecodeHelper:
    _extractor = RiichiResNetFeatures()
    _transform = transforms.Compose([transforms.Resize(224)])
    _target_transform = None

    @staticmethod
    def apply(sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        state, label = decode_record(sample["bin"])

        with torch.no_grad():
            x = DecodeHelper._extractor(state)["x"]
        y = torch.asarray(label, dtype=torch.long)

        if DecodeHelper._transform:
            x = DecodeHelper._transform(x)
        if DecodeHelper._target_transform:
            y = DecodeHelper._target_transform(y)

        return x, y

    @staticmethod
    def apply_with_mask(sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, label = decode_record(sample["bin"])

        with torch.no_grad():
            out = DecodeHelper._extractor(state)
            x = out["x"]
            legal_mask = out["legal_mask"]
        y = torch.asarray(label, dtype=torch.long)

        if DecodeHelper._transform:
            x = DecodeHelper._transform(x)
        if DecodeHelper._target_transform:
            y = DecodeHelper._target_transform(y)

        return x, y, legal_mask
    
def make_loader(pattern, batch_size, num_workers=4, shard_shuffle=True, class_conditional=True, prefetch_factor=1, seed=42):
    # 在旧版 webdataset 兼容接口下，ResampledShards 需要通过 resampled=True 打开，否则会因类型断言失败
    webdataset_kwargs = {"seed": seed}
    if shard_shuffle:
        webdataset_kwargs["resampled"] = True
    else:
        webdataset_kwargs["shardshuffle"] = False
    ds = (
        wds.WebDataset(pattern, **webdataset_kwargs)
        .shuffle(2000)  # 轻度预热，先打散样本键
        .decode()       # 我们自己解码，不用自动解码器
        .map(DecodeHelper.apply_with_mask if class_conditional else DecodeHelper.apply)
        .shuffle(100000)  # 片内大缓冲区乱序（关键！）
        .batched(batch_size, partial=False)
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
    )
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
        os.path.join(root, "webdataset/train/discard/riichi-{000000..004270}.tar"),
        batch_size=batch_size,
        num_workers=num_workers,
        shard_shuffle=True,
        class_conditional = class_conditional,
        prefetch_factor=2
    )
    ds_tst = make_loader(
        os.path.join(root, "webdataset/test/discard/riichi-{000000..000042}.tar"),
        batch_size=batch_size,
        num_workers=num_workers,
        shard_shuffle=True,
        class_conditional = class_conditional,
        prefetch_factor=1
    )
    return ds, ds_tst

if __name__ == "__main__":
    loader = make_loader(
        "/data/MajhongEnv/output/webdataset/train/discard/riichi-{000000..004270}.tar",
        batch_size=4,
        num_workers=4,
        shard_shuffle=True,
    )
    for batch in loader:
        x, y, m = batch
        print(x.shape, y.shape, m.shape)