"""
Reusable PyTorch feature extractor for Japanese Riichi Mahjong
----------------------------------------------------------------

This module builds a (C x 34 x 34) tensor suitable for a ResNet-style
model that predicts the *next discard* for the current player (ego view).

It implements the **Baseline (~20 channels)** spec from the previous
message, plus a few handy utilities and masks.

You can extend it by plugging in custom calculators (e.g. shanten, ukeire)
via the hooks in `ExtraCalcs`.

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple
import math
import torch
from shanten_dp import compute_ukeire_advanced


# ----------------------------
# Tile system helpers (34-tile)
# ----------------------------
# Indexing convention:
# 0..8:  m1-m9 (characters/manzu)
# 9..17: p1-p9 (dots/pinzu)
# 18..26: s1-s9 (bamboo/souzu)
# 27..33: winds+dragons [East,South,West,North,White,Green,Red]

NUM_TILES = 34
WIDTH = 1
NUM_FEATURES = 128
RIVER_LEN = 24
HAND_LEN = 14
DORA_MAX = 5

# Red fives are tracked via flags, not separate indices.


def is_numbered(tile: int) -> bool:
    return 0 <= tile <= 26


def suit_of(tile: int) -> Optional[int]:
    """Return suit id for numbered tiles: 0=m,1=p,2=s ; None for honors."""
    if 0 <= tile <= 8:
        return 0
    if 9 <= tile <= 17:
        return 1
    if 18 <= tile <= 26:
        return 2
    return None


def rank_of(tile: int) -> Optional[int]:
    """Return rank 1..9 for numbered tiles; None for honors."""
    if 0 <= tile <= 8:
        return tile + 1
    if 9 <= tile <= 17:
        return tile - 9 + 1
    if 18 <= tile <= 26:
        return tile - 18 + 1
    return None


def next_rank_wrap(r: int) -> int:
    return 1 if r == 9 else r + 1


def indicator_to_dora(ind: int) -> int:
    """Map a dora indicator tile index -> actual dora tile index (34-tile space).
    Honors wrap E->S->W->N->E and W->G->R->W for dragons.
    """
    if 0 <= ind <= 26:  # numbered
        s = suit_of(ind)
        r = rank_of(ind)
        assert s is not None and r is not None
        r2 = next_rank_wrap(r)
        base = 0 if s == 0 else 9 if s == 1 else 18
        return base + (r2 - 1)
    # honors
    if 27 <= ind <= 30:  # winds E,S,W,N
        return 27 + ((ind - 27 + 1) % 4)
    if 31 <= ind <= 33:  # dragons: Wht->Grn->Red->Wht
        return 31 + ((ind - 31 + 1) % 3)
    raise ValueError("Invalid tile index")


# ----------------------------
# Data schema
# ----------------------------
@dataclass
class PlayerPublic:
    """
    Public/visible state for one opponent (or self where applicable).
    All counts are in the 34-tile space.
    """
    river: List[int] = field(default_factory=list)  # discard order as 34-idx
    river_counts: Optional[Sequence[int]] = None   # length 34 (optional; built if None)
    meld_counts: Optional[Sequence[int]] = None    # length 34 (exposed tiles from chi/pon/kan)
    riichi: bool = False
    riichi_turn: int = 0  # turn number when riichi declared (if any)
    score: int = 0
    rank: int = -1


@dataclass
class ExtraCalcs:
    """
    Optional calculators. Provide callable attributes to enrich features
    without changing this file.
    - visible_count_hook(state) -> List[int] length 34 (override default)
    - remaining_count_hook(state) -> List[int] length 34
    """
    visible_count_hook: Optional[callable] = None
    remaining_count_hook: Optional[callable] = None


@dataclass
class RiichiState:
    """Snapshot at the exact moment before the current player's discard."""
    # Ego (current player)
    hand_counts: Sequence[int]                 # length 34, 0..4
    meld_counts_self: Optional[Sequence[int]] = None  # exposed tiles only (chi/pon/kan)
    riichi: bool = False
    river_self: List[int] = field(default_factory=list)
    river_self_counts: Optional[Sequence[int]] = None

    # Opponents (relative seats: Left, Across/Center, Right)
    left: PlayerPublic = field(default_factory=PlayerPublic)
    across: PlayerPublic = field(default_factory=PlayerPublic)
    right: PlayerPublic = field(default_factory=PlayerPublic)

    # Round/seat
    round_wind: int = 0        # 0:E,1:S,2:W,3:N
    seat_wind_self: int = 0    # 0:E,1:S,2:W,3:N
    dealer_self: bool = False

    # Progress & sticks
    turn_number: int = 0       # approx 0..24 (normalize later)
    honba: int = 0
    riichi_sticks: int = 0

    # Score and rank
    score: int = 0
    rank: int = -1

    # Dora
    dora_indicators: Sequence[int] = field(default_factory=list)
    aka5m: bool = False
    aka5p: bool = False
    aka5s: bool = False

    # Legal actions (optional)
    legal_discards_mask: Optional[Sequence[int]] = None  # len 34, 0/1
    legal_actions_mask: Optional[Sequence[int]] = None  # len 253, 0/1

    # Last droped tiles (for naki)
    last_draw_136: int = -1
    last_discarded_tile_136: int = -1
    last_discarder: int = -1

    # Computed features
    visible_counts: Sequence[int] = None
    remaining_counts: Sequence[int] = None
    shantens: Sequence[int] = None
    ukeires: Sequence[int] = None

    # Hooks / extra calculators
    extra: ExtraCalcs = field(default_factory=ExtraCalcs)


# ----------------------------
# Feature Extractor
# ----------------------------
class RiichiResNetFeatures(torch.nn.Module):
    """Builds a (C, 34, 34) tensor from `RiichiState`.

    Channels implemented (Baseline ~20 + a few small extras):
        1  hand_count (/4)
        2  hand_mask
        3  meld_self (/4)
        4  riichi_self (always 0; provided for symmetry/extension)
        5-7   river_count_{L,C,R} (/4)
        8-79 river sequences
        80-82  meld_{L,C,R} (/4)
        83-85 riichi_{L,C,R}
        86-89 round_wind one-hot (4ch)
        90-93 seat_wind_self one-hot (4ch)
        94 dealer_flag
        95 turn_number (/24)
        96 honba (/5)
        97 riichi_sticks (/5)
        98 dora_flag (any tile that is current dora)
        99 dora_indicator_mark
        100 aka5 flags for m/p/s
        101 legal_discard_mask (if provided, else derived from hand_count>0)

        --- intermediate features ---
        102 is_in_any_tuitsu?  (>=2 same tile)
        103 is_in_any_triplet? (>=3 same tile)
        104 is_in_any_taatsu?  (1, 2)
        105 is_in_any_shuntsu? (1, 2 ,3)
        106 is_surplus1? (remove all melds and taatsu, ascending order)
        107 is_surplus2? (remove all melds and taatsu, descending order)
        108 shanten_normal (global, /8)
        109 shanten_chiitoitsu (global, /6)
        110 shanten_kokushi (global, /13)
        111 ukeire_count (global, /60)
        112-114 genbutsu_to_{L,C,R} (per-tile)
        115 tile_4visible_flag (per-tile)
        116 tile_3visible_flag (per-tile)
        117 tile_2visible_flag (per-tile)
        118 dora_count_hand (global, /5) # in both hand and melds
        119-121 visible_dora_hand_{L,C,R} (global, /5) # in melds only
        122 visible_dora_total (global, /10)
        123 furiten_self (global, {0/1})
        124-126 riichi_turn_{L,C,R} (global, /24)

        # 127 possible discards & its shantens
        # 128 possible discards & its ukeires

    Total: 128 channels

    Notes:
      - Rivers can be provided as ordered lists; if `river_counts` is None
        we compute counts from the river list.
      - All per-tile features are broadcast along width to shape (34,34).
      - Global scalars are written as constant planes.
    """

    def __init__(self,
                 max_turns: int = 24,
                 max_sticks: int = 5,
                 use_constant_width: bool = True):
        super().__init__()
        self.max_turns = max_turns
        self.max_sticks = max_sticks
        self.use_constant_width = use_constant_width  # keep 34x34 canvas

    # ---------- utilities ----------
    @staticmethod
    def _to_tensor_1d(arr: Sequence[int], dtype=torch.float32) -> torch.Tensor:
        t = torch.as_tensor(arr, dtype=dtype)
        if t.numel() != NUM_TILES:
            raise ValueError(f"Expected length {NUM_TILES}, got {t.numel()}")
        return t

    @staticmethod
    def _broadcast_row(v: torch.Tensor) -> torch.Tensor:
        # v: (34,)
        return v.view(NUM_TILES, 1).expand(NUM_TILES, WIDTH)

    @staticmethod
    def _const_plane(val: float) -> torch.Tensor:
        return torch.full((NUM_TILES, WIDTH), float(val))

    @staticmethod
    def _one_hot_plane(index: int, num: int) -> torch.Tensor:
        # one-hot across 0..num-1 as constant planes stacked
        planes = []
        for i in range(num):
            planes.append(RiichiResNetFeatures._const_plane(1.0 if i == index else 0.0))
        return torch.stack(planes, dim=0)  # (num,34,34)

    @staticmethod
    def _counts_from_river(river: Sequence[int]) -> List[int]:
        counts = [0]*NUM_TILES
        for t in river:
            counts[t] += 1
        return counts

    @staticmethod
    def _one_hot_tile(t_34: int) -> List[int]:
        counts = [0]*NUM_TILES
        counts[t_34] += 1
        return counts
    
    @staticmethod
    def _default_visible_counts(state: RiichiState) -> List[int]:
        counts = [0] * NUM_TILES
        for i, c in enumerate(state.hand_counts):
            counts[i] += int(c)
        if state.meld_counts_self is not None:
            for i, c in enumerate(state.meld_counts_self):
                counts[i] += int(c)
        rc_self = state.river_self_counts if state.river_self_counts is not None else RiichiResNetFeatures._counts_from_river(state.river_self)
        for i, c in enumerate(rc_self):
            counts[i] += int(c)
        for opp in [state.left, state.across, state.right]:
            rc = opp.river_counts if opp.river_counts is not None else RiichiResNetFeatures._counts_from_river(opp.river)
            mc = opp.meld_counts or [0]*NUM_TILES
            for i, c in enumerate(rc):
                counts[i] += int(c)
            for i, c in enumerate(mc):
                counts[i] += int(c)
        for ind in state.dora_indicators:
            if 0 <= ind < NUM_TILES:
                counts[ind] += 1
        return [min(4, c) for c in counts]

    @staticmethod
    def _surplus_mask(counts: Sequence[int], ascending: bool) -> torch.Tensor:
        c = [int(x) for x in counts]
        for i in range(NUM_TILES):
            c[i] %= 3
        rng = range(NUM_TILES-2) if ascending else range(NUM_TILES-3, -1, -1)
        for i in rng:
            if suit_of(i) is None or i % 9 > 6:
                continue
            m = min(c[i], c[i+1], c[i+2])
            if m > 0:
                c[i] -= m; c[i+1] -= m; c[i+2] -= m
        rng2 = range(NUM_TILES) if ascending else range(NUM_TILES-1, -1, -1)
        for i in rng2:
            if suit_of(i) is None:
                continue
            for d in (1, 2):
                j = i + d if ascending else i - d
                if 0 <= j < NUM_TILES and suit_of(j) == suit_of(i):
                    m = min(c[i], c[j])
                    if m > 0:
                        c[i] -= m; c[j] -= m
        for i in range(NUM_TILES):
            c[i] %= 2
        return torch.tensor([1.0 if v > 0 else 0.0 for v in c], dtype=torch.float32)

    @staticmethod
    def _get_possible_moves(hand_34, remaining):
        # Return list of discards that drop shanten and maximize ukeire.
        # If none can lower shanten further than others, return those with max ukeire.
        shantens = [8]*NUM_TILES
        ukeires = [0]*NUM_TILES

        for i, cnt in enumerate(hand_34):
            if cnt <= 0:
                continue
            out = compute_ukeire_advanced(hand_34, i, remaining)
            s = out.get("shanten", 1_000_000)
            u = out.get("ukeire", -1)

            shantens[i] = s
            ukeires[i] = u

        return RiichiResNetFeatures._to_tensor_1d(shantens), RiichiResNetFeatures._to_tensor_1d(ukeires)

    # ---------- core ----------
    def forward(self, state: RiichiState) -> Dict[str, torch.Tensor]:
        """Construct feature planes for a given :class:`RiichiState`.

        This implementation aims to be reasonably fast as feature extraction is
        executed for every step in self-play or dataset generation.  The focus
        here is to keep the operations vectorised and avoid Python level loops
        where possible.
        """
        planes: List[torch.Tensor] = []

        # 1) Self hand
        hand = self._to_tensor_1d(state.hand_counts)
        hand_clamped = hand.clamp(min=0, max=4)
        hand_mask = (hand_clamped > 0).float()
        planes.append(self._broadcast_row(hand_clamped / 4.0))                  # 1 hand_count
        planes.append(self._broadcast_row(hand_mask))                           # 2 hand_mask

        meld_self = self._to_tensor_1d(state.meld_counts_self or [0]*NUM_TILES)
        planes.append(self._broadcast_row(meld_self.clamp(0, 4) / 4.0))         # 3 meld_self
        planes.append(self._const_plane(0.0))                                    # 4 riichi_self placeholder

        # 2) Opponents public (L, C, R)
        opps = [state.left, state.across, state.right]
        for idx, opp in enumerate(opps):
            river_counts = opp.river_counts
            if river_counts is None:
                river_counts = self._counts_from_river(opp.river)
            river = self._to_tensor_1d(river_counts)
            planes.append(self._broadcast_row(river.clamp(0, 4) / 4.0))          # 5-7 river_count
        
            # Add river 24C x 3 opps
            for irc in range(RIVER_LEN):
                if irc < len(opp.river):
                    planes.append(self._broadcast_row(self._to_tensor_1d(self._one_hot_tile(opp.river[irc]))))
                else:
                    planes.append(self._const_plane(0.0))                        # 8-79 rivers


        for opp in opps:
            meld_counts = opp.meld_counts or [0]*NUM_TILES
            meld = self._to_tensor_1d(meld_counts)
            planes.append(self._broadcast_row(meld.clamp(0, 4) / 4.0))           # 80-82 meld_{L,C,R}

        for opp in opps:
            planes.append(self._const_plane(1.0 if opp.riichi else 0.0))         # 83-85 riichi flags

        # 3) Round/seat
        planes.extend(list(self._one_hot_plane(state.round_wind, 4)))            # 86-89 round wind OH
        planes.extend(list(self._one_hot_plane(state.seat_wind_self, 4)))        # 90-93 seat wind OH
        planes.append(self._const_plane(1.0 if state.dealer_self else 0.0))      # 94 dealer flag

        # 4) Progress & sticks (normalized and clipped)
        tn = min(max(int(state.turn_number), 0), self.max_turns)
        planes.append(self._const_plane(tn / float(self.max_turns)))             # 95 turn_number
        hb = min(max(int(state.honba), 0), self.max_sticks)
        rs = min(max(int(state.riichi_sticks), 0), self.max_sticks)
        planes.append(self._const_plane(hb / float(self.max_sticks)))            # 96 honba
        planes.append(self._const_plane(rs / float(self.max_sticks)))            # 97 riichi_sticks

        # 5) Dora related
        is_dora = torch.zeros(NUM_TILES, dtype=torch.float32)
        ind_mark = torch.zeros(NUM_TILES, dtype=torch.float32)
        for ind in state.dora_indicators:
            try:
                d = indicator_to_dora(ind)
                is_dora[d] = 1.0
                ind_mark[ind] = 1.0
            except Exception:
                continue
        planes.append(self._broadcast_row(is_dora))                              # 98 dora flag
        planes.append(self._broadcast_row(ind_mark))                             # 99 indicator mark

        # 6) Aka5 flags (optional)
        aka = torch.zeros(NUM_TILES, dtype=torch.float32)
        if state.aka5m:
            aka[4] = 1.0   # m5 index = 0+ (5-1) = 4
        if state.aka5p:
            aka[13] = 1.0  # p5 index = 9+ (5-1) = 13
        if state.aka5s:
            aka[22] = 1.0  # s5 index = 18+ (5-1) = 22
        planes.append(self._broadcast_row(aka))                                  # 100 (packed as one plane)

        # 7) Legal discard mask (hand>0 by default)
        if state.legal_discards_mask is not None:
            legal = self._to_tensor_1d(state.legal_discards_mask)
        else:
            legal = hand_mask
        planes.append(self._broadcast_row(legal))                                # 101 legal mask

        # --- Intermediate features ---
        if state.visible_counts:
            visible_counts = state.visible_counts
        else:
            visible_counts = self._default_visible_counts(state)
        if state.remaining_counts:
            remaining_counts = state.remaining_counts
        else:
            remaining_counts = [max(0, 4 - v) for v in visible_counts]
        visible_tensor = torch.as_tensor(visible_counts, dtype=torch.float32)
        remaining_tensor = torch.as_tensor(remaining_counts, dtype=torch.float32)

        planes.append(self._broadcast_row((hand_clamped >= 2).float()))          # 102 tuitsu
        planes.append(self._broadcast_row((hand_clamped >= 3).float()))          # 103 triplet

        # Vectorised taatsu (open-ended shapes) detection per suit
        taatsu = torch.zeros(NUM_TILES, dtype=torch.float32)
        shuntsu = torch.zeros(NUM_TILES, dtype=torch.float32)
        for base in (0, 9, 18):  # m, p, s suits
            h = hand_mask[base:base+9]
            if h.sum() == 0:
                continue

            # Taatsu: pairs with gap 1 or 2
            mask = torch.zeros(9, dtype=torch.float32)
            pair1 = h[:-1] * h[1:]
            pair2 = h[:-2] * h[2:]
            mask[:-1] += pair1
            mask[1:] += pair1
            mask[:-2] += pair2
            mask[2:] += pair2
            taatsu[base:base+9] = (mask > 0).float()

            # Shuntsu: straight sequences
            sh_mask = torch.zeros(9, dtype=torch.float32)
            triplet = h[:-2] * h[1:-1] * h[2:]
            sh_mask[:-2] += triplet
            sh_mask[1:-1] += triplet
            sh_mask[2:] += triplet
            shuntsu[base:base+9] = (sh_mask > 0).float()

        planes.append(self._broadcast_row(taatsu))                               # 104 taatsu
        planes.append(self._broadcast_row(shuntsu))                              # 105 shuntsu

        planes.append(self._broadcast_row(self._surplus_mask(hand_clamped.tolist(), True)))   # 106 surplus1
        planes.append(self._broadcast_row(self._surplus_mask(hand_clamped.tolist(), False)))  # 107 surplus2

        # convert hand_clamped (tile counts) to a flat list of tile indices [0..33]
        hand_list = hand_clamped.to(torch.int64).tolist()
        last_draw = state.last_draw_136//4
        res = compute_ukeire_advanced(hand_list, last_draw, remaining_counts)
        sh_normal = res["explain"]["shanten_regular"]
        sh_chiitoi = res["explain"]["shanten_chiitoi"]
        sh_kokushi = res["explain"]["shanten_kokushi"]
        planes.append(self._const_plane(max(sh_normal, 0) / 8.0))                # 108 shanten normal
        planes.append(self._const_plane(max(sh_chiitoi, 0) / 6.0))               # 109 shanten chiitoi
        planes.append(self._const_plane(max(sh_kokushi, 0) / 13.0))              # 110 shanten kokushi

        ukeire = res['ukeire']
        ukeire_counts = [0] * NUM_TILES
        for t, cnt in res['tiles']:
            ukeire_counts[t] = cnt
        ukeire_counts = self._to_tensor_1d(ukeire_counts)
        planes.append(self._const_plane(min(ukeire, 60) / 60.0))                 # 111 ukeire count

        for opp in opps:
            rc = opp.river_counts if opp.river_counts is not None else self._counts_from_river(opp.river)
            gen = torch.tensor([1.0 if c > 0 else 0.0 for c in rc], dtype=torch.float32)
            planes.append(self._broadcast_row(gen))                              # 112-114 genbutsu

        planes.append(self._broadcast_row((visible_tensor >= 4).float()))        # 115 4 visible
        planes.append(self._broadcast_row((visible_tensor >= 3).float()))        # 116 3 visible
        planes.append(self._broadcast_row((visible_tensor >= 2).float()))        # 117 2 visible

        total_hand_meld = hand_clamped + meld_self
        dora_count = float((total_hand_meld * is_dora).sum().item())
        if state.aka5m and is_dora[4] == 0:
            dora_count += 1
        if state.aka5p and is_dora[13] == 0:
            dora_count += 1
        if state.aka5s and is_dora[22] == 0:
            dora_count += 1
        planes.append(self._const_plane(min(dora_count, 5) / 5.0))               # 118 dora count hand

        for opp in opps:
            meld = self._to_tensor_1d(opp.meld_counts or [0]*NUM_TILES)
            count = float((meld * is_dora).sum().item())
            planes.append(self._const_plane(min(count, 5) / 5.0))                # 119-121 visible dora in melds

        total_dora_visible = float((visible_tensor * is_dora).sum().item())
        if state.aka5m and is_dora[4] == 0:
            total_dora_visible += 1
        if state.aka5p and is_dora[13] == 0:
            total_dora_visible += 1
        if state.aka5s and is_dora[22] == 0:
            total_dora_visible += 1
        planes.append(self._const_plane(min(total_dora_visible, 10) / 10.0))      # 122 visible dora total

        furiten = 0
        # Need to choose a tile to discard first
        my_river_counts = self._to_tensor_1d(state.river_self_counts or self._counts_from_river(state.river_self))
        if torch.sum(ukeire_counts * my_river_counts).item() > 0:
            furiten = 1
        planes.append(self._const_plane(float(furiten)))                         # 123 furiten self

        for opp in opps:
            rt = opp.riichi_turn if opp.riichi_turn >= 0 else 0
            rt = max(0, min(rt, self.max_turns))
            planes.append(self._const_plane(rt / float(self.max_turns)))         # 124-126 riichi turn

        # --- Advanced features ---
        if state.shantens:
            shantens = self._to_tensor_1d(state.shantens).clamp(min=0, max=8)
            ukeires = self._to_tensor_1d(state.ukeires).clamp(min=0, max=60)
        else:
            hand_mask = (hand_clamped > 0).float()
            shantens, ukeires = RiichiResNetFeatures._get_possible_moves(hand_list, remaining_counts)
            shantens = shantens.clamp(min=0, max=8)
            ukeires = ukeires.clamp(min=0, max=60)
        planes.append(self._broadcast_row(shantens / 8.0)) # 127 possible discards & its shantens
        planes.append(self._broadcast_row(ukeires / 60.0)) # 128 possible discards & its ukeires

        # --- Features end ---
        x = torch.stack(planes, dim=0)  # (C,34,34)

        legal_actions = torch.as_tensor(state.legal_actions_mask)

        return {
            "x": x,                              # model input
            "legal_mask": legal_actions,                 # (253,)
            "meta": {
                "num_channels": x.shape[0],
                "spec": "baseline-128ch-253ac",
            },
        }


def get_action_index(t_34, type):
    """Map an action description to the flat action index used in ``print_all_actions``."""
    """t_34 can be the tile index or the (t_34, called_index) for chi. """
    action_type = type.lower() if isinstance(type, str) else type

    if action_type == "discard":
        return int(t_34)

    if action_type == "riichi":
        return 34 + int(t_34)

    if action_type == "chi":
        base, called = t_34
        base, called = int(base), int(called)
        suit = base // 9
        rank = base % 9
        offset = 68 + suit * 15
        if called == 0:
            local_a = rank + 1
            return offset + local_a
        elif called == 1:
            local_a = rank
            return offset + 8 + local_a
        elif called == 2:
            local_a = rank
            return offset + local_a
        raise ValueError(f"Unsupported chi shape: {t_34}")

    if action_type == "pon":
        base, called = t_34
        base = int(base)
        return 113 + base

    if action_type == "kan":
        base, called = t_34
        base = int(base)
        if called != None:
            return 147 + base
        else:
            return 215 + base

    if action_type == "chakan":
        base, called = t_34
        base = int(base)
        return 181 + base

    if action_type == "ryuukyoku":
        return 249

    if action_type == "ron":
        return 250

    if action_type == "tsumo":
        return 251

    if action_type in ("cancel", "pass"):
        return 252

    raise ValueError(f"Unsupported action type: {type}")


def get_action_from_index(i):
    # discard
    if i < 34:
        return (i, False)

    # riichi
    elif i < 68:
        return (i-34, True)

    # chi
    elif i < 113:
        pouts = []
        for r in range(3):
            for j in range(8):
                s = (r*9+j, r*9+j+1)
                pouts.append((s, True))
            for j in range(7):
                s = (r*9+j, r*9+j+2)
                pouts.append((s, True))
        return pouts[i-68]

    # pon
    elif i < 147:
        k = (i-113, i-113)
        return (k, True)

    # kan
    elif i < 181:
        k = (i-147, i-147, i-147)
        return (k, True)

    # chakan
    elif i < 215:
        k = (i-181,)
        return (k, True)

    # ankan
    elif i < 249:
        k = (i-215, i-215, i-215, i-215)
        return (k, True)

    # ryuukyoku
    elif i == 249:
        return (255, True)

    # ron
    elif i == 250:
        return (255, True)

    # tsumo
    elif i == 251:
        return (255, True)

    # cancel
    elif i == 252:
        return (255, False)

    else:
        return (-1, False)


def get_actions():
    pouts = []

    # discard
    for i in range(34):
        pouts.append((i, False))

    # riichi
    for i in range(34):
        pouts.append((i, True))

    # chi
    for r in range(3):
        for i in range(8):
            s = (r*9+i, r*9+i+1)
            pouts.append((s, True))
        for i in range(7):
            s = (r*9+i, r*9+i+2)
            pouts.append((s, True))

    # pon
    for i in range(34):
        k = (i, i)
        pouts.append((k, True))

    # kan
    for i in range(34):
        k = (i, i, i)
        pouts.append((k, True))

    # chakan
    for i in range(34):
        k = i
        pouts.append((k, True))

    # ankan
    for i in range(34):
        k = i
        pouts.append((k, True))

    # ryuukyoku
    pouts.append((255, True))

    # ron
    pouts.append((255, True))

    # tsumo
    pouts.append((255, True))

    # cancel
    pouts.append((255, False))

    return pouts


# ----------------------------
# Mini example / smoke test
# ----------------------------
if __name__ == "__main__":
    # Fabricate a tiny state and run the extractor
    state = RiichiState(
        hand_counts=[0]*NUM_TILES,
        meld_counts_self=[0]*NUM_TILES,
        left=PlayerPublic(river=[27, 5, 5, 31], riichi=False),
        across=PlayerPublic(river=[9, 10], riichi=True),
        right=PlayerPublic(river=[18], riichi=False),
        round_wind=0, seat_wind_self=1, dealer_self=False,
        turn_number=8, honba=1, riichi_sticks=0,
        dora_indicators=[3, 31], aka5m=True, aka5p=False, aka5s=True,
    )
    # Give the player a few tiles and legal mask
    state.hand_counts[0] = 1  # m1
    state.hand_counts[4] = 2  # m5
    state.hand_counts[27] = 1 # East
    state.legal_discards_mask = [1 if c>0 else 0 for c in state.hand_counts]
    state.legal_actions_mask = [1 if c<34 else 0 for c in range(253)]

    extractor = RiichiResNetFeatures()
    out = extractor(state)
    x = out["x"]
    legal = out["legal_mask"]
    print("Feature tensor:", x.shape, "channels=", x.shape[0])
    print("Legal mask", legal.shape, " sum:", out["legal_mask"].sum().item())
