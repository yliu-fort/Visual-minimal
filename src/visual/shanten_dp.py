# -*- coding: utf-8 -*-
from functools import lru_cache

# 牌表示：0-8m, 9-17p, 18-26s, 27-33z（字牌）
# hand: 长度34的数组，每项0..4
# remaining: 长度34的数组，牌山里剩余张数（可含副露信息修正）
# last_draw34: 丢弃前的“现摸牌”索引（0..33）；算法按题意先把它从手牌减1再评估
# 返回:
# {
#   "shanten": 最小向听（普通/七对/国士三者取最小）,
#   "ukeire":  受入合计张数,
#   "tiles":   list[(tile, cnt)] 逐张受入与剩余，tile为0..33,
#   "explain": {"best_mode": "normal|chiitoi|kokushi"}
# }
#
# 核心思路：
# - 普通手：分花色枚举(面子m, 搭子t, 对子p)的帕累托前沿，四类合并。
# - 普通手的改良牌：只测“邻域候选”（x, x±1, x±2）与已持有字牌；对每个候选，仅重算该花色DP并与其它花色的合并缓存拼装。
# - 七对、国士：无需试摸，直接推导改良集合。

# =========================
# 工具 & 常量
# =========================
# 经验：每项 ~1.4KB；10万项 ≈ 140MB；5万项 ≈ 70MB
SUIT_CACHE_MAX = 50_000     # ~70MB 量级
HONOR_CACHE_MAX = 10_000    # ~7-10MB 量级

def _suit_range(tile):
    """返回该tile所在花色的起止索引 [lo, hi] (含)，以及是否字牌"""
    if tile < 9: return (0, 8, False)
    if tile < 18: return (9, 17, False)
    if tile < 27: return (18, 26, False)
    return (27, 33, True)

def _is_same_suit(a, b):
    return _suit_range(a)[0] == _suit_range(b)[0]

def _neighbors_for_suit_index(idx_in_0_8):
    """给出同花色内的邻域候选索引（相对花色0..8），x, x±1, x±2 落在 0..8 内"""
    x = idx_in_0_8
    cand = {x}
    if x-1 >= 0: cand.add(x-1)
    if x-2 >= 0: cand.add(x-2)
    if x+1 <= 8: cand.add(x+1)
    if x+2 <= 8: cand.add(x+2)
    return cand

def _calc_shanten_from_mtp(m_total, t_total, pair_used):
    # 常见公式：sh = 8 - 2*m - min(4-m, t) - pair_used
    t_eff = min(4 - m_total, t_total)
    return 8 - 2*m_total - t_eff - (1 if pair_used else 0)

# =========================
# 花色DP：返回帕累托前沿的 (m,t,p) 状态集合
# =========================

def _pareto_prune(states):
    """去掉被支配的状态：m、t、p三个维度全不劣且至少一维更优的，保留前沿"""
    states = list(states)
    states.sort(reverse=True)  # 粗排减少比较
    kept = []
    for m,t,p in states:
        dominated = False
        for m2,t2,p2 in kept:
            if m2 >= m and t2 >= t and p2 >= p and (m2>m or t2>t or p2>p):
                dominated = True
                break
        if not dominated:
            kept.append((m,t,p))
    return kept

@lru_cache(maxsize=SUIT_CACHE_MAX)
def _enumerate_suit_states(counts_tuple):
    """
    对一个花色(9格)计数tuple，枚举所有 (m,t,p)：
      m: 面子数（刻/顺）
      t: 搭子数（12、23、13 这些两张搭子；不含对子）
      p: 对子数（用于后续合并时决定是否拿一个作雀头，其余对子可视为搭子）
    返回帕累托前沿集合 list[(m,t,p)]
    """
    counts = list(counts_tuple)

    @lru_cache(maxsize=None)
    def dfs(state_tuple):
        c = list(state_tuple)
        # 找到第一个非0位置
        i = next((k for k in range(9) if c[k] > 0), -1)
        if i == -1:
            return {(0,0,0)}  # 空
        res = set()

        # 1) 刻子
        if c[i] >= 3:
            c[i] -= 3
            for m,t,p in dfs(tuple(c)):
                res.add((m+1, t, p))
            c[i] += 3

        # 2) 顺子
        if i <= 6 and c[i+1] > 0 and c[i+2] > 0:
            c[i] -= 1; c[i+1] -= 1; c[i+2] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m+1, t, p))
            c[i] += 1; c[i+1] += 1; c[i+2] += 1

        # 3) 两面/边 & 嵌张搭子
        # (i,i+1)
        if i <= 7 and c[i+1] > 0:
            c[i] -= 1; c[i+1] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m, t+1, p))
            c[i] += 1; c[i+1] += 1
        # (i,i+2)
        if i <= 6 and c[i+2] > 0:
            c[i] -= 1; c[i+2] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m, t+1, p))
            c[i] += 1; c[i+2] += 1

        # 4) 对子（不立刻当雀头，先计为p，合并时决定是否用1个当雀头，其余当搭子）
        if c[i] >= 2:
            c[i] -= 2
            for m,t,p in dfs(tuple(c)):
                res.add((m, t, p+1))
            c[i] += 2

        # 5) 丢单张
        c[i] -= 1
        for m,t,p in dfs(tuple(c)):
            res.add((m, t, p))
        c[i] += 1

        return frozenset(_pareto_prune(res))

    return list(dfs(tuple(counts)))

@lru_cache(maxsize=HONOR_CACHE_MAX)
def _enumerate_honor_states(counts_tuple):
    """
    字牌(7格)版本：没有顺子，只有刻子/对子/单张
    返回帕累托前沿 list[(m,t,p)]，t来源于“未作为雀头的对子”可当搭子
    """
    counts = list(counts_tuple)

    @lru_cache(maxsize=None)
    def dfs(state_tuple, idx):
        if idx == 7:
            return {(0,0,0)}
        c = list(state_tuple)
        x = c[idx]
        res = set()

        # 刻子
        if x >= 3:
            c[idx] -= 3
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m+1, t, p))
            c[idx] += 3

        # 对子
        if x >= 2:
            c[idx] -= 2
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p+1))
            c[idx] += 2

        # 用掉1张
        if x >= 1:
            c[idx] -= 1
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p))
            c[idx] += 1
        else:
            # 0张，直接跳
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p))

        return frozenset(_pareto_prune(res))

    return list(dfs(tuple(counts), 0))

def _combine_four_groups(ms_states, ps_states, ss_states, z_states):
    """
    合并4类的帕累托，返回全局“最小普通手向听”及一个用于快速重算的缓存结构
    缓存包括：除去某一花色后的合并前缀，便于只重算单花色时快速拼回
    """
    # 把三门花色先两两合，最后合字牌；每步都做帕累托剪枝
    def merge(A, B):
        tmp = {}
        for m1,t1,p1 in A:
            for m2,t2,p2 in B:
                m = m1+m2; t = t1+t2; p = p1+p2
                # p里 >=1 可以当雀头，其余当搭子
                pair_used = 1 if p>0 else 0
                t_eff = t + max(0, p - pair_used)
                sh = _calc_shanten_from_mtp(min(m,4), t_eff, pair_used)
                # 使用 (m,t,p) 作为状态，先存最优sh，随后做剪枝
                key = (m,t,p)
                if key not in tmp or sh < tmp[key]:
                    tmp[key] = sh
        # 以 (m,t,p) 为维度的帕累托：按“计算最终sh的潜力”粗剪
        # 这里再转回列表，真正求最小sh时还要再遍历一次
        return list(tmp.keys())

    mp = merge(ms_states, ps_states)
    mps = merge(mp, ss_states)
    mpsz = merge(mps, z_states)

    # 计算全局最小普通手向听
    best_sh = +10**9
    for m,t,p in mpsz:
        pair_used = 1 if p>0 else 0
        t_eff = t + max(0, p - pair_used)
        sh = _calc_shanten_from_mtp(min(m,4), t_eff, pair_used)
        if sh < best_sh:
            best_sh = sh

    # 为“只改一门花色”做快速拼装缓存：
    # 预先把 “(另一门合并结果)”存起来，避免每次候选重算整套
    # 三门花色互相的合并结果：
    mp_cache = merge(ms_states, ps_states)     # m+p
    ps_cache = merge(ps_states, ss_states)     # p+s
    ms_cache = merge(ms_states, ss_states)     # m+s

    def merge_pair(A, B):
        """把两个已合并列表再合上去，返回 (m,t,p) 列表（不必保存sh）"""
        res = set()
        for m1,t1,p1 in A:
            for m2,t2,p2 in B:
                res.add((m1+m2, t1+t2, p1+p2))
        return list(res)

    # (m+p)+s
    mps_cache = merge_pair(mp_cache, ss_states)
    # (p+s)+m
    psm_cache = merge_pair(ps_cache, ms_states)
    # (m+s)+p
    msp_cache = merge_pair(ms_cache, ps_states)

    # 把三组合并上字牌
    other_than_m = merge_pair(ps_cache, z_states)  # (p+s)+z
    other_than_p = merge_pair(ms_cache, z_states)  # (m+s)+z
    other_than_s = merge_pair(mp_cache, z_states)  # (m+p)+z

    return best_sh, {
        "mpsz": mpsz,
        "other_than_m": other_than_m,
        "other_than_p": other_than_p,
        "other_than_s": other_than_s,
        "z_only": z_states,  # 方便仅改字牌时与(三门合并结果)拼
        "mp_cache": mp_cache,
        "ps_cache": ps_cache,
        "ms_cache": ms_cache
    }

def _best_sh_from_states(states):
    best = +10**9
    for m,t,p in states:
        pair_used = 1 if p>0 else 0
        t_eff = t + max(0, p - pair_used)
        sh = _calc_shanten_from_mtp(min(m,4), t_eff, pair_used)
        if sh < best: best = sh
    return best

# =========================
# 七对 & 国士：向听与改良集合
# =========================

def _chiitoi_shanten_and_improves(hand, remaining):
    pairs = sum(1 for c in hand if c >= 2)
    singles = sum(1 for c in hand if c == 1)
    distinct = pairs + singles
    # 七对向听
    sh = 6 - pairs + max(0, 7 - distinct)

    improve = set()
    if sh <= 6:
        # 1) 补成对子：已有1张的牌 -> 再来1张
        for t in range(34):
            if hand[t] == 1 and remaining[t] > 0 and hand[t] < 2:
                improve.add(t)
        # 2) 凑满7种：当 distinct < 7 时，任何“手里没有”的牌都能把 distinct +1
        if distinct < 7:
            for t in range(34):
                if hand[t] == 0 and remaining[t] > 0:
                    improve.add(t)
        # 注意：已有2+张的牌，再摸同张不会立刻降向听（只是浪费）
    return sh, improve

_TERMINALS_AND_HONORS = set([0,8,9,17,18,26] + list(range(27,34)))

def _kokushi_shanten_and_improves(hand, remaining):
    have = 0
    pair = 0
    need_set = []
    for t in _TERMINALS_AND_HONORS:
        if hand[t] > 0:
            have += 1
            if hand[t] >= 2:
                pair = 1
        else:
            need_set.append(t)
    sh = 13 - have - pair  # 13种一张+1对

    improve = set()
    if sh >= 0:
        if have < 13:
            # 缺的任意一张都能降向听
            for t in need_set:
                if remaining[t] > 0:
                    improve.add(t)
        if pair == 0:
            # 没有对时，任何已持有的幺九字再来一张也能降向听
            for t in _TERMINALS_AND_HONORS:
                if hand[t] > 0 and remaining[t] > 0 and hand[t] < 4:
                    improve.add(t)
    return sh, improve

# =========================
# 普通手：一次性DP→base，增量只改单花色→是否降向听
# =========================

def _normal_base_and_cache(hand):
    # 拆花色
    m_cnts = tuple(hand[0:9])
    p_cnts = tuple(hand[9:18])
    s_cnts = tuple(hand[18:27])
    z_cnts = tuple(hand[27:34])

    ms = _enumerate_suit_states(m_cnts)
    ps = _enumerate_suit_states(p_cnts)
    ss = _enumerate_suit_states(s_cnts)
    zs = _enumerate_honor_states(z_cnts)

    base_sh, cache = _combine_four_groups(ms, ps, ss, zs)
    cache["ms"] = ms; cache["ps"] = ps; cache["ss"] = ss; cache["zs"] = zs
    return base_sh, cache

def _normal_candidate_tiles(hand, remaining):
    cand = set()
    # 三门花色：对每张在手牌中的格子，加入 x, x±1, x±2（同花色）
    for base in (0,9,18):
        arr = hand[base:base+9]
        for i, c in enumerate(arr):
            if c == 0: continue
            for j in _neighbors_for_suit_index(i):
                t = base + j
                if hand[t] < 4 and remaining[t] > 0:
                    cand.add(t)
    # 字牌：只有已持有的才可能立刻改善（新摸一张只能变对子/刻子）
    for t in range(27,34):
        if hand[t] > 0 and hand[t] < 4 and remaining[t] > 0:
            cand.add(t)
    return cand

def _recompute_normal_with_one_tile_added(hand, cache, add_tile):
    """只改一个花色的DP，快速拼回全局最小普通手向听"""
    lo, hi, is_honor = _suit_range(add_tile)
    if is_honor:
        # 重算字牌DP
        z_cnts = list(hand[27:34])
        z_cnts[add_tile-27] += 1
        zs2 = _enumerate_honor_states(tuple(z_cnts))
        # 其它三门已合并缓存： (m+p)+s
        mps_cache = cache["other_than_z"] if "other_than_z" in cache else None
        if mps_cache is None:
            # 现算一次并挂到cache上，供后续复用
            mps_cache = []
            for a in _combine_four_groups(cache["ms"], cache["ps"], cache["ss"], [(0,0,0)])[1]["mpsz"]:
                mps_cache.append(a)
            cache["other_than_z"] = mps_cache
        merged = []
        for m1,t1,p1 in mps_cache:
            for m2,t2,p2 in zs2:
                merged.append((m1+m2, t1+t2, p1+p2))
        return _best_sh_from_states(merged)

    # 是三门花色之一
    base = lo
    arr = list(hand[lo:hi+1])
    arr[add_tile-lo] += 1
    states2 = _enumerate_suit_states(tuple(arr))

    # 取“其余三类合并上字牌”的缓存，并与 states2 拼
    if base == 0:   # 改m
        other = cache["other_than_m"]  # (p+s)+z
    elif base == 9: # 改p
        other = cache["other_than_p"]  # (m+s)+z
    else:           # 改s
        other = cache["other_than_s"]  # (m+p)+z

    merged = []
    for m1,t1,p1 in states2:
        for m2,t2,p2 in other:
            merged.append((m1+m2, t1+t2, p1+p2))
    return _best_sh_from_states(merged)

# =========================
# 主入口
# =========================

def compute_ukeire_advanced(hand, last_draw34, remaining):
    """
    进阶版受入计算（普通/七对/国士三合一），只对普通手做“邻域候选+单花色重算”。
    """
    # 先把“现摸”打出
    hand = hand[:]  # 复制以免原地污染
    if last_draw34 is not None:
        hand[last_draw34] -= 1

    # 1) 普通手 base + cache
    normal_sh, cache = _normal_base_and_cache(hand)

    # 为字牌的“其它三门+字牌=？”缓存 one-shot 填好
    # 方便之后字牌重算时使用
    if "other_than_z" not in cache:
        mps_cache = []
        for a in _combine_four_groups(cache["ms"], cache["ps"], cache["ss"], [(0,0,0)])[1]["mpsz"]:
            mps_cache.append(a)
        cache["other_than_z"] = mps_cache

    # 2) 七对 & 国士
    chiitoi_sh, chiitoi_improve = _chiitoi_shanten_and_improves(hand, remaining)
    kokushi_sh, kokushi_improve = _kokushi_shanten_and_improves(hand, remaining)

    base_sh_global = min(normal_sh, chiitoi_sh, kokushi_sh)

    # 3) 普通手候选（邻域裁剪 → 极少枚数）
    normal_cands = _normal_candidate_tiles(hand, remaining)

    # 4) 汇总三种路线的改良牌集合（tile -> 是否降向听）
    improve_tiles = set()
    # 4.1 七对
    if chiitoi_sh == base_sh_global:
        improve_tiles |= chiitoi_improve
    # 4.2 国士
    if kokushi_sh == base_sh_global:
        improve_tiles |= kokushi_improve
    # 4.3 普通手（只测邻域候选）
    if normal_sh == base_sh_global:
        for t in normal_cands:
            if hand[t] >= 4 or remaining[t] <= 0:
                continue
            new_sh = _recompute_normal_with_one_tile_added(hand, cache, t)
            if new_sh < normal_sh:
                improve_tiles.add(t)

    # 5) 统计受入
    tiles_list = sorted((t, remaining[t]) for t in improve_tiles if remaining[t] > 0)
    ukeire = sum(cnt for _, cnt in tiles_list)
    mode = "normal" if base_sh_global == normal_sh else ("chiitoi" if base_sh_global == chiitoi_sh else "kokushi")
    return {
        "shanten": base_sh_global - (((14 - sum(hand))//3)*2),
        "ukeire": ukeire,
        "tiles": tiles_list,
        "explain": {"best_mode": mode, 
                    "shanten_regular": normal_sh, 
                    "shanten_chiitoi": chiitoi_sh, 
                    "shanten_kokushi": kokushi_sh}
    }

# =========================
# 示例：
# =========================
if __name__ == "__main__":
    hand = [0]*34
    # m: 123 456 678  -> 9张
    hand[0]+=1;hand[1]+=1;hand[2]+=1
    hand[3]+=1;hand[4]+=1;hand[5]+=1
    hand[6]+=1;hand[7]+=1;hand[8]+=1
    # p: 55          -> 2张
    hand[9+4] += 2
    # s: 23          -> 2张
    hand[18+1] += 1; hand[18+2] += 1
    # z: 白          -> 1张（索引 31；27:东 28:南 29:西 30:北 31:白 32:发 33:中）
    hand[27+4] += 1

    # 总数 9 + 2 + 2 + 1 = 14 张（包含现摸）
    assert sum(hand) == 14

    last_draw = 9 + 4  # 刚摸的是 5p（索引13）
    remaining = [4]*34
    remaining[9+4] = 2  # 牌山中 5p 还剩两张
    res = compute_ukeire_advanced(hand, last_draw, remaining)
    print(res, sum(hand))
