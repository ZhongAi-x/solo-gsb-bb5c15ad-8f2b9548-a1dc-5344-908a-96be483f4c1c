"""检查点到访分析：找出轨迹进入每个检查点圆内的连续时间区间。

判定规则（每个检查点独立进行，不修改原始轨迹）：

- 按原始 trkseg 的相邻点构造采样线段：把线段端点投影到以检查点圆心为
  原点的局部等距圆柱平面（米），解 |a + t·(b−a)|² = r² 得到线段位于
  圆内的比例区间 t ∈ (0,1)，进出时间按线段比例在两端采样时间之间
  线性插值；即使两端都在圆外，只要线段穿越圆（判别式 > 0）也会识别；
  相切（判别式 = 0）只是瞬时接触，不计到访；
- 时间必须严格递增（零时差或倒序抛 ValueError）；相邻采样间隔超过
  10 分钟视为记录中断，断档处不构造线段，也不跨断档合并；
  trkseg 边界同样不插值、不合并；
- 每个检查点独立把相接（或重叠）的圆内区间合并为一次到访；
  轨迹起/止在圆内时，到访端点直接取该连续记录的端点（采样点）时间；
  仅列出正时长的到访，未到访的检查点返回空列表。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .geo import EARTH_RADIUS_M

DEFAULT_RADIUS_M = 50.0   # 检查点默认半径（米）
MIN_RADIUS_M = 1.0        # 最小半径（米）
MAX_RADIUS_M = 1000.0     # 最大半径（米）
MAX_CHECKPOINTS = 10      # 检查点数量上限
MAX_GAP_S = 600.0         # 记录中断阈值（秒）：超过 10 分钟不插值、不合并

_MERGE_EPS_S = 1e-3       # 区间相接判定的浮点容差（秒）


@dataclass
class Checkpoint:
    """一个圆形检查点。"""

    lat: float
    lon: float
    radius_m: float = DEFAULT_RADIUS_M
    name: str = ""


@dataclass
class CheckpointVisit:
    """一次到访：轨迹持续位于检查点圆内的合并后区间。"""

    seg_index: int          # 到访所在的原始 trkseg 序号（从 1 开始）
    start_time: datetime    # 进入时刻（线段与圆周交点的插值时刻；起点在圆内则取采样点时间）
    end_time: datetime      # 离开时刻（终点在圆内则取采样点时间）
    duration_s: float       # 停留秒数（正时长）


@dataclass
class CheckpointResult:
    """单个检查点的到访结果。"""

    name: str
    lat: float
    lon: float
    radius_m: float
    visits: list = field(default_factory=list)  # list[CheckpointVisit]，按时间先后排列


def _inside_intervals(ax: float, ay: float, dx: float, dy: float,
                      radius: float) -> list:
    """线段（圆心局部米制投影中，a 为端点、d=b−a 为方向）位于圆内的
    比例区间列表 [(t_enter, t_exit)]，t ∈ [0,1]；相切（判别式 = 0）不计。
    """
    A = dx * dx + dy * dy
    r2 = radius * radius
    if A == 0.0:
        # 零长度线段（相邻采样点重合的原地停留）：点在圆内则整段时长都算在圆内
        return [(0.0, 1.0)] if ax * ax + ay * ay < r2 else []
    B = 2.0 * (ax * dx + ay * dy)
    C = ax * ax + ay * ay - r2
    disc = B * B - 4.0 * A * C
    if C < 0.0:
        # 起点在圆内：至多一个 (0,1) 内的出口；判别式 ≤ 0 时整段都在圆内
        if disc <= 0.0:
            return [(0.0, 1.0)]
        t_exit = (-B + math.sqrt(disc)) / (2.0 * A)
        if t_exit <= 0.0:
            return []
        return [(0.0, min(1.0, t_exit))]
    # 起点在圆上或圆外：只有穿越（两个不同实根）才有正长度圆内区间
    if disc <= 0.0:
        return []  # 无交点或相切（相切不进入圆内，忽略）
    sq = math.sqrt(disc)
    t_enter = (-B - sq) / (2.0 * A)
    t_exit = (-B + sq) / (2.0 * A)
    if t_enter >= 1.0 or t_exit <= 0.0:
        return []
    return [(max(0.0, t_enter), min(1.0, t_exit))]


def _interp_time(a, b, t: float):
    """按线段比例 t 在两端采样时间之间线性插值。"""
    return a.time + (b.time - a.time) * t


def _checkpoint_visits(segments: list, cp: Checkpoint) -> list:
    """单个检查点的到访区间：逐 trkseg、逐采样线段求圆内区间并合并。"""
    ky = math.radians(EARTH_RADIUS_M)                  # 每度纬度的米数
    kx = ky * math.cos(math.radians(cp.lat))           # 每度经度的米数

    def xy(p):
        return (p.lon - cp.lon) * kx, (p.lat - cp.lat) * ky

    intervals = []  # [(seg_index, start_time, end_time)]
    for seg_idx, seg in enumerate(segments, start=1):
        for i, (a, b) in enumerate(zip(seg, seg[1:]), start=1):
            dt = (b.time - a.time).total_seconds()
            if dt <= 0:
                raise ValueError(
                    f"第 {seg_idx} 个 trkseg 中第 {i}→{i + 1} 个轨迹点的时间未递增"
                    "（零时差或倒序），无法分析检查点到访"
                )
            if dt > MAX_GAP_S:
                continue  # 记录中断：不构造线段、不插值
            ax, ay = xy(a)
            bx, by = xy(b)
            for t0, t1 in _inside_intervals(ax, ay, bx - ax, by - ay, cp.radius_m):
                intervals.append((
                    seg_idx,
                    a.time if t0 == 0.0 else _interp_time(a, b, t0),
                    b.time if t1 == 1.0 else _interp_time(a, b, t1),
                ))

    # 同一检查点的连续圆内区间合并：只在同一 trkseg 内相接（或重叠）时合并——
    # 相邻采样线段的圆内区间共端点（同一采样点在圆内）即相接；
    # 跨 trkseg 不合并，断档处两侧区间时间不连续也不会相接
    merged: list = []
    for seg_idx, t0, t1 in intervals:
        if (merged and merged[-1][0] == seg_idx
                and (t0 - merged[-1][2]).total_seconds() <= _MERGE_EPS_S):
            merged[-1] = (seg_idx, merged[-1][1], max(merged[-1][2], t1))
        else:
            merged.append((seg_idx, t0, t1))

    return [
        CheckpointVisit(
            seg_index=seg_idx,
            start_time=t0,
            end_time=t1,
            duration_s=(t1 - t0).total_seconds(),
        )
        for seg_idx, t0, t1 in merged
        if (t1 - t0).total_seconds() > 0.0  # 仅列出正时长到访
    ]


def find_checkpoint_visits(segments: list, checkpoints: list) -> list:
    """对每个检查点分别分析到访区间，返回 list[CheckpointResult]，顺序与输入一致。

    不修改原始轨迹；时间未递增时抛出 ValueError，由接口层转为明确提示。
    """
    results = []
    for idx, cp in enumerate(checkpoints, start=1):
        visits = _checkpoint_visits(segments, cp)
        results.append(CheckpointResult(
            name=cp.name or f"检查点 {idx}",
            lat=cp.lat,
            lon=cp.lon,
            radius_m=cp.radius_m,
            visits=visits,
        ))
    return results
