"""检查点到访分析的单元测试。

可直接运行：python3 tests/test_checkpoint.py
或使用 pytest：python3 -m pytest tests/
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoint import (
    DEFAULT_RADIUS_M,
    MAX_GAP_S,
    Checkpoint,
    find_checkpoint_visits,
)
from app.geo import EARTH_RADIUS_M, TrackPoint

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)

# 测试轨迹围绕 (31.0, 121.0)
HOME = (31.0, 121.0)
KY = math.radians(EARTH_RADIUS_M)                  # 每度纬度的米数
KX = KY * math.cos(math.radians(HOME[0]))          # 每度经度的米数
DEG_LAT_PER_M = 1.0 / KY


def make_seg(points):
    """points: [(lat, lon, 秒偏移), ...]"""
    return [
        TrackPoint(lat=la, lon=lo, ele=None, time=T0 + timedelta(seconds=s))
        for la, lo, s in points
    ]


def north(meters, base=HOME):
    """base 正北方向约 meters 米的位置。"""
    return (base[0] + meters * DEG_LAT_PER_M, base[1])


def east(meters, base=HOME):
    """base 正东方向约 meters 米的位置。"""
    return (base[0], base[1] + meters / KX)


def move(p0, p1, t0, t1, step):
    """从 p0 匀速移动到 p1 的分段（按 step 秒加密采样）。"""
    pts = []
    t = t0
    while t < t1:
        f = t / (t1 - t0)
        pts.append((p0[0] + (p1[0] - p0[0]) * f,
                    p0[1] + (p1[1] - p0[1]) * f, t))
        t += step
    pts.append((p1[0], p1[1], t1))
    return make_seg(pts)


def cp(radius=50.0, at=HOME, name=""):
    return Checkpoint(lat=at[0], lon=at[1], radius_m=radius, name=name)


def names(results):
    return [r.name for r in results]


def test_passing_through_enter_leave_times():
    """匀速东西向穿过圆心：一次到访，进出时间与解析解一致。"""
    seg = move(east(-200), east(200), 0, 400, step=100)
    res = find_checkpoint_visits([seg], [cp(50.0)])
    assert len(res) == 1
    visits = res[0].visits
    assert len(visits) == 1
    v = visits[0]
    # 速度 = 400 m / 400 s = 1 m/s，半径 50 m → 进入于 150 s、离开于 250 s
    assert abs((v.start_time - T0).total_seconds() - 150.0) < 1e-6
    assert abs((v.end_time - T0).total_seconds() - 250.0) < 1e-6
    assert abs(v.duration_s - 100.0) < 1e-6
    assert v.seg_index == 1


def test_both_endpoints_outside_but_crossing_detected():
    """相邻两点都在圆外但线段穿圆（两个采样点、单线段）：仍须识别。"""
    seg = make_seg([(east(-100)[0], east(-100)[1], 0),
                    (east(100)[0], east(100)[1], 200)])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    v = res[0].visits
    assert len(v) == 1
    assert abs((v[0].start_time - T0).total_seconds() - 50.0) < 1e-6
    assert abs((v[0].end_time - T0).total_seconds() - 150.0) < 1e-6


def test_near_miss_not_counted():
    """线段与圆保持距离（最近处 51 米 > 半径 50 米）：不到访。"""
    seg = make_seg([(north(51)[0], east(-100, north(51))[1], 0),
                    (north(51)[0], east(100, north(51))[1], 200)])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    assert res[0].visits == []


def test_grazing_path_counted():
    """最近处 49 米（< 半径 50 米）的擦边路径：识别为一次到访。"""
    seg = make_seg([(north(49)[0], east(-100, north(49))[1], 0),
                    (north(49)[0], east(100, north(49))[1], 200)])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    assert len(res[0].visits) == 1


def test_exact_tangent_quadratic_not_counted():
    """判别式恰为 0 的严格相切：圆内区间为空（相切不计）。"""
    from app.checkpoint import _inside_intervals
    # 沿 x 轴方向、起点 (0, 50)，半径 50：最近距离恰为半径
    assert _inside_intervals(0.0, 50.0, 100.0, 0.0, 50.0) == []


def test_start_and_end_inside_use_sample_endpoint_times():
    """起/止在圆内：到访端点取该连续记录的采样端点时间，不外推。"""
    seg = make_seg([
        (HOME[0], HOME[1], 0),
        (north(20)[0], north(20)[1], 60),
        (north(40)[0], north(40)[1], 120),
    ])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    v = res[0].visits
    assert len(v) == 1
    assert v[0].start_time == T0
    assert v[0].end_time == T0 + timedelta(seconds=120)
    assert abs(v[0].duration_s - 120.0) < 1e-6


def test_exit_and_reenter_two_visits():
    """离开圆后再次进入：合并连续圆内区间后得到两次独立到访。"""
    seg = make_seg([
        (HOME[0], HOME[1], 0),                       # 圆内
        (north(100)[0], north(100)[1], 100),         # 圆外
        (HOME[0], HOME[1], 200),                     # 圆内
        (north(100)[0], north(100)[1], 300),         # 圆外
    ])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    visits = res[0].visits
    assert len(visits) == 2
    # 每次到访 [圆内端点时间, 50 米边界插值时刻]：段速 100 m / 100 s = 1 m/s
    assert visits[0].start_time == T0
    assert abs((visits[0].end_time - T0).total_seconds() - 50.0) < 1e-6
    assert abs((visits[1].start_time - T0).total_seconds() - 150.0) < 1e-6
    assert abs((visits[1].end_time - T0).total_seconds() - 250.0) < 1e-6


def test_consecutive_inside_edges_merge():
    """连续采样线段都在圆内（共端点相接）：合并为一次到访。"""
    seg = make_seg([
        (HOME[0], HOME[1], 0),
        (north(10)[0], north(10)[1], 30),
        (north(20)[0], north(20)[1], 60),
        (north(30)[0], north(30)[1], 90),
    ])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    visits = res[0].visits
    assert len(visits) == 1
    assert visits[0].start_time == T0
    assert visits[0].end_time == T0 + timedelta(seconds=90)
    assert abs(visits[0].duration_s - 90.0) < 1e-6


def test_trkseg_boundary_not_merged():
    """不跨 trkseg 合并：两段在圆内但分段结束/重新开始，分别列出两次到访。"""
    segments = [
        make_seg([(HOME[0], HOME[1], 0), (HOME[0], HOME[1], 60)]),
        make_seg([(HOME[0], HOME[1], 120), (HOME[0], HOME[1], 180)]),
    ]
    res = find_checkpoint_visits(segments, [cp(50.0)])
    visits = res[0].visits
    assert len(visits) == 2
    assert visits[0].seg_index == 1
    assert visits[1].seg_index == 2
    assert abs(visits[0].duration_s - 60.0) < 1e-6
    assert abs(visits[1].duration_s - 60.0) < 1e-6


def test_gap_not_interpolated_or_merged():
    """超过 10 分钟的采样间隔：不构造线段、不插值，断档两侧分别到访。"""
    seg = make_seg([
        (HOME[0], HOME[1], 0),
        (HOME[0], HOME[1], 60),
        (HOME[0], HOME[1], 60 + MAX_GAP_S + 1),   # 断档
        (HOME[0], HOME[1], 60 + MAX_GAP_S + 61),
    ])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    visits = res[0].visits
    assert len(visits) == 2
    assert abs(visits[0].duration_s - 60.0) < 1e-6
    assert abs(visits[1].duration_s - 60.0) < 1e-6


def test_gap_exactly_10min_interpolated():
    """采样间隔恰好 10 分钟（600 秒）：不算断档，正常插值线段。"""
    # 从圆外到圆外、单线段穿过圆，耗时恰好 600 秒
    seg = make_seg([(east(-100)[0], east(-100)[1], 0),
                    (east(100)[0], east(100)[1], MAX_GAP_S)])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    visits = res[0].visits
    assert len(visits) == 1
    # 半程在圆内：[150, 450]（速度 1/3 m/s）
    assert abs((visits[0].start_time - T0).total_seconds() - 150.0) < 1e-6
    assert abs((visits[0].end_time - T0).total_seconds() - 450.0) < 1e-6


def test_zero_duration_not_listed():
    """仅瞬时接触（单点在圆内、前后线段均不穿入正长度区间）：不列出。"""
    # 圆内点与南北两个圆外点等距共线：两段都只是从圆外到圆内边界……
    # 构造真正的瞬时情形：圆内一个点，但进出线段在该点停留时长为 0
    # （采样点本身在圆内时，前后线段各给出 (0..t_exit) / (t_enter..1) 正区间，
    #  会合并为正时长到访；这里改为单点 trkseg：无线段 → 不到访）
    seg = make_seg([(HOME[0], HOME[1], 0)])
    res = find_checkpoint_visits([seg], [cp(50.0)])
    assert res[0].visits == []


def test_non_increasing_time_rejected():
    """trkseg 内时间未递增：抛出 ValueError 并指明分段与点序号。"""
    seg = make_seg([
        (HOME[0], HOME[1], 0),
        (HOME[0], HOME[1], 60),
        (HOME[0], HOME[1], 60),   # 零时差
    ])
    try:
        find_checkpoint_visits([seg], [cp(50.0)])
    except ValueError as exc:
        assert "未递增" in str(exc) and "第 2→3 个轨迹点" in str(exc)
    else:
        raise AssertionError("零时差应报明确错误")


def test_multiple_checkpoints_independent_and_ordered():
    """多个检查点独立分析，结果顺序与输入一致，未到访者为空列表与默认命名。"""
    seg = move(east(-300), east(300), 0, 600, step=100)  # 1 m/s 东西向
    c1 = cp(50.0, at=HOME)                       # 穿过
    c2 = cp(50.0, at=north(500), name="北山脊")  # 不经过
    c3 = cp(50.0, at=east(200))                  # 末段穿过
    res = find_checkpoint_visits([seg], [c1, c2, c3])
    assert len(res) == 3
    assert names(res) == ["检查点 1", "北山脊", "检查点 3"]
    assert len(res[0].visits) == 1
    assert res[1].visits == []
    assert len(res[2].visits) == 1
    # c1 圆心 HOME：进入于 250 s、离开于 350 s
    v1 = res[0].visits[0]
    assert abs((v1.start_time - T0).total_seconds() - 250.0) < 1e-6
    assert abs((v1.end_time - T0).total_seconds() - 350.0) < 1e-6
    # c3 圆心在东侧 200 m：进入于 450 s、离开于 550 s
    v3 = res[2].visits[0]
    assert abs((v3.start_time - T0).total_seconds() - 450.0) < 1e-6
    assert abs((v3.end_time - T0).total_seconds() - 550.0) < 1e-6
    assert abs(v3.duration_s - 100.0) < 1e-6


def test_default_radius_constant():
    """默认半径 50 米常量。"""
    assert DEFAULT_RADIUS_M == 50.0


def test_does_not_mutate_track():
    """分析不修改原始轨迹点。"""
    seg = move(east(-200), east(200), 0, 400, step=100)
    before = [(p.lat, p.lon, p.time) for p in seg]
    find_checkpoint_visits([seg], [cp(50.0)])
    after = [(p.lat, p.lon, p.time) for p in seg]
    assert before == after


def test_empty_checkpoints():
    """无检查点：返回空结果列表。"""
    seg = make_seg([(HOME[0], HOME[1], 0), (HOME[0], HOME[1], 60)])
    assert find_checkpoint_visits([seg], []) == []


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
