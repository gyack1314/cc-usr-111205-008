"""多通道受限配准的编排与逐段解释报告。

本模块只在原始采样点之间建立对齐关系：水平/垂直步表示某一序列的采样点
被重复使用，绝不生成插值后的新观测；配准代价只描述两过程在锁定标准化
口径下的相似程度，不用于判断降雨因果。
"""
from fractions import Fraction

from .costs import build_local_costs
from .dp import enumerate_paths, minimum_sum_path, solve, solve_material

DECLARATIONS = (
    '路径只复用原始等间隔采样点，不构造插值后的新观测',
    '配准代价仅描述锁定口径下的过程相似性，不据配准好坏判断降雨因果',
    '局部代价只在双方均已观测的共同通道上计算，缺测不补零',
)


def _q(x):
    """有理量序列化：精确分数字符串加浮点近似；浮点量原样返回。"""
    if isinstance(x, Fraction):
        return {'exact': str(x), 'approx': float(x)}
    return x


def _reuse_counts(path, axis):
    """统计某一序列各采样点在路径中的出现次数，只回列重复使用项。"""
    counts = {}
    for cell in path:
        k = cell[axis]
        counts[k] = counts.get(k, 0) + 1
    return {k: c for k, c in sorted(counts.items()) if c > 1}


def _steps_and_segments(lc, result):
    """把点对路径翻译成逐步步法与伸缩分段解释。"""
    path = result.path
    states = result.states
    steps = []
    segments = []
    run_kind = None      # 'H' / 'V' / 'D'
    run_start = 0        # 分段起点在 path 中的位置
    for t in range(1, len(path)):
        i0, j0 = path[t - 1]
        i1, j1 = path[t]
        if (i1 - i0, j1 - j0) == (1, 1):
            kind, zh = 'diagonal', '斜步'
            explain = f'参考时刻 {i1} 与候选时刻 {j1} 同步对齐，两序列各推进一步'
        elif (j1 - j0) == 0:
            kind, zh = 'horizontal', '水平步'
            explain = (f'参考时刻 {i1} 对齐到候选时刻 {j1}：候选序列停留在 '
                       f'j={j1}，参考序列独自推进，候选采样点被重复使用')
        else:
            kind, zh = 'vertical', '垂直步'
            explain = (f'参考时刻 {i1} 对齐到候选时刻 {j1}：参考序列停留在 '
                       f'i={i1}，候选序列独自推进，参考采样点被重复使用')
        steps.append({'from': [i0, j0], 'to': [i1, j1],
                      'kind': kind, 'explanation': explain})
        cur = 'D' if kind == 'diagonal' else kind[0].upper()
        if cur != run_kind:
            if run_kind is not None:
                segments.append(_segment(lc, path, states, run_start,
                                         t - 1, run_kind))
            run_kind, run_start = cur, t - 1
    segments.append(_segment(lc, path, states, run_start,
                             len(path) - 1, run_kind))
    return steps, segments


def _segment(lc, path, states, a, b, kind):
    """汇总 path[a..b] 这一同类分段（a 为分段前的衔接点）。"""
    if kind == 'D':
        return {
            'kind': 'synchronized',
            'points': [list(c) for c in path[a:b + 1]],
            'explanation': (f'参考 i={path[a][0]}..{path[b][0]} 与候选 '
                            f'j={path[a][1]}..{path[b][1]} 逐点斜步同步，'
                            '两序列各推进一步')}
    if kind == 'H':
        j = path[a][1]
        return {
            'kind': 'candidate_held',
            'points': [list(c) for c in path[a:b + 1]],
            'held_candidate_index': j,
            'reused_reference_indices': [c[0] for c in path[a:b + 1]],
            'run_length': b - a,
            'explanation': (f'连续 {b - a} 个水平步：候选 j={j} 被复用 '
                            f'{b - a + 1} 次，对齐参考 i={path[a][0]}..'
                            f'{path[b][0]}，参考过程的这一段被“放慢”对齐')}
    i = path[a][0]
    return {
        'kind': 'reference_held',
        'points': [list(c) for c in path[a:b + 1]],
        'held_reference_index': i,
        'reused_candidate_indices': [c[1] for c in path[a:b + 1]],
        'run_length': b - a,
        'explanation': (f'连续 {b - a} 个垂直步：参考 i={i} 被复用 '
                        f'{b - a + 1} 次，对齐候选 j={path[a][1]}..'
                        f'{path[b][1]}，候选过程的这一段被“放慢”对齐')}


def _warp_summary(lc, result, steps, segments):
    n_diag = sum(1 for s in steps if s['kind'] == 'diagonal')
    n_h = sum(1 for s in steps if s['kind'] == 'horizontal')
    n_v = sum(1 for s in steps if s['kind'] == 'vertical')
    max_h = max((seg['run_length'] for seg in segments
                 if seg['kind'] == 'candidate_held'), default=0)
    max_v = max((seg['run_length'] for seg in segments
                 if seg['kind'] == 'reference_held'), default=0)
    return {
        'reference_points': lc.n,
        'candidate_points': lc.m,
        'path_points': result.length,
        'diagonal_steps': n_diag,
        'horizontal_steps': n_h,
        'vertical_steps': n_v,
        'max_horizontal_run': max_h,
        'max_vertical_run': max_v,
        'run_limit': lc.run_limit,
        'index_band': lc.index_band,
        'interpretation': (
            f'斜步 {n_diag} 次表示两序列同步推进；水平步 {n_h} 次表示候选'
            f'时刻被复用（参考独自推进），垂直步 {n_v} 次表示参考时刻被'
            f'复用（候选独自推进）。水平/垂直连续步均不超过 run_limit='
            f'{lc.run_limit}（实测最大 {max_h}/{max_v}），且所有点对满足'
            f' |i-j|<={lc.index_band}，伸缩强度因此受限，不能把长时间小雨'
            '任意拉伸成短时暴雨')}


def _pair_records(lc, result):
    records = []
    cumulative = Fraction(0) if lc.exact else 0.0
    for (i, j) in result.path:
        p = lc.pairs[(i, j)]
        cumulative += p.cost
        records.append({
            'i': i, 'j': j,
            'local_cost': _q(p.cost),
            'cumulative_cost': _q(cumulative),
            'weight_sum': _q(p.weight_sum),
            'effective_channels': list(p.channels),
            'missing_channels': [c for c in lc.names
                                 if c not in p.channels],
        })
    return records, cumulative


def _infeasible_reason(lc):
    """区分首末点缺支持与带/run_limit 造成的不可达。"""
    n, m = lc.n, lc.m
    for (i, j), label in ((0, 0), '入口 (0,0)'), ((n - 1, m - 1),
                                                  f'出口 ({n - 1},{m - 1})'):
        if not lc.in_band(i, j):
            return f'{label} 落在索引带 |i-j|<={lc.index_band} 之外'
        p = lc.pairs[(i, j)]
        if not p.usable:
            return (f'{label} 共同通道仅 {len(p.channels)} 个'
                    f'（{list(p.channels)}），低于 min_channels='
                    f'{lc.min_channels}，缺测不得补零')
    # 首末可用但 DP 仍不可达：带或 run_limit 切断了全部合法走法
    return (f'首末点对可用，但在索引带 |i-j|<={lc.index_band} 与 '
            f'run_limit={lc.run_limit} 约束下不存在从 (0,0) 到 '
            f'({n - 1},{m - 1}) 的合法单调路径')


def _wrong_objective_check(lc, result):
    """对照“先最小化总和再除长度”的错误口径。"""
    minsum = minimum_sum_path(lc)
    opt_ratio = result.ratio
    wrong_ratio = minsum.ratio
    return {
        'minimum_sum_path': [list(c) for c in minsum.path],
        'minimum_sum': _q(minsum.cost),
        'its_mean_cost': _q(wrong_ratio),
        'optimal_mean_cost': _q(opt_ratio),
        'misselects': wrong_ratio > opt_ratio,
        'note': ('该路径只最小化代价总和；若其平均代价高于最优平均代价，'
                 '说明“先求最小总和再除长度”会选错路径')}, minsum


def register(v, exact=None, include_wrong_objective=True,
             include_coverage=True):
    """主入口：返回可 JSON 序列化的配准报告 dict。"""
    lc, out = solve_material(v, exact=exact)
    report = {
        'mode': 'exact_rational' if lc.exact else 'float',
        'objective': 'min_P sum(local_cost) / path_points',
        'declarations': list(DECLARATIONS),
        'config': {
            'channels': lc.names,
            'min_channels': lc.min_channels,
            'index_band': lc.index_band,
            'run_limit': lc.run_limit,
        },
    }
    if out.status == 'infeasible':
        report['status'] = 'infeasible'
        report['infeasible_reason'] = _infeasible_reason(lc)
        report['bounds'] = None
        if include_coverage:
            report['coverage_losses'] = lc.coverage_losses()
        return report

    if out.result is None:
        # 预算在首轮 DP 完成前耗尽：没有任何候选路径，明确标注而非伪造
        report['status'] = out.status
        report['bounds'] = None
        report['dp_evals'] = out.evals
        report['states_used'] = out.states_used
        report['state_budget'] = lc.state_budget
        report['note'] = '状态预算在首个完整 DP 评估前耗尽，无候选路径可报'
        if include_coverage:
            report['coverage_losses'] = lc.coverage_losses()
        return report

    result = out.result
    records, cumulative = _pair_records(lc, result)
    steps, segments = _steps_and_segments(lc, result)
    report.update({
        'status': out.status,
        'bounds': {
            'lower': _q(out.lower),
            'upper': _q(out.upper),
            'declared_tolerance': _q(out.tolerance),
        },
        'dp_evals': out.evals,
        'states_used': out.states_used,
        'state_budget': lc.state_budget,
        'path': records,
        'path_indices': [list(c) for c in result.path],
        'cumulative_cost': _q(cumulative),
        'length': result.length,
        'mean_cost': _q(result.ratio),
        'reuse_counts': {
            'reference': _reuse_counts(result.path, 0),
            'candidate': _reuse_counts(result.path, 1),
        },
        'steps': steps,
        'warp_segments': segments,
        'time_warp_explanation': _warp_summary(lc, result, steps, segments),
    })
    if include_coverage:
        report['coverage_losses'] = lc.coverage_losses()
        report['fully_missing_channels'] = [
            c for c in lc.names
            if all(c not in lc.pairs[cell].channels for cell in result.path)]
    if include_wrong_objective:
        report['wrong_objective_comparison'], _ = _wrong_objective_check(
            lc, result)
    return report


def brute_force_check(v):
    """小序列穷举全部合法路径，返回与 Dinkelbach 结果的核对报告。"""
    lc = build_local_costs(v, exact=True)
    all_paths = enumerate_paths(lc)
    if not all_paths:
        return {'feasible': False,
                'reason': _infeasible_reason(lc),
                'paths': []}
    winner = all_paths[0]
    solved = solve(build_local_costs(v, exact=True))
    opt = solved.result
    agree = (opt.ratio == winner.ratio and opt.length == winner.length
             and opt.path == winner.path)
    return {
        'feasible': True,
        'legal_path_count': len(all_paths),
        'exhaustive_winner': {
            'path': [list(c) for c in winner.path],
            'mean_cost': _q(winner.ratio),
            'length': winner.length,
        },
        'dinkelbach_winner': {
            'path': [list(c) for c in opt.path],
            'mean_cost': _q(opt.ratio),
            'length': opt.length,
        },
        'agree': agree,
        'all_paths_ranked': [
            {'path': [list(c) for c in r.path],
             'mean_cost': _q(r.ratio),
             'length': r.length}
            for r in all_paths],
    }
