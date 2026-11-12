"""局部点对代价与共同通道支持。

点对 (i, j) 的局部代价只在双方均已观测的通道上计算：

    c(i,j) = sum_c w_c * ((r_ic - center_c)/scale_c - (q_jc - center_c)/scale_c)^2
             / sum_{c in 共同通道} w_c

缺测（None）绝不补零；共同通道数少于 min_channels 的点对不可用。
入口/出口点对同样必须可用，否则无可行路径。
"""
from .io import number, positive, integer


def _observed(row, name):
    """返回该采样点已观测的通道名集合，None 视为缺测。"""
    out = set()
    for c, x in row.items():
        if x is None:
            continue
        number(x)
        out.add(c)
    return out


def validate_material(v):
    """校验资料结构与锁定参数，返回标准化后的内部表示。

    返回 (names, centers, scales, weights, reference, candidate,
          min_channels, index_band, run_limit, tolerance, state_budget)，
    其中两个序列保留 None 缺测，数值不做替换。
    """
    if not isinstance(v, dict):
        raise ValueError('顶层资料必须为对象')
    channels = v['channels']
    if not isinstance(channels, dict) or not channels:
        raise ValueError('通道配置必须为非空对象')
    names, centers, scales, weights = [], {}, {}, {}
    for name, cfg in channels.items():
        if not isinstance(name, str) or not name:
            raise ValueError('通道名必须非空字符串')
        names.append(name)
        centers[name] = number(cfg['center'])
        scales[name] = positive(cfg['scale'])
        weights[name] = positive(cfg['weight'])
    min_channels = integer(v['min_channels'], 1)
    if min_channels > len(names):
        raise ValueError('min_channels 超过通道总数')
    index_band = integer(v['index_band'], 0)
    run_limit = integer(v['run_limit'], 1)
    solver = v.get('solver', {})
    if not isinstance(solver, dict):
        raise ValueError('solver 必须为对象')
    tolerance = positive(solver.get('tolerance', 1e-10))
    state_budget = integer(solver.get('state_budget', 100_000), 1)
    series = {}
    for key in ('reference', 'candidate'):
        rows = v[key]
        if not isinstance(rows, (list, tuple)) or len(rows) < 1:
            raise ValueError(f'{key} 必须为非空序列')
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(names):
                raise ValueError(f'{key} 通道引用不齐')
        series[key] = rows
    return (names, centers, scales, weights, series['reference'],
            series['candidate'], min_channels, index_band, run_limit,
            tolerance, state_budget)


def build_local_costs(v, exact=False):
    """构造全部点对的局部代价与有效通道信息。

    exact=True 时使用 fractions.Fraction 精确有理运算（要求原始值为 int，
    scale/weight/center 也为整数或 Fraction 可接受的量）；否则使用浮点。
    返回 dict，键为 (i, j)，值为 LocalPair。索引带在路径层另行过滤，
    这里先保留所有点对，便于解释与覆盖损失统计。
    """
    from fractions import Fraction
    (names, centers, scales, weights, reference, candidate,
     min_channels, index_band, run_limit, tolerance, state_budget) = validate_material(v)

    def num(x):
        if exact:
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError('精确模式只接受整数观测值')
            return Fraction(x)
        return float(x)

    ctr = {c: num(centers[c]) for c in names}
    scl = {c: (Fraction(scales[c]) if exact else float(scales[c])) for c in names}
    wgt = {c: (Fraction(weights[c]) if exact else float(weights[c])) for c in names}
    zero = Fraction(0) if exact else 0.0
    pairs = {}
    ref_obs = [_observed(row, 'reference') for row in reference]
    cand_obs = [_observed(row, 'candidate') for row in candidate]
    for i, rrow in enumerate(reference):
        for j, qrow in enumerate(candidate):
            common = sorted(ref_obs[i] & cand_obs[j], key=names.index)
            if len(common) < min_channels:
                pairs[(i, j)] = LocalPair(None, tuple(common), zero, False)
                continue
            wsum = zero
            acc = zero
            for c in common:
                zr = (num(rrow[c]) - ctr[c]) / scl[c]
                zq = (num(qrow[c]) - ctr[c]) / scl[c]
                d = zr - zq
                w = wgt[c]
                acc += w * d * d
                wsum += w
            pairs[(i, j)] = LocalPair(acc / wsum, tuple(common), wsum, True)
    return LocalCosts(
        names=names, centers=centers, scales=scales, weights=weights,
        reference=reference, candidate=candidate, pairs=pairs,
        ref_obs=ref_obs, cand_obs=cand_obs,
        min_channels=min_channels, index_band=index_band, run_limit=run_limit,
        tolerance=tolerance, state_budget=state_budget, exact=exact)


class LocalPair:
    """单个点对的局部信息。

    usable：共同通道是否达到 min_channels；cost 为不可用时为 0（占位）；
    channels 为双方均观测的通道名（按配置顺序）；weight_sum 为共同权重和。
    """

    __slots__ = ('cost', 'channels', 'weight_sum', 'usable')

    def __init__(self, cost, channels, weight_sum, usable):
        self.cost = cost
        self.channels = channels
        self.weight_sum = weight_sum
        self.usable = usable

    def __repr__(self):
        return (f'LocalPair(cost={self.cost}, channels={self.channels!r}, '
                f'usable={self.usable})')


class LocalCosts:
    """局部代价表与资料上下文。"""

    __slots__ = ('names', 'centers', 'scales', 'weights', 'reference',
                 'candidate', 'pairs', 'ref_obs', 'cand_obs',
                 'min_channels', 'index_band', 'run_limit',
                 'tolerance', 'state_budget', 'exact')

    def __init__(self, **kw):
        for k, val in kw.items():
            setattr(self, k, val)

    @property
    def n(self):
        return len(self.reference)

    @property
    def m(self):
        return len(self.candidate)

    def in_band(self, i, j):
        return abs(i - j) <= self.index_band

    def cell_usable(self, i, j):
        return self.in_band(i, j) and self.pairs[(i, j)].usable

    def coverage_losses(self):
        """逐点对列出相对全通道的覆盖损失，供报告使用。"""
        total = len(self.names)
        out = []
        for i in range(self.n):
            for j in range(self.m):
                p = self.pairs[(i, j)]
                missing = tuple(c for c in self.names if c not in p.channels)
                if missing or not p.usable:
                    out.append({
                        'i': i, 'j': j,
                        'effective_channels': list(p.channels),
                        'missing_channels': list(missing),
                        'in_band': self.in_band(i, j),
                        'usable': p.usable,
                        'reason': (f'共同通道 {len(p.channels)}/{total}，'
                                   f'要求至少 {self.min_channels}')})
        return out
