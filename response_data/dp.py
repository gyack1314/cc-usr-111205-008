"""受伸缩限制的路径动态规划与分式规划。

合法路径从 (0,0) 到 (n-1,m-1)，步长只允许 (1,1) 斜步、(1,0) 水平步、
(0,1) 垂直步；水平/垂直各自的连续步数不得超过 run_limit，转换步方向或
斜步都使上一方向的连续计数归零。目标是

    min_P  sum_{(i,j) in P} c(i,j) / |P|

即全局最小平均代价，而不是先最小化总和再除以长度。

状态按格点 (i,j) 的到达方式拆分：D（经斜步到达或起点）、H_k（连续 k 个
水平步到达，1<=k<=R）、V_k（连续 k 个垂直步到达，1<=k<=R）。水平步可
从 D 或任意 V_k 转入（转向，落到 H_1），也可从 H_k（k<R）延续；垂直步
对称。给定比例试探值 lam 时，在增广代价 c-lam 上做普通最短路即可，这
就是增广长度 DP；Dinkelbach 迭代利用
F(lam)=min_P(sum c - lam*|P|) 的符号更新比例上下界。

子路径层面只按净增广代价剪枝（同层并列前缀全部保留），因为前缀的比例、
长度都不决定拼上同一后缀后的全路径排序；比例/长度/字典序决胜只在终点
进行，避免误删全局最优路径的前缀。
"""
from fractions import Fraction

from .costs import build_local_costs

# 状态标签：('D', 0)、('H', k)、('V', k)
D = ('D', 0)


class BudgetExhausted(RuntimeError):
    """参数化 DP 生成的状态数超过 state_budget。"""


class PathResult:
    """一次参数化 DP 的最优路径结果。"""

    __slots__ = ('path', 'states', 'net', 'cost', 'length', 'lam')

    def __init__(self, path, states, net, cost, length, lam):
        self.path = path          # tuple[(i,j), ...]，完整点对序列
        self.states = states      # 与 path 等长的到达状态标签
        self.net = net            # sum(cost - lam)
        self.cost = cost          # sum cost
        self.length = length      # 点对数
        self.lam = lam

    @property
    def ratio(self):
        return self.cost / self.length


def _dominant(candidates, tie_mode):
    """返回候选中净增广代价最小的全部 PathResult（并列全保留）。

    子路径层面不能按比例或长度剪枝：前缀的比例/长度不决定拼上同一后缀
    后的全路径排序，任何额外决胜键都可能误删全局最优路径的前缀。因此
    每个状态保留全部最小净代价前缀，决胜推迟到终点（题面规则）或迭代
    选路（比例规则）时进行。
    """
    alive = [c for c in candidates if c is not None]
    if not alive:
        return []
    floor = min(c.net for c in alive)
    return [c for c in alive if c.net == floor]


def parametric_dp(lc, lam, state_counter=None, state_cap=None,
                  tie_mode='ratio'):
    """在增广代价 c-lam 上求受 run_limit 约束的最短路。

    每个 (格点, 到达状态) 保留全部最小净代价前缀；返回 PathResult。
    tie_mode 只决定终点的决胜口径：
    'ratio'：(比例, 长度, 字典序)，供 Dinkelbach 迭代选路；
    'sum'：(长度, 字典序)，供最小总和对照与题面同分决胜。
    无可行路径时返回 None；状态数超过 state_cap 抛 BudgetExhausted。
    """
    n, m, R = lc.n, lc.m, lc.run_limit
    # table[i][j][label] = 最小净代价前缀 PathResult 列表
    table = [[{} for _ in range(m)] for _ in range(n)]

    if not lc.cell_usable(0, 0) or not lc.cell_usable(n - 1, m - 1):
        return None

    def add(i, j, label, new_cands):
        kept = _dominant(new_cands, tie_mode)
        if not kept:
            return
        cell = table[i][j]
        if label in cell:
            # 必须把新旧候选合并后统一取最小净代价层；分别取组内最小
            # 再比较会丢掉跨组并列的最小净代价前缀
            merged = _dominant(cell[label] + kept, tie_mode)
            if state_counter is not None:
                state_counter[0] += len(merged) - len(cell[label])
            cell[label] = merged
        else:
            cell[label] = kept
            if state_counter is not None:
                state_counter[0] += len(kept)

    start = lc.pairs[(0, 0)]
    add(0, 0, D, [PathResult(((0, 0),), (D,), start.cost - lam,
                             start.cost, 1, lam)])

    def extend(prev, i, j, label):
        if not (0 <= i < n and 0 <= j < m) or not lc.cell_usable(i, j):
            return None
        pair = lc.pairs[(i, j)]
        return PathResult(
            prev.path + ((i, j),),
            prev.states + (label,),
            prev.net + pair.cost - lam,
            prev.cost + pair.cost,
            prev.length + 1,
            lam)

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            if not lc.cell_usable(i, j):
                continue
            if (state_counter is not None and state_cap is not None
                    and state_counter[0] >= state_cap):
                raise BudgetExhausted
            # 斜步 (i-1,j-1)：任意到达状态均可，斜步重置计数 -> D
            if i > 0 and j > 0:
                src = table[i - 1][j - 1]
                cands = [extend(p, i, j, D)
                         for plist in src.values() for p in plist]
                add(i, j, D, cands)
            # 水平步 (i-1,j) -> H_1：来源可以是 D、任意 V_k（转向重置），
            # 或 H_k（k<R，连续计数 +1）。
            if i > 0:
                src = table[i - 1][j]
                by_label = {}
                def collect(label_src, out_label):
                    if label_src not in src:
                        return
                    by_label.setdefault(out_label, []).extend(
                        extend(p, i, j, out_label)
                        for p in src[label_src])
                collect(D, ('H', 1))
                for k in range(1, R + 1):
                    collect(('V', k), ('H', 1))  # 转向，落到 H_1
                for k in range(1, R):
                    collect(('H', k), ('H', k + 1))  # 连续，计数 +1
                for label, cands in by_label.items():
                    add(i, j, label, cands)
            # 垂直步 (i,j-1) -> V_1：来源可以是 D、任意 H_k（转向重置），
            # 或 V_k（k<R，连续计数 +1）。
            if j > 0:
                src = table[i][j - 1]
                by_label = {}
                def collect_v(label_src, out_label):
                    if label_src not in src:
                        return
                    by_label.setdefault(out_label, []).extend(
                        extend(p, i, j, out_label)
                        for p in src[label_src])
                collect_v(D, ('V', 1))
                for k in range(1, R + 1):
                    collect_v(('H', k), ('V', 1))  # 转向，落到 V_1
                for k in range(1, R):
                    collect_v(('V', k), ('V', k + 1))  # 连续，计数 +1
                for label, cands in by_label.items():
                    add(i, j, label, cands)

    end = table[n - 1][m - 1]
    if not end:
        return None
    finals = [p for plist in end.values() for p in plist]
    floor = min(p.net for p in finals)
    # 决胜只在最小净代价层内进行：比例最优对应 F(lam) 的最小净代价，
    # 净代价更大的路径不参与同分决胜
    finals = [p for p in finals if p.net == floor]
    if tie_mode == 'ratio':
        finals.sort(key=lambda r: (r.ratio, r.length, r.path))
    else:
        finals.sort(key=lambda r: (r.length, r.path))
    return finals[0]


def minimum_sum_path(lc, state_counter=None, state_cap=None):
    """最小总代价路径（lam=0、sum 决胜），用于展示“先求最小和再除长度”
    的错误口径。"""
    zero = Fraction(0) if lc.exact else 0.0
    return parametric_dp(lc, zero, state_counter, state_cap,
                         tie_mode='sum')


def _min_cell_cost(lc):
    """最优平均代价的合法初始下界：任一可行路径平均 >= 最小局部代价。"""
    return min(p.cost for (i, j), p in lc.pairs.items()
               if lc.cell_usable(i, j))


class SolveOut:
    """求解输出：状态、界、容差、候选路径、DP 评估次数与迭代历史。"""

    __slots__ = ('result', 'status', 'evals', 'states_used', 'lower',
                 'upper', 'tolerance', 'history')

    def __init__(self, result, status, evals, states_used, lower, upper,
                 tolerance, history=None):
        self.result = result
        self.status = status          # exact / optimal_within_tolerance /
        #                            # budget_exhausted / infeasible
        self.evals = evals
        self.states_used = states_used
        self.lower = lower
        self.upper = upper
        self.tolerance = tolerance
        self.history = history or []


def solve(lc, progress=None):
    """Dinkelbach 分式规划求全局最小平均代价路径。

    exact 模式（Fraction）在有限步精确终止，上下界重合；浮点模式按
    lc.tolerance 停止，并报告上下界与声明容差。state_budget 限制累计
    DP 状态数，预算耗尽时只返回当前候选并标注 budget_exhausted。
    progress 若给出，按每次 DP 评估回调 (lam, result)。
    """
    states_used = [0]
    cap = lc.state_budget
    evals = [0]

    def run(lam):
        evals[0] += 1
        res = parametric_dp(lc, lam, states_used, cap)
        if progress is not None:
            progress(lam, res)
        return res

    def run_final(lam):
        # 收敛后的决胜评估：同净代价按短路径、字典序（题面同分规则）
        evals[0] += 1
        res = parametric_dp(lc, lam, states_used, cap, tie_mode='sum')
        if progress is not None:
            progress(lam, res)
        return res

    try:
        first = minimum_sum_path(lc, states_used, cap)
    except BudgetExhausted:
        return SolveOut(None, 'budget_exhausted', evals[0],
                        states_used[0], None, None, lc.tolerance)
    evals[0] += 1
    if first is None:
        return SolveOut(None, 'infeasible', evals[0], states_used[0],
                        None, None, lc.tolerance)

    lower = _min_cell_cost(lc)
    upper = first.ratio
    best = first  # 已访问路径中比例最小者
    history = [first]

    if lc.exact:
        # 标准 Dinkelbach：lam 取上一轮最优路径的比例（初始为最小和
        # 路径比例）。F(lam)<0 时更新 lam=新路径比例（严格递减的上界）；
        # F(lam)=0 时用严格更小的探针确认没有比例更小的路径，再按题面
        # 同分规则做最终决胜。
        lam = upper
        seen = {lam}
        while True:
            try:
                res = run(lam)
            except BudgetExhausted:
                return SolveOut(best, 'budget_exhausted', evals[0],
                                states_used[0], lower, upper,
                                Fraction(0), history)
            history.append(res)
            if res.net == 0:
                probe = (lower + lam) / 2
                if probe < lam:
                    try:
                        check = run(probe)
                    except BudgetExhausted:
                        return SolveOut(best, 'budget_exhausted', evals[0],
                                        states_used[0], lower, lam,
                                        Fraction(0), history)
                    history.append(check)
                    if check.net < 0:
                        # 探针下仍有比例更小的路径：lam 不是最优，
                        # 回到标准 Dinkelbach 更新
                        new_lam = check.ratio
                        upper = min(upper, new_lam)
                        best = check if check.ratio <= best.ratio else best
                        seen.add(new_lam)
                        lam = new_lam
                        continue
                    # check.net >= 0：探针处无更优路径，确认 lam 最优
                    if check.net > 0:
                        lower = max(lower, probe)
                try:
                    winner = run_final(lam)
                except BudgetExhausted:
                    winner = res
                return SolveOut(winner, 'exact', evals[0], states_used[0],
                                lam, lam, Fraction(0), history)
            if res.net < 0:
                new_lam = res.ratio
                upper = min(upper, new_lam)
                if res.ratio <= best.ratio:
                    best = res
                if new_lam in seen:  # 防御性保护：有理输入下不应发生
                    return SolveOut(best, 'exact', evals[0],
                                    states_used[0], lower, upper,
                                    Fraction(0), history)
                seen.add(new_lam)
                lam = new_lam
            else:
                # F(lam)>0：所有路径比例 > lam，lam 是合法下界；
                # 回到当前上界继续标准迭代
                lower = max(lower, lam)
                lam = upper
    else:
        # 浮点混合迭代：F(lam)<0 时用新路径比例收紧上界（Dinkelbach），
        # F(lam)>=0 时收紧下界（二分）。只有当上下界宽度 <= eps 才声明
        # 达标，保证 upper-lower <= 声明容差，而不是在宽松下界处提前停止。
        lam = upper
        eps = lc.tolerance
        while True:
            if upper - lower <= eps:
                try:
                    best = run_final(upper)
                except BudgetExhausted:
                    pass
                return SolveOut(best, 'optimal_within_tolerance', evals[0],
                                states_used[0], lower, upper, eps, history)
            try:
                res = run(lam)
            except BudgetExhausted:
                return SolveOut(best, 'budget_exhausted', evals[0],
                                states_used[0], lower, upper, eps, history)
            history.append(res)
            zero_tol = eps * max(1.0, abs(lam)) * res.length
            if res.net < -zero_tol:
                new_lam = res.ratio
                upper = min(upper, new_lam)
                if res.ratio <= best.ratio:
                    best = res
                lam = upper
            else:
                # F(lam)>=0（含数值零）：lam 是合法下界，按二分继续收紧
                lower = max(lower, lam)
                lam = (lower + upper) / 2


def enumerate_paths(lc):
    """小序列穷举全部合法路径（精确有理），仅用于核对与测试。

    返回按 (比例, 长度, 完整点对序列字典序) 排序的 PathResult 列表；
    无可行路径时返回空列表。
    """
    if not lc.exact:
        raise ValueError('穷举核对必须使用 exact=True 的局部代价表')
    n, m, R = lc.n, lc.m, lc.run_limit
    if not lc.cell_usable(0, 0) or not lc.cell_usable(n - 1, m - 1):
        return []
    out = []

    def dfs(i, j, h_run, v_run, path, states, cost):
        if i == n - 1 and j == m - 1:
            length = len(path)
            out.append(PathResult(tuple(path), tuple(states), None,
                                  cost, length, None))
            return
        if i + 1 < n and j + 1 < m and lc.cell_usable(i + 1, j + 1):
            p = lc.pairs[(i + 1, j + 1)]
            dfs(i + 1, j + 1, 0, 0, path + [(i + 1, j + 1)],
                states + [D], cost + p.cost)
        if i + 1 < n and h_run < R and lc.cell_usable(i + 1, j):
            p = lc.pairs[(i + 1, j)]
            dfs(i + 1, j, h_run + 1, 0, path + [(i + 1, j)],
                states + [('H', h_run + 1)], cost + p.cost)
        if j + 1 < m and v_run < R and lc.cell_usable(i, j + 1):
            p = lc.pairs[(i, j + 1)]
            dfs(i, j + 1, 0, v_run + 1, path + [(i, j + 1)],
                states + [('V', v_run + 1)], cost + p.cost)

    dfs(0, 0, 0, 0, [(0, 0)], [D], lc.pairs[(0, 0)].cost)
    out.sort(key=lambda r: (r.ratio, r.length, r.path))
    return out


def solve_material(v, exact=None):
    """便捷入口：资料 dict -> (LocalCosts, SolveOut)。"""
    if exact is None:
        exact = _all_integer(v)
    lc = build_local_costs(v, exact=exact)
    return lc, solve(lc)


def _all_integer(v):
    for key in ('reference', 'candidate'):
        for row in v[key]:
            for x in row.values():
                if x is not None and (isinstance(x, bool)
                                      or not isinstance(x, int)):
                    return False
    for cfg in v['channels'].values():
        for k in ('center', 'scale', 'weight'):
            x = cfg[k]
            if isinstance(x, bool) or not isinstance(x, int):
                return False
    return True
