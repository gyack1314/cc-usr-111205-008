"""受限最小平均代价配准的核对测试。

包含：小序列穷举所有合法路径与 DP/Dinkelbach 对照、“先求最小总和再除
长度”选错的反例、同分决胜规则、run_limit/索引带约束、缺测覆盖损失、
不可行情形、预算耗尽、浮点模式，以及通道重命名与锁定参数等价换单位的
不变性证明式测试。
"""
import copy
import unittest
from fractions import Fraction

from response_data.costs import build_local_costs
from response_data.dp import (enumerate_paths, minimum_sum_path,
                              parametric_dp, solve)
from response_data.register import brute_force_check, register


# 3x3 反例代价表（单通道、center=0、scale=1、weight=1）：
#   c(0,0)=0 c(0,1)=1 c(0,2)=4
#   c(1,0)=9 c(1,1)=0 c(1,2)=1
#   c(2,0)=9 c(2,1)=9 c(2,2)=0
GRID_33 = [[0, 1, 4], [9, 0, 1], [9, 9, 0]]


def material_from_grid(grid, run_limit=5, index_band=10,
                       min_channels=1, state_budget=100_000,
                       tolerance=1e-12):
    """构造单通道整数资料（值全零），随后用 override_costs 写入显式代价。"""
    n = len(grid)
    m = len(grid[0])
    return {
        'channels': {'x': {'center': 0, 'scale': 1, 'weight': 1}},
        'reference': [{'x': 0} for _ in range(n)],
        'candidate': [{'x': 0} for _ in range(m)],
        'min_channels': min_channels,
        'index_band': index_band,
        'run_limit': run_limit,
        'solver': {'tolerance': tolerance, 'state_budget': state_budget},
    }


def make_lc(grid, run_limit=5, index_band=10, min_channels=1,
            state_budget=100_000, exact=True):
    """从显式整数/分数代价表直接构造 LocalCosts 测试桩。"""
    v = material_from_grid(grid, run_limit=run_limit,
                           index_band=index_band,
                           min_channels=min_channels,
                           state_budget=state_budget)
    lc = build_local_costs(v, exact=exact)
    n, m = len(grid), len(grid[0])
    for i in range(n):
        for j in range(m):
            val = grid[i][j]
            lc.pairs[(i, j)].cost = (Fraction(val) if exact else float(val))
            lc.pairs[(i, j)].usable = True
            lc.pairs[(i, j)].channels = ('x',)
    return lc


class ExhaustiveOracleTests(unittest.TestCase):
    def test_dp_matches_exhaustive_on_small_series(self):
        import random
        random.seed(7)
        for n in range(2, 5):
            for m in range(2, 5):
                for R in (1, 2, 3):
                    for band in (0, 1, max(n, m)):
                        with self.subTest(n=n, m=m, R=R, band=band):
                            grid = [[random.randint(0, 9)
                                     for _ in range(m)] for _ in range(n)]
                            lc = make_lc(grid, run_limit=R, index_band=band)
                            ranked = enumerate_paths(lc)
                            if not ranked:
                                # 穷举无合法路径时 DP 也必须报不可行
                                self.assertIsNone(
                                    parametric_dp(lc, Fraction(0)))
                                continue
                            out = solve(lc)
                            self.assertEqual(out.status, 'exact')
                            winner = ranked[0]
                            self.assertEqual(out.result.ratio, winner.ratio)
                            self.assertEqual(out.result.length, winner.length)
                            self.assertEqual(out.result.path, winner.path)
                            # 每个穷举路径的比例都不小于胜出比例
                            for r in ranked:
                                self.assertGreaterEqual(r.ratio, winner.ratio)

    def test_bruteforce_report_agrees_on_project_data(self):
        for name in ('normal.json', 'one-channel-missing.json'):
            import json
            from pathlib import Path
            v = json.loads(Path(f'data/{name}').read_text(encoding='utf8'))
            check = brute_force_check(v)
            self.assertTrue(check['feasible'])
            self.assertTrue(check['agree'])

    def test_exhaustive_respects_band_and_run_limit(self):
        # run_limit=1 时任何路径都不得含连续两个同向水平/垂直步
        lc = make_lc(GRID_33, run_limit=1, index_band=10)
        for r in enumerate_paths(lc):
            h = vrun = 0
            for a, b in zip(r.path, r.path[1:]):
                step = (b[0] - a[0], b[1] - a[1])
                if step == (1, 0):
                    h += 1
                    vrun = 0
                elif step == (0, 1):
                    vrun += 1
                    h = 0
                else:
                    h = vrun = 0
                self.assertLessEqual(h, 1)
                self.assertLessEqual(vrun, 1)
                self.assertLessEqual(abs(b[0] - b[1]), lc.index_band)


class WrongObjectiveTests(unittest.TestCase):
    def test_misselect_counterexample(self):
        # 显式代价表构造的反例（穷举 3x3 全部 13 条合法路径核验）：
        #   c(0,0)=0 c(0,1)=1 c(0,2)=0
        #   c(1,0)=0 c(1,1)=2 c(1,2)=2
        #   c(2,0)=0 c(2,1)=2 c(2,2)=0
        # 最小总和路径严格唯一：对角 [00,11,22] 总和 2、平均 2/3；
        # 比例最优路径是 5 点绕行路径，总和 2、平均 2/5 < 2/3。
        grid = [[0, 1, 0], [0, 2, 2], [0, 2, 0]]
        lc = make_lc(grid)
        ranked = enumerate_paths(lc)
        winner = ranked[0]
        minsum = minimum_sum_path(lc)
        self.assertEqual(minsum.path, ((0, 0), (1, 1), (2, 2)))
        self.assertEqual(minsum.cost, Fraction(2))
        self.assertEqual(minsum.ratio, Fraction(2, 3))
        self.assertEqual(winner.ratio, Fraction(2, 5))
        self.assertEqual(winner.path,
                         ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)))
        self.assertLess(winner.ratio, minsum.ratio)
        # Dinkelbach 必须选到比例最优而非最小总和路径
        out = solve(lc)
        self.assertEqual(out.result.ratio, Fraction(2, 5))
        self.assertEqual(out.result.path, winner.path)
        self.assertNotEqual(out.result.path, minsum.path)
        # 报告层的错误口径对照必须标注选错
        from response_data.register import _wrong_objective_check
        comparison, _ = _wrong_objective_check(lc, out.result)
        self.assertTrue(comparison['misselects'])
        self.assertEqual(Fraction(comparison['its_mean_cost']['exact']),
                         Fraction(2, 3))

    def test_tie_shorter_path_wins(self):
        # 2x2 全零代价：对角路径 2 点、比例 0；绕行路径 3 点、比例 0，
        # 同比例下短路径胜出
        lc = make_lc([[0, 0], [0, 0]])
        out = solve(lc)
        self.assertEqual(out.result.ratio, Fraction(0))
        self.assertEqual(out.result.length, 2)
        self.assertEqual(out.result.path, ((0, 0), (1, 1)))
        # 穷举榜中同比例的绕行路径确实更长
        ranked = enumerate_paths(lc)
        detours = [r for r in ranked if r.ratio == Fraction(0)
                   and r.path != ((0, 0), (1, 1))]
        self.assertTrue(detours)
        self.assertTrue(all(r.length == 3 for r in detours))

    def test_tie_lexicographic_winner(self):
        # 2x3 全零代价：所有合法路径比例同为 0。同比例先比长度（对角+
        # 垂直最短 3 点），同长度再按完整点对序列字典序：
        # [(0,0),(0,1),(1,2)] 与 [(0,0),(1,1),(1,2)] 等长，
        # 首差在第二个点 (0,1) < (1,1)，前者胜出。
        lc = make_lc([[0, 0, 0], [0, 0, 0]])
        ranked = enumerate_paths(lc)
        best = ranked[0]
        tied = [r for r in ranked if r.ratio == best.ratio
                and r.length == best.length]
        self.assertGreaterEqual(len(tied), 2)
        out = solve(lc)
        self.assertEqual(out.result.path, tied[0].path)
        self.assertEqual(out.result.path, ((0, 0), (0, 1), (1, 2)))
        # 含 (1,0) 的等长路径字典序必然更大
        self.assertNotIn((1, 0), out.result.path)


class ConstraintAndSupportTests(unittest.TestCase):
    def test_missing_value_is_not_zero(self):
        v = material_from_grid(GRID_33)
        v['candidate'][1]['x'] = None
        lc = build_local_costs(v, exact=True)
        for i in range(3):
            self.assertFalse(lc.pairs[(i, 1)].usable)
            self.assertEqual(lc.pairs[(i, 1)].channels, ())
        # 其余点对不受影响
        self.assertTrue(lc.pairs[(0, 0)].usable)

    def test_min_channels_gate(self):
        v = material_from_grid(GRID_33, min_channels=2)
        # 单通道资料：min_channels=2 时没有任何点对可用
        with self.assertRaises(ValueError):
            build_local_costs(v)
        v['min_channels'] = 1
        lc = build_local_costs(v, exact=True)
        self.assertTrue(all(p.usable for p in lc.pairs.values()))

    def test_endpoint_missing_is_infeasible(self):
        v = material_from_grid(GRID_33)
        v['reference'][0]['x'] = None  # 入口失去全部支持
        report = register(v)
        self.assertEqual(report['status'], 'infeasible')
        self.assertIn('入口 (0,0)', report['infeasible_reason'])

    def test_band_makes_endpoint_unreachable(self):
        import json
        from pathlib import Path
        v = json.loads(Path('data/unreachable-endpoint.json')
                       .read_text(encoding='utf8'))
        report = register(v)
        self.assertEqual(report['status'], 'infeasible')
        self.assertIn('索引带', report['infeasible_reason'])
        self.assertIsNone(report['bounds'])

    def test_run_limit_can_block_all_paths(self):
        # 2x5、run_limit=1、带宽 3：首末点 (0,0)/(1,4) 均在带内
        # （|1-4|=3<=3），但“连续同向步至多 1”切断了全部交替走法，
        # 穷举与 DP 均判不可行，原因必须落在 run_limit 而非索引带。
        grid = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        v = material_from_grid(grid, run_limit=1, index_band=3)
        lc = build_local_costs(v, exact=True)
        self.assertEqual(enumerate_paths(lc), [])
        self.assertIsNone(parametric_dp(lc, Fraction(0)))
        report = register(v)
        self.assertEqual(report['status'], 'infeasible')
        self.assertIn('run_limit', report['infeasible_reason'])
        self.assertNotIn('落在索引带', report['infeasible_reason'])

    def test_direction_change_resets_run_counter(self):
        # 契约：转换步方向即重置连续计数。2x4、run_limit=1、带宽足够宽时，
        # 垂直/水平交替走法合法，路径必须存在且每个同向连续步恰为 1。
        grid = [[0, 0, 0, 0], [0, 0, 0, 0]]
        v = material_from_grid(grid, run_limit=1, index_band=10)
        lc = build_local_costs(v, exact=True)
        ranked = enumerate_paths(lc)
        self.assertTrue(ranked)
        for r in ranked:
            h = vrun = 0
            for a, b in zip(r.path, r.path[1:]):
                step = (b[0] - a[0], b[1] - a[1])
                if step == (1, 0):
                    h += 1
                    vrun = 0
                elif step == (0, 1):
                    vrun += 1
                    h = 0
                else:
                    h = vrun = 0
                self.assertLessEqual(h, 1)
                self.assertLessEqual(vrun, 1)
        # DP 与穷举在该约束下选出同一胜者
        from response_data.dp import solve as _solve
        self.assertEqual(_solve(lc).result.path, ranked[0].path)

    def test_fully_missing_channel_keeps_path_with_coverage_loss(self):
        import json
        from pathlib import Path
        v = json.loads(Path('data/one-channel-missing.json')
                       .read_text(encoding='utf8'))
        report = register(v)
        self.assertEqual(report['status'], 'exact')
        self.assertEqual(report['fully_missing_channels'], ['inland'])
        # 每个点对的有效通道都只有 coast，且报告逐对列出
        for rec in report['path']:
            self.assertEqual(rec['effective_channels'], ['coast'])
            self.assertEqual(rec['missing_channels'], ['inland'])

    def test_pair_records_report_support_and_cumulative(self):
        report = register(material_from_grid(GRID_33))
        cum = Fraction(0)
        for rec in report['path']:
            cum += Fraction(rec['local_cost']['exact'])
            self.assertEqual(Fraction(rec['cumulative_cost']['exact']), cum)
            self.assertEqual(rec['effective_channels'], ['x'])
        self.assertEqual(Fraction(report['cumulative_cost']['exact']), cum)


class BudgetTests(unittest.TestCase):
    def test_budget_exhausted_reports_candidate_only(self):
        v = material_from_grid(
            [[0] * 8 for _ in range(8)], state_budget=3)
        report = register(v)
        self.assertEqual(report['status'], 'budget_exhausted')
        # 预算在首轮 DP 即耗尽：没有已完成候选，明确标注而非伪造结果
        self.assertNotIn('path', report)

    def test_budget_during_iteration_keeps_candidate_and_bounds(self):
        # 预算允许首轮（最小和）DP 完成、但在后续 Dinkelbach 评估中耗尽：
        # 必须保留当前候选路径与已知上下界，而不是伪装成最优或清空结果
        import random
        random.seed(3)
        grid = [[random.randint(0, 9) for _ in range(8)] for _ in range(8)]
        lc = make_lc(grid, run_limit=3, index_band=7, exact=True)
        full = solve(lc)
        self.assertEqual(full.status, 'exact')
        lc.state_budget = 700
        mid = solve(lc)
        self.assertEqual(mid.status, 'budget_exhausted')
        self.assertIsNotNone(mid.result)
        self.assertTrue(mid.lower <= full.result.ratio <= mid.upper
                        or mid.lower is None)
        self.assertGreaterEqual(mid.evals, 2)


class FloatModeTests(unittest.TestCase):
    def test_float_matches_exact_path(self):
        v = material_from_grid(GRID_33)
        exact = register(v, exact=True)
        flt = register(v, exact=False)
        self.assertEqual(flt['status'], 'optimal_within_tolerance')
        self.assertEqual(flt['path_indices'], exact['path_indices'])
        self.assertAlmostEqual(
            flt['mean_cost'], exact['mean_cost']['approx'], places=10)
        self.assertLessEqual(
            flt['bounds']['upper'] - flt['bounds']['lower'],
            flt['bounds']['declared_tolerance'] + 1e-18)

    def test_float_accepts_fractional_observations(self):
        v = material_from_grid(GRID_33)
        v['reference'][1]['x'] = 0.5
        report = register(v)  # 自动选择浮点模式
        self.assertEqual(report['mode'], 'float')
        self.assertEqual(report['status'], 'optimal_within_tolerance')
        self.assertTrue(report['path_indices'])


class InvarianceTests(unittest.TestCase):
    def _two_channel_material(self):
        return {
            'channels': {
                'coast': {'center': 0, 'scale': 10, 'weight': 2},
                'inland': {'center': 1, 'scale': 5, 'weight': 1},
            },
            'reference': [
                {'coast': 0, 'inland': 1},
                {'coast': 5, 'inland': 3},
                {'coast': 20, 'inland': None},
                {'coast': 3, 'inland': 2},
            ],
            'candidate': [
                {'coast': 0, 'inland': 1},
                {'coast': 2, 'inland': None},
                {'coast': 6, 'inland': 2},
                {'coast': 18, 'inland': 4},
                {'coast': 4, 'inland': 2},
            ],
            'min_channels': 1, 'index_band': 2, 'run_limit': 2,
            'solver': {'tolerance': 1e-10, 'state_budget': 100_000},
        }

    def test_channel_rename_and_reorder_invariance(self):
        v = self._two_channel_material()
        base = register(v)
        w = copy.deepcopy(v)
        # 同时重命名并交换通道顺序（inland 放到首位）
        rename = {'coast': 'station_A', 'inland': 'station_B'}
        w['channels'] = {rename[k]: cfg for k, cfg in
                         [('inland', v['channels']['inland']),
                          ('coast', v['channels']['coast'])]}
        for key in ('reference', 'candidate'):
            w[key] = [{rename[k]: val for k, val in row.items()}
                      for row in v[key]]
        moved = register(w)
        self.assertEqual(moved['path_indices'], base['path_indices'])
        self.assertEqual(moved['mean_cost'], base['mean_cost'])
        self.assertEqual(moved['length'], base['length'])
        # 有效通道名同步重命名，数量与缺测结构不变
        for r0, r1 in zip(base['path'], moved['path']):
            self.assertEqual(len(r0['effective_channels']),
                             len(r1['effective_channels']))
            self.assertEqual(r0['missing_channels'] == [],
                             r1['missing_channels'] == [])

    def test_locked_unit_change_invariance(self):
        # 等价换单位：x' = a*x + b（a>0），center' = a*center + b，
        # scale' = a*scale；权重同时乘正因子 g。标准化值与加权平均不变。
        v = self._two_channel_material()
        base = register(v)
        a, b, g = 3, 7, 4

        def conv(x):
            return None if x is None else a * x + b

        w = copy.deepcopy(v)
        for name, cfg in w['channels'].items():
            cfg['center'] = a * cfg['center'] + b
            cfg['scale'] = a * cfg['scale']
            cfg['weight'] = g * cfg['weight']
        for key in ('reference', 'candidate'):
            w[key] = [{k: conv(x) for k, x in row.items()}
                      for row in v[key]]
        equiv = register(w)
        self.assertEqual(equiv['path_indices'], base['path_indices'])
        self.assertEqual(equiv['mean_cost'], base['mean_cost'])
        self.assertEqual(equiv['length'], base['length'])
        for r0, r1 in zip(base['path'], equiv['path']):
            self.assertEqual(r0['local_cost'], r1['local_cost'])
            self.assertEqual(r0['effective_channels'], r1['effective_channels'])

    def test_invariance_under_float_mode(self):
        v = self._two_channel_material()
        base = register(v, exact=False)
        a, b, g = Fraction(2), Fraction(5), Fraction(3)
        w = copy.deepcopy(v)
        for name, cfg in w['channels'].items():
            cfg['center'] = float(a * cfg['center'] + b)
            cfg['scale'] = float(a * cfg['scale'])
            cfg['weight'] = float(g * cfg['weight'])
        for key in ('reference', 'candidate'):
            w[key] = [{k: (None if x is None
                           else float(a * x + b)) for k, x in row.items()}
                      for row in v[key]]
        equiv = register(w, exact=False)
        self.assertEqual(equiv['path_indices'], base['path_indices'])
        self.assertAlmostEqual(equiv['mean_cost'], base['mean_cost'],
                               places=9)


class ValidationTests(unittest.TestCase):
    def test_invalid_scale_rejected(self):
        import json
        from pathlib import Path
        v = json.loads(Path('data/invalid-domain.json')
                       .read_text(encoding='utf8'))
        with self.assertRaises(ValueError):
            register(v)

    def test_exact_mode_rejects_float_input(self):
        v = material_from_grid(GRID_33)
        v['reference'][0]['x'] = 0.1
        with self.assertRaises(ValueError):
            build_local_costs(v, exact=True)


if __name__ == '__main__':
    unittest.main()
