"""可运行的配准方法核对演示（无第三方依赖）。

在项目根目录执行：

    python -m response_data.demo

演示内容：
1. 小序列穷举所有合法路径，逐条列出比例并与 Dinkelbach/DP 胜者对照；
2. “先求最小总和再除长度”选错路径的反例；
3. 同分决胜规则（先短路径、再完整点对序列字典序）；
4. 通道同时重命名、按锁定参数等价换单位不改变配准；
5. 缺测不补零与全通道缺测时的覆盖损失报告。
"""
from fractions import Fraction

from .costs import build_local_costs
from .dp import enumerate_paths, minimum_sum_path, solve
from .register import register

# 反例代价表（3x3，单通道）：最小总和路径与最小平均路径分离
MISSELECT_GRID = [[0, 1, 0], [0, 2, 2], [0, 2, 0]]


def _grid_material(grid, run_limit=5, index_band=10):
    n, m = len(grid), len(grid[0])
    return {
        'channels': {'x': {'center': 0, 'scale': 1, 'weight': 1}},
        'reference': [{'x': 0} for _ in range(n)],
        'candidate': [{'x': 0} for _ in range(m)],
        'min_channels': 1, 'index_band': index_band,
        'run_limit': run_limit,
        'solver': {'tolerance': 1e-12, 'state_budget': 100_000},
    }


def _stub_costs(v, grid):
    lc = build_local_costs(v, exact=True)
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            lc.pairs[(i, j)].cost = Fraction(val)
            lc.pairs[(i, j)].usable = True
            lc.pairs[(i, j)].channels = ('x',)
    return lc


def demo_exhaustive():
    print('=' * 72)
    print('1. 小序列穷举所有合法路径，核对比例目标与同分规则')
    print('-' * 72)
    grid = MISSELECT_GRID
    v = _grid_material(grid)
    lc = _stub_costs(v, grid)
    ranked = enumerate_paths(lc)
    print(f'网格 {len(grid)}x{len(grid[0])}，合法路径 {len(ranked)} 条：')
    for k, r in enumerate(ranked, 1):
        print(f'  {k:2d}. 比例={str(r.ratio):>5}  长度={r.length}  '
              f'路径={[list(c) for c in r.path]}')
    out = solve(lc)
    winner = ranked[0]
    assert (out.result.ratio, out.result.length, out.result.path) == (
        winner.ratio, winner.length, winner.path)
    print(f'Dinkelbach 胜者：{[list(c) for c in out.result.path]} '
          f'比例={out.result.ratio}，与穷举榜首完全一致 ✓')


def demo_wrong_objective():
    print('=' * 72)
    print('2. “先求最小总和再除长度”会选错路径')
    print('-' * 72)
    v = _grid_material(MISSELECT_GRID)
    lc = _stub_costs(v, MISSELECT_GRID)
    minsum = minimum_sum_path(lc)
    opt = solve(lc).result
    print(f'最小总和路径：{[list(c) for c in minsum.path]}')
    print(f'  总和={minsum.cost}，长度={minsum.length}，'
          f'其平均={minsum.ratio}')
    print(f'最小平均路径：{[list(c) for c in opt.path]}')
    print(f'  总和={opt.cost}，长度={opt.length}，其平均={opt.ratio}')
    assert minsum.ratio > opt.ratio
    print(f'结论：{minsum.ratio} > {opt.ratio}，最小总和路径的平均代价'
          '严格更大，先求最小和再除长度选错 ✓')


def demo_tie_rules():
    print('=' * 72)
    print('3. 同分决胜：平均相同先短路径，再按完整 (i,j) 序列字典序')
    print('-' * 72)
    lc = _stub_costs(_grid_material([[0, 0, 0], [0, 0, 0]]),
                     [[0, 0, 0], [0, 0, 0]])
    ranked = enumerate_paths(lc)
    print('2x3 全零代价，所有路径比例均为 0，穷举榜前部：')
    for r in ranked[:4]:
        print(f'  长度={r.length}  路径={[list(c) for c in r.path]}')
    winner = solve(lc).result
    print(f'胜者：{[list(c) for c in winner.path]}（同比例最短，'
          '同长度字典序最小）✓')


def demo_invariance():
    print('=' * 72)
    print('4. 通道重命名 / 锁定参数等价换单位不改变配准')
    print('-' * 72)
    base_v = {
        'channels': {
            'coast': {'center': 0, 'scale': 10, 'weight': 2},
            'inland': {'center': 1, 'scale': 5, 'weight': 1}},
        'reference': [
            {'coast': 0, 'inland': 1}, {'coast': 5, 'inland': 3},
            {'coast': 20, 'inland': None}, {'coast': 3, 'inland': 2}],
        'candidate': [
            {'coast': 0, 'inland': 1}, {'coast': 2, 'inland': None},
            {'coast': 6, 'inland': 2}, {'coast': 18, 'inland': 4},
            {'coast': 4, 'inland': 2}],
        'min_channels': 1, 'index_band': 2, 'run_limit': 2,
        'solver': {'tolerance': 1e-10, 'state_budget': 100_000},
    }
    base = register(base_v)

    # (a) 同时重命名并交换通道顺序
    rename = {'coast': 'station_A', 'inland': 'station_B'}
    renamed = {
        'channels': {
            rename['inland']: base_v['channels']['inland'],
            rename['coast']: base_v['channels']['coast']},
        'reference': [{rename[k]: x for k, x in row.items()}
                      for row in base_v['reference']],
        'candidate': [{rename[k]: x for k, x in row.items()}
                      for row in base_v['candidate']],
        'min_channels': 1, 'index_band': 2, 'run_limit': 2,
        'solver': base_v['solver']}
    moved = register(renamed)
    print('(a) 通道同时重命名+换序：')
    print(f'  路径相同？{moved["path_indices"] == base["path_indices"]}，'
          f'平均代价相同？{moved["mean_cost"] == base["mean_cost"]} ✓')

    # (b) 等价换单位 x'=a*x+b（a>0），权重同乘正因子 g
    a, b, g = 3, 7, 4
    unit = {
        'channels': {
            c: {'center': a * cfg['center'] + b,
                'scale': a * cfg['scale'], 'weight': g * cfg['weight']}
            for c, cfg in base_v['channels'].items()},
        'reference': [{k: (None if x is None else a * x + b)
                       for k, x in row.items()}
                      for row in base_v['reference']],
        'candidate': [{k: (None if x is None else a * x + b)
                       for k, x in row.items()}
                      for row in base_v['candidate']],
        'min_channels': 1, 'index_band': 2, 'run_limit': 2,
        'solver': base_v['solver']}
    equiv = register(unit)
    same = (equiv['path_indices'] == base['path_indices']
            and equiv['mean_cost'] == base['mean_cost'])
    print(f'(b) x -> {a}x+{b}、scale 同比例、权重 x{g}：')
    print(f'  路径相同？{equiv["path_indices"] == base["path_indices"]}，'
          f'每个点对局部代价逐格相同？'
          f'{all(p0["local_cost"] == p1["local_cost"] for p0, p1 in zip(base["path"], equiv["path"]))} ✓')
    assert same


def demo_missing_support():
    print('=' * 72)
    print('5. 缺测不补零：全通道缺测仍可有路径，但须列覆盖损失')
    print('-' * 72)
    import json
    from pathlib import Path
    v = json.loads(Path('data/one-channel-missing.json')
                   .read_text(encoding='utf8'))
    r = register(v)
    print(f'状态={r["status"]}，路径长度={r["length"]}，'
          f'路径全程缺测通道={r["fully_missing_channels"]}')
    print('首个点对的有效通道与缺测通道：',
          r['path'][0]['effective_channels'],
          r['path'][0]['missing_channels'])
    print('复用统计：', r['reuse_counts'])
    assert r['fully_missing_channels'] == ['inland']
    assert all(rec['effective_channels'] == ['coast'] for rec in r['path'])


def main():
    demo_exhaustive()
    demo_wrong_objective()
    demo_tie_rules()
    demo_invariance()
    demo_missing_support()
    print('=' * 72)
    print('全部演示核对通过。')


if __name__ == '__main__':
    main()
