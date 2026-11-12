"""资料摘要与受限最小平均代价配准的命令行入口。"""
import argparse
import json

from .io import load
from .domain import summarize
from .register import brute_force_check, register


def _json(x):
    return json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False,
                      default=str)


def main():
    p = argparse.ArgumentParser(
        description='多站雨型受限最小平均代价配准（离线演练资料）')
    p.add_argument('command', choices=['inspect', 'register', 'bruteforce'])
    p.add_argument('file')
    p.add_argument('--float', action='store_true',
                   help='强制浮点模式（默认整数资料走精确有理模式）')
    a = p.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        print(_json({'source_sha256': digest, 'summary': summarize(v)}))
    elif a.command == 'register':
        report = register(v, exact=False if a.float else None)
        report['source_sha256'] = digest
        print(_json(report))
    else:
        check = brute_force_check(v)
        check['source_sha256'] = digest
        print(_json(check))


if __name__ == '__main__':
    main()
