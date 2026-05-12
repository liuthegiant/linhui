#!/usr/bin/env python3
"""Print MoE router alpha summaries from TopoMoE runs."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def _fmt(xs):
    return '[' + ', '.join(f'{float(x):.4f}' for x in xs) + ']'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='logs_topomoe', help='log root to scan')
    ap.add_argument('--limit', type=int, default=0, help='optional max number of files')
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, '**', '*_moe_alpha_summary.json'), recursive=True))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f'No *_moe_alpha_summary.json found under {args.root!r}')
        return
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            s = json.load(fp)
        names = s.get('expert_names', [])
        mean = s.get('mean', [])
        std = s.get('std', [])
        print('\n====', f, '====')
        print('experts:', names)
        print('mean:   ', _fmt(mean))
        print('std:    ', _fmt(std))
        alpha_file = f.replace('_moe_alpha_summary.json', '_moe_alpha.npy')
        if os.path.exists(alpha_file):
            a = np.load(alpha_file)
            winner = a.argmax(axis=1)
            ratio = np.bincount(winner, minlength=len(names)) / max(len(winner), 1)
            entropy = float(-(a * np.log(a + 1e-8)).sum(axis=1).mean())
            print('argmax: ', _fmt(ratio))
            print('entropy:', f'{entropy:.4f}')

if __name__ == '__main__':
    main()
