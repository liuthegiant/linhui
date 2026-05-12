#!/usr/bin/env python3
"""Collect TopoMoE prediction_scores.txt into per-config mean/std tables."""
from __future__ import annotations
import argparse, csv, glob, os, re, statistics as stats
from collections import defaultdict


def parse_line(line: str):
    p = [x.strip() for x in line.strip().split(',')]
    if not p or len(p) < 4:
        return None
    # Estimation wrapper: name, mode, Masked MAE, v, RMSE, v, MAPE, v
    if 'Masked MAE' in p:
        try:
            return {
                'task': 'estimation', 'name': p[0], 'mode': p[1],
                'MAE': float(p[p.index('Masked MAE') + 1]),
                'RMSE': float(p[p.index('RMSE') + 1]),
                'MAPE': float(p[p.index('MAPE') + 1]),
            }
        except Exception:
            return None
    # Forecast wrapper: all pred steps, name, mode, MSE, RMSE, MAE, MAPE, v, v, v, v
    if p[0] == 'all pred steps' and len(p) >= 11:
        try:
            return {
                'task': 'forecast', 'name': p[1], 'mode': p[2],
                'MSE': float(p[7]), 'RMSE': float(p[8]), 'MAE': float(p[9]), 'MAPE': float(p[10]),
            }
        except Exception:
            return None
    return None


def norm_config(tag: str):
    tag = re.sub(r'_seed\d+$', '', tag)
    tag = re.sub(r'^seed\d+_', '', tag)
    return tag


def mean_std(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return '', ''
    if len(vals) == 1:
        return f'{vals[0]:.4f}', '0.0000'
    return f'{stats.mean(vals):.4f}', f'{stats.stdev(vals):.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='logs_topomoe', help='log root to scan')
    ap.add_argument('--csv', default='', help='optional output CSV path')
    args = ap.parse_args()

    records = []
    for f in sorted(glob.glob(os.path.join(args.root, '**', '*_prediction_scores.txt'), recursive=True)):
        # keep latest record per task/mode within a score file, because reruns append
        latest = {}
        with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                rec = parse_line(line)
                if rec:
                    latest[(rec['task'], rec['mode'])] = rec
        for rec in latest.values():
            tag = os.path.basename(os.path.dirname(f))
            rec['run_tag'] = tag
            rec['config'] = norm_config(tag)
            rec['path'] = f
            records.append(rec)

    if not records:
        print(f'No parsable *_prediction_scores.txt found under {args.root!r}')
        return

    if args.csv:
        fields = ['task','mode','config','run_tag','MSE','RMSE','MAE','MAPE','path']
        with open(args.csv, 'w', newline='', encoding='utf-8') as fp:
            w = csv.DictWriter(fp, fieldnames=fields)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k, '') for k in fields})
        print('wrote', args.csv)

    groups = defaultdict(list)
    for r in records:
        groups[(r['task'], r['mode'], r['config'])].append(r)

    print('\n| task | mode | config | n | MAE | RMSE | MAPE | MSE |')
    print('|---|---|---|---:|---:|---:|---:|---:|')
    for key in sorted(groups):
        task, mode, cfg = key
        rs = groups[key]
        n = len(rs)
        m_mae, s_mae = mean_std([r['MAE'] for r in rs if 'MAE' in r])
        m_rmse, s_rmse = mean_std([r['RMSE'] for r in rs if 'RMSE' in r])
        m_mape, s_mape = mean_std([r['MAPE'] for r in rs if 'MAPE' in r])
        m_mse, s_mse = mean_std([r['MSE'] for r in rs if 'MSE' in r])
        def cell(m, s):
            return '' if m == '' else f'{m} ± {s}'
        print(f'| {task} | {mode} | {cfg} | {n} | {cell(m_mae,s_mae)} | {cell(m_rmse,s_rmse)} | {cell(m_mape,s_mape)} | {cell(m_mse,s_mse)} |')

if __name__ == '__main__':
    main()
