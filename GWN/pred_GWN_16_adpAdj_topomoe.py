from __future__ import annotations
import os, sys, time
from datetime import datetime
from typing import List
import numpy as np
import torch
import torch.nn as nn

import pred_GWN_16_adpAdj as base
from topo_moe_utils import SparseExpertFusion, add_moe_regularization, build_topology_embedding, default_moe_context, dense_adjacency_from_supports, save_alpha_report

_ORIG_GET_ARGV = base.get_argv
_ORIG_TRAIN = base.trainModel
_ORIG_TEST = base.testModel
MOE_MODES = {'sparse_moe', 'topo_moe'}


def _bool(x): return bool(int(x)) if not isinstance(x, bool) else x

def _arg(name, idx, default, cast):
    if os.environ.get(name, '') != '': return cast(os.environ[name])
    return cast(sys.argv[idx]) if len(sys.argv) > idx else default


_EXPERT_ALIASES = {
    'temporal': 'temporal', 'temp': 'temporal', 'tmp': 'temporal', 'scpt': 'temporal',
    'geometric': 'geometric', 'geo': 'geometric', 'geom': 'geometric',
    'topology': 'topology', 'topo': 'topology', 'lap': 'topology', 'laplacian': 'topology',
}


def _parse_experts(raw):
    parts = str(raw).replace('+', ',').replace('|', ',').split(',')
    out = []
    for p in parts:
        key = p.strip().lower()
        if not key:
            continue
        if key not in _EXPERT_ALIASES:
            raise ValueError(f'Unknown MOE expert {p!r}. Use temporal/geometric/topology, or aliases scpt/geo/topo.')
        val = _EXPERT_ALIASES[key]
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError('MOE_EXPERTS is empty. Example: MOE_EXPERTS=temporal,geometric,topology')
    return out


def _expert_arg(default_spec, argv_idx):
    raw = os.environ.get('MOE_EXPERTS', '').strip()
    if not raw and len(sys.argv) > argv_idx:
        raw = sys.argv[argv_idx].strip()
    if not raw:
        raw = default_spec
    return _parse_experts(raw)


def get_argv_topomoe():
    _ORIG_GET_ARGV()
    P = base.P
    # original pred_GWN_16_adpAdj.py consumes argv[1]..argv[34]
    P.TOPO_LAP_K = _arg('TOPO_LAP_K', 35, 16, int)
    P.MOE_TOP_K = _arg('MOE_TOP_K', 36, 2, int)
    P.MOE_TAU = _arg('MOE_TAU', 37, 1.0, float)
    P.MOE_LB_REG = _arg('MOE_LB_REG', 38, 1e-3, float)
    P.MOE_SMOOTH_REG = _arg('MOE_SMOOTH_REG', 39, 1e-3, float)
    P.MOE_DELTA_REG = _arg('MOE_DELTA_REG', 40, getattr(P, 'DELTA_REG', 0.0), float)
    P.MOE_USE_CTX = _arg('MOE_USE_CTX', 41, 1, _bool)
    default_experts = 'temporal,geometric,topology' if P.FUSION_MODE == 'topo_moe' else 'temporal,geometric'
    P.MOE_EXPERTS = _expert_arg(default_experts, 42)
    P.MOE_FORCE_EXPERT = os.environ.get('MOE_FORCE_EXPERT', '').strip()
    P.MOE_INIT_TEMPORAL_BIAS = float(os.environ.get('MOE_INIT_TEMPORAL_BIAS', '1.0'))
    P.TOPO_FORCE_RECOMPUTE = _bool(os.environ.get('TOPO_FORCE_RECOMPUTE', '0'))
    print('[TopoMoE forecast]', {k: getattr(P, k) for k in ['FUSION_MODE','MOE_EXPERTS','TOPO_LAP_K','MOE_TOP_K','MOE_TAU','MOE_LB_REG','MOE_SMOOTH_REG','MOE_DELTA_REG','MOE_USE_CTX']})


def _enabled():
    P = base.P
    return bool(P.IS_PRETRN) and P.PRETRN_MODE == 'dual' and P.FUSION_MODE in MOE_MODES


def _A_np():
    return dense_adjacency_from_supports(base.load_adj(base.P.ADJPATH, base.P.ADJTYPE, base.P.DATANAME), symmetrize=True)


def _topo_embed(D):
    P = base.P
    A = _A_np()
    path = os.path.join(P.PATH, f'topology_embed_D{D}_lap{P.TOPO_LAP_K}.npz')
    Z = build_topology_embedding(A, embed_dim=D, lap_k=P.TOPO_LAP_K, cache_path=path, force_recompute=P.TOPO_FORCE_RECOMPUTE)
    return torch.tensor(Z, dtype=torch.float32, device=base.device).detach()


def _inputs():
    P = base.P
    # We load the two pretrained branches to keep dimensions identical to the original dual-pretraining path.
    temp = base._get_temporal_full_embed().detach()
    geo = base._get_geometric_full_embed().detach()
    A = _A_np()

    pool = {
        'temporal': temp,
        'geometric': geo,
    }
    if 'topology' in P.MOE_EXPERTS:
        pool['topology'] = _topo_embed(temp.shape[0])

    experts, names = [], []
    for name in P.MOE_EXPERTS:
        if name not in pool:
            raise ValueError(f'Expert {name!r} is unavailable. Available experts: {sorted(pool)}')
        experts.append(pool[name])
        names.append(name)

    ctx = torch.tensor(default_moe_context(A, has_temporal=('temporal' in names)), dtype=torch.float32, device=base.device) if P.MOE_USE_CTX else None
    adj_dense = torch.tensor(A, dtype=torch.float32, device=base.device)
    return experts, names, ctx, adj_dense


def _make(experts, names, ctx):
    P = base.P
    return SparseExpertFusion(embed_dim=experts[0].shape[0], n_experts=len(experts), ctx_dim=0 if ctx is None else ctx.shape[1],
                              hidden_dim=P.GATE_HIDDEN, top_k=P.MOE_TOP_K, temperature=P.MOE_TAU,
                              expert_names=names, init_temporal_bias=P.MOE_INIT_TEMPORAL_BIAS).to(base.device)


def _run(fusion, experts, ctx, adj_dense, return_aux=True):
    P = base.P
    avail = None
    if P.MOE_FORCE_EXPERT:
        if P.MOE_FORCE_EXPERT not in fusion.expert_names:
            raise ValueError(f'MOE_FORCE_EXPERT={P.MOE_FORCE_EXPERT} not in {fusion.expert_names}')
        idx = fusion.expert_names.index(P.MOE_FORCE_EXPERT)
        avail = torch.zeros((experts[0].shape[1], len(experts)), dtype=torch.float32, device=base.device)
        avail[:, idx] = 1.0
    smooth_adj = adj_dense if P.MOE_SMOOTH_REG > 0 else None
    return fusion(experts, ctx=ctx, avail_mask=avail, adj_dense=smooth_adj, return_aux=return_aux)


def trainModel_topomoe(name, mode, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a, spatialSplit_unseen, spatialSplit_allNod):
    if not _enabled():
        return _ORIG_TRAIN(name, mode, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a, spatialSplit_unseen, spatialSplit_allNod)
    P = base.P
    print('trainModel Started with Sparse/Topo MoE ...', time.ctime())
    model = base.getModel(name, base.device)
    criterion = nn.L1Loss()
    experts, names, ctx, adj_dense = _inputs()
    fusion = _make(experts, names, ctx)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(fusion.parameters()), lr=P.LEARN, weight_decay=P.weight_decay)
    with torch.no_grad():
        full, aux = _run(fusion, experts, ctx, adj_dense, True)
    train_embed = full[:, spatialSplit_unseen.i_trn]
    print('experts', names, 'initial alpha mean', aux['alpha'].mean(0).detach().cpu().numpy())
    min_u, min_a = np.inf, np.inf
    epoch_secs, batch_secs = [], []
    s_time = datetime.now()
    for epoch in range(P.EPOCH):
        st = datetime.now(); loss_sum = 0.0; n = 0
        model.train(); fusion.train()
        for x, y in train_iter:
            tb = time.perf_counter(); optimizer.zero_grad()
            full, aux = _run(fusion, experts, ctx, adj_dense, True)
            train_embed = full[:, spatialSplit_unseen.i_trn]
            pred = model(x.to(base.device), adj_train, train_embed)
            task_loss = criterion(pred, y.to(base.device))
            loss = add_moe_regularization(task_loss, aux, P)
            loss.backward(); optimizer.step()
            loss_sum += task_loss.item() * y.shape[0]; n += y.shape[0]
            batch_secs.append(time.perf_counter() - tb)
        train_loss = loss_sum / max(n, 1)
        model.eval(); fusion.eval()
        with torch.no_grad():
            full, aux_eval = _run(fusion, experts, ctx, adj_dense, True)
            train_embed = full[:, spatialSplit_unseen.i_trn]
            val_u_embed = full[:, spatialSplit_unseen.i_val]
            val_a_embed = full[:, spatialSplit_allNod.i_val]
            val_u_loss = base.evaluateModel(model, criterion, val_u_iter, adj_val_u, val_u_embed)
            val_a_loss = base.evaluateModel(model, criterion, val_a_iter, adj_val_a, val_a_embed)
        if val_u_loss < min_u:
            min_u = val_u_loss
            torch.save(model.state_dict(), f'{P.PATH}/{name}_u.pt')
            torch.save(fusion.state_dict(), f'{P.PATH}/{name}_fusion_u.pt')
            save_alpha_report(P.PATH, f'{name}_u', aux_eval['alpha'], names)
        if val_a_loss < min_a:
            min_a = val_a_loss
            torch.save(model.state_dict(), f'{P.PATH}/{name}_a.pt')
            torch.save(fusion.state_dict(), f'{P.PATH}/{name}_fusion_a.pt')
            save_alpha_report(P.PATH, f'{name}_a', aux_eval['alpha'], names)
        epoch_secs.append((datetime.now() - st).total_seconds())
        am = aux_eval['alpha'].mean(0).detach().cpu().numpy()
        print('epoch', epoch, 'time', int(epoch_secs[-1]), 'train', train_loss, 'val_u', val_u_loss, 'val_a', val_a_loss, 'alpha_mean', am,
              'lb', aux_eval['load_balance_loss'].item(), 'smooth', aux_eval['smooth_loss'].item(), 'delta', aux_eval['delta_l2'].item())
        with open(f'{P.PATH}/{name}_log.txt', 'a') as f:
            f.write(f'epoch,{epoch},train_loss,{train_loss:.10f},val_u_loss,{val_u_loss:.10f},val_a_loss,{val_a_loss:.10f},alpha_mean,{am.tolist()}\n')
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', e_time - s_time)
    if hasattr(base, 'PAPER_TIMING'):
        base.PAPER_TIMING['main_train_sec'] = float((e_time - s_time).total_seconds())
        base.PAPER_TIMING['main_train_epoch_time_mean_sec'] = float(sum(epoch_secs) / len(epoch_secs)) if epoch_secs else 0.0
        base.PAPER_TIMING['main_train_epoch_sec_list'] = epoch_secs
        base._put_iter_stats('main_train_iter', batch_secs)
        base.PAPER_TIMING['forecast_model_params'] = base._trainable_params(model)
        base.PAPER_TIMING['fusion_module_params'] = base._trainable_params(fusion)
    score = base.evaluateModel(model, criterion, train_iter, adj_train, train_embed)
    with open(f'{P.PATH}/{name}_prediction_scores.txt', 'a') as f:
        f.write('%s, %s, %s, %.10e, %.10f\n' % (name, mode, 'MAE on train', score, score))
    print('min_val_u_loss', min_u, 'min_val_a_loss', min_a)


def testModel_topomoe(name, mode, test_iter, adj_tst, spatialsplit):
    if not _enabled():
        return _ORIG_TEST(name, mode, test_iter, adj_tst, spatialsplit)
    P = base.P
    criterion = nn.L1Loss(); t_all = time.perf_counter()
    print('Model Testing', mode, 'with Sparse/Topo MoE ...', time.ctime())
    model = base.getModel(name, base.device)
    model.load_state_dict(torch.load(f'{P.PATH}/{name}{mode[-2:]}.pt', map_location=base.device)); model.eval()
    experts, names, ctx, adj_dense = _inputs()
    fusion = _make(experts, names, ctx)
    ckpt = f'{P.PATH}/{name}_fusion{mode[-2:]}.pt'
    if os.path.exists(ckpt): fusion.load_state_dict(torch.load(ckpt, map_location=base.device))
    else: print('fusion checkpoint not found:', ckpt)
    fusion.eval()
    with torch.no_grad():
        full, aux = _run(fusion, experts, ctx, adj_dense, True)
        tst_embed = full[:, spatialsplit.i_tst]
        save_alpha_report(P.PATH, f'{name}_{mode}', aux['alpha'], names)
    t0 = time.perf_counter()
    torch_score, eval_times = base.evaluateModel(model, criterion, test_iter, adj_tst, tst_embed, return_batch_times=True)
    t1 = time.perf_counter()
    YS_pred, pred_times = base.predictModel(model, test_iter, adj_tst, tst_embed, return_batch_times=True)
    t2 = time.perf_counter()
    if hasattr(base, '_put_iter_stats'):
        base._put_iter_stats(f'{mode}_eval_batch', eval_times); base._put_iter_stats(f'{mode}_predict_batch', pred_times)
    YS = test_iter.dataset.tensors[1].cpu().numpy()
    shape = np.squeeze(YS).shape
    YS = base.scaler.inverse_transform(np.squeeze(YS).reshape(-1, YS.shape[2])).reshape(shape)
    YS_pred = base.scaler.inverse_transform(np.squeeze(YS_pred).reshape(-1, YS_pred.shape[2])).reshape(shape)
    np.save(f'{P.PATH}/{P.MODELNAME}_{mode}_{name}_prediction.npy', YS_pred)
    np.save(f'{P.PATH}/{P.MODELNAME}_{mode}_{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = base.Metrics.evaluate(YS, YS_pred)
    with open(f'{P.PATH}/{name}_prediction_scores.txt', 'a') as f:
        print('all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f' % (name, mode, MSE, RMSE, MAE, MAPE))
        f.write('%s, %s, Torch MSE, %.10e, %.10f\n' % (name, mode, torch_score, torch_score))
        f.write('all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n' % (name, mode, MSE, RMSE, MAE, MAPE))
        for i in range(P.TIMESTEP_OUT):
            m = base.Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
            f.write('%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n' % (i + 1, name, mode, *m))
    if hasattr(base, 'PAPER_TIMING'):
        n_test = len(test_iter.dataset); pred_sec = max(t2 - t1, 1e-12)
        base.PAPER_TIMING[f'{mode}_eval_forward_sec'] = float(t1 - t0)
        base.PAPER_TIMING[f'{mode}_predict_forward_sec'] = float(t2 - t1)
        base.PAPER_TIMING[f'{mode}_predict_throughput_samples_per_s'] = float(n_test / pred_sec)
        base.PAPER_TIMING[f'{mode}_num_test_samples'] = int(n_test)
        base.PAPER_TIMING[f'{mode}_all_sec'] = float(time.perf_counter() - t_all)
    print('alpha mean', aux['alpha'].mean(0).detach().cpu().numpy(), 'experts', names)


base.get_argv = get_argv_topomoe
base.trainModel = trainModel_topomoe
base.testModel = testModel_topomoe

if __name__ == '__main__':
    base.main()
